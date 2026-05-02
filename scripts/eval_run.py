"""Run a golden-set evaluation against a live Ollama+Qdrant corpus.

Run:
  uv run python -m scripts.eval_run --pdf data/papers/<file>.pdf \
      --golden data/golden/v1.yaml

For contextual retrieval (Anthropic-style situating blurbs prepended to each
chunk before embedding/BM25), pick a provider:

  # Cloud (OpenRouter, requires RAG_OPENROUTER_API_KEY):
  uv run python -m scripts.eval_run --pdf <pdf> --contextualize \
      --contextualize-provider openrouter \
      --contextualize-model openai/gpt-4o-mini

  # Local (Ollama, no API key — needs `ollama pull <model>` first):
  uv run python -m scripts.eval_run --pdf <pdf> --contextualize \
      --contextualize-provider ollama \
      --contextualize-model qwen2.5:7b

Requires `docker compose up -d qdrant ollama` and `ollama pull bge-m3`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.eval.golden_set import load_golden_set
from src.eval.judges import LLMJudge
from src.eval.report import write_run_json, write_run_markdown
from src.eval.runner import evaluate
from src.eval.storage import make_engine, write_eval_run
from src.ingestion.captioner import OllamaVisionCaptioner
from src.ingestion.pipeline import ingest_paper
from src.llm.ollama_chat import OllamaChatClient
from src.llm.openrouter import OpenRouterClient
from src.llm.protocol import LLMClient
from src.observability.logging import configure_logging, get_logger
from src.prompts.loader import load_prompt_by_name
from src.rag.bm25 import Bm25Index
from src.rag.generate import Generator
from src.rag.query_expansion import QueryExpander
from src.rag.rerank import BgeReranker
from src.rag.retrievers.multi_query import ExpansionMode, MultiQueryRetriever
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.protocol import Retriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Paper


def _build_llm(provider: str, *, ollama_url: str, num_ctx: int | None = None) -> LLMClient:
    """Construct an LLMClient by provider name. Used for generator + judge."""
    if provider == "openrouter":
        api_key = os.environ.get("RAG_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "openrouter provider requires RAG_OPENROUTER_API_KEY (or OPENROUTER_API_KEY)"
            )
        return OpenRouterClient(api_key=api_key)
    if provider == "ollama":
        return OllamaChatClient(base_url=ollama_url, num_ctx=num_ctx)
    raise SystemExit(f"unknown provider: {provider!r} (expected 'openrouter' or 'ollama')")


async def _main(
    *,
    pdf_paths: list[Path],
    golden_path: Path,
    qdrant_url: str,
    ollama_url: str,
    output_dir: Path,
    top_k: int,
    collection: str,
    contextualize: bool,
    contextualize_provider: str,
    contextualize_model: str,
    contextualize_concurrency: int,
    contextualize_num_ctx: int | None,
    generate: bool,
    generator_provider: str,
    generator_model: str,
    refusal_score_threshold: float | None,
    judge: bool,
    judge_provider: str,
    judge_model: str,
    judge_num_ctx: int | None,
    rerank: bool,
    rerank_model: str,
    rerank_input_size: int,
    rerank_device: str | None,
    postgres_dsn: str | None,
    extract_figures: bool,
    extract_tables: bool,
    vlm_caption_model: str | None,
    query_expansion: bool,
    query_expansion_mode: str,
    query_expansion_provider: str,
    query_expansion_model: str,
    query_expansion_n: int,
) -> None:
    log = get_logger("scripts.eval_run")
    log.info(
        "eval_cli.start",
        pdfs=[str(p) for p in pdf_paths],
        golden=str(golden_path),
        contextualize=contextualize,
    )

    embedder = OllamaBgeEmbedder(base_url=ollama_url)
    vectorstore = QdrantVectorStore(url=qdrant_url, collection_name=collection, dim=embedder.dim)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    contextualizer_llm: LLMClient | None = None
    if contextualize:
        contextualizer_llm = _build_llm(
            contextualize_provider, ollama_url=ollama_url, num_ctx=contextualize_num_ctx
        )
        print(f"Contextualizing chunks via {contextualize_provider} with {contextualize_model}")

    vlm_captioner_obj: OllamaVisionCaptioner | None = None
    if vlm_caption_model:
        vlm_captioner_obj = OllamaVisionCaptioner(base_url=ollama_url, model=vlm_caption_model)
        print(f"VLM captioning figures with {vlm_caption_model}")

    chunks_by_id: dict[str, Chunk] = {}
    paper_ids: list[str] = []
    for pdf_path in pdf_paths:
        paper = Paper(paper_id=pdf_path.stem, title=pdf_path.stem, pdf_path=pdf_path)
        ingested = await ingest_paper(
            paper=paper,
            embedder=embedder,
            vectorstore=vectorstore,
            bm25=bm25,
            contextualizer_llm=contextualizer_llm,
            contextualizer_model=contextualize_model if contextualize else None,
            contextualizer_concurrency=contextualize_concurrency,
            extract_figures_enabled=extract_figures,
            extract_tables_enabled=extract_tables,
            vlm_captioner=vlm_captioner_obj,
        )
        for chunk in ingested.chunks:
            chunks_by_id[chunk.chunk_id] = chunk
        paper_ids.append(paper.paper_id)
        print(f"Ingested {ingested.chunk_count} chunks from {pdf_path.name}")
        if contextualize:
            with_ctx = sum(1 for c in ingested.chunks if c.context)
            print(f"Contextualized {with_ctx}/{ingested.chunk_count} chunks")

    reranker_obj: BgeReranker | None = None
    if rerank:
        reranker_obj = BgeReranker(model_name=rerank_model, device=rerank_device)
        print(
            f"Reranking top-{rerank_input_size} with {rerank_model} (device={rerank_device or 'auto'})"
        )

    pipeline_retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        reranker=reranker_obj,
        rerank_input_size=rerank_input_size,
    )

    retriever: Retriever = pipeline_retriever
    if query_expansion:
        # CLI's `choices=("rewrite","hyde","combo")` already validates the value;
        # the cast is safe — Literal narrowing isn't supported on tuple choices.
        mode_typed: ExpansionMode = query_expansion_mode  # type: ignore[assignment]
        qe_llm = _build_llm(query_expansion_provider, ollama_url=ollama_url)
        expander = QueryExpander(llm=qe_llm, model=query_expansion_model)
        retriever = MultiQueryRetriever(
            base=pipeline_retriever,
            expander=expander,
            mode=mode_typed,
            n_rewrites=query_expansion_n,
        )
        print(
            f"Query expansion via {query_expansion_provider} {query_expansion_model} "
            f"(mode={query_expansion_mode}, n_rewrites={query_expansion_n})"
        )

    golden_set = load_golden_set(golden_path)
    print(
        f"Loaded golden set {golden_set.name} {golden_set.version} ({len(golden_set.queries)} queries)"
    )

    generator_obj: Generator | None = None
    if generate:
        generator_llm = _build_llm(generator_provider, ollama_url=ollama_url)
        generator_obj = Generator(
            llm=generator_llm,
            prompt=load_prompt_by_name("answer"),
            model=generator_model,
            refusal_score_threshold=refusal_score_threshold,
        )
        print(f"Generating answers via {generator_provider} with {generator_model}")

    judge_obj: LLMJudge | None = None
    if judge:
        judge_llm = _build_llm(judge_provider, ollama_url=ollama_url, num_ctx=judge_num_ctx)
        judge_obj = LLMJudge(
            llm=judge_llm,
            model=judge_model,
            faithfulness_prompt=load_prompt_by_name("judge_faithfulness"),
            answer_relevance_prompt=load_prompt_by_name("judge_answer_relevance"),
            context_precision_prompt=load_prompt_by_name("judge_context_precision"),
        )
        print(f"Judging answers via {judge_provider} with {judge_model}")

    run = await evaluate(
        retriever=retriever,
        golden_set=golden_set,
        generator=generator_obj,
        judge=judge_obj,
        top_k=top_k,
        config={
            "retriever": "pipeline",
            "rerank": rerank,
            "rerank_model": rerank_model if rerank else None,
            "rerank_input_size": rerank_input_size if rerank else None,
            "top_k": top_k,
            "paper_ids": paper_ids,
            "embedding_model": "bge-m3",
            "embedding_dim": embedder.dim,
            "contextualize": contextualize,
            "contextualize_provider": contextualize_provider if contextualize else None,
            "contextualize_model": contextualize_model if contextualize else None,
            "contextualize_num_ctx": contextualize_num_ctx if contextualize else None,
            "generate": generate,
            "generator_provider": generator_provider if generate else None,
            "generator_model": generator_model if generate else None,
            "refusal_score_threshold": refusal_score_threshold,
            "judge": judge,
            "judge_provider": judge_provider if judge else None,
            "judge_model": judge_model if judge else None,
            "judge_num_ctx": judge_num_ctx if judge else None,
            "extract_figures": extract_figures,
            "extract_tables": extract_tables,
            "vlm_caption_model": vlm_caption_model,
            "query_expansion": query_expansion,
            "query_expansion_mode": query_expansion_mode if query_expansion else None,
            "query_expansion_model": query_expansion_model if query_expansion else None,
            "query_expansion_n": query_expansion_n if query_expansion else None,
        },
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"run-{timestamp}.json"
    md_path = output_dir / f"run-{timestamp}.md"
    write_run_json(run, json_path)
    write_run_markdown(run, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if postgres_dsn:
        engine = make_engine(postgres_dsn)
        write_eval_run(run, engine=engine)
        engine.dispose()
        print(f"Stored run {run.run_id} in {postgres_dsn.rsplit('@', 1)[-1]}")

    log.info("eval_cli.done", run_id=run.run_id, json=str(json_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a golden-set evaluation end-to-end.")
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        nargs="+",
        help="One or more PDFs. All chunks are ingested into a unified corpus.",
    )
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v1.yaml"))
    parser.add_argument("--qdrant", default="http://localhost:6333")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/runs"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collection", default="eval_phase1")
    parser.add_argument(
        "--contextualize",
        action="store_true",
        help="Run Anthropic-style contextual retrieval (LLM blurb per chunk).",
    )
    parser.add_argument(
        "--contextualize-provider",
        choices=("openrouter", "ollama"),
        default="openrouter",
        help="LLM provider for contextualization. 'ollama' is local (no API key).",
    )
    parser.add_argument(
        "--contextualize-model",
        default=None,
        help=(
            "Model used for contextualization. Defaults: "
            "'openai/gpt-4o-mini' (openrouter), 'qwen2.5:7b' (ollama)."
        ),
    )
    parser.add_argument(
        "--contextualize-concurrency",
        type=int,
        default=4,
        help="Concurrent in-flight contextualizer LLM calls.",
    )
    parser.add_argument(
        "--contextualize-num-ctx",
        type=int,
        default=None,
        help=(
            "Override Ollama context window (num_ctx). Default 4096 truncates "
            "long paper prompts; bump to 8192/16384 for cleaner contextual signal "
            "(VRAM-bound). Ignored for openrouter provider."
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate answers per query with an LLM (citations parsed from output).",
    )
    parser.add_argument(
        "--generator-provider",
        choices=("openrouter", "ollama"),
        default="ollama",
        help="LLM provider for answer generation.",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Generator model. Defaults: 'openai/gpt-4o-mini' (openrouter), 'qwen2.5:7b' (ollama).",
    )
    parser.add_argument(
        "--refusal-score-threshold",
        type=float,
        default=None,
        help=(
            "If set, Generator refuses (returns a zero-citation 'cannot answer' Answer) "
            "when ALL top-K retrieved chunks have rerank score < this value. "
            "Empirically calibrated per ADR 0006. Default: off (no gate)."
        ),
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Score generations with LLM-as-judge metrics (faithfulness, answer "
            "relevance, context precision). Requires --generate for the first two; "
            "context_precision works without it."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        choices=("openrouter", "ollama"),
        default="ollama",
        help="LLM provider for judging.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model. Defaults: 'openai/gpt-4o-mini' (openrouter), 'qwen2.5:7b' (ollama).",
    )
    parser.add_argument(
        "--judge-num-ctx",
        type=int,
        default=None,
        help="Override Ollama num_ctx for the judge. Bump for long context_precision prompts.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Enable cross-encoder reranking on top of hybrid (BM25+dense+RRF). "
            "Standard SOTA RAG pattern: retrieve large pool, rerank to top-K."
        ),
    )
    parser.add_argument(
        "--rerank-model",
        default="BAAI/bge-reranker-v2-m3",
        help="HF cross-encoder model name. Default is the BGE m3 reranker (~600 MB).",
    )
    parser.add_argument(
        "--rerank-input-size",
        type=int,
        default=50,
        help="Top-N RRF candidates fed into the reranker. More = better recall, slower.",
    )
    parser.add_argument(
        "--rerank-device",
        default=None,
        help="Device for the reranker ('cuda', 'cpu'). Default: auto (cuda if available).",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=os.environ.get("RAG_POSTGRES_DSN"),
        help=(
            "Postgres DSN to persist the EvalRun. Defaults to env RAG_POSTGRES_DSN. "
            "Use a full SQLAlchemy URL: 'postgresql+psycopg://rag:rag@localhost:5432/rag'. "
            "Pass empty string '' to skip storage even when env is set."
        ),
    )
    parser.add_argument(
        "--extract-figures",
        action="store_true",
        help="Phase 2: extract embedded figures via PyMuPDF and add as Chunks (caption-text-only, no VLM yet).",
    )
    parser.add_argument(
        "--extract-tables",
        action="store_true",
        help="Phase 2: extract tables via PyMuPDF and add as Chunks (markdown-rendered).",
    )
    parser.add_argument(
        "--vlm-caption-model",
        default=None,
        help=(
            "Phase 2.1: vision Ollama model used to caption extracted figures "
            "(e.g. 'gemma3:4b', 'qwen2.5vl:7b', 'llava-llama3:8b'). When set, "
            "Figure.vlm_caption is filled and figure_to_chunk uses it as the "
            "indexable text. Requires --extract-figures."
        ),
    )
    parser.add_argument(
        "--query-expansion",
        action="store_true",
        help=(
            "Phase 2.2: wrap PipelineRetriever in MultiQueryRetriever — generate "
            "query variants via LLM, retrieve for each, fuse with RRF. Helps on "
            "multi-hop and term-mismatch queries."
        ),
    )
    parser.add_argument(
        "--query-expansion-mode",
        choices=("rewrite", "hyde", "combo"),
        default="rewrite",
        help=(
            "rewrite=LLM paraphrases of the query; hyde=hypothetical answer passage "
            "embedded for retrieval; combo=both."
        ),
    )
    parser.add_argument(
        "--query-expansion-provider",
        choices=("openrouter", "ollama"),
        default="ollama",
        help="LLM provider for the expander.",
    )
    parser.add_argument(
        "--query-expansion-model",
        default=None,
        help="Expander model. Defaults: 'openai/gpt-4o-mini' (openrouter), 'qwen2.5:7b' (ollama).",
    )
    parser.add_argument(
        "--query-expansion-n",
        type=int,
        default=3,
        help="Number of rewrites to request (rewrite/combo modes). Default 3.",
    )
    args = parser.parse_args()

    _provider_default_model = {
        "openrouter": "openai/gpt-4o-mini",
        "ollama": "qwen2.5:7b",
    }
    contextualize_model = (
        args.contextualize_model or _provider_default_model[args.contextualize_provider]
    )
    generator_model = args.generator_model or _provider_default_model[args.generator_provider]
    judge_model = args.judge_model or _provider_default_model[args.judge_provider]
    query_expansion_model = (
        args.query_expansion_model or _provider_default_model[args.query_expansion_provider]
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = Path("logs") / f"eval-{timestamp}.log"
    configure_logging(level="INFO", env="local", log_file=log_file)
    print(f"Logging JSON to {log_file}")

    asyncio.run(
        _main(
            pdf_paths=args.pdf,
            golden_path=args.golden,
            qdrant_url=args.qdrant,
            ollama_url=args.ollama,
            output_dir=args.output_dir,
            top_k=args.top_k,
            collection=args.collection,
            contextualize=args.contextualize,
            contextualize_provider=args.contextualize_provider,
            contextualize_model=contextualize_model,
            contextualize_concurrency=args.contextualize_concurrency,
            contextualize_num_ctx=args.contextualize_num_ctx,
            generate=args.generate,
            generator_provider=args.generator_provider,
            generator_model=generator_model,
            refusal_score_threshold=args.refusal_score_threshold,
            judge=args.judge,
            judge_provider=args.judge_provider,
            judge_model=judge_model,
            judge_num_ctx=args.judge_num_ctx,
            rerank=args.rerank,
            rerank_model=args.rerank_model,
            rerank_input_size=args.rerank_input_size,
            rerank_device=args.rerank_device,
            postgres_dsn=args.postgres_dsn or None,
            extract_figures=args.extract_figures,
            extract_tables=args.extract_tables,
            vlm_caption_model=args.vlm_caption_model,
            query_expansion=args.query_expansion,
            query_expansion_mode=args.query_expansion_mode,
            query_expansion_provider=args.query_expansion_provider,
            query_expansion_model=query_expansion_model,
            query_expansion_n=args.query_expansion_n,
        )
    )
