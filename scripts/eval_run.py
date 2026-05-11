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
from src.ingestion.captioner import OllamaVisionCaptioner, OpenRouterVisionCaptioner
from src.ingestion.pipeline import ingest_paper
from src.ingestion.visual import render_pages
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
from src.rag.retrievers.routing import RoutingRetriever
from src.rag.retrievers.visual import build_visual_retriever
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
    paper_id_filter: bool = False,
    region_number_boost: bool = False,
    rerank_length_norm: bool = False,
    rerank_length_threshold: int = 300,
    rerank_length_penalty: float = 0.5,
    vlm_caption_provider: str = "ollama",
    cascade: bool = False,
    cascade_threshold: float | None = None,
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
    judge_n_samples: int,
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
    router: bool,
    visual_model: str,
    visual_device: str,
    pages_dir: Path,
    pages_dpi: int,
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

    vlm_captioner_obj: OllamaVisionCaptioner | OpenRouterVisionCaptioner | None = None
    if vlm_caption_model:
        if vlm_caption_provider == "openrouter":
            api_key = os.environ.get("RAG_OPENROUTER_API_KEY") or os.environ.get(
                "OPENROUTER_API_KEY"
            )
            if not api_key:
                raise SystemExit(
                    "--vlm-caption-provider=openrouter requires RAG_OPENROUTER_API_KEY "
                    "(or OPENROUTER_API_KEY) in the environment."
                )
            vlm_captioner_obj = OpenRouterVisionCaptioner(api_key=api_key, model=vlm_caption_model)
        else:
            vlm_captioner_obj = OllamaVisionCaptioner(base_url=ollama_url, model=vlm_caption_model)
        print(f"VLM captioning figures with {vlm_caption_provider}/{vlm_caption_model}")

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
        reranker_obj = BgeReranker(
            model_name=rerank_model,
            device=rerank_device,
            length_norm=rerank_length_norm,
            length_threshold=rerank_length_threshold,
            length_penalty=rerank_length_penalty,
        )
        suffix = (
            f" + length_norm(threshold={rerank_length_threshold}, penalty={rerank_length_penalty})"
            if rerank_length_norm
            else ""
        )
        print(
            f"Reranking top-{rerank_input_size} with {rerank_model} "
            f"(device={rerank_device or 'auto'}){suffix}"
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

    if router:
        # ADR 0008 router: wrap the text retriever (already query-expanded if
        # requested) with RoutingRetriever so figure/table/multi_hop queries
        # get RRF-fused with the visual leg at page granularity. Render PDF
        # pages first (idempotent — render_pages caches), then build ColQwen2
        # page embeddings (slow, GPU-heavy).
        pages_by_paper: dict[str, list[tuple[int, Path]]] = {}
        for pdf_path in pdf_paths:
            paper_id = pdf_path.stem
            rendered = render_pages(paper_id, pdf_path, out_dir=pages_dir, dpi=pages_dpi)
            pages_by_paper[paper_id] = [(p.page_number, p.image_path) for p in rendered]
        print(
            f"Building visual retriever ({visual_model}, device={visual_device}) "
            f"over {sum(len(v) for v in pages_by_paper.values())} pages..."
        )
        # On shared 8 GB GPUs anything loaded in Ollama (bge-m3 embedder ~1.2 GB,
        # gemma3:4b VLM ~3.3 GB if --vlm-caption-model was active, etc.) competes
        # with ColQwen2's bf16 load and segfaults rather than OOMs cleanly. Query
        # /api/ps for all currently-resident models, then evict each via the
        # appropriate endpoint (embeddings for embedders, generate for chat/VLM
        # models — both honor keep_alive: 0). Only matters on cuda.
        if visual_device.startswith("cuda"):
            import httpx as _httpx  # local import keeps module-level deps clean

            try:
                async with _httpx.AsyncClient(timeout=10.0) as _ev_client:
                    ps = await _ev_client.get(f"{ollama_url}/api/ps")
                    loaded = [m["name"] for m in (ps.json().get("models") or [])]
                    for model_name in loaded:
                        # bge-m3 is an embedder; everything else is chat/VLM.
                        # Both endpoints accept keep_alive=0 to force eviction.
                        if "bge-m3" in model_name or "embed" in model_name:
                            await _ev_client.post(
                                f"{ollama_url}/api/embeddings",
                                json={"model": model_name, "prompt": "x", "keep_alive": 0},
                            )
                        else:
                            await _ev_client.post(
                                f"{ollama_url}/api/generate",
                                json={
                                    "model": model_name,
                                    "prompt": "x",
                                    "stream": False,
                                    "keep_alive": 0,
                                    "options": {"num_predict": 1},
                                },
                            )
                if loaded:
                    print(
                        f"Evicted {len(loaded)} Ollama model(s) ({', '.join(loaded)}) "
                        "before ColQwen2 cuda load."
                    )
                else:
                    print("No Ollama models resident; skipping eviction step.")
            except _httpx.HTTPError as e:
                print(f"Ollama eviction skipped ({e!r}); continuing.")
        visual_retriever = await build_visual_retriever(
            pages_by_paper, model_name=visual_model, device=visual_device
        )
        if cascade:
            if cascade_threshold is None:
                raise SystemExit("--cascade requires --cascade-threshold (float)")
            retriever = RoutingRetriever(
                text=retriever,
                visual=visual_retriever,
                mode="cascade",
                cascade_confidence_threshold=cascade_threshold,
            )
            print(
                f"Routing enabled (cascade mode, threshold={cascade_threshold}) — text leg "
                "first; visual leg fires only when text confidence is below the threshold."
            )
        else:
            retriever = RoutingRetriever(text=retriever, visual=visual_retriever)
            print(
                "Routing enabled (category mode) — text leg + ColQwen2 visual leg fused "
                "per query category."
            )

    if region_number_boost:
        from src.rag.retrievers.region_boost import RegionNumberBoostRetriever

        retriever = RegionNumberBoostRetriever(base=retriever)
        print(
            "Region-number boost enabled — chunks whose text starts with "
            "'Table N:' / 'Figure N:' bubble to the top when the query "
            "explicitly references that number."
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
            n_samples=judge_n_samples,
        )
        suffix = f" x {judge_n_samples} samples" if judge_n_samples > 1 else ""
        print(f"Judging answers via {judge_provider} with {judge_model}{suffix}")

    run = await evaluate(
        retriever=retriever,
        golden_set=golden_set,
        generator=generator_obj,
        judge=judge_obj,
        top_k=top_k,
        paper_id_filter=paper_id_filter,
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
            "judge_n_samples": judge_n_samples if judge else None,
            "extract_figures": extract_figures,
            "extract_tables": extract_tables,
            "vlm_caption_model": vlm_caption_model,
            "vlm_caption_provider": vlm_caption_provider if vlm_caption_model else None,
            "query_expansion": query_expansion,
            "query_expansion_mode": query_expansion_mode if query_expansion else None,
            "query_expansion_model": query_expansion_model if query_expansion else None,
            "query_expansion_n": query_expansion_n if query_expansion else None,
            "router": router,
            "visual_model": visual_model if router else None,
            "visual_device": visual_device if router else None,
            "paper_id_filter": paper_id_filter,
            "region_number_boost": region_number_boost,
            "rerank_length_norm": rerank_length_norm,
            "rerank_length_threshold": rerank_length_threshold if rerank_length_norm else None,
            "rerank_length_penalty": rerank_length_penalty if rerank_length_norm else None,
            "cascade": cascade,
            "cascade_threshold": cascade_threshold if cascade else None,
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
        "--judge-n-samples",
        type=int,
        default=1,
        help=(
            "Multi-seed judge averaging (B2). When >1, each metric is sampled "
            "N times in parallel at temperature=0.7 and the score is the mean; "
            "GenerationMetrics gain *_std fields with the sample stddev. "
            "Eliminates single-call judge variance (e.g. q33 in run 196ac0f8786f). "
            "Cost: Nx judge tokens. Default 1 = previous behavior."
        ),
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
        "--rerank-length-norm",
        action="store_true",
        help=(
            "ADR 0009 follow-up: subtract a smooth length penalty from "
            "rerank scores so caption-stub chunks (~80 chars of PDF caption) "
            "stop crowding rich text chunks (~1200 chars). 0 penalty above "
            "--rerank-length-threshold; scales linearly to "
            "--rerank-length-penalty at len=0."
        ),
    )
    parser.add_argument(
        "--rerank-length-threshold",
        type=int,
        default=300,
        help=(
            "Char-length threshold above which no length penalty is applied. "
            "Calibrated to leave q8-style legitimately short answers (~250 chars) "
            "with a small penalty and caption stubs (<150 chars) heavily penalised."
        ),
    )
    parser.add_argument(
        "--rerank-length-penalty",
        type=float,
        default=0.5,
        help=(
            "Maximum length penalty (subtracted from raw rerank score at len=0). "
            "0.5 is calibrated for bge-reranker-v2-m3's [-5, 5] logit range."
        ),
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
        help="Extract embedded figures via PyMuPDF and add as Chunks (caption-text-only, no VLM unless --vlm-caption-model is set).",
    )
    parser.add_argument(
        "--extract-tables",
        action="store_true",
        help="Extract tables via PyMuPDF and add as Chunks (markdown-rendered).",
    )
    parser.add_argument(
        "--vlm-caption-model",
        default=None,
        help=(
            "Vision model used to caption extracted figures. Provider-dependent "
            "shape: with --vlm-caption-provider=ollama (default), an Ollama tag "
            "like 'gemma3:4b' / 'qwen2.5vl:7b'. With openrouter, an OpenRouter "
            "model id like 'openai/gpt-4o-mini' or 'anthropic/claude-3.5-sonnet'. "
            "When set, Figure.vlm_caption is filled and figure_to_chunk uses it "
            "as the indexable text. Requires --extract-figures."
        ),
    )
    parser.add_argument(
        "--vlm-caption-provider",
        choices=("ollama", "openrouter"),
        default="ollama",
        help=(
            "Backend for VLM captioning. 'ollama' is local + free + fast on a "
            "small GPU but mediocre on technical figures (gemma3:4b hallucinates "
            "'heatmap' / 'gene expression' on a scaling-law plot). 'openrouter' "
            "is cloud + costs ~$0.02-0.10 per ingestion pass on the v3 corpus "
            "but produces fidelitous captions. Requires RAG_OPENROUTER_API_KEY."
        ),
    )
    parser.add_argument(
        "--query-expansion",
        action="store_true",
        help=(
            "Wrap PipelineRetriever in MultiQueryRetriever — generate query "
            "variants via LLM, retrieve for each, fuse with RRF. Helps on "
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
    parser.add_argument(
        "--router",
        action="store_true",
        help=(
            "ADR 0008: wrap the text retriever in RoutingRetriever, build a "
            "ColQwen2 visual leg, and fuse text+visual via RRF at page "
            "granularity for figure/table/multi_hop queries. GPU-heavy first run."
        ),
    )
    parser.add_argument(
        "--visual-model",
        default="vidore/colqwen2-v1.0",
        help=(
            "Visual model checkpoint for the routing leg. Default fits an "
            "8 GB GPU; bump to colqwen2.5-v0.2 / colqwen3 on roomier hardware."
        ),
    )
    parser.add_argument(
        "--visual-device",
        default="cuda",
        help="torch device for the visual model. Use 'cpu' if no GPU available.",
    )
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=Path("data/pages"),
        help="Where to cache rendered PDF page PNGs. Idempotent — re-runs reuse cached files.",
    )
    parser.add_argument(
        "--pages-dpi",
        type=int,
        default=150,
        help="DPI for rendered page PNGs. Higher = better visual signal, larger files.",
    )
    parser.add_argument(
        "--paper-id-filter",
        action="store_true",
        help=(
            "Eval-side fairness knob (ADR 0009 follow-up): scope retrieval to "
            "the GoldenQuery.paper_id so paper-specific queries don't bleed "
            "candidates from unrelated papers. Mirrors what a real user "
            "implicitly knows ('I'm asking about paper X'); production "
            "callers don't pass a paper hint, so this only affects evals."
        ),
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help=(
            "ADR 0010: cascade routing mode. Run text leg first; only invoke "
            "the visual leg if top-1 rerank score < --cascade-threshold. "
            "Cuts ColQwen2 invocations on confident text queries. Requires "
            "--router."
        ),
    )
    parser.add_argument(
        "--cascade-threshold",
        type=float,
        default=None,
        help=(
            "Cascade confidence threshold. Calibrated per-corpus via "
            "scripts/calibrate_cascade.py. Required when --cascade is set."
        ),
    )
    parser.add_argument(
        "--region-number-boost",
        action="store_true",
        help=(
            "Reorder retrieval results so chunks whose text starts with "
            "'Table N:' or 'Figure N:' bubble to the top when the query "
            "explicitly references that table/figure number. ADR 0009 "
            "follow-up; closes 'wrong region picked' failures (q29-style)."
        ),
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
            paper_id_filter=args.paper_id_filter,
            region_number_boost=args.region_number_boost,
            rerank_length_norm=args.rerank_length_norm,
            rerank_length_threshold=args.rerank_length_threshold,
            rerank_length_penalty=args.rerank_length_penalty,
            vlm_caption_provider=args.vlm_caption_provider,
            cascade=args.cascade,
            cascade_threshold=args.cascade_threshold,
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
            judge_n_samples=args.judge_n_samples,
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
            router=args.router,
            visual_model=args.visual_model,
            visual_device=args.visual_device,
            pages_dir=args.pages_dir,
            pages_dpi=args.pages_dpi,
        )
    )
