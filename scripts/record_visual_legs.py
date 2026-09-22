"""Record the visual leg's results for a golden set, for replay in the CI gate.

`scripts/eval_retrieval_ci.py` runs the text leg live on the CI runner and
replays the visual leg from the fixture this writes (`src/eval/replay.py`), so
the served hybrid arm is gated without a GPU or the page index in git.
Re-record when the page index, the visual model or the golden set changes.

Needs the visual collection, which ships in the Cloud Build overlay rather than
in git (ADR 0028), and about 5 GB of RAM in bf16 on CPU:

  uv run python -m scripts.record_visual_legs \\
      --snapshot qdrant_local --golden data/golden/v3.yaml \\
      --out data/eval/fixtures/visual-legs-v3.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import torch

from src.eval.golden_set import load_golden_set
from src.eval.replay import fixture_entry
from src.observability.logging import configure_logging
from src.rag.retrievers.visual import VisualRetriever, load_visual_model
from src.rag.visual_store import QdrantVisualStore
from src.types import Query

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


async def _main(args: argparse.Namespace) -> None:
    golden = load_golden_set(args.golden)
    with tempfile.TemporaryDirectory() as tmp:
        # The working tree's meta.json can lack the visual collection (a local
        # `spectrarag serve` rewrites it), and embedded Qdrant ignores any
        # collection meta.json does not list. Open a copy registered with
        # meta.visual.json, the committed two-collection registry.
        snapshot = Path(tmp) / "qdrant"
        shutil.copytree(args.snapshot, snapshot)
        shutil.copyfile(snapshot / "meta.visual.json", snapshot / "meta.json")
        store = QdrantVisualStore(url=f"path:{snapshot}", collection_name=args.collection)
        n_pages = await store.count()
        if n_pages == 0:
            raise SystemExit(f"{args.collection!r} is empty in {args.snapshot}")
        model, processor = await load_visual_model(
            args.model, args.device, dtype=_DTYPES[args.dtype]
        )
        retriever = VisualRetriever(
            model=model, processor=processor, store=store, device=args.device
        )

        queries = []
        for q in golden.queries:
            paper_filter = q.paper_id if args.paper_id_filter and q.paper_id else None
            filters = {"paper_id": paper_filter} if paper_filter else {}
            results = await retriever.retrieve(
                Query(text=q.text, top_k=args.depth, filters=filters)
            )
            queries.append(fixture_entry(q.text, paper_filter, results))
            print(f"{q.query_id}: {len(results)} pages")
        await store.close()

    fixture = {
        "golden": f"{golden.name}/{golden.version}",
        "collection": args.collection,
        "visual_model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "n_pages": n_pages,
        "depth": args.depth,
        "paper_id_filter": args.paper_id_filter,
        "queries": queries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(queries)} queries to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("qdrant_local"))
    parser.add_argument("--collection", default="rag_corpus_visual")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v3.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/fixtures/visual-legs-v3.json"))
    parser.add_argument("--model", default="vidore/colqwen2-v1.0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument(
        "--paper-id-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scope each query to its golden paper, as the CI gate does.",
    )
    configure_logging(level="WARNING", env="local")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
