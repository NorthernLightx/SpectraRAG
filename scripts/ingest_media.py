"""Ingest recordings: talk videos and audio, as documents of time segments.

A video's pages are its slide segments, each with a keyframe; an audio file's
pages are time windows over its transcript (src/ingestion/media.py). Keyframes
and the segment manifest go under `--pages-dir`; transcript chunks go into
`--collection`, embedded by the same builder the eval uses for the profile.

A recording's document id is its file name without the extension, sanitised
like an upload's. One recording that fails to ingest is reported and the batch
goes on; the exit status is non-zero when any failed.

Usage:
    uv run python -m scripts.ingest_media \\
        --media-dir ./recordings --pages-dir data/recordings \\
        --qdrant path:./qdrant_media --collection recordings
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path

import src  # noqa: F401  -- loads .env
from src.config.settings import load_settings
from src.ingestion.media import WhisperTranscriber, manifest_path
from src.ingestion.pipeline import ingest_media
from src.rag.bm25 import Bm25Index
from src.rag.retrieval_config import RetrievalConfig, build_embedder
from src.rag.vectorstore import QdrantVectorStore

_MEDIA_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
}


def doc_id_for(path: Path) -> str:
    """The document id of a recording: its stem, restricted to the characters
    an uploaded PDF's id may hold (src/api/routes/ingest.py)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", path.stem) or "recording"


def plan(paths: Iterable[Path]) -> dict[str, Path]:
    """Doc id -> recording. Two recordings with one id would overwrite each
    other's pages and chunks, so that is refused before any work starts."""
    out: dict[str, Path] = {}
    clashes: list[str] = []
    for path in sorted(paths):
        doc_id = doc_id_for(path)
        if doc_id in out:
            clashes.append(f"{doc_id}: {out[doc_id].name} and {path.name}")
        out[doc_id] = path
    if clashes:
        raise ValueError("recordings share a document id: " + "; ".join(clashes))
    return out


async def ingest_all(
    items: Mapping[str, Path], ingest_one: Callable[[str, Path], Awaitable[None]]
) -> list[str]:
    """Ingest every recording; return one line per recording that failed."""
    failures: list[str] = []
    for doc_id, path in items.items():
        try:
            await ingest_one(doc_id, path)
        except Exception as exc:  # a bad file is reported, the batch goes on
            failures.append(f"{doc_id}: {type(exc).__name__}: {exc}")
            print(f"  {doc_id}: FAILED {type(exc).__name__}: {exc}")
    return failures


async def main(args: argparse.Namespace) -> int:
    found = [p for p in args.media_dir.iterdir() if p.suffix.lower() in _MEDIA_SUFFIXES]
    items = plan(found)
    if args.only:
        items = {d: p for d, p in items.items() if d in set(args.only)}
    todo = {
        d: p
        for d, p in items.items()
        if args.fresh or not manifest_path(args.pages_dir, d).exists()
    }
    print(f"{len(items)} recordings, {len(items) - len(todo)} already ingested, {len(todo)} to go")
    if not todo:
        return 0

    config = RetrievalConfig.from_settings(load_settings(profile=args.profile))
    embedder = await asyncio.to_thread(build_embedder, config, ollama_url=args.ollama)
    vectorstore = QdrantVectorStore(
        url=args.qdrant, collection_name=args.collection, dim=embedder.dim
    )
    # Chunk ids depend on the segmentation, so a re-ingest starts from an empty
    # collection rather than leaving the old segmentation's chunks behind.
    if args.fresh:
        await vectorstore.delete_collection()
    await vectorstore.ensure_collection()
    transcriber = WhisperTranscriber(args.asr_model, threads=args.threads, language=args.language)

    async def ingest_one(doc_id: str, path: Path) -> None:
        started = time.monotonic()
        result = await ingest_media(
            doc_id=doc_id,
            media_path=path,
            embedder=embedder,
            vectorstore=vectorstore,
            bm25=Bm25Index(),
            transcriber=transcriber,
            pages_dir=args.pages_dir,
        )
        pages = len({c.page_numbers[0] for c in result.chunks})
        print(
            f"  {doc_id}: {result.chunk_count} chunks on {pages} pages "
            f"in {time.monotonic() - started:.0f} s"
        )

    failures = await ingest_all(todo, ingest_one)
    if failures:
        print(f"{len(failures)} of {len(todo)} recordings failed:")
        for line in failures:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument(
        "--qdrant", required=True, help="Qdrant URL, or path:<dir> for an embedded store"
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--profile", default="cpu", help="settings profile the embedder comes from")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument(
        "--language", default=None, help="spoken language code, for example en; default detects it"
    )
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for transcription")
    parser.add_argument("--only", nargs="*", help="document ids to ingest, default all")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="drop the collection and re-ingest every recording; cached transcripts are reused",
    )
    parsed = parser.parse_args()
    if parsed.fresh and parsed.only:
        parser.error("--fresh drops the whole collection; it cannot be combined with --only")
    raise SystemExit(asyncio.run(main(parsed)))
