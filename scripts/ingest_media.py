"""Ingest recordings: talk videos and audio, as documents of time segments.

A video's pages are its slide segments, each with a keyframe; an audio file's
pages are time windows over its transcript (src/ingestion/media.py). Keyframes
and the segment manifest go under `--pages-dir`; transcript chunks go into
`--collection`, embedded by the same builder the eval uses for the profile.

Usage:
    uv run python -m scripts.ingest_media \\
        --media-dir data/mcif/MCIF_DATA/LONG_VIDEOS --pages-dir data/mcif/pages \\
        --qdrant path:data/mcif/qdrant --collection mcif_text
"""

from __future__ import annotations

import argparse
import asyncio
import time
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


async def main(args: argparse.Namespace) -> None:
    videos = sorted(p for p in args.media_dir.iterdir() if p.suffix.lower() in _MEDIA_SUFFIXES)
    if args.only:
        videos = [p for p in videos if p.stem in set(args.only)]
    todo = [p for p in videos if args.fresh or not manifest_path(args.pages_dir, p.stem).exists()]
    print(
        f"{len(videos)} recordings, {len(videos) - len(todo)} already ingested, {len(todo)} to go"
    )
    if not todo:
        return

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
    transcriber = WhisperTranscriber(args.asr_model, threads=args.threads)

    for video in todo:
        started = time.monotonic()
        result = await ingest_media(
            doc_id=video.stem,
            media_path=video,
            embedder=embedder,
            vectorstore=vectorstore,
            bm25=Bm25Index(),
            transcriber=transcriber,
            pages_dir=args.pages_dir,
        )
        pages = len({c.page_numbers[0] for c in result.chunks})
        print(
            f"  {video.stem}: {result.chunk_count} chunks on {pages} pages "
            f"in {time.monotonic() - started:.0f} s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--qdrant", default="path:data/mcif/qdrant")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--profile", default="cpu", help="settings profile the embedder comes from")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for transcription")
    parser.add_argument("--only", nargs="*", help="video stems to ingest, default all")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="drop the collection and re-ingest every video; cached transcripts are reused",
    )
    parsed = parser.parse_args()
    if parsed.fresh and parsed.only:
        parser.error("--fresh drops the whole collection; it cannot be combined with --only")
    asyncio.run(main(parsed))
