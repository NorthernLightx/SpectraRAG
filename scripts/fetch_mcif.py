"""Fetch MCIF: long-form QA questions, reference answers and the talk videos.

MCIF (arXiv 2507.19634, huggingface.co/datasets/FBK-MT/MCIF, CC-BY 4.0) holds
100 ACL 2023 talks with human-written transcripts, and 220 human-written QA
pairs on 21 of them. Its labels carry no timestamps; `build_mcif_golden` turns
the QA pairs into candidates for a human to add time spans to.

Questions and references stay in the HuggingFace cache. The videos of the QA
talks go under `--out` for ingestion.

Usage:
    uv run python -m scripts.fetch_mcif --out data/mcif
"""

from __future__ import annotations

import argparse
import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

HF_REPO_ID = "FBK-MT/MCIF"
# Pinned so an upstream edit cannot silently change the question set.
HF_REVISION = "e24065b919758263cfe5d157057278affe76ea7b"
QUESTIONS_FILE = "long_fixedprompt/test-00000-of-00001.parquet"
REFERENCES_FILE = "MCIF.long.en.ref.xml.gz"
VIDEO_DIR = "MCIF_DATA/LONG_VIDEOS"


@dataclass(frozen=True)
class McifSample:
    """One QA pair from the reference file. `id` joins it to its question."""

    id: str
    iid: str
    talk: str
    qa_type: str
    qa_origin: str
    reference: str


def qa_samples(xml_text: str) -> list[McifSample]:
    """The QA samples of a MCIF reference file, in file order."""
    # A pinned upstream file. ElementTree resolves no external entities, and the
    # bundled Expat (2.4.1 or later) caps entity expansion.
    root = ET.fromstring(xml_text.encode("utf-8"))
    samples: list[McifSample] = []
    for sample in root.iter("sample"):
        if sample.get("task") != "QA":
            continue
        samples.append(
            McifSample(
                id=sample.get("id", ""),
                iid=sample.get("iid", ""),
                talk=Path(sample.findtext("video_path") or "").stem,
                qa_type=sample.get("qa_type", ""),
                qa_origin=sample.get("qa_origin", ""),
                reference=(sample.findtext("reference") or "").strip(),
            )
        )
    return samples


def download(filename: str, local_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            HF_REPO_ID,
            filename,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=local_dir,
        )
    )


def load_qa_samples() -> list[McifSample]:
    with gzip.open(download(REFERENCES_FILE), "rt", encoding="utf-8") as fh:
        return qa_samples(fh.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/mcif"))
    args = parser.parse_args()

    samples = load_qa_samples()
    talks = sorted({s.talk for s in samples})
    print(f"{len(samples)} QA samples over {len(talks)} talks")
    for talk in talks:
        path = download(f"{VIDEO_DIR}/{talk}.mp4", local_dir=args.out)
        print(f"  {path}")


if __name__ == "__main__":
    main()
