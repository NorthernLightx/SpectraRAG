"""Fetch QMSum's AMI (Product) meetings, AMI's word timings and, optionally, audio.

QMSum (github.com/Yale-LILY/QMSum, MIT) supplies the queries and turn labels;
the AMI Meeting Corpus (groups.inf.ed.ac.uk/ami, CC BY 4.0) supplies the timed
words and the headset-mix audio. `build_qmsum_golden` joins them.

Everything lands under `--out` (gitignored): `qmsum/<split>/<meeting>.json`,
`annotations/words/*.words.xml` + `annotations/corpusResources/meetings.xml`,
and with `--audio` `audio/<meeting>.wav`, named by meeting id so the ingested
document id is the meeting id. Files already present are kept.

Usage:
    uv run python -m scripts.fetch_ami --split test [--audio]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

# Pinned so an upstream edit cannot silently change the query set.
QMSUM_REVISION = "83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e"
_QMSUM_TREE = f"https://api.github.com/repos/Yale-LILY/QMSum/git/trees/{QMSUM_REVISION}?recursive=1"
_QMSUM_RAW = f"https://raw.githubusercontent.com/Yale-LILY/QMSum/{QMSUM_REVISION}/"
_AMI = "https://groups.inf.ed.ac.uk/ami/"
_ANNOTATIONS_ZIP = "ami_public_manual_1.6.2.zip"


def _download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)


def qmsum_meetings(split: str) -> list[str]:
    with urllib.request.urlopen(_QMSUM_TREE) as resp:
        tree = json.load(resp)["tree"]
    prefix = f"data/Product/{split}/"
    return sorted(
        Path(t["path"]).stem
        for t in tree
        if t["path"].startswith(prefix) and t["path"].endswith(".json")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ami"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument(
        "--audio", action="store_true", help="also fetch each meeting's headset mix"
    )
    args = parser.parse_args()

    meetings = qmsum_meetings(args.split)
    print(f"{len(meetings)} QMSum Product meetings in {args.split}")
    for m in meetings:
        _download(
            f"{_QMSUM_RAW}data/Product/{args.split}/{m}.json",
            args.out / "qmsum" / args.split / f"{m}.json",
        )

    zip_path = args.out / _ANNOTATIONS_ZIP
    _download(f"{_AMI}AMICorpusAnnotations/{_ANNOTATIONS_ZIP}", zip_path)
    ann = args.out / "annotations"
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            wanted = name.startswith("words/") or name == "corpusResources/meetings.xml"
            if wanted and not (ann / name).exists():
                zf.extract(name, ann)

    if args.audio:
        for i, m in enumerate(meetings, start=1):
            _download(
                f"{_AMI}AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav",
                args.out / "audio" / f"{m}.wav",
            )
            print(f"  audio {i}/{len(meetings)} {m}")


if __name__ == "__main__":
    main()
