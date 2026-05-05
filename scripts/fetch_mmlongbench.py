"""Fetch MMLongBench-Doc — QA parquet + whatever PDFs are publicly hosted.

The benchmark publishes 1091 QAs over 135 documents; only ~18 PDFs are bundled
on HuggingFace (the rest are web-sourced reports the authors can't redistribute
under their licences). For our purposes a partial corpus is fine — we run the
eval on the subset whose PDFs we successfully fetched, which still gives ~145
questions across 18 docs.

Usage:
    .venv/Scripts/python.exe -m scripts.fetch_mmlongbench

Outputs:
    data/mmlongbench/qa.parquet             — full 1091-row QA table (~smallish)
    data/mmlongbench/documents/*.pdf        — whatever HuggingFace has
    data/mmlongbench/missing.txt            — doc_ids referenced by QAs but not
                                              available locally; fetch manually
                                              if needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import src  # noqa: F401  -- loads .env

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

OUT_DIR = Path("data/mmlongbench")
DOCS_DIR = OUT_DIR / "documents"
HF_REPO_ID = "yubo2333/MMLongBench-Doc"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. QA parquet via the datasets API. Lazily import so users without
    #    `datasets` installed get a clear error.
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets library not installed — run `uv add datasets` first.") from None

    print(f"Loading QA pairs from HuggingFace ({HF_REPO_ID})...")
    ds = load_dataset(HF_REPO_ID, split="train")
    qa_path = OUT_DIR / "qa.parquet"
    ds.to_parquet(str(qa_path))
    print(f"  Wrote {qa_path} ({len(ds)} rows)")

    # 2. PDF documents — list the HF repo's `/documents/` folder and download
    #    whatever's there. Use huggingface_hub directly because the datasets
    #    API doesn't expose the raw documents/ subdirectory.
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub library not installed — should ship with `datasets`."
        ) from None

    api = HfApi()
    print("Listing documents/ on HuggingFace...")
    files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    pdf_files = [f for f in files if f.startswith("documents/") and f.endswith(".pdf")]
    print(f"  Found {len(pdf_files)} PDFs on HuggingFace.")

    fetched: list[str] = []
    for fname in pdf_files:
        local = DOCS_DIR / Path(fname).name
        if local.exists():
            fetched.append(local.name)
            continue
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=fname,
                repo_type="dataset",
                local_dir=str(OUT_DIR),
            )
        except Exception as exc:
            print(f"  [skip] {fname}: {exc}")
            continue
        # hf_hub_download mirrors the repo path; move the file up to documents/
        downloaded = OUT_DIR / fname
        if downloaded.exists() and downloaded != local:
            downloaded.rename(local)
        fetched.append(local.name)

    print(f"  Fetched {len(fetched)} PDFs into {DOCS_DIR}")

    # 3. Compare against the doc_ids referenced by QAs and write missing.txt.
    referenced = sorted({row["doc_id"] for row in ds})
    have = {p.name for p in DOCS_DIR.glob("*.pdf")}
    missing = [d for d in referenced if d not in have]
    miss_path = OUT_DIR / "missing.txt"
    miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(
        f"\nReferenced doc_ids: {len(referenced)} | locally available: "
        f"{len(referenced) - len(missing)} | missing: {len(missing)}"
    )
    print(f"  Missing list written to {miss_path}")
    print(
        f"  Of {len(ds)} total QAs, "
        f"{sum(1 for row in ds if row['doc_id'] in have)} are runnable now."
    )


if __name__ == "__main__":
    main()
