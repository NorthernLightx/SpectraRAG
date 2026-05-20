"""Compare installed Docling + classifier preset against what's available
on PyPI / HuggingFace. Looking for a newer figure-classifier release."""
from __future__ import annotations
import json
import urllib.request
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def main() -> None:
    # 1) Current Docling preset
    from docling.datamodel.picture_classification_options import DocumentPictureClassifierOptions

    print("=== Docling presets (this install) ===")
    for pid in DocumentPictureClassifierOptions.list_preset_ids():
        p = DocumentPictureClassifierOptions.get_preset(pid)
        spec = p.model_spec
        print(f"  preset_id={pid}")
        print(f"    repo_id={spec.repo_id}")
        print(f"    revision={spec.revision}")
        print(f"    name={spec.name}")

    # 2) Installed Docling version + PyPI latest
    import importlib.metadata as md

    print(f"\n=== Docling release ===")
    print(f"  installed: {md.version('docling')}")
    data = json.loads(urllib.request.urlopen("https://pypi.org/pypi/docling/json", timeout=10).read())
    info = data["info"]
    print(f"  latest on PyPI: {info['version']}")
    releases = sorted(data["releases"].keys(), reverse=True)
    print(f"  recent versions:")
    for v in releases[:8]:
        print(f"    {v}")

    # 3) HuggingFace tags on the classifier repo
    print(f"\n=== HuggingFace tags for DocumentFigureClassifier ===")
    repo = "docling-project/DocumentFigureClassifier"
    try:
        hf = json.loads(
            urllib.request.urlopen(
                f"https://huggingface.co/api/models/{repo}/refs", timeout=10
            ).read()
        )
        for t in hf.get("branches", []):
            print(f"  branch: {t['name']}")
        for t in hf.get("tags", []):
            print(f"  tag: {t['name']}")
        # Also list any -v2.0 / -v2.5 / -v3.0 variants in the org
        print(f"\n=== docling-project repos matching 'FigureClassifier' ===")
        org = json.loads(
            urllib.request.urlopen(
                "https://huggingface.co/api/models?author=docling-project&search=Classifier",
                timeout=10,
            ).read()
        )
        for m in org:
            print(f"  {m['id']}")
    except Exception as e:
        print(f"  HF probe failed: {e}")


if __name__ == "__main__":
    main()
