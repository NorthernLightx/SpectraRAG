"""ADR 0021 probe: how does Docling's text output compare to PyMuPDF + chunk_pages?

Same evidence-first discipline as the figure/table probe. Take one paper
we know well, dump:
  - Docling text blocks (with `label`, page, bbox, length, text snippet).
  - The current PyMuPDF + ADR-0017 chunk output for the same paper.
Read both side by side, decide whether to wire Docling text into
`pipeline.py` (ADR 0021).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.ingestion.chunking import chunk_pages
from src.ingestion.pdf import extract_pages

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = Path("data/papers/2604.22753v1.pdf")
SHOW = 30  # how many items to print from each side


def _docling_doc():
    opts = PdfPipelineOptions()
    opts.images_scale = 2.0
    opts.generate_picture_images = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return converter.convert(PDF).document


def main() -> None:
    print(f"=== Docling text blocks for {PDF.name} ===\n")
    doc = _docling_doc()
    texts = list(getattr(doc, "texts", []))
    print(f"total text blocks: {len(texts)}\n")

    label_counts: Counter[str] = Counter(getattr(t, "label", "?") for t in texts)
    print("label distribution:")
    for label, n in label_counts.most_common():
        print(f"  {str(label):>20}: {n}")
    print()

    print(f"first {SHOW} blocks (label · page · bbox-y · len · snippet):")
    for i, t in enumerate(texts[:SHOW]):
        provs = getattr(t, "prov", None) or []
        page = bbox = None
        if provs:
            page = getattr(provs[0], "page_no", None)
            bbox = getattr(provs[0], "bbox", None)
        y_top = f"y={getattr(bbox, 't', '?')!r}->{getattr(bbox, 'b', '?')!r}" if bbox else "y=?"
        text = " ".join(getattr(t, "text", "").split())[:80]
        label = getattr(t, "label", "?")
        print(
            f"  [{i:>3}] {str(label):<18} p{page or '?'} {y_top} len={len(getattr(t, 'text', ''))}  {text!r}"
        )
    print()

    # Stats: avg block length, multi-page presence
    lengths = [len(getattr(t, "text", "")) for t in texts]
    if lengths:
        print(
            f"text-block lengths: mean={sum(lengths) / len(lengths):.1f}  "
            f"median={sorted(lengths)[len(lengths) // 2]}  "
            f"min={min(lengths)}  max={max(lengths)}\n"
        )

    # Compare to PyMuPDF + ADR-0017 chunk output for the same paper.
    print(f"=== PyMuPDF + chunk_pages output (ADR 0017) for {PDF.name} ===\n")
    pages = extract_pages(PDF.stem, PDF)
    chunks = chunk_pages(pages)
    print(f"total chunks: {len(chunks)}\n")
    print(f"first {SHOW} chunks (section · pages · len · snippet):")
    for i, c in enumerate(chunks[:SHOW]):
        text = " ".join(c.text.split())[:80]
        print(
            f"  [{i:>3}] section={c.section!r:<40} pp={c.page_numbers} len={len(c.text)}  {text!r}"
        )
    print()

    # Length stats
    chunk_lengths = [len(c.text) for c in chunks]
    if chunk_lengths:
        print(
            f"chunk lengths: mean={sum(chunk_lengths) / len(chunk_lengths):.1f}  "
            f"median={sorted(chunk_lengths)[len(chunk_lengths) // 2]}  "
            f"min={min(chunk_lengths)}  max={max(chunk_lengths)}"
        )


if __name__ == "__main__":
    main()
