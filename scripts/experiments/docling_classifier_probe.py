"""Enable Docling's built-in picture classifier on one paper and dump the
labels it assigns. Probes whether `do_picture_classification=True` is a
drop-in replacement for our string-heuristic role classifier.

Run:
  python scripts/experiments/docling_classifier_probe.py 2604.28181v1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper_id", help="e.g. 2604.28181v1 (Microsoft-logo-everywhere paper)")
    ap.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    args = ap.parse_args()

    pdf = args.papers_dir / f"{args.paper_id}.pdf"
    if not pdf.exists():
        sys.exit(f"missing: {pdf}")

    # Try transformers engine with torch.compile/dynamo disabled — avoids
    # the Triton requirement that bit us on Windows.
    import os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch._dynamo as _dyn
    _dyn.config.suppress_errors = True
    _dyn.disable()

    opts = PdfPipelineOptions()
    opts.images_scale = 2.0
    opts.generate_picture_images = True
    opts.do_picture_classification = True

    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    t0 = time.perf_counter()
    res = conv.convert(pdf)
    elapsed = time.perf_counter() - t0
    doc = res.document
    pics = list(getattr(doc, "pictures", []))
    print(f"convert took {elapsed:.1f}s, pictures: {len(pics)}")
    print()

    # Dump the structure of the first picture's annotations / meta so we
    # know which field to read.
    if pics:
        p0 = pics[0]
        print("--- first pic introspection:")
        for fld in ("annotations", "meta"):
            val = getattr(p0, fld, None)
            print(f"  {fld}: {val!r}")
        print()

    for idx, pic in enumerate(pics, start=1):
        provs = getattr(pic, "prov", None) or []
        page = getattr(provs[0], "page_no", "?") if provs else "?"
        # bbox area
        bb = getattr(provs[0], "bbox", None) if provs else None
        area = 0.0
        if bb is not None:
            try:
                area = abs((bb.r - bb.l) * (bb.t - bb.b))
            except Exception:
                pass
        # caption
        caption = ""
        try:
            caption = (pic.caption_text(doc) or "")[:60]
        except Exception:
            pass
        # classifications: pic.annotations is the docling output for the
        # classifier. Each annotation has a kind discriminator and (for the
        # classifier) a `predicted_classes` list of {class_name, confidence}.
        labels = []
        for ann in getattr(pic, "annotations", []) or []:
            kind = getattr(ann, "kind", None) or getattr(ann, "type", None)
            if str(kind) == "classification" or hasattr(ann, "predicted_classes"):
                preds = getattr(ann, "predicted_classes", []) or []
                top = preds[0] if preds else None
                if top is not None:
                    labels.append(f"{top.class_name}({top.confidence:.2f})")
        labels_str = ",".join(labels) or "(none)"
        print(f"  p{page:>3}  area={area:>7.0f}  labels={labels_str:<35}  caption={caption!r}")


if __name__ == "__main__":
    main()
