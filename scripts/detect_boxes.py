"""CLI: run Surya detection on a PDF, output per-page normalized bboxes as JSON to stdout.

Usage:
    uv run scripts/detect_boxes.py input.pdf [--dpi 200] [--pages 1-5] [--detect-batch-size 20]

Output (stdout):
    {
        "pages": [
            {"page": 0, "width": 595.0, "height": 842.0, "boxes": [[x0,y0,x1,y1], ...]},
            ...
        ]
    }
"""
import argparse
import io
import json
import os
import sys

from PIL import Image

os.environ.setdefault("TQDM_DISABLE", "1")


def main():
    parser = argparse.ArgumentParser(description="Surya detection only — output bbox JSON")
    parser.add_argument("input", help="Path to PDF file")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for rendering (default: 200)")
    parser.add_argument("--pages", type=str, default=None, help="Page range e.g. 1-3,5")
    parser.add_argument("--detect-batch-size", type=int, default=20, help="Pages per batch")
    parser.add_argument("--text-threshold", type=float, default=None, help="Override DETECTOR_TEXT_THRESHOLD (default 0.6)")
    parser.add_argument("--blank-threshold", type=float, default=None, help="Override DETECTOR_BLANK_THRESHOLD (default 0.35)")
    args = parser.parse_args()

    if args.text_threshold is not None or args.blank_threshold is not None:
        from surya.settings import settings
        if args.text_threshold is not None:
            settings.DETECTOR_TEXT_THRESHOLD = args.text_threshold
        if args.blank_threshold is not None:
            settings.DETECTOR_BLANK_THRESHOLD = args.blank_threshold

    import fitz
    from pdf_ocr.core.aligner import HybridAligner

    doc = fitz.open(args.input)
    total_pages = len(doc)
    pages_arg = args.pages

    if pages_arg:
        indices = set()
        for part in pages_arg.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                a = int(a.strip()) - 1
                b = int(b.strip()) - 1
                indices.update(range(max(0, a), min(total_pages, b + 1)))
            else:
                i = int(part.strip()) - 1
                if 0 <= i < total_pages:
                    indices.add(i)
        page_indices = sorted(indices)
    else:
        page_indices = list(range(total_pages))

    page_images: dict[int, bytes] = {}
    page_sizes: dict[int, tuple] = {}

    for pg in page_indices:
        page = doc[pg]
        pix = page.get_pixmap(dpi=args.dpi)
        img_data = pix.tobytes("png")
        page_images[pg] = img_data
        page_sizes[pg] = (page.rect.width, page.rect.height)

    doc.close()

    aligner = HybridAligner()
    image_bytes_list = [page_images[pg] for pg in page_indices]

    batch_size = args.detect_batch_size
    all_boxes: dict[int, list] = {}

    for batch_start in range(0, len(image_bytes_list), batch_size):
        batch_end = min(batch_start + batch_size, len(image_bytes_list))
        batch_images = image_bytes_list[batch_start:batch_end]
        batch_pages = page_indices[batch_start:batch_end]

        results = aligner.get_detected_boxes_batch(batch_images)
        for pg, boxes in zip(batch_pages, results):
            all_boxes[pg] = [list(b) for b in boxes] if boxes else []

    output = {"pages": []}
    for pg in sorted(all_boxes.keys()):
        pw, ph = page_sizes[pg]
        output["pages"].append({
            "page": pg,
            "width": pw,
            "height": ph,
            "boxes": all_boxes[pg],
        })

    json.dump(output, sys.stdout, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
