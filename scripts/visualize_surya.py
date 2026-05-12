"""Visualize Surya text-detection bboxes overlaid on rendered PDF pages.

Usage:
    uv run scripts/visualize_surya.py input.pdf [--dpi 200] [--page 0] [--output out.png]
    uv run scripts/visualize_surya.py input.pdf --all  (visualize every page)

Output: PNG image(s) with Surya bboxes drawn in red over the page render.
"""

import argparse
import os
import sys

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description="Visualize Surya bboxes on PDF pages")
    parser.add_argument("input", help="Path to PDF file")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--page", type=int, default=0, help="Page index (0-based) to visualize")
    parser.add_argument("--all", action="store_true", help="Visualize every page")
    parser.add_argument("--output", help="Output PNG path (for single page)")
    parser.add_argument("--out-dir", help="Output directory (for --all)")
    parser.add_argument("--text-threshold", type=float, default=None, help="Override DETECTOR_TEXT_THRESHOLD (default 0.6)")
    parser.add_argument("--blank-threshold", type=float, default=None, help="Override DETECTOR_BLANK_THRESHOLD (default 0.35)")
    parser.add_argument("--detect-batch-size", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")

    # Apply threshold overrides before importing Surya
    if args.text_threshold is not None or args.blank_threshold is not None:
        from surya.settings import settings
        if args.text_threshold is not None:
            settings.DETECTOR_TEXT_THRESHOLD = args.text_threshold
            print(f"[surya] DETECTOR_TEXT_THRESHOLD = {args.text_threshold}", file=sys.stderr)
        if args.blank_threshold is not None:
            settings.DETECTOR_BLANK_THRESHOLD = args.blank_threshold
            print(f"[surya] DETECTOR_BLANK_THRESHOLD = {args.blank_threshold}", file=sys.stderr)

    import fitz
    from surya.detection import DetectionPredictor

    doc = fitz.open(args.input)
    total_pages = len(doc)

    if args.all:
        page_indices = list(range(total_pages))
    else:
        if args.page < 0 or args.page >= total_pages:
            print(f"ERROR: page {args.page} out of range (0-{total_pages - 1})", file=sys.stderr)
            sys.exit(1)
        page_indices = [args.page]

    # Render pages to images
    page_images: dict[int, Image.Image] = {}
    page_sizes: dict[int, tuple] = {}
    for pg in page_indices:
        pix = doc[pg].get_pixmap(dpi=args.dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images[pg] = img
        page_sizes[pg] = (pix.width, pix.height)
    doc.close()

    # Run Surya detection
    predictor = DetectionPredictor()
    image_list = list(page_images.values())
    pg_list = list(page_images.keys())
    all_boxes: dict[int, list] = {}

    for batch_start in range(0, len(image_list), args.detect_batch_size):
        batch_end = min(batch_start + args.detect_batch_size, len(image_list))
        batch_images = image_list[batch_start:batch_end]
        batch_pages = pg_list[batch_start:batch_end]
        predictions = predictor(batch_images)
        for pg, pred in zip(batch_pages, predictions):
            boxes = []
            for bbox in (pred.bboxes or []):
                boxes.append(list(bbox.bbox))
            boxes.sort(key=lambda b: (b[1], b[0]))
            all_boxes[pg] = boxes

    # Draw bboxes on images
    for pg in sorted(all_boxes.keys()):
        img = page_images[pg].copy()
        draw = ImageDraw.Draw(img)
        pw, ph = page_sizes[pg]
        boxes = all_boxes[pg]

        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            # Label box index for spatial reference
            draw.text((x0 + 2, y0), str(i), fill="yellow")

        # Write PDF page dimensions in corner
        pdf_pw = pw / args.dpi * 72
        pdf_ph = ph / args.dpi * 72
        draw.text((10, 10), f"Page {pg}  DPI={args.dpi}  {pdf_pw:.0f}x{pdf_ph:.0f}pt  Boxes={len(boxes)}", fill="green")

        if args.all:
            out_dir = args.out_dir or "."
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"page_{pg:04d}.png")
        else:
            out_path = args.output or f"page_{pg}.png"

        img.save(out_path)
        print(f"Saved: {out_path} ({len(boxes)} boxes)", file=sys.stderr)

    if args.all:
        print(f"Total pages processed: {len(page_indices)}", file=sys.stderr)


if __name__ == "__main__":
    main()
