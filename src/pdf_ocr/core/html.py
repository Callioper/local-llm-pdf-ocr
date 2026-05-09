"""
HTMLHandler - emit OCR results as a self-contained HTML document.

Layout mirrors `PDFHandler`'s sandwich PDF: each page is shown as a
background image with OCR text overlaid as invisible, absolutely-
positioned ``<span>``s. Browsers select / Ctrl+F-search / copy the
text exactly as if it were a real text layer, while still showing the
original page image untouched.

Design choices
--------------
* **Self-contained output.** Every page image is inlined as a base64
  JPEG ``data:`` URL — no sidecar files, no broken links if the user
  moves the HTML around.
* **Letter-spacing on monospace by default.** Font is sized to the
  bbox height; ``letter-spacing`` is computed so the rendered text
  width fills the bbox horizontally. This keeps the *selection*
  extents (what a user gets when they click-and-drag in the browser)
  aligned with the visible text on the underlying image — the
  practical ceiling for invisible-text alignment without per-glyph
  positions from the OCR pipeline.
* **Two alternative modes** are still available via ``mode=``:
  ``"full-height"`` keeps the font at bbox height and lets the text
  overflow if too long; ``"scaled"`` shrinks the font so neither
  dimension overflows (smaller text, but the text region in the DOM
  matches the rendered glyphs more tightly).
* **Same edge-case handling as the PDF writer.** The aligner's
  ``[0,0,1,1]`` full-page fallback bbox and ``"\\n"``-joined
  multi-line bbox content are routed through the shared helpers in
  ``core/_layout.py`` so HTML and PDF outputs treat them identically.

Co-authored-by: Milan Hauth <milahu@milahu.duckdns.org>  (PR #8 prototype)
"""

from __future__ import annotations

import base64
import html as _html
import io
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from PIL import Image, ImageSequence

from pdf_ocr.core._layout import is_full_page_fallback, split_multi_line_bbox
from pdf_ocr.core.pdf import _is_image_path

# --- mode names -----------------------------------------------------------
MODE_LETTER_SPACING = "letter-spacing"
MODE_FULL_HEIGHT = "full-height"
MODE_SCALED = "scaled"
_VALID_MODES = frozenset({MODE_LETTER_SPACING, MODE_FULL_HEIGHT, MODE_SCALED})

# Approximate width/height ratio for browser monospace fonts (Courier-
# family). Used to convert font-size into character width when computing
# letter-spacing or scale. Typical browser monospace stacks land near
# 0.6; the exact value depends on the user agent's default font, which
# we don't control. Treated as a tunable class-level constant.
_MONOSPACE_FONT_ASPECT = 0.6

_JPEG_QUALITY = 80

_PAGE_CSS = """\
body { margin: 0; background: #f5f5f5; }
div.page {
  position: relative;
  background-repeat: no-repeat;
  background-size: 100% 100%;
  margin: 0 auto 1em auto;
  outline: solid 1px #d0d0d0;
}
span.line {
  position: absolute;
  color: transparent;
  white-space: nowrap;
  font-family: monospace;
  line-height: 1;
  /* Keep invisibility intact even when the browser is in dark mode —
     a previous draft applied `filter: invert()` to the page div, which
     interacts badly with `color: transparent` in Chromium and renders
     the glyph outlines as a faint inverted color. The OCR HTML preserves
     the source page's appearance regardless of OS theme; users who want
     dark theming for documents can use a browser extension. */
}
"""


class HTMLHandler:
    """Output writer mirroring ``PDFHandler.embed_structured_text``."""

    def __init__(self, mode: str = MODE_LETTER_SPACING):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"unknown HTML mode {mode!r}; "
                f"expected one of {sorted(_VALID_MODES)}"
            )
        self.mode = mode

    def embed_structured_text(
        self,
        input_path: str,
        output_path: str,
        pages_data: dict,
        dpi: int = 200,
    ) -> None:
        """Render the OCR'd `pages_data` as HTML next to `input_path` images.

        Parameters mirror ``PDFHandler.embed_structured_text`` so this
        class can drop into ``OCRPipeline(output_writer=...)``.
        """
        title = Path(input_path).stem or "OCR output"
        pages = list(self._iter_page_images(input_path, dpi))
        html = self._render_html(title, pages, pages_data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    # --- input dispatch ---------------------------------------------------

    def _iter_page_images(
        self, input_path: str, dpi: int,
    ) -> Iterator[tuple[int, bytes, float, float]]:
        """Yield (page_index, jpeg_bytes, width_px, height_px) per page."""
        if _is_image_path(input_path):
            yield from self._pages_from_image(input_path)
        else:
            yield from self._pages_from_pdf(input_path, dpi)

    @staticmethod
    def _pages_from_pdf(
        input_path: str, dpi: int,
    ) -> Iterator[tuple[int, bytes, float, float]]:
        """Rasterize each PDF page at `dpi` and emit JPEG bytes + dims."""
        doc = fitz.open(input_path)
        try:
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("jpg", jpg_quality=_JPEG_QUALITY)
                # Use rendered pixel dimensions so absolute-positioned
                # spans align with the rasterized background image.
                yield page_num, img_bytes, float(pix.width), float(pix.height)
        finally:
            doc.close()

    @staticmethod
    def _pages_from_image(
        input_path: str,
    ) -> Iterator[tuple[int, bytes, float, float]]:
        """Iterate frames of a single- or multi-frame image (TIFF, etc.)."""
        with Image.open(input_path) as src:
            for page_num, frame in enumerate(ImageSequence.Iterator(src)):
                img = frame.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
                yield page_num, buf.getvalue(), float(img.width), float(img.height)

    # --- HTML rendering ---------------------------------------------------

    def _render_html(self, title: str, pages, pages_data: dict) -> str:
        out = io.StringIO()
        out.write("<!doctype html>\n<html>\n<head>\n")
        out.write('<meta charset="utf-8" />\n')
        out.write(f"<title>{_html.escape(title)}</title>\n")
        out.write("<style>\n")
        out.write(_PAGE_CSS)
        out.write("</style>\n</head>\n<body>\n")
        for page_num, img_bytes, width, height in pages:
            self._render_page(
                out, page_num, img_bytes, width, height,
                pages_data.get(page_num, []),
            )
        out.write("</body>\n</html>\n")
        return out.getvalue()

    def _render_page(self, out, page_num, img_bytes, width, height, items):
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        out.write(
            f'<div class="page" data-page="{page_num + 1}" '
            f'style="width:{_num(width)}px;height:{_num(height)}px;'
            f"background-image:url('{data_url}')\">\n"
        )
        for rect_coords, text in items:
            self._render_box(
                out, list(rect_coords), text or "", width, height,
            )
        out.write("</div>\n")

    def _render_box(self, out, rect_coords, text, page_width, page_height):
        text = text.strip()
        if not text:
            return

        # The aligner's [0,0,1,1] full-page fallback bbox: render in a
        # small inset rect, with each '\n'-separated line on its own row.
        if is_full_page_fallback(rect_coords, text):
            inset = [
                10.0 / page_width,
                10.0 / page_height,
                (page_width - 10.0) / page_width,
                (page_height - 10.0) / page_height,
            ]
            for sub_rect, line in split_multi_line_bbox(inset, text):
                self._emit_span(out, sub_rect, line, page_width, page_height)
            return

        # Real bbox with multi-line content (grounded VLM joined lines):
        # split vertically, emit one span per line.
        if "\n" in text:
            sub = split_multi_line_bbox(rect_coords, text)
            if len(sub) > 1:
                for sub_rect, line in sub:
                    self._emit_span(out, sub_rect, line, page_width, page_height)
                return
            if sub:
                text = sub[0][1]

        self._emit_span(out, rect_coords, text, page_width, page_height)

    def _emit_span(self, out, rect_coords, text, page_width, page_height):
        nx0, ny0, nx1, ny1 = rect_coords
        x = nx0 * page_width
        y = ny0 * page_height
        w = (nx1 - nx0) * page_width
        h = (ny1 - ny0) * page_height
        if w <= 0 or h <= 0 or not text:
            return

        sizing = self._span_sizing_style(text, w, h)
        # Escape the minimum HTML special chars so the text is parsed
        # correctly even though it's rendered transparent.
        safe_text = _html.escape(text, quote=False)
        out.write(
            f'<span class="line" style="left:{_num(x)}px;top:{_num(y)}px;'
            f'width:{_num(w)}px;{sizing}">{safe_text}</span>\n'
        )

    def _span_sizing_style(self, text: str, w: float, h: float) -> str:
        """Choose font-size + letter-spacing for one span per `self.mode`."""
        n_chars = max(1, len(text))

        if self.mode == MODE_LETTER_SPACING:
            # Font sized to bbox height; spread characters to fill width
            # via letter-spacing. Negative spacing is allowed — characters
            # overlap visually but selection extents still span the box.
            font_size = h
            natural_width = n_chars * font_size * _MONOSPACE_FONT_ASPECT
            letter_spacing = (w - natural_width) / n_chars
            return (
                f"font-size:{_num(font_size)}px;"
                f"letter-spacing:{_num(letter_spacing)}px;"
                f"height:{_num(h)}px;"
            )

        if self.mode == MODE_FULL_HEIGHT:
            # Font sized to bbox height regardless of resulting width.
            return f"font-size:{_num(h)}px;height:{_num(h)}px;"

        # MODE_SCALED: pick the smaller of width-fit and height-fit so
        # neither dimension overflows.
        char_width_for_height = h * _MONOSPACE_FONT_ASPECT
        char_width_for_width = w / n_chars
        char_width = min(char_width_for_height, char_width_for_width)
        font_size = char_width / _MONOSPACE_FONT_ASPECT
        return (
            f"font-size:{_num(font_size)}px;"
            f"height:{_num(h)}px;"
        )


def _num(n: float) -> str:
    """Render `n` as an int if it has no fractional part, else 4 decimals."""
    if isinstance(n, int):
        return str(n)
    if n == int(n):
        return str(int(n))
    return f"{n:.4f}".rstrip("0").rstrip(".")
