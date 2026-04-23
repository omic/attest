"""PDF text extraction with OCR fallback.

Uses pdfplumber for text-layer extraction, falls back to
pytesseract + pdf2image for scanned/image PDFs.

Usage:
    extractor = PDFExtractor()
    pages = extractor.extract_text("document.pdf")
    for page in pages:
        print(page.page_number, page.text[:100], page.method)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum characters to consider a page as having usable text
_MIN_TEXT_CHARS = 50


@dataclass
class PageText:
    """Extracted text from a single PDF page."""

    page_number: int
    text: str
    method: str  # "pdfplumber" or "ocr"
    confidence_modifier: float  # 1.0 for text-layer, 0.8 for OCR

    @property
    def has_text(self) -> bool:
        return len(self.text.strip()) >= _MIN_TEXT_CHARS


@dataclass
class PDFMetadata:
    """Metadata about a PDF file."""

    path: str
    page_count: int
    text_pages: int
    ocr_pages: int
    total_chars: int


class PDFExtractor:
    """Extract text from PDF files with OCR fallback.

    Tries pdfplumber first for each page. If a page yields fewer
    than 50 characters, falls back to OCR via pdf2image + pytesseract.
    """

    def __init__(self, ocr_fallback: bool = True, dpi: int = 300) -> None:
        self.ocr_fallback = ocr_fallback
        self.dpi = dpi
        self._ocr_available: bool | None = None

    def _check_ocr(self) -> bool:
        """Check if OCR dependencies are available."""
        if self._ocr_available is not None:
            return self._ocr_available
        try:
            import pytesseract  # noqa: F401
            from pdf2image import convert_from_path  # noqa: F401

            self._ocr_available = True
        except ImportError:
            self._ocr_available = False
            logger.info("OCR dependencies not installed (pytesseract/pdf2image)")
        return self._ocr_available

    def extract_text(self, path: str | Path) -> list[PageText]:
        """Extract text from all pages of a PDF.

        Returns a list of PageText, one per page. Pages with no
        extractable text are included with empty text strings.
        """
        import pdfplumber

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages: list[PageText] = []

        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text() or ""

                if len(text.strip()) >= _MIN_TEXT_CHARS:
                    pages.append(PageText(
                        page_number=page_num,
                        text=text,
                        method="pdfplumber",
                        confidence_modifier=1.0,
                    ))
                elif self.ocr_fallback and self._check_ocr():
                    ocr_text = self._ocr_page(str(path), page_num)
                    pages.append(PageText(
                        page_number=page_num,
                        text=ocr_text,
                        method="ocr",
                        confidence_modifier=0.8,
                    ))
                else:
                    pages.append(PageText(
                        page_number=page_num,
                        text=text,
                        method="pdfplumber",
                        confidence_modifier=1.0,
                    ))

        return pages

    def extract_tables(self, path: str | Path, page_number: int | None = None) -> list[list[list[str | None]]]:
        """Extract tables from a PDF.

        Returns a list of tables, each table is a list of rows,
        each row is a list of cell values.
        """
        import pdfplumber

        path = Path(path)
        all_tables: list[list[list[str | None]]] = []

        with pdfplumber.open(str(path)) as pdf:
            target_pages = [pdf.pages[page_number - 1]] if page_number else pdf.pages
            for page in target_pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)

        return all_tables

    def get_metadata(self, path: str | Path) -> PDFMetadata:
        """Get metadata about a PDF without full extraction."""
        import pdfplumber

        path = Path(path)
        with pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            text_pages = 0
            ocr_pages = 0
            total_chars = 0

            for page in pdf.pages:
                text = page.extract_text() or ""
                chars = len(text.strip())
                total_chars += chars
                if chars >= _MIN_TEXT_CHARS:
                    text_pages += 1
                else:
                    ocr_pages += 1

        return PDFMetadata(
            path=str(path),
            page_count=page_count,
            text_pages=text_pages,
            ocr_pages=ocr_pages,
            total_chars=total_chars,
        )

    def _ocr_page(self, pdf_path: str, page_number: int) -> str:
        """OCR a single page using pdf2image + pytesseract."""
        try:
            import pytesseract
            from pdf2image import convert_from_path

            images = convert_from_path(
                pdf_path,
                first_page=page_number,
                last_page=page_number,
                dpi=self.dpi,
            )
            if not images:
                return ""
            text = pytesseract.image_to_string(images[0])
            return text
        except Exception as exc:
            logger.warning("OCR failed for %s page %d: %s", pdf_path, page_number, exc)
            return ""
