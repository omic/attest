"""Two-zone OCR for DOJ production-stamped pages.

DOJ releases frequently carry TWO overlapping Bates stamps in the top
header band: the court's original filing header and the DOJ production
header stamped on top of it. Neither is a redaction — both carry useful
metadata — but naive tesseract OCR on the overlap produces garbage.

Strategy:
  1. Split each page image into a header strip (top N pixels) and a body.
  2. OCR the body with tesseract (fast, free, high accuracy on clean text).
  3. OCR the header strip with a vision LLM capable of disentangling
     overlapping stamps (Gemini 2.5 Flash or similar). Output is structured:
     a list of {case, doc, filed, page_of} dicts, one per visible stamp.

Usage:
    from attestdb.extraction.header_ocr import HeaderOCR

    ocr = HeaderOCR(provider="gemini")   # or None → tesseract-only
    result = ocr.extract_page("DOJ-OGR-00008901.jpg")
    print(result.body_text)          # clean body via tesseract
    print(result.header_stamps)      # list of parsed stamp dicts
    print(result.bates_number)       # parsed from filename or footer
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_ROUTER_PATH = Path(__file__).parent / "router.json"


@lru_cache(maxsize=1)
def _load_router() -> dict:
    try:
        return json.loads(_ROUTER_PATH.read_text())
    except Exception as e:
        logger.warning("router.json unreadable (%s); using defaults", e)
        return {}


def _fragment_ratio(text: str) -> float:
    """Words of length <=2 / total words. High values suggest column-scramble."""
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return 0.0
    short = sum(1 for w in words if len(w) <= 2)
    return short / len(words)


def _pdftotext_page(path: Path, page_num: int, layout: bool = True) -> str:
    """Extract one page's text layer. Returns '' if pdftotext missing or fails."""
    if not shutil.which("pdftotext"):
        return ""
    args = ["pdftotext", "-f", str(page_num), "-l", str(page_num), "-q"]
    if layout:
        args.append("-layout")
    args += [str(path), "-"]
    try:
        out = subprocess.run(args, capture_output=True, timeout=15, check=False)
        return out.stdout.decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("pdftotext failed on %s p%d: %s", path, page_num, e)
        return ""

# Default header/footer heights in pixels, assuming ~300-DPI legal pages.
# Tuned against DOJ-OGR productions (US v. Maxwell, 1:20-cr-00330-AJN).
DEFAULT_HEADER_PX = 80
DEFAULT_FOOTER_PX = 60

# Minimum body text length to consider body OCR successful.
_MIN_BODY_CHARS = 100

# Regex for parsing court-style Bates stamps.
# Matches "Case 1:20-cr-00330-AJN Document 590 Filed 02/08/22 Page 2 of 11"
_STAMP_RE = re.compile(
    r"Case\s+(?P<case>[\w:.-]+)\s+"
    r"Document\s+(?P<doc>\d+)\s+"
    r"Filed\s+(?P<filed>\d{2}/\d{2}/\d{2,4})\s+"
    r"Page\s+(?P<page>\d+)\s+of\s+(?P<total>\d+)",
    re.IGNORECASE,
)

# DOJ production Bates pattern: DOJ-OGR-00008901 (or variants).
_BATES_RE = re.compile(r"\b(DOJ[-_][A-Z]+[-_]\d{6,})\b", re.IGNORECASE)


@dataclass
class HeaderStamp:
    """One parsed Bates/filing stamp from the header band."""

    case: str = ""
    doc: str = ""
    filed: str = ""          # MM/DD/YY or MM/DD/YYYY as printed
    page: int = 0
    total: int = 0
    raw: str = ""            # original text span, for debugging


@dataclass
class PageOCRResult:
    """Combined two-zone OCR output for a single page."""

    page_file: str
    body_text: str = ""
    body_method: str = ""     # "tesseract" | "pdftotext-layout" | "vlm:*" | "failed"
    body_confidence: float = 0.0
    header_stamps: list[HeaderStamp] = field(default_factory=list)
    header_method: str = ""   # "tesseract" | "gemini" | "none"
    bates_number: str = ""    # from filename or footer
    warnings: list[str] = field(default_factory=list)
    router_strategy: str = ""      # pdf_text_layer | scanned_typed | ...
    fragment_ratio: float = 0.0
    body_chars: int = 0

    @property
    def primary_case(self) -> str:
        """First (most likely court-original) case number, if any."""
        return self.header_stamps[0].case if self.header_stamps else ""

    @property
    def primary_doc(self) -> str:
        return self.header_stamps[0].doc if self.header_stamps else ""


class HeaderOCR:
    """Two-zone OCR orchestrator.

    Args:
        provider: Vision-LLM provider name for header disentanglement.
            Expected values: "gemini", "openai", or None (tesseract-only).
        header_px: Header strip height in pixels.
        footer_px: Footer strip height in pixels (searched for Bates).
        model: Optional model override for the vision call.
    """

    def __init__(
        self,
        provider: str | None = "gemini",
        header_px: int = DEFAULT_HEADER_PX,
        footer_px: int = DEFAULT_FOOTER_PX,
        model: str | None = None,
    ):
        self.provider = provider
        self.header_px = header_px
        self.footer_px = footer_px
        self.model = model
        self._client = None
        self._client_model: str | None = None
        if provider:
            self._init_vision_client()

    # ------------------------------------------------------------------
    # Client init
    # ------------------------------------------------------------------
    def _init_vision_client(self) -> None:
        """Lazy-init the vision LLM client via the intelligence layer.

        Falls back to tesseract-only mode if ``attestdb.intelligence``
        isn't installed or the requested provider isn't configured.
        """
        try:
            from attestdb.intelligence.llm_client import make_vision_client
        except ImportError:
            logger.info(
                "attestdb.intelligence not installed; header OCR falls back to tesseract"
            )
            self.provider = None
            return

        client, model = make_vision_client(self.provider)
        if client is None:
            logger.warning(
                "Vision provider %s not available; header OCR falls back to tesseract",
                self.provider,
            )
            self.provider = None
            return

        self._client = client
        self._client_model = self.model or model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_page(self, page_file: str | Path, page_num: int = 1) -> PageOCRResult:
        """Run two-zone OCR on a single page image (TIFF, JPG, PNG, PDF).

        page_num: 1-indexed page within a multi-page PDF. Ignored for single-
        page images. Phase-1 DOJ PDFs bundle multiple bates pages per file, so
        callers must pass the correct page index from the OPT bates offset.
        """
        path = Path(page_file)
        result = PageOCRResult(page_file=str(path))

        # Bates number is sometimes right there in the filename (DOJ-OGR-*)
        m = _BATES_RE.search(path.stem)
        if m:
            result.bates_number = m.group(1).upper()

        try:
            from PIL import Image
        except ImportError:
            result.warnings.append("Pillow not installed; install `investigation` extras")
            return result

        router = _load_router()

        # PDF text-layer probe: if a clean text layer exists, prefer it.
        if path.suffix.lower() == ".pdf":
            strat = router.get("pipelines", {}).get("pdf_text_layer", {})
            text = _pdftotext_page(path, page_num, layout=True)
            frag = _fragment_ratio(text)
            chars = len(text.strip())
            if chars >= 50 and frag <= 0.2:
                result.body_text = text.strip()
                result.body_method = "pdftotext-layout"
                result.body_confidence = 1.0
                result.router_strategy = "pdf_text_layer"
                result.fragment_ratio = round(frag, 3)
                result.body_chars = chars
                # still try to parse header stamps from image below; load img
                try:
                    img = self._load_image(path, page_num=page_num)
                except Exception as e:
                    result.warnings.append(f"could not open for header: {e}")
                    return result
                return self._finish_header_and_footer(img, result)

        try:
            img = self._load_image(path, page_num=page_num)
        except Exception as e:
            result.warnings.append(f"could not open: {e}")
            return result

        width, height = img.size
        header_box = (0, 0, width, min(self.header_px, height))
        footer_top = max(height - self.footer_px, 0)
        footer_box = (0, footer_top, width, height)
        body_box = (0, header_box[3], width, footer_top)

        header_img = img.crop(header_box)
        body_img = img.crop(body_box)
        footer_img = img.crop(footer_box)

        # Body OCR via tesseract (PSM 6 by default per router, env override)
        result.body_text, result.body_confidence, result.body_method = \
            self._ocr_body(body_img)
        result.router_strategy = "scanned_typed"
        result.fragment_ratio = round(_fragment_ratio(result.body_text), 3)
        result.body_chars = len(result.body_text.strip())

        # VLM fallback: only on thin text. frag>0.28 alone is a false positive on
        # line-numbered transcripts (PSM 6 clean output measures frag≈0.23-0.36
        # because "1"–"25" line numbers are ≤2 chars); require frag to co-occur
        # with low chars so we don't escalate every clean transcript page.
        strat = router.get("pipelines", {}).get("scanned_typed", {})
        trigger_fallback = (
            result.body_chars < 200
            or (result.fragment_ratio > 0.40 and result.body_chars < 500)
        )
        if trigger_fallback:
            result.warnings.append(
                f"body thin: {result.body_chars}c frag={result.fragment_ratio}"
            )
            if self.provider and self._client:
                vlm_text = self._vlm_body_transcribe(img)
                if vlm_text and len(vlm_text.strip()) > result.body_chars:
                    result.body_text = vlm_text
                    result.body_method = f"vlm:{self.provider}"
                    result.body_confidence = 0.0
                    result.fragment_ratio = round(_fragment_ratio(vlm_text), 3)
                    result.body_chars = len(vlm_text.strip())

        return self._finish_header_and_footer(img, result)

    def _finish_header_and_footer(self, img, result: "PageOCRResult") -> "PageOCRResult":
        """Run header-stamp parsing and footer bates extraction on full-page image."""
        width, height = img.size
        header_box = (0, 0, width, min(self.header_px, height))
        footer_top = max(height - self.footer_px, 0)
        footer_box = (0, footer_top, width, height)
        header_img = img.crop(header_box)
        footer_img = img.crop(footer_box)

        # Header OCR: tesseract first, LLM if two stamps suspected
        header_text_naive, _, _ = self._ocr_body(header_img)
        header_stamps_naive = self._parse_stamps(header_text_naive)
        if header_stamps_naive and not self._overlap_suspected(header_text_naive):
            result.header_stamps = header_stamps_naive
            result.header_method = "tesseract"
        elif self.provider and self._client:
            llm_stamps = self._ocr_header_vision(header_img)
            result.header_stamps = llm_stamps or header_stamps_naive
            result.header_method = "gemini" if llm_stamps else "tesseract"
        else:
            result.header_stamps = header_stamps_naive
            result.header_method = "tesseract" if header_stamps_naive else "none"

        # Footer: Bates number fallback if not from filename
        if not result.bates_number:
            footer_text, _, _ = self._ocr_body(footer_img)
            fm = _BATES_RE.search(footer_text)
            if fm:
                result.bates_number = fm.group(1).upper()

        return result

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_image(self, path: Path, page_num: int = 1):
        """Open image file; for PDFs, rasterize the requested page (1-indexed)."""
        from PIL import Image

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError as e:
                raise RuntimeError("pdf2image not installed") from e
            pages = convert_from_path(
                str(path), first_page=page_num, last_page=page_num, dpi=300,
            )
            if not pages:
                raise RuntimeError(f"PDF page {page_num} not found in {path}")
            return pages[0]
        return Image.open(str(path))

    # ------------------------------------------------------------------
    # Tesseract body OCR
    # ------------------------------------------------------------------
    def _ocr_body(self, img) -> tuple[str, float, str]:
        try:
            import pytesseract
        except ImportError:
            return "", 0.0, "failed"
        try:
            psm = os.environ.get("ATTEST_TESSERACT_PSM", "6")
            config = f"--psm {psm}"
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT, config=config
            )
            text_parts = [w for w in data["text"] if w.strip()]
            confs = [int(c) for c in data["conf"] if isinstance(c, (int, str)) and str(c).lstrip("-").isdigit() and int(c) >= 0]
            text = " ".join(text_parts)
            # Normalize newlines: multiple spaces -> single, preserve paragraph hints
            text = re.sub(r"\s+", " ", text).strip()
            confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
            return text, round(confidence, 3), "tesseract"
        except Exception as e:
            logger.warning("tesseract body OCR failed: %s", e)
            return "", 0.0, "failed"

    # ------------------------------------------------------------------
    # Stamp parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_stamps(text: str) -> list[HeaderStamp]:
        stamps: list[HeaderStamp] = []
        for m in _STAMP_RE.finditer(text):
            stamps.append(HeaderStamp(
                case=m.group("case"),
                doc=m.group("doc"),
                filed=m.group("filed"),
                page=int(m.group("page")),
                total=int(m.group("total")),
                raw=m.group(0),
            ))
        return stamps

    @staticmethod
    def _overlap_suspected(text: str) -> bool:
        """Heuristic: does the header OCR look like stamps are overlapping?

        Indicators:
        - "Case" appears more than once in a very short string
        - Bizarre letter combinations from stacked text (e.g. 'CaseS2D29')
        - Multiple 'Filed' or 'Document' tokens
        """
        if not text:
            return False
        case_hits = len(re.findall(r"\bCase\b", text, re.IGNORECASE))
        filed_hits = len(re.findall(r"\bFiled\b", text, re.IGNORECASE))
        doc_hits = len(re.findall(r"\bDocument\b", text, re.IGNORECASE))
        # When two headers overlap, tesseract emits concatenated-garbage tokens:
        # e.g. "CaseS2D29-00830GPAEJN" where 'Case' runs directly into a mix
        # of uppercase letters and digits with no space. A single clean header
        # would always have "Case 1:20-cr-..." with a space after 'Case'.
        jumbled = bool(re.search(
            r"\b(?:Case|Document|Filed|Page)[A-Z0-9]{2,}", text
        )) or bool(re.search(r"\d[A-Z]{2,}\d", text))
        return case_hits > 1 or filed_hits > 1 or doc_hits > 1 or jumbled

    # ------------------------------------------------------------------
    # Vision-LLM header disentanglement
    # ------------------------------------------------------------------
    def _ocr_header_vision(self, header_img) -> list[HeaderStamp]:
        """Send the header strip to a vision LLM, parse JSON stamps."""
        if not self._client:
            return []

        buf = io.BytesIO()
        header_img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        prompt = (
            "This image is the top header band of a court-filed document page. "
            "It may contain one OR TWO Bates/filing stamps printed overlapping "
            "each other. Transcribe every visible stamp separately. "
            "Return STRICTLY a JSON object of the form "
            '{"stamps": [{"case": "...", "doc": "...", "filed": "MM/DD/YY", '
            '"page": N, "total": N}, ...]} — no prose, no code fences. '
            "Use empty strings / 0 for fields you cannot read."
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._client_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                temperature=0.0,
                max_tokens=400,
            )
        except Exception as e:
            logger.warning("vision header OCR call failed: %s", e)
            return []

        raw = (resp.choices[0].message.content or "").strip()
        # Strip possible ```json fences
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("vision header OCR returned non-JSON: %s", raw[:200])
            return []

        stamps: list[HeaderStamp] = []
        for s in data.get("stamps", []):
            stamps.append(HeaderStamp(
                case=str(s.get("case", "")),
                doc=str(s.get("doc", "")),
                filed=str(s.get("filed", "")),
                page=int(s.get("page") or 0),
                total=int(s.get("total") or 0),
                raw=json.dumps(s),
            ))
        return stamps

    def _vlm_body_transcribe(self, page_img) -> str:
        """Send full-page image to VLM for transcription + description.

        Used when tesseract returns empty on image-only scanned PDFs.
        Returns a transcription of any visible text plus a brief structural
        description (e.g. form type, handwriting vs print, visible dates).
        """
        if not self._client:
            return ""
        buf = io.BytesIO()
        page_img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        prompt = (
            "This is a single page from a DOJ document production. "
            "Transcribe every piece of visible text — printed, "
            "handwritten, stamped, or redacted. Preserve line breaks and "
            "tabular structure. After the transcription, on a new line "
            "starting with 'DESCRIPTION:', give a one-sentence description "
            "of what the page appears to be (e.g. flight manifest, letter, "
            "financial ledger, photograph, blank page, redaction cover)."
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._client_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                temperature=0.0,
                max_tokens=2000,
            )
        except Exception as e:
            logger.warning("VLM body transcription failed: %s", e)
            return ""
        return (resp.choices[0].message.content or "").strip()
