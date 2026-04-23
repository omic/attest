"""Group consecutive Bates-stamped pages into logical documents.

DOJ productions ship as one file per page (DOJ-OGR-NNNNNNNN.tif/jpg).
A "document" is a run of consecutive pages sharing the same court
(case_number, document_number) header stamp. This module clusters pages
into documents using HeaderOCR output, with fallback strategies for
pages whose headers couldn't be parsed.

Usage:
    from attestdb.extraction.page_grouper import group_pages
    from attestdb.extraction.header_ocr import HeaderOCR

    ocr = HeaderOCR(provider="gemini")
    headers = [ocr.extract_page(p) for p in sorted_page_paths]
    documents = group_pages(headers)
    for doc in documents:
        print(doc.doc_id, doc.page_count, doc.bates_range)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from attestdb.extraction.header_ocr import PageOCRResult

logger = logging.getLogger(__name__)

_BATES_NUM_RE = re.compile(r"(\d+)$")


@dataclass
class DocumentGroup:
    """A logical document reassembled from consecutive pages."""

    doc_id: str                       # "{case}:{doc}" e.g. "1:20-cr-00330-AJN:590"
    case_number: str
    document_number: str
    total_pages_declared: int         # from header "Page X of Y"
    filed_on: str = ""                # earliest filing date seen
    page_files: list[str] = field(default_factory=list)  # in order
    bates_numbers: list[str] = field(default_factory=list)
    stamps_seen: list[dict] = field(default_factory=list)  # raw header stamps
    gaps: list[tuple[int, int]] = field(default_factory=list)  # missing page ranges
    notes: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.page_files)

    @property
    def is_complete(self) -> bool:
        return (
            self.total_pages_declared > 0
            and self.page_count == self.total_pages_declared
            and not self.gaps
        )

    @property
    def bates_range(self) -> str:
        if not self.bates_numbers:
            return ""
        return f"{self.bates_numbers[0]} — {self.bates_numbers[-1]}"


def _bates_seq(bates: str) -> int:
    """Extract the trailing numeric portion of a Bates number, or -1."""
    if not bates:
        return -1
    m = _BATES_NUM_RE.search(bates)
    return int(m.group(1)) if m else -1


def group_pages(
    headers: list[PageOCRResult],
    *,
    max_gap: int = 2,
) -> list[DocumentGroup]:
    """Cluster page headers into DocumentGroup objects.

    Walks the input in the order given (assumed: Bates-ascending). A new
    document starts whenever (case, doc) changes or the header-declared
    `page` number resets to 1. Pages with unreadable headers attach to the
    current document if their Bates number is contiguous within `max_gap`.

    Args:
        headers: PageOCRResult in Bates order.
        max_gap: Max Bates-sequence gap before declaring a boundary.

    Returns:
        List of DocumentGroup in input order.
    """
    documents: list[DocumentGroup] = []
    current: DocumentGroup | None = None
    last_bates_seq = -999

    for h in headers:
        seq = _bates_seq(h.bates_number)
        case = h.primary_case
        doc = h.primary_doc
        declared_total = h.header_stamps[0].total if h.header_stamps else 0
        page_in_doc = h.header_stamps[0].page if h.header_stamps else 0

        starts_new_doc = False
        if current is None:
            starts_new_doc = True
        elif case and doc:
            # Header is readable — authoritative boundary signal
            if (case, doc) != (current.case_number, current.document_number):
                starts_new_doc = True
            elif page_in_doc == 1 and current.page_count > 0:
                # Explicit new document starting at page 1 of same case? unusual but possible
                starts_new_doc = True
        else:
            # Header unreadable — use Bates contiguity only
            if seq >= 0 and last_bates_seq >= 0 and (seq - last_bates_seq) > max_gap:
                starts_new_doc = True
                current.notes.append(
                    f"boundary inferred from bates gap at {h.bates_number}"
                )

        if starts_new_doc:
            if current is not None:
                _finalize_document(current)
                documents.append(current)
            current = DocumentGroup(
                doc_id=_make_doc_id(case, doc, h),
                case_number=case,
                document_number=doc,
                total_pages_declared=declared_total,
                filed_on=h.header_stamps[0].filed if h.header_stamps else "",
            )

        assert current is not None
        current.page_files.append(h.page_file)
        if h.bates_number:
            current.bates_numbers.append(h.bates_number)
        if h.header_stamps:
            current.stamps_seen.append({
                "case": case, "doc": doc,
                "filed": h.header_stamps[0].filed,
                "page": page_in_doc,
                "total": declared_total,
                "bates": h.bates_number,
            })
        if not case or not doc:
            current.notes.append(f"{h.bates_number or h.page_file}: header unreadable")
        if seq >= 0:
            last_bates_seq = seq

    if current is not None:
        _finalize_document(current)
        documents.append(current)

    return documents


def _finalize_document(doc: DocumentGroup) -> None:
    """Compute gaps and validate page count after all pages attached."""
    seen_pages = sorted({
        s["page"] for s in doc.stamps_seen if s.get("page")
    })
    if doc.total_pages_declared > 0:
        missing: list[int] = []
        for p in range(1, doc.total_pages_declared + 1):
            if p not in seen_pages:
                missing.append(p)
        # Collapse consecutive missing pages into (start, end) ranges
        if missing:
            start = prev = missing[0]
            for p in missing[1:]:
                if p == prev + 1:
                    prev = p
                else:
                    doc.gaps.append((start, prev))
                    start = prev = p
            doc.gaps.append((start, prev))


def _make_doc_id(case: str, doc: str, header: PageOCRResult) -> str:
    """Canonical document ID. Falls back to Bates if header unreadable."""
    if case and doc:
        return f"{case}:{doc}"
    if header.bates_number:
        return f"bates:{header.bates_number}"
    return f"unknown:{Path(header.page_file).stem}"


def page_order_by_bates(page_files: list[str]) -> list[str]:
    """Sort page files by numeric Bates sequence in the filename."""
    def sort_key(p: str) -> tuple[int, str]:
        stem = Path(p).stem
        seq = _bates_seq(stem)
        return (seq if seq >= 0 else 10**12, stem)
    return sorted(page_files, key=sort_key)
