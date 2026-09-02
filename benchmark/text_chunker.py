from __future__ import annotations

"""
Chunking helpers for Granite paper preprocessing.

Target methods:
- boundary: fixed-size character chunking used by the current baseline
- structure_aware: reserved for section/paragraph-aware splitting
- adaptive_verified: reserved for verification-aware adaptive splitting

Sprint 0 note:
- equation_table_safe is the temporary internal name for the legacy
  equation/table-protection path that will later be folded into
  adaptive_verified.
"""

import re
from typing import Literal

ChunkMethod = Literal["boundary", "equation_table_safe", "structure_aware", "adaptive_verified"]

_NUMBERED_HEADER_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*\.?)\s+(.+?)\s*$")
_TITLE_CASE_HEADER_RE = re.compile(r"^[A-Z][A-Za-z0-9/&,\-() ]{1,79}$")
_CAPTION_PREFIX_RE = re.compile(r"^\s*(figure|table|algorithm)\s+\d+", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*\.?\s+\S+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_DIGIT_TOKEN_RE = re.compile(r"\S*\d\S*")
_EQUATION_LINE_RE = re.compile(r"(?:^|\s)(?:y|x|f\(x\)|h\(x\)|w\d*|relu|avg pool|fc \d+|conv\d+)\b|[=+\-*/]|→|<-|->", re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"(?:\S*\d\S*\s+){2,}\S*")
_FACT_TOKEN_RE = re.compile(r"\b(?:\d+(?:\.\d+)?%?|\d+[x×]\d+|[A-Z][A-Za-z0-9_-]{2,})\b")
_SUMMARY_SECTION_RE = re.compile(r"summary:\s*", re.IGNORECASE)
_COVERAGE_SECTION_RE = re.compile(r"coverage:\s*", re.IGNORECASE)


def _find_section_headers(text: str) -> list[int]:
    """
    Returns character offsets of lines that look like real section
    headers (e.g. '3.2 Residual Learning', '4. Experiments'), NOT
    numbered captions or list items.

    A line qualifies as a header if ALL of:
      - starts with 1-2 dot-separated numbers OR is short Title Case text
      - line length <= 80 chars
      - does not end in a period, comma, or colon
      - the following line does not start with the same number pattern
        (rules out numbered lists)
    Reject lines containing "Figure", "Table", "Algorithm" immediately
    followed by a number (those are captions, not headers).
    """

    offsets: list[int] = []
    running_offset = 0
    lines = text.splitlines(keepends=True)

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            running_offset += len(raw_line)
            continue

        if len(line) > 80 or line.endswith((".", ",", ":")):
            running_offset += len(raw_line)
            continue

        if len(_DIGIT_TOKEN_RE.findall(line)) >= 2:
            running_offset += len(raw_line)
            continue

        if _CAPTION_PREFIX_RE.match(line):
            running_offset += len(raw_line)
            continue

        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""

        numbered_match = _NUMBERED_HEADER_RE.match(line)
        if numbered_match:
            number_prefix = numbered_match.group(1).rstrip(".")
            title_part = numbered_match.group(2).strip()

            if not re.search(r"[A-Za-z]", title_part):
                running_offset += len(raw_line)
                continue

            if ":" in title_part:
                running_offset += len(raw_line)
                continue

            if _CAPTION_PREFIX_RE.match(title_part):
                running_offset += len(raw_line)
                continue

            if next_line and _LIST_ITEM_RE.match(next_line):
                next_match = _NUMBERED_HEADER_RE.match(next_line)
                if next_match and next_match.group(1).rstrip(".") == number_prefix:
                    running_offset += len(raw_line)
                    continue

            offsets.append(running_offset)
            running_offset += len(raw_line)
            continue

        if _TITLE_CASE_HEADER_RE.match(line):
            words = [word for word in re.split(r"\s+", line) if word]
            alphabetic_words = [word for word in words if re.search(r"[A-Za-z]", word)]
            if (
                1 <= len(words) <= 8
                and alphabetic_words
                and (len(words) > 1 or (len(words) == 1 and alphabetic_words[0].isalpha()))
                and sum(word[:1].isupper() for word in alphabetic_words if word[:1].isalpha()) >= max(1, len(alphabetic_words) // 2)
            ):
                offsets.append(running_offset)

        running_offset += len(raw_line)

    return offsets


def _find_paragraph_breaks(text: str) -> list[int]:
    return [match.start() for match in re.finditer(r"(?:\r?\n){2,}", text)]


def _find_protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    running_offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if not stripped:
            running_offset += len(raw_line)
            continue

        is_caption = bool(_CAPTION_PREFIX_RE.match(stripped))
        digit_tokens = _DIGIT_TOKEN_RE.findall(stripped)
        is_table_row = bool(_TABLE_ROW_RE.search(stripped)) or len(digit_tokens) >= 3
        is_equation = bool(_EQUATION_LINE_RE.search(stripped)) and len(stripped) <= 120

        if is_caption or is_table_row or is_equation:
            spans.append((running_offset, running_offset + len(raw_line.rstrip("\r\n"))))

        running_offset += len(raw_line)

    return spans


def _is_inside_protected_span(offset: int, protected_spans: list[tuple[int, int]]) -> bool:
    return any(start < offset < end for start, end in protected_spans)


def _nearest_safe_cut(offset: int, protected_spans: list[tuple[int, int]], fallback: int) -> int:
    for start, end in protected_spans:
        if start < offset < end:
            distance_to_start = abs(offset - start)
            distance_to_end = abs(end - offset)
            candidate = start if distance_to_start <= distance_to_end else end
            return candidate if candidate > 0 else fallback
    return offset


def _select_cut_point(
    text: str,
    target_end: int,
    search_start: int,
    search_end: int,
    *,
    protected_spans: list[tuple[int, int]] | None = None,
) -> int:
    candidates: list[int] = []
    protected = protected_spans or []

    header_offsets = [
        offset
        for offset in _find_section_headers(text)
        if search_start < offset <= search_end and not _is_inside_protected_span(offset, protected)
    ]
    if header_offsets:
        candidates.extend(header_offsets)

    paragraph_offsets = [
        offset
        for offset in _find_paragraph_breaks(text)
        if search_start < offset <= search_end and not _is_inside_protected_span(offset, protected)
    ]
    if paragraph_offsets:
        candidates.extend(paragraph_offsets)

    if candidates:
        return _nearest_safe_cut(min(candidates, key=lambda offset: abs(target_end - offset)), protected, target_end)

    sentence_candidates = [
        match.end()
        for match in _SENTENCE_BOUNDARY_RE.finditer(text[search_start:search_end])
    ]
    if sentence_candidates:
        absolute = [
            search_start + offset
            for offset in sentence_candidates
            if not _is_inside_protected_span(search_start + offset, protected)
        ]
        if absolute:
            return _nearest_safe_cut(min(absolute, key=lambda offset: abs(target_end - offset)), protected, target_end)

    return _nearest_safe_cut(target_end, protected, target_end)


def _split_structure_aware(
    text: str,
    chunk_size: int,
    *,
    protected_spans: list[tuple[int, int]] | None = None,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    text_length = len(text)
    search_radius = max(200, chunk_size // 5)
    protected = protected_spans or []

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        if target_end >= text_length:
            chunks.append(text[start:])
            break

        search_start = min(text_length, max(start + 1, target_end - search_radius))
        search_end = min(text_length, target_end + search_radius)
        cut_point = _select_cut_point(
            text,
            target_end,
            search_start,
            search_end,
            protected_spans=protected,
        )

        if cut_point <= start:
            cut_point = target_end

        chunks.append(text[start:cut_point])
        start = cut_point

    return chunks


def _split_equation_table_safe(text: str, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    protected_spans = _find_protected_spans(text)
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        if target_end >= text_length:
            chunks.append(text[start:])
            break

        cut_point = _nearest_safe_cut(target_end, protected_spans, target_end)
        if cut_point <= start:
            cut_point = target_end

        chunks.append(text[start:cut_point])
        start = cut_point

    return chunks


def verify_chunk_summary(chunk_text: str, summary_text: str) -> dict[str, object]:
    normalized_summary = _COVERAGE_SECTION_RE.split(_SUMMARY_SECTION_RE.sub("", summary_text), maxsplit=1)[0]
    chunk_facts = []
    seen = set()
    for match in _FACT_TOKEN_RE.findall(chunk_text):
        token = match.strip()
        if token not in seen:
            chunk_facts.append(token)
            seen.add(token)

    if not chunk_facts:
        return {"coverage": 1.0, "flag": "OK", "facts_missing": []}

    summary_lower = normalized_summary.lower()
    facts_missing = [fact for fact in chunk_facts if fact.lower() not in summary_lower]
    covered = len(chunk_facts) - len(facts_missing)
    coverage = round(covered / len(chunk_facts), 3)
    return {
        "coverage": coverage,
        "flag": "OK" if coverage >= 0.8 else "FACTS_DROPPED",
        "facts_missing": facts_missing,
    }


def split_into_chunks(text: str, chunk_size: int, *, method: ChunkMethod = "boundary") -> list[str]:
    """
    Split text into chunks according to the requested method.

    Sprint 0 behavior:
    - boundary preserves the current baseline exactly via fixed-size
      character slicing.
    - equation_table_safe is a temporary alias that intentionally
      matches the current slicing behavior until the legacy protected
      logic is reintroduced in a later sprint.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if method not in {"boundary", "equation_table_safe", "structure_aware", "adaptive_verified"}:
        raise ValueError(f"Unsupported chunking method: {method}")

    if method == "boundary":
        return [text[index:index + chunk_size] for index in range(0, len(text), chunk_size)]

    if method == "equation_table_safe":
        return _split_equation_table_safe(text, chunk_size)

    if method == "structure_aware":
        return _split_structure_aware(text, chunk_size)

    if method == "adaptive_verified":
        return _split_structure_aware(text, chunk_size, protected_spans=_find_protected_spans(text))

    raise NotImplementedError(f"Chunking method '{method}' is not implemented yet")
