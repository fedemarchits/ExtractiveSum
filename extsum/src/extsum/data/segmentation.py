"""Sentence segmentation.

Two splitters behind one entry point, ``sentence_split(text, dataset_name)``:

- news-style text (XSum, CNN/DailyMail): NLTK punkt, with a regex fallback;
- legislative bills (BillSum): structure-aware. Each enumerated provision is
  one unit and a section header is attached to the sentence that follows it,
  so the model never has to select a bare "(a) IN GENERAL.--" line.

Silver labels are sentence indices, so segmentation is part of the dataset:
changing it changes every label. Re-generate the silver files after any edit.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple


def _try_nltk_sent_tokenize(text: str) -> List[str]:
    """Sentence segmentation using NLTK Punkt."""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass


        return [
            s.strip()
            for s in sent_tokenize(text)
            if s.strip()
        ]

    except Exception:
        return []


def _regex_sentence_split(text: str) -> List[str]:
    """Fallback deterministic sentence splitter."""
    chunks = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# ---------------------------------------------------------------------------
# BillSum: structure-aware segmentation
# ---------------------------------------------------------------------------
# Bills are not prose. Collapsing whitespace and running Punkt (the XSum/CNN-DM
# path) turns section headers into fragments ("SEC.", "2.", "FINDINGS.") and
# leaves whole enumerations, whose items end in ";" rather than ".", glued
# into one several-hundred-word "sentence". The line layout carries the real
# structure, so BillSum is segmented from it before whitespace is normalized.

# "SECTION 1. SHORT TITLE." / "SEC. 2." (US and California formats), including
# headers of quoted amendment text: "``SEC. 123. DEFINITIONS."
_BILL_HEADER_RE = re.compile(
    r"^[`'\"“]*(?:SECTION|SEC\.|Sec\.)\s+\d+[A-Za-z\-]*\."
)

# A unit that is only a section label, e.g. "Sec. 2." split off by Punkt.
_BILL_LABEL_ONLY_RE = re.compile(
    r"^[`'\"“]*(?:SECTION|SEC|Section|Sec)\.?(?:\s+\d+[A-Za-z\-]*\.)?$"
)

# A unit with no words of its own: the closing "''." of quoted amendment text,
# optionally followed by a list connective ("''; and").
_BILL_PUNCT_ONLY_RE = re.compile(r"^[\W_]*(?:(?:and|or)[\W_]*)?$")

# Enumerators opening a provision: (a), (1), (A), (ii), (aa) ..., also inside
# quoted amendment text: "``(1) In general.--"
_BILL_ENUM_RE = re.compile(
    r"^[`'\"“]*\((?:\d{1,3}|[A-Za-z]{1,2}|[ivxlcIVXLC]{1,6})\)"
)

# A line after which a new enumerated provision may legitimately start. This
# separates a real item from a hard-wrapped cross-reference such as
# "... described in paragraph\n(2) of subsection (a)".
_BILL_BREAK_END_RE = re.compile(
    r"(?:[.;:][\'\"`\u2019\u201d]*|--|[;,]\s*(?:and|or))$"
)

# Abbreviations Punkt wrongly treats as sentence ends in legal citations.
_BILL_ABBREV_END_RE = re.compile(
    r"(?:^|[\s(`'\"])(?:Sec|Secs|No|Nos|Stat|Stats|Pub|L|Rev|Res|Con|Ch|Art|"
    r"Cl|Div|Pt|Par|Para|Subsec|Subpar|Inc|Co|Corp|Ltd|vs|v|U\.S\.C|U\.S|"
    r"C\.F\.R|H\.R|H\.J|S\.J|Fed|Reg|St|Mr|Mrs|Ms|Dr|Jr|Sr|Gen|Rep|Sen|Gov|"
    r"et seq|et al)\.$",
    re.IGNORECASE,
)

# Abbreviations that never end a sentence, whatever follows.
_BILL_ALWAYS_MERGE_RE = re.compile(
    r"(?:^|[\s(])(?:Pub|L|H\.R|H\.J|S\.J|vs|v)\.$",
    re.IGNORECASE,
)


def _is_bill_header(line: str) -> bool:
    """True for a section-header line, false for "Sec. 2. The Secretary shall"."""
    match = _BILL_HEADER_RE.match(line)

    if not match:
        return False

    title = line[match.end():].strip()

    if not re.search(r"[a-z]", title):
        # "SEC. 2. FINDINGS." or a bare California "SEC. 2."
        return True

    # Title-case header of quoted amendment text: "``Sec. 3. Definitions."
    words = title.split()

    return (
        len(words) <= 8
        and title.endswith(".")
        and all(
            word[0].isupper()
            for word in words
            if len(word) > 3
        )
    )


def _join_wrapped_lines(lines: Sequence[str]) -> str:
    """Join hard-wrapped lines, repairing words hyphenated across a wrap."""
    text = ""

    for line in lines:
        if not text:
            text = line
        elif text.endswith("-") and not text.endswith("--"):
            # "Pick-" + "Sloan" -> "Pick-Sloan"
            text += line
        else:
            text += " " + line

    return text


def _bill_blocks(text: str) -> List[Tuple[bool, str]]:
    """Group the lines of a bill into ``(is_header, text)`` structural blocks.

    A block is a section header, an enumerated provision, or a paragraph.
    Hard-wrapped continuation lines are joined back onto their block.
    """
    blocks: List[Tuple[bool, str]] = []
    current: List[str] = []
    current_is_header = False
    header_open = False
    previous_blank = True

    def flush() -> None:
        nonlocal current, current_is_header

        if current:
            blocks.append(
                (current_is_header, _join_wrapped_lines(current))
            )

        current = []
        current_is_header = False

    for raw_line in str(text).splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            flush()
            header_open = False
            previous_blank = True
            continue

        if header_open:
            # A header title wrapped onto a second line.
            current.append(line)
            header_open = not line.endswith(".")
            previous_blank = False
            continue

        is_header = _is_bill_header(line)

        starts_block = (
            previous_blank
            or is_header
            or current_is_header
            or (
                bool(_BILL_ENUM_RE.match(line))
                and bool(current)
                and bool(_BILL_BREAK_END_RE.search(current[-1]))
            )
        )

        if starts_block:
            flush()

        current.append(line)
        previous_blank = False

        if is_header:
            current_is_header = True
            header_open = not line.endswith(".")

    flush()

    return blocks


def _split_bill_block(block: str) -> List[str]:
    """Split one structural block into sentences, repairing citation splits."""
    pieces = _try_nltk_sent_tokenize(block) or _regex_sentence_split(block)

    merged: List[str] = []

    for piece in pieces:
        if merged:
            previous = merged[-1]

            continues_citation = bool(
                _BILL_ABBREV_END_RE.search(previous)
            ) and bool(re.match(r"[\da-z(]", piece))

            if (
                continues_citation
                or _BILL_ALWAYS_MERGE_RE.search(previous)
                or _BILL_LABEL_ONLY_RE.match(previous)
                or _BILL_PUNCT_ONLY_RE.match(piece)
                or re.match(r"[a-z),;]", piece)
            ):
                merged[-1] = f"{previous} {piece}"
                continue

        merged.append(piece)

    return merged


def bill_sentence_split(text: str) -> List[str]:
    """Segment a bill into extractable units using its line structure.

    - Every enumerated provision, "(a)", "(1)", "(A)", ..., is its own unit, so
      ";"-terminated list items are no longer fused into one giant sentence.
    - Prose inside a provision is still split into sentences.
    - A section header is attached to the unit that follows it instead of being
      emitted as a stand-alone fragment.
    """
    units: List[str] = []
    pending_header = ""

    for is_header, block in _bill_blocks(text):
        if is_header:
            pending_header = (
                f"{pending_header} {block}".strip()
            )
            continue

        sentences = _split_bill_block(block)

        if not sentences:
            continue

        if units and _BILL_PUNCT_ONLY_RE.match(sentences[0]):
            # A stray closing quote on its own line belongs to what it closes.
            units[-1] = f"{units[-1]} {sentences[0]}"
            sentences = sentences[1:]

            if not sentences:
                continue

        if pending_header:
            sentences[0] = f"{pending_header} {sentences[0]}"
            pending_header = ""

        units.extend(sentences)

    if pending_header:
        units.append(pending_header)

    return [
        unit.strip()
        for unit in units
        if unit.strip()
    ]


def sentence_split(
    text: str,
    dataset_name: str | None = None,
) -> List[str]:
    """Split raw document text into sentences.

    the heuristic study used pre-segmented ACLSum sentences.
    The document task receives raw XSum/CNN-DailyMail/BillSum documents and therefore performs
    deterministic sentence segmentation before applying the unchanged
    local-score heuristic.

    BillSum uses the structure-aware ``bill_sentence_split``; XSum and
    CNN/DailyMail keep the original whitespace-collapse + Punkt path, so their
    existing silver labels are unaffected.
    """
    if str(dataset_name or "").strip().lower() == "billsum":
        return bill_sentence_split(text)

    text = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    if not text:
        return []

    sentences = _try_nltk_sent_tokenize(text)

    if sentences:
        return sentences

    return _regex_sentence_split(text)
