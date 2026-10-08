"""Small text helpers: number/negation/modality extraction and fuzzy span location."""
from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

_UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000, "lakh": 100_000,
           "crore": 10_000_000, "k": 1_000}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
             "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "fifteenth": 15, "twentieth": 20,
             "thirtieth": 30}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) - {"k"}

NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot", "nope",
             "nowhere"}
MODALS = {"will", "would", "shall", "should", "must", "might", "may", "can", "could", "maybe", "perhaps",
          "probably", "definitely"}

_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def number_values(text: str) -> Counter:
    """Multiset of numeric values mentioned, whether written as digits or words.

    "twenty five" == "25", "Q three" == "Q3", "four hundred" == "400", "EC2" == "easy two".
    """
    values: Counter = Counter()
    words = re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)*(?:st|nd|rd|th)?|[^\sA-Za-z\d]", text.lower())
    current: float | None = None
    total = 0.0

    def flush():
        nonlocal current, total
        if current is not None or total:
            values[_fmt(total + (current or 0))] += 1
        current, total = None, 0.0

    for w in words:
        if w in _UNITS:
            current = (current or 0) + _UNITS[w]
        elif w in _TENS:
            current = (current or 0) + _TENS[w]
        elif w in ("hundred",) and current is not None:
            current *= 100
        elif w in _SCALES and w != "k" and current is not None:
            total += current * _SCALES[w]
            current = None
        elif w == "and" and current is not None:
            continue
        elif w in _ORDINALS:
            flush()
            values[_fmt(_ORDINALS[w])] += 1
        elif re.fullmatch(r"\d+(?:[.,]\d+)*(?:st|nd|rd|th)?", w):
            flush()
            num = re.sub(r"(st|nd|rd|th)$", "", w).replace(",", "")
            try:
                values[_fmt(float(num))] += 1
            except ValueError:
                values[num] += 1
        else:
            flush()
    flush()
    return values


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def negation_count(text: str) -> int:
    toks = [t.lower() for t in tokens(text)]
    return sum(1 for t in toks if t in NEGATIONS or t.endswith("n't"))


def modal_counter(text: str) -> Counter:
    toks = [t.lower() for t in tokens(text)]
    out: Counter = Counter()
    for t in toks:
        if t in MODALS:
            out[t] += 1
        elif t.endswith("'ll"):
            out["will"] += 1
        elif t in ("won't", "wont"):
            out["will"] += 1
        elif t.endswith("'d"):
            out["would"] += 1
    return out


def locate(span: str, text: str, threshold: float = 90.0) -> tuple[int, int] | None:
    """Find ``span`` in ``text``: exact, then case-insensitive, then fuzzy (token aligned)."""
    if not span:
        return None
    idx = text.find(span)
    if idx != -1:
        return idx, idx + len(span)
    low = text.lower().find(span.lower())
    if low != -1:
        return low, low + len(span)
    norm_span = re.sub(r"\s+", " ", span.strip(" .,;:!?\"'")).lower()
    low = text.lower().find(norm_span)
    if norm_span and low != -1:
        return low, low + len(norm_span)
    res = fuzz.partial_ratio_alignment(norm_span, text.lower())
    if res and res.score >= threshold:
        a, b = res.dest_start, res.dest_end
        # Snap to word boundaries.
        while a > 0 and text[a - 1].isalnum():
            a -= 1
        while b < len(text) and text[b].isalnum():
            b += 1
        return a, b
    return None


def normalise_for_match(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace — for fuzzy evidence matching."""
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())
