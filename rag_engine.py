"""Module 3: grounded note matching over context_notes.csv.

For one flagged route-week, return either a note that genuinely explains the cost rise for
that route and that week, or nothing (verdict "unexplained"). A wrong citation scores worse
than no citation, so every step fails closed:

1. Each note is parsed once into a NoteRecord: which routes it covers, the dates it is in
   force (its "window"), and whether it describes a cost increase at all.
2. The notes are embedded into a small TF-IDF vector index (cosine similarity).
3. Per flagged row, retrieval runs with a metadata filter on route, then two hard filters:
   the week must overlap the note's window, and the note's effect must be "increase".
   Similarity only orders notes that already passed every filter, so it can never promote
   an ineligible note into matched_note_id.

Everything here is plain Python and numpy, so the same input gives the same match on
every run and every machine.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ALL_ROUTES = "All Routes"

# A note that starts a lasting change ("starting this week") is treated as in force for this
# many weeks. After 8 weeks the change is fully inside the own-history baseline, so it can
# no longer be the reason a week looks unusual.
OPEN_ENDED_WEEKS = 8

# How far (in days) a rejected note may sit from a flagged week and still be mentioned as
# a "closest note" in the reason text, and how many to mention. They never fill matched_note_id.
NEAREST_NOTE_MAX_GAP_DAYS = 28
NEAREST_NOTE_LIMIT = 2

Effect = Literal["increase", "no_effect", "improvement", "informational", "out_of_scope"]

# Checked in this order; the first pattern that matches decides the effect. Negations come
# first so that "costs were not significantly affected" can never read as an increase.
EFFECT_RULES: list[tuple[Effect, tuple[str, ...]]] = [
    ("out_of_scope", ("not part of this dataset", "not in this dataset", "outside this dataset")),
    ("no_effect", ("not significantly affected", "were not affected", "not affected",
                   "without a rate change", "no impact on cost", "no change in rates")),
    ("improvement", ("returned to normal", "improved road", "improved conditions",
                     "back to normal", "resurfacing work was completed")),
    ("informational", ("remained stable", "remained normal", "no significant disruptions",
                       "no major disruptions")),
    ("increase", ("higher", "surcharge", "rose", "pushing up", "increase", "hike",
                  "more expensive", "detour", "price rise", "costlier")),
]

REJECT_REASONS: dict[str, str] = {
    "no_effect": "says costs were not affected",
    "improvement": "describes conditions improving or returning to normal, not a cost increase",
    "informational": "reports stable conditions with no disruption, which is not a reason for a cost rise",
    "out_of_scope": "concerns routes that are not part of this dataset",
}

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
RANGE_PATTERN = re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})\s*(?:to|-|–|until|through)\s*{_MONTH}\s+(\d{{1,2}})\b", re.I)


@dataclass(frozen=True)
class NoteRecord:
    note_id: str
    date: date
    applies_to: str
    text: str
    window_start: date
    window_end: date
    window_rule: str
    effect: Effect

    @property
    def is_all_routes(self) -> bool:
        return self.applies_to == ALL_ROUTES

    def covers_route(self, route: str) -> bool:
        return self.is_all_routes or self.applies_to == route

    def window_label(self) -> str:
        if self.window_start == self.window_end:
            return self.window_start.isoformat()
        return f"{self.window_start.isoformat()} to {self.window_end.isoformat()}"


@dataclass(frozen=True)
class MatchResult:
    verdict: Literal["justified", "unexplained"]
    matched_note_id: str | None
    note: NoteRecord | None
    candidates: tuple[str, ...]                    # eligible note ids, best first
    rejected: dict[str, str] = field(default_factory=dict)  # note_id -> why it does not count
    nearest_rejected: tuple[NoteRecord, ...] = ()           # closest rejected notes, closest first


class NotesValidationError(ValueError):
    """Raised when context_notes.csv breaks its input contract."""


# --------------------------------------------------------------------------- note parsing

def load_notes(path: str | Path) -> list[NoteRecord]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"note_id", "date", "applies_to", "note"} - set(raw.columns)
    if missing:
        raise NotesValidationError(f"Missing columns in notes file: {sorted(missing)}")
    if raw["note_id"].duplicated().any():
        raise NotesValidationError("Duplicate note_id in notes file")
    notes = []
    for row in raw.itertuples(index=False):
        try:
            note_date = date.fromisoformat(row.date.strip())
        except ValueError as exc:
            raise NotesValidationError(f"{row.note_id}: unparseable date {row.date!r}") from exc
        notes.append(parse_note(row.note_id.strip(), note_date, row.applies_to.strip(), row.note.strip()))
    return sorted(notes, key=lambda n: n.note_id)


def parse_note(note_id: str, note_date: date, applies_to: str, text: str) -> NoteRecord:
    start, end, rule = note_window(note_date, text)
    return NoteRecord(note_id, note_date, applies_to, text, start, end, rule, classify_effect(text))


def classify_effect(text: str) -> Effect:
    lowered = text.lower()
    for effect, phrases in EFFECT_RULES:
        if any(p in lowered for p in phrases):
            return effect
    return "informational"  # fail closed: a note we cannot read as an increase never justifies


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def note_window(note_date: date, text: str) -> tuple[date, date, str]:
    """Dates a note is in force, by the first rule that applies.

    1. An explicit range in the text ("from Feb 24 to Mar 8"), year taken from the date column.
    2. "starting this week" / "from this week": OPEN_ENDED_WEEKS weeks from the note's Monday.
    3. "this quarter": the calendar quarter containing the date.
    4. Otherwise the Monday-to-Sunday week containing the date.
    """
    lowered = text.lower()
    found = RANGE_PATTERN.search(lowered)
    if found:
        m1, d1, m2, d2 = found.groups()
        start = date(note_date.year, MONTHS[m1[:3]], int(d1))
        end_year = note_date.year + (1 if MONTHS[m2[:3]] < MONTHS[m1[:3]] else 0)
        return start, date(end_year, MONTHS[m2[:3]], int(d2)), "explicit date range in note"
    if "starting this week" in lowered or "from this week" in lowered:
        start = monday_of(note_date)
        return start, start + timedelta(weeks=OPEN_ENDED_WEEKS, days=-1), f"lasting change, {OPEN_ENDED_WEEKS} weeks from note week"
    if "this quarter" in lowered:
        q_start = date(note_date.year, 3 * ((note_date.month - 1) // 3) + 1, 1)
        q_end_month = q_start.month + 2
        next_month = date(q_start.year + (q_end_month == 12), q_end_month % 12 + 1, 1)
        return q_start, next_month - timedelta(days=1), "calendar quarter of note"
    start = monday_of(note_date)
    return start, start + timedelta(days=6), "Monday-to-Sunday week of note"


# --------------------------------------------------------------------------- vector index

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it its of on or so "
    "that the this to was were what when where which who why will with".split()
)


def _stem(token: str) -> str:
    """Crude suffix stripping so 'flood', 'floods' and 'flooding' share a term."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower().replace("-", " ")) if t not in STOPWORDS]


class TfidfIndex:
    """Tiny in-memory vector index: smoothed TF-IDF vectors, L2-normalised, cosine by dot product.

    Ten notes do not need an external vector database; this keeps the run free, offline and
    byte-for-byte reproducible. The interface (fit on documents, score a query) is what a
    FAISS or Chroma index would sit behind if the notes file grew.
    """

    def __init__(self, documents: list[str]):
        tokenised = [tokenize(d) for d in documents]
        self.vocab = {t: i for i, t in enumerate(sorted({t for doc in tokenised for t in doc}))}
        n_docs = len(documents)
        doc_freq = Counter(t for doc in tokenised for t in set(doc))
        self.idf = np.array([math.log((1 + n_docs) / (1 + doc_freq[t])) + 1 for t in self.vocab])
        self.matrix = np.vstack([self._vector(doc) for doc in tokenised]) if documents else np.zeros((0, len(self.vocab)))

    def _vector(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for token, count in Counter(tokens).items():
            if token in self.vocab:
                vec[self.vocab[token]] = count
        vec *= self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def scores(self, query: str) -> np.ndarray:
        # Rounded so floating-point noise across machines can never reorder near-ties.
        return np.round(self.matrix @ self._vector(tokenize(query)), 6)


# --------------------------------------------------------------------------- matching

def query_text(route: str, week_of: date) -> str:
    origin, _, destination = route.partition("-")
    return (f"{route} {origin} {destination} corridor freight cost rise higher trip costs "
            f"surcharge fuel diesel toll weather flood festival week of {week_of:%B %Y}")


def overlaps(week_of: date, note: NoteRecord) -> bool:
    week_end = week_of + timedelta(days=6)
    return week_of <= note.window_end and note.window_start <= week_end


def gap_days(week_of: date, note: NoteRecord) -> int:
    week_end = week_of + timedelta(days=6)
    if overlaps(week_of, note):
        return 0
    return (note.window_start - week_end).days if note.window_start > week_end else (week_of - note.window_end).days


class NoteMatcher:
    def __init__(self, notes: list[NoteRecord]):
        self.notes = notes
        self.by_id = {n.note_id: n for n in notes}
        self.index = TfidfIndex([f"{n.applies_to}. {n.text}" for n in notes])

    @classmethod
    def from_csv(cls, path: str | Path) -> "NoteMatcher":
        return cls(load_notes(path))

    def retrieve(self, query: str, route: str | None = None, top_k: int | None = None) -> list[tuple[NoteRecord, float]]:
        """Similarity search with an optional route metadata filter; ties break on note_id."""
        scored = [(n, float(s)) for n, s in zip(self.notes, self.index.scores(query))
                  if route is None or n.covers_route(route)]
        scored.sort(key=lambda pair: (-pair[1], pair[0].note_id))
        return scored[:top_k] if top_k else scored

    def match(self, route: str, week_of: date, rose_vs_own_history: bool = True) -> MatchResult:
        """Find the note that explains this route-week's rise, or report why none does.

        rose_vs_own_history=False means the row was flagged only for its gap to similar routes.
        A note that applies to All Routes raises the peers too, so it cannot explain that gap.
        """
        week_of = week_of.date() if isinstance(week_of, pd.Timestamp) else week_of
        rejected: dict[str, str] = {}
        eligible: list[tuple[NoteRecord, float]] = []

        for note in self.notes:
            if not note.covers_route(route):
                rejected[note.note_id] = f"applies to {note.applies_to}, not this route"
        for note, score in self.retrieve(query_text(route, week_of), route=route):
            reasons = []
            if not overlaps(week_of, note):
                reasons.append(f"only covers {note.window_label()}, not this week")
            if note.effect != "increase":
                reasons.append(REJECT_REASONS[note.effect])
            elif note.is_all_routes and not rose_vs_own_history:
                reasons.append("applies to all routes, so it would raise similar routes too and cannot "
                               "explain this route's gap to them")
            if reasons:
                rejected[note.note_id] = ", and it ".join(reasons)
            else:
                eligible.append((note, score))

        # Route-specific notes beat All Routes notes; then similarity; then note_id.
        eligible.sort(key=lambda pair: (pair[0].is_all_routes, -pair[1], pair[0].note_id))
        if eligible:
            best = eligible[0][0]
            return MatchResult("justified", best.note_id, best, tuple(n.note_id for n, _ in eligible), rejected)

        return MatchResult("unexplained", None, None, (), rejected, self._nearest_rejected(route, week_of))

    def _nearest_rejected(self, route: str, week_of: date) -> tuple[NoteRecord, ...]:
        nearby = sorted((gap_days(week_of, n), n.is_all_routes, n.note_id, n) for n in self.notes
                        if n.covers_route(route) and gap_days(week_of, n) <= NEAREST_NOTE_MAX_GAP_DAYS)
        return tuple(item[-1] for item in nearby[:NEAREST_NOTE_LIMIT])
