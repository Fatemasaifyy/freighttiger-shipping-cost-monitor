"""Stretch goal: ask questions about the findings in plain English.

    python qa.py "Why did Chennai-Bangalore get pricier in March 2025?"
    python qa.py                     # interactive prompt; type 'quit' to leave

Answers come only from the assistant's own results (output/all_route_weeks.csv, written by
runner.py) and the notes those results cite. The question is parsed for a route or city, a
month and a year; the matching rows are summarised with their numbers, verdicts and note ids.
Questions with no route or date fall back to searching the notes index, then show which
flagged rows cite the notes found. Nothing is generated beyond what the results contain.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from rag_engine import NoteMatcher
from runner import DEFAULT_NOTES, DEFAULT_OUTPUT, DEFAULT_SHIPMENTS, analyse, write_outputs

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december"]
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
FLAGGED = ("Yes", "No (justified)")


class FindingsQA:
    def __init__(self, table: pd.DataFrame, matcher: NoteMatcher):
        self.table = table.copy()
        self.table["week_of"] = pd.to_datetime(self.table["week_of"])
        self.table["matched_note_id"] = self.table["matched_note_id"].fillna("")
        self.table["reason"] = self.table["reason"].fillna("")
        self.matcher = matcher
        self.routes = sorted(self.table["route"].unique())

    @classmethod
    def load(cls, out_dir: Path = DEFAULT_OUTPUT, notes_path: Path = DEFAULT_NOTES) -> "FindingsQA":
        results = out_dir / "all_route_weeks.csv"
        if not results.exists():
            output, table, log = analyse(DEFAULT_SHIPMENTS, notes_path)
            write_outputs(output, table, log, out_dir)
        return cls(pd.read_csv(results, dtype={"matched_note_id": str}, keep_default_na=False,
                               na_values=[""]), NoteMatcher.from_csv(notes_path))

    # ------------------------------------------------------------------ parsing

    def find_routes(self, question: str) -> list[str]:
        q = question.lower()
        exact = [r for r in self.routes if r.lower() in q]
        if exact:
            return exact
        cities = {c for r in self.routes for c in r.split("-")}
        named = [c for c in cities if re.search(rf"\b{c.lower()}\b", q)]
        if len(named) >= 2:
            both = [r for r in self.routes if all(c in r.split("-") for c in named[:2])]
            if both:
                return both
        return [r for r in self.routes if any(c in r.split("-") for c in named)]

    @staticmethod
    def find_month_year(question: str) -> tuple[int | None, int | None]:
        # a word counts as a month if it is a prefix of a month name at least 3 letters long
        # ("mar", "sept", "december"), so "market" or "maybe" do not match
        month_num = None
        for word in re.findall(r"[a-z]+", question.lower()):
            hits = [i for i, m in enumerate(MONTH_NAMES, start=1) if len(word) >= 3 and m.startswith(word)]
            if hits:
                month_num = hits[0]
                break
        year = YEAR_PATTERN.search(question)
        return month_num, int(year.group(1)) if year else None

    # ------------------------------------------------------------------ answering

    def answer(self, question: str) -> str:
        routes = self.find_routes(question)
        month, year = self.find_month_year(question)
        rows = self.table
        if routes:
            rows = rows[rows["route"].isin(routes)]
        if month:
            rows = rows[rows["week_of"].dt.month == month]
        if year:
            rows = rows[rows["week_of"].dt.year == year]

        if not routes and not month and not year:
            return self._answer_from_notes(question)
        if rows.empty:
            return "I have no results for that route and period (data covers 2024-01-01 to 2025-12-22)."

        scope = self._scope(routes, month, year)
        flagged = rows[rows["flagged"].isin(FLAGGED)].sort_values(["week_of", "route"])
        wants = question.lower()
        if "unexplained" in wants:
            flagged = flagged[flagged["flagged"] == "Yes"]
        elif "justified" in wants or "explained" in wants:
            flagged = flagged[flagged["flagged"] == "No (justified)"]

        if flagged.empty:
            return self._no_flags(rows, scope)
        lines = [f"{scope}: {len(flagged)} flagged week(s)."]
        for _, r in flagged.head(12).iterrows():
            note = f", note {r['matched_note_id']}" if r["matched_note_id"] else ""
            verdict = "justified" if r["flagged"] == "No (justified)" else "unexplained"
            lines.append(
                f"- {r['route']}, week of {r['week_of']:%Y-%m-%d}: {r['cost_per_tonne_km']:.2f} per tonne-km, "
                f"{r['pct_vs_own_history']:+.1f}% vs its past average, {r['pct_vs_peers']:+.1f}% vs similar routes. "
                f"Verdict: {verdict}{note}. {r['reason']}"
            )
        if len(flagged) > 12:
            lines.append(f"... and {len(flagged) - 12} more; narrow the question by month or route.")
        return "\n".join(lines)

    @staticmethod
    def _scope(routes: list[str], month: int | None, year: int | None) -> str:
        where = ", ".join(routes) if routes else "All routes"
        when = " ".join(p for p in [MONTH_NAMES[month - 1].title() if month else "", str(year) if year else ""] if p)
        return f"{where}{' in ' + when if when else ''}"

    @staticmethod
    def _no_flags(rows: pd.DataFrame, scope: str) -> str:
        own = rows["pct_vs_own_history"].dropna()
        peers = rows["pct_vs_peers"].dropna()
        return (
            f"{scope}: no week was flagged. Cost per tonne-km averaged {rows['cost_per_tonne_km'].mean():.2f} over "
            f"{len(rows)} route-week(s); change vs own history ranged {own.min():+.1f}% to {own.max():+.1f}% and "
            f"vs similar routes {peers.min():+.1f}% to {peers.max():+.1f}%, inside the flag thresholds."
        )

    def _answer_from_notes(self, question: str) -> str:
        flagged = self.table[self.table["flagged"].isin(FLAGGED)]
        hits = [(n, s) for n, s in self.matcher.retrieve(question, top_k=3) if s > 0]
        if not hits:
            counts = flagged["flagged"].value_counts().to_dict()
            unexplained = flagged[flagged["flagged"] == "Yes"]["route"].value_counts().to_dict()
            return (f"{len(flagged)} route-weeks were flagged: {counts.get('No (justified)', 0)} justified by a note, "
                    f"{counts.get('Yes', 0)} unexplained. Unexplained flags by route: {unexplained}. "
                    f"Ask about a route or a month for detail.")
        lines = ["Closest notes to your question, and the flagged weeks that cite them:"]
        for note, score in hits:
            cited = flagged[flagged["matched_note_id"] == note.note_id]
            weeks = ", ".join(f"{r} {w:%Y-%m-%d}" for r, w in zip(cited["route"], cited["week_of"])) or "no flagged week cites it"
            lines.append(f"- {note.note_id} ({note.date}, {note.applies_to}, similarity {score:.2f}): {note.text} -> {weeks}")
        return "\n".join(lines)


def main(argv: list[str]) -> None:
    qa = FindingsQA.load()
    if argv:
        print(qa.answer(" ".join(argv)))
        return
    print("Ask about the findings (e.g. 'Why did Chennai-Bangalore get pricier in March 2025?'). Type 'quit' to leave.")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print(qa.answer(question))


if __name__ == "__main__":
    main(sys.argv[1:])
