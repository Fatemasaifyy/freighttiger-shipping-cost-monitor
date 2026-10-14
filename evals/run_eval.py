"""Eval harness: score the assistant against hand labels and audit every output row.

    python evals/run_eval.py

Checks, all of which must pass:
  1. Note reading     each of the 10 notes gets the hand-labelled effect and window.
  2. Labeled weeks    23 hand-labelled route-weeks get the expected verdict and note id.
  3. Format contract  output columns match sample_output_format exactly, and the three sample
                      rows reproduce their numeric columns character for character.
  4. Grounding audit  every justified row cites a note that covers that route, overlaps that
                      week and describes a cost increase; every unexplained row has no note id.
  5. Baselines        own-history and peer baselines recomputed for all 728 route-weeks with a
                      plain loop (independent of the pandas code) agree to 1e-9.
It also prints how many rows each nearby threshold would flag, for context.

Writes output/eval_report.md and exits with status 1 if any check fails.
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flagging import apply_flags  # noqa: E402
from rag_engine import NoteMatcher, overlaps  # noqa: E402
from runner import DEFAULT_NOTES, DEFAULT_SHIPMENTS, OUTPUT_COLUMNS, analyse  # noqa: E402

EVALS = ROOT / "evals"
SAMPLE = ROOT / "data" / "sample_output_format_v2.csv"
REPORT = ROOT / "output" / "eval_report.md"


def check_notes(matcher: NoteMatcher) -> tuple[list[str], int]:
    labels = pd.read_csv(EVALS / "labeled_notes.csv", dtype=str)
    failures = []
    for row in labels.itertuples(index=False):
        note = matcher.by_id[row.note_id]
        got = (note.effect, note.window_start.isoformat(), note.window_end.isoformat())
        want = (row.expected_effect, row.expected_window_start, row.expected_window_end)
        if got != want:
            failures.append(f"{row.note_id}: expected {want}, got {got}")
    return failures, len(labels)


def check_labeled_weeks(table: pd.DataFrame) -> tuple[list[str], int]:
    labels = pd.read_csv(EVALS / "labeled_route_weeks.csv", dtype=str, keep_default_na=False)
    indexed = table.set_index(["route", table["week_of"].dt.strftime("%Y-%m-%d")])
    failures = []
    for row in labels.itertuples(index=False):
        got_row = indexed.loc[(row.route, row.week_of)]
        got = (got_row["flagged"], got_row["matched_note_id"])
        want = (row.expected_flagged, row.expected_note_id)
        if got != want:
            failures.append(f"{row.route} {row.week_of}: expected {want}, got {got} ({row.why})")
    return failures, len(labels)


def read_sample() -> pd.DataFrame:
    """The sample file's third row has an unquoted comma inside `reason` ("(N006, 2025-09-22)"),
    so a strict CSV parser sees 9 fields. Everything past the 7th field is joined back into reason."""
    with open(SAMPLE, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    fixed = [r[:7] + [",".join(r[7:])] for r in body]
    return pd.DataFrame(fixed, columns=header)


def check_format(output: pd.DataFrame) -> tuple[list[str], int]:
    sample = read_sample()
    failures = []
    if list(output.columns) != list(sample.columns) or list(output.columns) != OUTPUT_COLUMNS:
        failures.append(f"columns differ: {list(output.columns)} vs {list(sample.columns)}")
    numeric = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]
    for s in sample.itertuples(index=False):
        got = output[(output["route"] == s.route) & (output["week_of"] == s.week_of)]
        if got.empty:
            failures.append(f"sample row {s.route} {s.week_of} missing from output")
            continue
        for col in numeric:
            if got.iloc[0][col] != getattr(s, col):
                failures.append(f"sample row {s.route} {s.week_of} {col}: {got.iloc[0][col]!r} != {getattr(s, col)!r}")
    pct = re.compile(r"^[+-]\d+\.\d% vs ")
    for r in output.itertuples(index=False):
        if date.fromisoformat(r.week_of).weekday() != 0:
            failures.append(f"{r.route} {r.week_of}: week_of is not a Monday")
        if not re.fullmatch(r"\d+\.\d{2}", r.cost_per_tonne_km):
            failures.append(f"{r.route} {r.week_of}: cost not 2 dp")
        if not (pct.match(r.vs_own_history) and pct.match(r.vs_similar_routes)):
            failures.append(f"{r.route} {r.week_of}: percentage text format")
    return failures, len(sample) + len(output)


def check_grounding(output: pd.DataFrame, matcher: NoteMatcher) -> tuple[list[str], int]:
    failures = []
    for r in output.itertuples(index=False):
        week = date.fromisoformat(r.week_of)
        if r.flagged == "No (justified)":
            note = matcher.by_id.get(r.matched_note_id)
            if note is None:
                failures.append(f"{r.route} {r.week_of}: cites unknown note {r.matched_note_id!r}")
                continue
            if not note.covers_route(r.route):
                failures.append(f"{r.route} {r.week_of}: {note.note_id} is for {note.applies_to}")
            if not overlaps(week, note):
                failures.append(f"{r.route} {r.week_of}: {note.note_id} window {note.window_label()} misses the week")
            if note.effect != "increase":
                failures.append(f"{r.route} {r.week_of}: {note.note_id} does not describe an increase")
            if note.note_id not in r.reason:
                failures.append(f"{r.route} {r.week_of}: reason does not name {note.note_id}")
        elif r.flagged == "Yes":
            if r.matched_note_id:
                failures.append(f"{r.route} {r.week_of}: unexplained row carries note id {r.matched_note_id}")
        else:
            failures.append(f"{r.route} {r.week_of}: unexpected flagged value {r.flagged!r}")
    return failures, len(output)


def check_baselines(table: pd.DataFrame) -> tuple[list[str], int]:
    values = {(r.route, r.week_of.date()): (r.route_type, r.cost_per_tonne_km) for r in table.itertuples()}
    failures = []
    for r in table.itertuples():
        week = r.week_of.date()
        prior = [values[(r.route, week - timedelta(weeks=k))][1] for k in range(1, 9)
                 if (r.route, week - timedelta(weeks=k)) in values]
        peers = [v for (route, w), (rtype, v) in values.items() if w == week and rtype == r.route_type and route != r.route]
        own = sum(prior) / len(prior) if prior else float("nan")
        peer = sum(peers) / len(peers) if peers else float("nan")
        for name, want, got in [("own", own, r.own_history_avg), ("peer", peer, r.peer_avg)]:
            if pd.isna(want) and pd.isna(got):
                continue
            if pd.isna(want) or pd.isna(got) or abs(want - got) > 1e-9:
                failures.append(f"{r.route} {week} {name}: expected {want}, got {got}")
        if len(prior) != r.own_history_weeks:
            failures.append(f"{r.route} {week}: own_history_weeks {r.own_history_weeks} != {len(prior)}")
    return failures, len(table)


def threshold_table(table: pd.DataFrame) -> str:
    lines = ["| own-history threshold | peer threshold | rows flagged |", "|---|---|---|"]
    for own, peer in [(10, 15), (15, 15), (15, 20), (20, 20), (20, 25)]:
        n = int(apply_flags(table, own, peer)["is_flagged"].sum())
        marker = " (chosen)" if (own, peer) == (15, 20) else ""
        lines.append(f"| +{own}% | +{peer}% | {n}{marker} |")
    return "\n".join(lines)


def main() -> int:
    output, table, _ = analyse(DEFAULT_SHIPMENTS, DEFAULT_NOTES)
    matcher = NoteMatcher.from_csv(DEFAULT_NOTES)
    checks = [
        ("Note reading (effect + window)", *check_notes(matcher)),
        ("Hand-labelled route-weeks (verdict + note id)", *check_labeled_weeks(table)),
        ("Output format contract and sample rows", *check_format(output)),
        ("Grounding audit of every output row", *check_grounding(output, matcher)),
        ("Independent baseline recomputation", *check_baselines(table)),
    ]

    lines = ["# Eval report", "", "Generated by `python evals/run_eval.py`.", "",
             "| check | items | result |", "|---|---|---|"]
    for name, failures, n in checks:
        lines.append(f"| {name} | {n} | {'PASS' if not failures else f'FAIL ({len(failures)})'} |")
    for name, failures, _ in checks:
        if failures:
            lines += ["", f"## Failures: {name}", *[f"- {f}" for f in failures]]
    counts = output["flagged"].value_counts().to_dict()
    lines += ["", "## Run summary", "",
              f"- Route-weeks analysed: {len(table)}",
              f"- Flagged: {len(output)} ({counts.get('Yes', 0)} unexplained, {counts.get('No (justified)', 0)} justified)",
              f"- Notes cited: {', '.join(sorted(set(output['matched_note_id']) - {''})) or 'none'}",
              "", "## Threshold sensitivity (context, not a pass/fail check)", "", threshold_table(table), ""]

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 1 if any(f for _, f, _ in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
