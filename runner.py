"""Module 4: end-to-end run.

    shipments -> weekly cost per tonne-km -> baselines -> flags -> note matching -> reasons -> CSV

Usage:
    python runner.py                       # template wording, zero LLM calls
    python runner.py --llm ollama          # wording by a local open-source model (temperature 0)
    python runner.py --llm anthropic       # wording by claude-haiku-4-5 (temperature 0)

Writes to the output folder (default output/):
    flagged_routes.csv    the graded file, same columns, order and formats as sample_output_format
    all_route_weeks.csv   every route-week with its numbers, flag and verdict (used by qa.py)
    token_log.json        LLM calls, input/output tokens and cost for this run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from baselines import compute_baselines
from explain import ReasonWriter, RowFacts, usage_report
from flagging import OWN_HISTORY_THRESHOLD_PCT, apply_flags
from preprocessing import build_weekly_table
from rag_engine import NoteMatcher

ROOT = Path(__file__).resolve().parent
DEFAULT_SHIPMENTS = ROOT / "data" / "shipment_records.csv"
DEFAULT_NOTES = ROOT / "data" / "context_notes.csv"
DEFAULT_OUTPUT = ROOT / "output"

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

FLAGGED_UNEXPLAINED = "Yes"
FLAGGED_JUSTIFIED = "No (justified)"
NOT_FLAGGED = "No"


def format_pct(value: float, suffix: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"n/a (no baseline) {suffix}"
    return f"{value:+.1f}% {suffix}"


def analyse(shipments_path: Path, notes_path: Path, llm: str = "none") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the whole pipeline. Returns (graded output rows, full route-week table, token log)."""
    table = apply_flags(compute_baselines(build_weekly_table(shipments_path)))
    matcher = NoteMatcher.from_csv(notes_path)
    writer = ReasonWriter(llm)

    table["flagged"] = NOT_FLAGGED
    table["matched_note_id"] = ""
    table["reason"] = ""
    all_facts = []
    for idx in table.index[table["is_flagged"]]:
        row = table.loc[idx]
        rose_vs_own = row["pct_vs_own_history"] >= OWN_HISTORY_THRESHOLD_PCT
        match = matcher.match(row["route"], row["week_of"].date(), rose_vs_own_history=rose_vs_own)
        facts = RowFacts(
            route=row["route"],
            week_of=row["week_of"].date(),
            cost_per_tonne_km=row["cost_per_tonne_km"],
            pct_vs_own_history=row["pct_vs_own_history"],
            pct_vs_peers=row["pct_vs_peers"],
            own_history_weeks=int(row["own_history_weeks"]),
            flag_basis=row["flag_basis"],
            match=match,
        )
        all_facts.append(facts)
        table.at[idx, "flagged"] = FLAGGED_JUSTIFIED if match.verdict == "justified" else FLAGGED_UNEXPLAINED
        table.at[idx, "matched_note_id"] = match.matched_note_id or ""
        table.at[idx, "reason"] = writer.reason(facts)

    output = to_output_format(table[table["is_flagged"]])
    return output, table, usage_report(writer, all_facts)


def to_output_format(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "route": rows["route"],
        "week_of": rows["week_of"].dt.strftime("%Y-%m-%d"),
        "cost_per_tonne_km": rows["cost_per_tonne_km"].map(lambda v: f"{v:.2f}"),
        "vs_own_history": rows["pct_vs_own_history"].map(lambda v: format_pct(v, "vs this route's past average")),
        "vs_similar_routes": rows["pct_vs_peers"].map(lambda v: format_pct(v, "vs similar-length routes this week")),
        "flagged": rows["flagged"],
        "matched_note_id": rows["matched_note_id"],
        "reason": rows["reason"],
    })
    return out[OUTPUT_COLUMNS].sort_values(["week_of", "route"]).reset_index(drop=True)


def write_outputs(output: pd.DataFrame, table: pd.DataFrame, token_log: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / "flagged_routes.csv", index=False, lineterminator="\n")

    full = table.drop(columns=["is_flagged"]).copy()
    full["week_of"] = full["week_of"].dt.strftime("%Y-%m-%d")
    full.round(6).to_csv(out_dir / "all_route_weeks.csv", index=False, lineterminator="\n")

    # newline="\n" keeps every output byte-identical on Windows, macOS and Linux
    with open(out_dir / "token_log.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(token_log, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Flag unusual weekly shipping cost rises per route.")
    parser.add_argument("--shipments", type=Path, default=DEFAULT_SHIPMENTS)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm", choices=["none", "ollama", "anthropic"], default="none",
                        help="who words the reason column; verdicts and note ids never depend on it")
    args = parser.parse_args(argv)

    try:
        output, table, token_log = analyse(args.shipments, args.notes, args.llm)
    except RuntimeError as exc:
        parser.error(str(exc))
    write_outputs(output, table, token_log, args.out)

    counts = output["flagged"].value_counts().to_dict()
    run = token_log["this_run"]
    print(f"Route-weeks analysed: {len(table)}  |  flagged: {len(output)} "
          f"({counts.get(FLAGGED_UNEXPLAINED, 0)} unexplained, {counts.get(FLAGGED_JUSTIFIED, 0)} justified)")
    print(f"LLM: {run['provider']}  |  calls: {run['llm_calls']}  |  tokens in/out: "
          f"{run['total_input_tokens']}/{run['total_output_tokens']}  |  cost: ${run['estimated_cost_usd']}")
    if run["failed_requests"]:
        print(f"Warning: the {args.llm} LLM could not be reached ({token_log['failed_requests'][0]['error']}); "
              f"template wording was used. Verdicts and note ids are unaffected.")
    print(f"Wrote {args.out / 'flagged_routes.csv'}, all_route_weeks.csv and token_log.json")


if __name__ == "__main__":
    main()
