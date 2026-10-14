"""Reproducibility check: three untouched runs on the same input must agree.

    python evals/reproducibility_check.py                 # template wording
    python evals/reproducibility_check.py --llm ollama    # with LLM wording

Each run is a fresh `python runner.py` process writing to its own folder. The check compares
every column except `reason` (flags, numbers, verdicts, note ids), which must be identical,
and also reports whether the whole files are byte-identical. With LLM wording the reason text
may differ between runs; nothing else may.

Writes output/reproducibility_report.md and exits with status 1 if the graded columns differ.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output" / "reproducibility_report.md"
RUNS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["none", "ollama", "anthropic"], default="none")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, RUNS + 1):
            out = Path(tmp) / f"run{i}"
            subprocess.run([sys.executable, str(ROOT / "runner.py"), "--out", str(out), "--llm", args.llm],
                           check=True, capture_output=True, text=True)
            flagged = out / "flagged_routes.csv"
            results.append({
                "run": i,
                "flagged_sha": sha256(flagged),
                "all_weeks_sha": sha256(out / "all_route_weeks.csv"),
                "graded": pd.read_csv(flagged, dtype=str, keep_default_na=False).drop(columns=["reason"]),
            })

    base = results[0]["graded"]
    graded_same = all(r["graded"].equals(base) for r in results[1:])
    bytes_same = len({r["flagged_sha"] for r in results}) == 1 and len({r["all_weeks_sha"] for r in results}) == 1

    lines = [
        "# Reproducibility check", "",
        f"Three fresh runs of `python runner.py --llm {args.llm}` on the same input.", "",
        "| run | flagged_routes.csv sha256 | all_route_weeks.csv sha256 |", "|---|---|---|",
        *[f"| {r['run']} | `{r['flagged_sha'][:16]}…` | `{r['all_weeks_sha'][:16]}…` |" for r in results],
        "",
        f"- Flags, numbers, verdicts and note ids identical across runs: **{'yes' if graded_same else 'NO'}** "
        f"({len(base)} flagged rows, every column except `reason` compared)",
        f"- Files byte-identical across runs (including `reason`): **{'yes' if bytes_same else 'no'}**",
        "",
    ]
    if args.llm != "none":
        lines.append("LLM wording runs at temperature 0. Any difference in `reason` text is wording only; "
                     "the verdict and note id are fixed before the model is called.")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if graded_same else 1


if __name__ == "__main__":
    sys.exit(main())
