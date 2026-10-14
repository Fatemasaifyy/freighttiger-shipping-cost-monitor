# Reproducibility check

Three fresh runs of `python runner.py --llm none` on the same input.

| run | flagged_routes.csv sha256 | all_route_weeks.csv sha256 |
|---|---|---|
| 1 | `4db84a86fb6dc8c9…` | `22ac9324615d4c80…` |
| 2 | `4db84a86fb6dc8c9…` | `22ac9324615d4c80…` |
| 3 | `4db84a86fb6dc8c9…` | `22ac9324615d4c80…` |

- Flags, numbers, verdicts and note ids identical across runs: **yes** (19 flagged rows, every column except `reason` compared)
- Files byte-identical across runs (including `reason`): **yes**

