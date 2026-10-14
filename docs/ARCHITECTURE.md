# Architecture

Every number, flag, verdict and note id is produced by deterministic Python. An LLM, when
switched on, only words the `reason` column, and its text is checked against the facts before
it is used.

| Module | Input | Output | Key rule |
|---|---|---|---|
| `preprocessing.py` | `shipment_records.csv` | one row per route-week | `week_of` = Monday; pooled `sum(cost) / sum(tonnes x km)`; invalid rows raise `DataValidationError` |
| `baselines.py` | weekly table | + `own_history_avg`, `own_history_weeks`, `peer_avg`, `peer_routes`, `% deltas` | own: 8 calendar weeks strictly before; peer: same route_type, same week, self excluded |
| `flagging.py` | baseline table | + `is_flagged`, `flag_basis` | rising (own > 0) and (own >= +15% or peer >= +20%) |
| `rag_engine.py` | `context_notes.csv`, route, week | `MatchResult` (verdict, note id, rejected notes with reasons, nearest notes) | route filter, window overlap, effect must be "increase", All Routes notes cannot explain a peer-only flag; TF-IDF similarity only ranks survivors |
| `explain.py` | `RowFacts` | reason text, `UsageLog` | template by default; optional Ollama / Anthropic at temperature 0, guarded by `check_reason` |
| `runner.py` | file paths, `--llm` | `output/flagged_routes.csv`, `all_route_weeks.csv`, `token_log.json` | output columns and formats copied from the sample |
| `qa.py` | `output/all_route_weeks.csv`, notes | plain-English answers | answers only from results; note search falls back to the vector index |

## Note records

| id | applies to | effect | window | can justify? |
|---|---|---|---|---|
| N001 | Chennai-Bangalore | increase (flood detours) | 2025-02-24 to 2025-03-08 (explicit range) | yes |
| N002 | Ahmedabad-Mumbai | increase (festival surcharge) | 2025-01-20 to 2025-01-26 (note week) | yes |
| N003 | All Routes | increase (diesel, 5-7%) | 2025-05-05 to 2025-06-29 ("starting this week", 8 weeks) | yes, except for peer-only flags |
| N004 | All Routes | out of scope | 2024-03-11 week | no |
| N005 | Mumbai-Delhi | no effect | 2024-07-29 week | no |
| N006 | All Routes | informational (stable) | Q3 2025 | no |
| N007 | Delhi-Jaipur | improvement | 2024-05-20 week | no |
| N008 | Kolkata-Bhubaneswar | informational (no disruption) | Q2 2025 | no |
| N009 | Chennai-Bangalore | improvement (back to normal) | 2025-03-17 week | no |
| N010 | All Routes | no effect (no rate change) | 2025-10-27 week | no |

## Failure behaviour

- Bad shipment or note rows: the run stops with a message naming example ids.
- No eligible note: verdict `unexplained`, blank note id.
- LLM unreachable, missing key, or text that fails the checks: template wording, logged in
  `token_log.json`; verdicts unchanged.
