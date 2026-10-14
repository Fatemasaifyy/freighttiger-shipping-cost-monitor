# Shipping Cost Watcher

An assistant that reads shipment records, tracks each route's weekly cost per tonne-km, flags
weeks where the cost is rising in a way that looks out of the ordinary, and checks a set of
context notes for a real reason before calling a rise unexplained. Every flag comes with a
plain-English reason, and you can ask questions about the findings from the command line.

## Quick start

Needs Python 3.10 or newer.

```bash
pip install -r requirements.txt

python runner.py                    # full run, writes output/
python evals/run_eval.py            # eval harness, writes output/eval_report.md
python evals/reproducibility_check.py   # 3 fresh runs compared, writes output/reproducibility_report.md
python -m pytest                    # 121 unit and integration tests

python qa.py "Why did Chennai-Bangalore get pricier in March 2025?"
python qa.py                        # interactive questions, type 'quit' to leave
```

No API key, network or model download is needed for any of the above.

## What to hand in, and where it is

| Deliverable asked for in the brief | File |
|---|---|
| Working code | `preprocessing.py`, `baselines.py`, `flagging.py`, `rag_engine.py`, `explain.py`, `runner.py`, `qa.py` |
| Output CSV matching `sample_output_format.csv` | `output/flagged_routes.csv` |
| README: how it works, how it was built, how it was checked | this file |
| Reproducibility check across 3 runs | `output/reproducibility_report.md` (re-create with `python evals/reproducibility_check.py`) |
| Token / cost log for one full run | `output/token_log.json` |
| Eval harness and labelled set | `evals/run_eval.py`, `evals/labeled_route_weeks.csv`, `evals/labeled_notes.csv`, result in `output/eval_report.md` |

## Results on the given data

728 route-weeks (7 routes x 104 weeks, 2024-01-01 to 2025-12-22) were analysed and **19 were
flagged: 3 justified by a note, 16 unexplained.**

| Route | Weeks flagged | Verdict |
|---|---|---|
| Ahmedabad-Mumbai | 2025-01-20 | Justified by N002 (festival surcharge that week) |
| Chennai-Bangalore | 2025-02-24, 2025-03-03 | Justified by N001 (flooding, Feb 24 to Mar 8) |
| Chennai-Bangalore | 2025-03-10, 2025-03-17 | Unexplained: still high after N001's window ended on Mar 8, and N009 says the route was back to normal |
| Delhi-Jaipur | 2024-11-11, 2024-11-18 | Unexplained: no note covers it |
| Mumbai-Pune | 12 weeks from 2025-06-23 to 2025-12-22 | Unexplained: a steady creep away from similar routes (from about +20% to +39%); the nearby notes (N003, N006, N010) say nothing that explains it |

All three rows in `sample_output_format.csv` are reproduced exactly: same week, cost, both
percentages, flag value and note id.

## How it works

```
data/shipment_records.csv
   │
   ▼  preprocessing.py   validate rows, week_of = Monday of the shipment's Monday-to-Sunday week,
   │                      one row per route-week: sum(cost) / sum(tonnes x km)
   ▼  baselines.py       own history: mean of the 8 weeks strictly before this one
   │                      similar routes: mean of the other same-route_type routes this week
   ▼  flagging.py        rising and out of the ordinary? (rule below)
   ▼  rag_engine.py      data/context_notes.csv -> parsed notes + vector index
   │                      route filter -> window filter -> "does it describe an increase?" -> rank
   ▼  explain.py         reason text (template, or optional LLM checked against the facts)
   ▼  runner.py          output/flagged_routes.csv, all_route_weeks.csv, token_log.json
      qa.py              questions answered from those results
```

### 1. Weekly cost per tonne-km (`preprocessing.py`)

- A route is `origin-destination`. Routes are directional, so A-B and B-A are different routes.
  The given `route_type` is used as is.
- `week_of` is the Monday of the Monday-to-Sunday week that contains `shipment_date`. Bucketing
  happens before any number is computed.
- The weekly figure is pooled: total freight cost for the route that week divided by the total
  tonne-km (`sum(cost) / sum(quantity_tonnes x distance_km)`). That is the brief's "total freight
  cost / (quantity x distance)", and it reproduces the sample rows to the published precision.
  An average of per-shipment ratios does not.
- Bad input fails loudly (missing columns, unknown route_type, duplicate ids, non-numeric or
  non-positive quantity/distance, bad dates, a route with two route_types) instead of dropping
  rows silently, because a dropped row would quietly change the numbers.

### 2. Baselines (`baselines.py`), exactly as graded

- **vs own history**: mean of the route's weekly values over the 8 weeks strictly before
  `week_of`. The current week never enters its own baseline.
- **vs similar routes**: mean of the same week's values across every other route with the same
  `route_type`. The route itself is excluded.
- Both are simple means of weekly route values; this is what reproduces the sample percentages.

**Fewer than 8 prior weeks.** Every route starts on 2024-01-01, so the first 8 weeks of each
route (2024-01-01 to 2024-02-19) have 0 to 7 prior weeks. Those weeks use whatever prior weeks
exist, with no padding or extrapolation, and `own_history_weeks` in
`output/all_route_weeks.csv` records how many were used. The very first week has no own-history
baseline, so it shows `n/a` and can never be flagged as rising. No flagged row falls in this
period. The data has no gaps after that, so every later week uses a full 8.

Peer groups in this data: Short = Mumbai-Pune, Delhi-Jaipur; Long = Delhi-Chennai,
Mumbai-Delhi (so the peer is the one other route); Medium = Ahmedabad-Mumbai,
Chennai-Bangalore, Kolkata-Bhubaneswar (peer = mean of the other two).

### 3. The flag rule (`flagging.py`)

A route-week is flagged when the cost is **rising** (above the route's own trailing average) and
**out of the ordinary**: at least +15% vs its own history, or at least +20% vs similar routes.

Why those numbers:

- Week-to-week movement vs own history is small: the standard deviation is 3-6% per route. +15%
  is far outside normal noise, while an across-the-board 5-7% diesel rise (N003) correctly stays
  under it.
- Gaps to peers are partly structural. Mumbai-Pune normally runs about 11% above Delhi-Jaipur and
  Delhi-Jaipur about 10% below. A peer threshold of +20% avoids flagging a route just for being
  itself, and still catches every sample row (Mumbai-Pune 2025-09-15 is +9.2% / +23.6%).
- Requiring "rising" stops a route that is pricey next to peers but cheaper than its own recent
  past from being flagged.

The thresholds are parameters. `output/eval_report.md` shows how many rows nearby settings would
flag (13 to 27), for context.

### 4. Finding a real reason in the notes (`rag_engine.py`)

A wrong citation scores worse than none, so matching fails closed.

1. **Read each note once.** Each note becomes a record with the routes it covers, the dates it
   is in force, and its effect. Windows: an explicit range in the text wins (N001 "Feb 24 to
   Mar 8"); "starting this week" means 8 weeks from that Monday (after 8 weeks the change is
   inside the own-history baseline and can no longer make a week look unusual); "this quarter"
   means that calendar quarter; otherwise the Monday-to-Sunday week of the note's date. Effect:
   phrases such as "not significantly affected", "without a rate change", "returned to normal",
   "remained stable", "not part of this dataset" are checked first, so a note saying costs did
   not move can never read as an increase. Only N001, N002 and N003 describe an increase.
2. **Vector index.** The notes are embedded with TF-IDF and searched by cosine similarity
   (`TfidfIndex`). With 10 notes this is all in memory, free and offline; the same interface
   would sit in front of FAISS or Chroma if the notes grew.
3. **Per flagged row**: retrieval with a route metadata filter (the note's `applies_to` is this
   route or `All Routes`), then two hard filters: the week must overlap the note's window, and
   the note must describe an increase. A note that applies to All Routes is also not accepted
   for a row flagged only for its gap to similar routes, because a nationwide change raises the
   peers as well and cannot explain the gap. Similarity only ranks notes that passed every
   filter (route-specific before All Routes, then similarity, then note id), so it can never
   promote an ineligible note.
4. **No eligible note means unexplained** with a blank `matched_note_id`. The reason still says
   which nearby notes were considered and why each does not count (wrong window, says costs
   were not affected, and so on), but those notes never fill `matched_note_id`.

### 5. Reasons, and where an LLM fits (`explain.py`)

The verdict and note id are settled before any wording is produced. By default the `reason`
column is written from fixed templates filled with the row's numbers and the note's own text and
dates, which needs no LLM at all. Every verdict can be decided by the deterministic pipeline, so
spending tokens on it would be "unlimited spend for a fixed answer".

LLM wording is optional:

```bash
python runner.py --llm ollama       # open-source model served locally by Ollama, free
python runner.py --llm anthropic    # claude-haiku-4-5, needs: pip install anthropic, ANTHROPIC_API_KEY
```

- Both run at **temperature 0** (Ollama also with a fixed seed). claude-haiku-4-5 is used for
  the hosted option because it is cheap and still accepts temperature 0; the newer hosted models
  no longer take sampling parameters.
- The model is given only the row's facts and the note that matching already picked.
- Its answer is checked: it must start with the right verdict, cite the matched note, mention
  no other note, and use no percentage that is not in the facts. If any check fails, or the
  model cannot be reached, the template is used and the log says so. The model can change
  wording, never a verdict, a note id or a number.
- Model and host are configurable with `OLLAMA_MODEL` (default `llama3.1:8b`), `OLLAMA_HOST`
  and `ANTHROPIC_MODEL`.

### 6. Questions about the findings (`qa.py`, stretch goal)

`qa.py` answers from the assistant's own results (`output/all_route_weeks.csv`) and nothing
else. It picks out a route or city names, a month and a year, then summarises the matching
weeks with their numbers, verdicts, note ids and reasons. If nothing was flagged it says so and
gives the range of movement. A question with no route or date ("anything about flooding?")
searches the notes index and shows which flagged weeks cite the notes it finds.

```
$ python qa.py "Why did Chennai-Bangalore get pricier in March 2025?"
Chennai-Bangalore in March 2025: 3 flagged week(s).
- Chennai-Bangalore, week of 2025-03-03: 3.48 per tonne-km, +22.8% vs its past average, +32.0% vs similar routes. Verdict: justified, note N001. ...
- Chennai-Bangalore, week of 2025-03-10: ... Verdict: unexplained. ...
```

## Output files

- `output/flagged_routes.csv`: the graded file. Columns, order and formats follow
  `sample_output_format.csv`: `route, week_of, cost_per_tonne_km` (2 dp),
  `vs_own_history` ("+35.5% vs this route's past average"),
  `vs_similar_routes` ("+21.0% vs similar-length routes this week"),
  `flagged` (`Yes` = flagged and unexplained, `No (justified)` = flagged but a note explains it),
  `matched_note_id` (blank when unexplained), `reason`. One row per flagged route-week, like the
  sample, sorted by week then route.
- `output/all_route_weeks.csv`: every route-week with its raw numbers, baselines, number of prior
  weeks used, flag and verdict. Unflagged weeks have `flagged = No`.
- `output/token_log.json`: see below.

Note on the sample file: its third row has an unquoted comma inside `reason`
("(N006, 2025-09-22)"), so strict CSV readers see 9 fields there. Our file quotes any `reason`
that contains a comma, which is standard CSV and reads back into exactly 8 columns.

## Cost of running it

`output/token_log.json` for one full run over the whole shipment file (default settings):

| | |
|---|---|
| LLM calls | 0 |
| Input tokens | 0 |
| Output tokens | 0 |
| Cost | $0.00 |

Nothing in the verdict path needs a model, so the default run makes no calls. The same file
also records what turning on LLM wording would cost, estimated from the exact prompts the run
would send (about 4 characters per token, 80 output tokens per reason): **19 calls, about 4,700
input and 1,500 output tokens, $0 with Ollama or about $0.012 with claude-haiku-4-5** at $1 /
$5 per million tokens. A run with `--llm` writes measured token counts from the API responses,
per call, in the same file.

## How I built it

Tools: Python 3, pandas, numpy and pytest. There are no external services, and the default run
needs no API key or model download.

1. **Read the inputs before writing logic.** The data has 7 routes with 104 complete weeks each,
   and 10 notes. Only three of the notes describe a cost increase; the rest say costs were not
   affected, conditions were normal, or the routes are outside the dataset.
2. **Pinned the numbers to the sample file first.** I tried the plausible readings of "cost per
   tonne-km" and "average", and kept only the ones that reproduce all three sample rows: a
   pooled weekly metric and simple means for both baselines. An average of per-shipment ratios
   misses the sample values.
3. **Built in pipeline order, with tests at each step:** preprocessing, then baselines, the flag
   rule, note matching, reasons, the runner, and finally Q&A. Each module only consumes the
   previous module's table, so it can be tested on small hand-made frames.
4. **Chose the flag thresholds from the data.** I looked at how much each route normally moves
   week to week and how far apart peers normally sit, then checked the result against the
   sample rows and the hand labels (see "The flag rule" above).
5. **Kept the model out of the decision path.** Verdicts and note ids come from explicit rules.
   An LLM can only reword the explanation, and its text is checked before it is used.

## How I convinced myself it was right

1. **Unit tests** (`python -m pytest`, 121 tests): validation and Monday bucketing edge cases
   (Sunday vs Monday, year boundaries, gaps), the 8-week window with fewer than 8 weeks and with
   gaps, peer exclusion and lone routes, every note window rule and window edge, every negation
   phrase, directional routes, the All-Routes/peer-gap rule, ranking ties, the LLM guard (wrong
   note, extra note, invented number, unreachable model), and that verdicts are identical with
   and without an LLM.
2. **Labelled set** (`evals/labeled_route_weeks.csv`, 23 route-weeks; `evals/labeled_notes.csv`,
   all 10 notes): written by hand from reading the notes and the brief, each with a one-line
   reason. It covers the three sample rows, both edges of the flood window, the two Chennai
   weeks after it, weeks where a note says costs did not change (N005, N010), the diesel week on
   every route, and a route's first week.
3. **Grounding audit** of every output row: each justified row's note must cover that route,
   overlap that week, describe an increase and be named in the reason; each unexplained row
   must have a blank note id.
4. **Independent recomputation** of both baselines for all 728 route-weeks with a plain loop
   that shares no code with `baselines.py`; they agree to 1e-9.
5. **Format contract**: columns equal the sample's, and the three sample rows match character
   for character in every column except the free-text `reason`.

`python evals/run_eval.py` runs checks 2 to 5 and currently passes all of them
(`output/eval_report.md`).

**Reproducibility.** `python evals/reproducibility_check.py` runs `runner.py` three times as
fresh processes and compares the outputs. Flags, numbers, verdicts and note ids are identical,
and the files are byte-identical (same SHA-256) because nothing in the default run is random.
The same check on Windows produced the same hashes as on Linux.
With `--llm`, the check compares every column except `reason`; temperature is 0, and the
verdict and note id are fixed before the model is called, so wording cannot move them.

## Decisions and trade-offs

- **Strict note windows.** Chennai-Bangalore stays expensive for two weeks after N001's stated
  end date (Mar 8). It is tempting to stretch N001 over the recovery, but the brief says a wrong
  window does not count and N009 says the route was back to normal, so those two weeks are
  unexplained and flagged for review. The reason text says exactly why N001 does not cover them.
- **All Routes notes vs gaps to peers.** Mumbai-Pune 2025-06-23 sits inside the diesel note's
  window (N003) and its rise vs its own history (+6.2%) is in N003's 5-7% range, but it was
  flagged for its +20.6% gap to similar routes. Diesel raises those routes too, so N003 cannot
  explain the gap and the week is unexplained.
- **Rules over a model for reading notes.** Ten short notes are read by explicit phrase and date
  rules that are reviewable, testable, and identical on every run. An LLM could read messier
  notes, but would then need its own checks; the rules fail closed (anything not clearly an
  increase can never justify a flag).
- **TF-IDF instead of a neural embedding model.** It needs no download, runs anywhere and is
  exactly reproducible. With 10 notes the hard filters do most of the work, and retrieval
  orders what survives and powers note search in `qa.py`.
- **"Starting this week" lasts 8 weeks.** An open-ended note needs some end; 8 weeks matches the
  own-history window, after which the change is part of the baseline.
- **Only flagged rows in the graded CSV**, mirroring the sample. The full table is in
  `all_route_weeks.csv`.

## Project layout

```
README.md
requirements.txt
preprocessing.py   ingestion, validation, weekly bucketing, cost per tonne-km
baselines.py       own-history and similar-route baselines
flagging.py        flag rule
rag_engine.py      note parsing, vector index, grounded matching
explain.py         reason text, optional LLM writers, token and cost log
runner.py          end-to-end run and CLI
qa.py              question answering over the results
data/              shipment_records.csv, context_notes.csv, sample_output_format_v2.csv (as given)
evals/             run_eval.py, reproducibility_check.py, labelled sets
tests/             pytest suite
output/            results of the latest run (CSV, token log, eval and reproducibility reports)
docs/ARCHITECTURE.md  module-by-module design notes
```
