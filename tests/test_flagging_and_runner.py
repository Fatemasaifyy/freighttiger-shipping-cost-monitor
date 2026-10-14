from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import explain
from explain import ReasonWriter, RowFacts, check_reason, template_reason
from flagging import apply_flags
from qa import FindingsQA
from rag_engine import NoteMatcher, parse_note
from runner import OUTPUT_COLUMNS, analyse, format_pct, write_outputs

ROOT = Path(__file__).resolve().parents[1]
SHIPMENTS = ROOT / "data" / "shipment_records.csv"
NOTES = ROOT / "data" / "context_notes.csv"


# ---------------------------------------------------------------- flag rule

def flag_table(rows):
    return pd.DataFrame(rows, columns=["pct_vs_own_history", "pct_vs_peers"])


@pytest.mark.parametrize("own, peer, flagged, basis", [
    (15.0, 0.0, True, "own history"),
    (14.9, 19.9, False, ""),
    (9.2, 23.6, True, "similar routes"),            # the Mumbai-Pune sample row
    (35.5, 21.0, True, "own history and similar routes"),
    (-0.1, 30.0, False, ""),                        # pricey vs peers but not rising
    (np.nan, 50.0, False, ""),                      # first week: no history, cannot be rising
    (6.0, np.nan, False, ""),
])
def test_flag_rule(own, peer, flagged, basis):
    out = apply_flags(flag_table([(own, peer)]))
    assert bool(out.loc[0, "is_flagged"]) is flagged
    assert out.loc[0, "flag_basis"] == basis


def test_thresholds_are_parameters():
    out = apply_flags(flag_table([(11.0, 0.0)]), own_threshold=10, peer_threshold=50)
    assert out.loc[0, "is_flagged"]


# ---------------------------------------------------------------- formatting

def test_format_pct():
    assert format_pct(35.4567, "vs x") == "+35.5% vs x"
    assert format_pct(-0.04, "vs x") == "-0.0% vs x"
    assert format_pct(float("nan"), "vs x") == "n/a (no baseline) vs x"


# ---------------------------------------------------------------- reasons and LLM guard

def facts_for(route, week, own, peer, rose=True, notes=None):
    matcher = NoteMatcher.from_csv(NOTES) if notes is None else NoteMatcher(notes)
    return RowFacts(route, week, 3.0, own, peer, 8, "own history", matcher.match(route, week, rose))


def test_template_cites_note_and_window_when_justified():
    text = template_reason(facts_for("Ahmedabad-Mumbai", date(2025, 1, 20), 29.5, 22.5))
    assert text.startswith("Justified") and "N002" in text and "2025-01-20" in text


def test_template_names_closest_note_when_unexplained():
    text = template_reason(facts_for("Mumbai-Pune", date(2025, 9, 15), 9.2, 23.6, rose=False))
    assert text.startswith("Unexplained") and "N006" in text


@pytest.mark.parametrize("text, ok", [
    ("Justified: note N002 says a festival surcharge hit the Ahmedabad-Mumbai corridor that week.", True),
    ("Justified: a festival surcharge hit the corridor.", False),                     # no note id
    ("Justified: note N002 and N003 explain it.", False),                             # extra note
    ("Unexplained: no note supports it.", False),                                     # wrong verdict
    ("Justified: note N002 drove a 45% surcharge.", False),                           # invented number
    ("Justified: note N002 explains the +29.5% rise.", True),
])
def test_llm_reason_checks(text, ok):
    facts = facts_for("Ahmedabad-Mumbai", date(2025, 1, 20), 29.5, 22.5)
    assert (check_reason(text, facts) == []) is ok


class FakeWriter:
    provider, model, temperature = "fake", "fake-model", 0.0
    input_usd_per_mtok, output_usd_per_mtok = 1.0, 5.0

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, system, prompt):
        self.calls += 1
        return self.reply, 100, 20


def test_llm_text_used_and_logged_when_it_passes_checks(monkeypatch):
    good = "Justified: note N002 (2025-01-20) records a festival surcharge on this corridor."
    monkeypatch.setattr(explain, "make_writer", lambda provider: FakeWriter(good))
    writer = ReasonWriter("fake")
    assert writer.reason(facts_for("Ahmedabad-Mumbai", date(2025, 1, 20), 29.5, 22.5)) == good
    summary = writer.log.summary()
    assert (summary["llm_calls"], summary["total_input_tokens"], summary["total_output_tokens"]) == (1, 100, 20)
    assert summary["estimated_cost_usd"] == pytest.approx(100 / 1e6 * 1.0 + 20 / 1e6 * 5.0)


def test_llm_text_replaced_by_template_when_it_cites_wrong_note(monkeypatch):
    monkeypatch.setattr(explain, "make_writer", lambda provider: FakeWriter("Justified: note N003 explains it."))
    writer = ReasonWriter("fake")
    facts = facts_for("Ahmedabad-Mumbai", date(2025, 1, 20), 29.5, 22.5)
    assert writer.reason(facts) == template_reason(facts)
    assert writer.log.summary()["calls_where_llm_text_failed_checks"] == 1


def test_unreachable_llm_falls_back_and_stops_calling(monkeypatch):
    class Broken(FakeWriter):
        def complete(self, system, prompt):
            self.calls += 1
            raise ConnectionError("down")

    broken = Broken("")
    monkeypatch.setattr(explain, "make_writer", lambda provider: broken)
    writer = ReasonWriter("fake")
    facts = facts_for("Ahmedabad-Mumbai", date(2025, 1, 20), 29.5, 22.5)
    assert writer.reason(facts) == template_reason(facts)
    assert writer.reason(facts) == template_reason(facts)
    assert broken.calls == 1
    assert writer.log.summary()["failed_requests"] == 1


# ---------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def run():
    return analyse(SHIPMENTS, NOTES)


def test_output_contract(run):
    output, _, _ = run
    assert list(output.columns) == OUTPUT_COLUMNS
    assert set(output["flagged"]) <= {"Yes", "No (justified)"}
    justified = output[output["flagged"] == "No (justified)"]
    assert (justified["matched_note_id"] != "").all()
    assert (output.loc[output["flagged"] == "Yes", "matched_note_id"] == "").all()


def test_verdicts_do_not_depend_on_llm(run, monkeypatch):
    output, _, _ = run
    monkeypatch.setattr(explain, "make_writer", lambda provider: FakeWriter("anything at all"))
    with_llm, _, log = analyse(SHIPMENTS, NOTES, llm="fake")
    cols = [c for c in OUTPUT_COLUMNS if c != "reason"]
    pd.testing.assert_frame_equal(output[cols], with_llm[cols])
    assert log["this_run"]["llm_calls"] == len(output)


def test_written_files_round_trip(run, tmp_path):
    output, table, log = run
    write_outputs(output, table, log, tmp_path)
    back = pd.read_csv(tmp_path / "flagged_routes.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(back, output.astype(str))
    assert (tmp_path / "token_log.json").exists()


# ---------------------------------------------------------------- Q&A

@pytest.fixture(scope="module")
def qa(run, tmp_path_factory):
    output, table, log = run
    out = tmp_path_factory.mktemp("qa")
    write_outputs(output, table, log, out)
    return FindingsQA.load(out, NOTES)


def test_qa_route_and_month(qa):
    answer = qa.answer("Why did Chennai-Bangalore get pricier in March 2025?")
    assert "N001" in answer and "2025-03-10" in answer and "unexplained" in answer


def test_qa_city_pair_in_any_case(qa):
    assert qa.find_routes("what about mumbai to pune?") == ["Mumbai-Pune"]


def test_qa_quiet_period(qa):
    assert "no week was flagged" in qa.answer("How did Delhi-Chennai do in June 2024?")


def test_qa_note_search(qa):
    assert "N001" in qa.answer("anything about flooding?")


def test_qa_parses_months():
    assert FindingsQA.find_month_year("pricier in Sept 2025") == (9, 2025)
    assert FindingsQA.find_month_year("in march") == (3, None)
