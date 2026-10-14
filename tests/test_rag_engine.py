from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from rag_engine import (
    NoteMatcher,
    NotesValidationError,
    TfidfIndex,
    classify_effect,
    load_notes,
    note_window,
    overlaps,
    parse_note,
)

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "data" / "context_notes.csv"


def matcher_for(*notes):
    return NoteMatcher([parse_note(*n) for n in notes])


# ---------------------------------------------------------------- effect classification

@pytest.mark.parametrize("text, effect", [
    ("Heavy flooding forced longer detours and higher trip costs.", "increase"),
    ("Transporters applied a temporary surcharge.", "increase"),
    ("Diesel prices rose nationwide.", "increase"),
    ("Maintenance caused delays; costs were not significantly affected.", "no_effect"),
    ("Compliance costs were absorbed without a rate change.", "no_effect"),
    ("The route returned to normal conditions after repairs.", "improvement"),
    ("Demand remained stable this quarter with no major disruptions.", "informational"),
    ("A toll plaza opened, but the affected routes are not part of this dataset.", "out_of_scope"),
    ("Something happened on the highway.", "informational"),
])
def test_classify_effect(text, effect):
    assert classify_effect(text) == effect


def test_negation_beats_increase_words():
    # mentions "higher" but says costs did not change
    assert classify_effect("Fuel was higher but costs were not affected.") == "no_effect"


# ---------------------------------------------------------------- windows

def test_explicit_range_overrides_date():
    start, end, _ = note_window(date(2025, 2, 24), "disrupted movement from Feb 24 to Mar 8")
    assert (start, end) == (date(2025, 2, 24), date(2025, 3, 8))


def test_range_crossing_new_year():
    start, end, _ = note_window(date(2024, 12, 28), "closed from Dec 28 to Jan 3")
    assert (start, end) == (date(2024, 12, 28), date(2025, 1, 3))


def test_starting_this_week_is_eight_weeks():
    start, end, _ = note_window(date(2025, 5, 7), "prices rose starting this week")
    assert start == date(2025, 5, 5)
    assert end == date(2025, 6, 29)


def test_quarter_window():
    start, end, _ = note_window(date(2025, 11, 9), "nothing unusual this quarter")
    assert (start, end) == (date(2025, 10, 1), date(2025, 12, 31))


def test_default_is_the_notes_monday_to_sunday_week():
    start, end, _ = note_window(date(2025, 1, 22), "a festival surcharge")
    assert (start, end) == (date(2025, 1, 20), date(2025, 1, 26))


def test_overlap_edges():
    note = parse_note("N1", date(2025, 1, 22), "A-B", "surcharge")
    assert overlaps(date(2025, 1, 20), note)
    assert not overlaps(date(2025, 1, 27), note)  # the next Monday
    assert not overlaps(date(2025, 1, 13), note)  # week ends Sunday the 19th


# ---------------------------------------------------------------- matching

def test_route_specific_note_justifies_its_own_route_only():
    m = matcher_for(("N1", date(2025, 1, 20), "A-B", "temporary surcharge"))
    assert m.match("A-B", date(2025, 1, 20)).matched_note_id == "N1"
    other = m.match("B-A", date(2025, 1, 20))  # routes are directional
    assert other.verdict == "unexplained" and other.matched_note_id is None
    assert "applies to A-B" in other.rejected["N1"]


def test_wrong_window_does_not_count():
    m = matcher_for(("N1", date(2025, 1, 20), "A-B", "temporary surcharge"))
    result = m.match("A-B", date(2025, 1, 27))
    assert result.verdict == "unexplained"
    assert "only covers" in result.rejected["N1"]
    assert [n.note_id for n in result.nearest_rejected] == ["N1"]


def test_note_saying_costs_unaffected_does_not_count():
    m = matcher_for(("N1", date(2025, 1, 20), "A-B", "delays, but costs were not significantly affected"))
    result = m.match("A-B", date(2025, 1, 20))
    assert result.verdict == "unexplained"
    assert result.rejected["N1"] == "says costs were not affected"


def test_all_routes_note_cannot_explain_a_peer_only_flag():
    m = matcher_for(("N1", date(2025, 5, 5), "All Routes", "diesel prices rose starting this week"))
    assert m.match("A-B", date(2025, 5, 12), rose_vs_own_history=True).matched_note_id == "N1"
    peer_only = m.match("A-B", date(2025, 5, 12), rose_vs_own_history=False)
    assert peer_only.verdict == "unexplained"
    assert "all routes" in peer_only.rejected["N1"]


def test_route_specific_note_ranks_above_all_routes_note():
    m = matcher_for(
        ("N1", date(2025, 5, 5), "All Routes", "diesel prices rose starting this week, higher costs"),
        ("N2", date(2025, 5, 5), "A-B", "temporary surcharge on the A-B corridor"),
    )
    result = m.match("A-B", date(2025, 5, 5))
    assert result.matched_note_id == "N2"
    assert result.candidates == ("N2", "N1")


def test_nearest_rejected_ignores_far_notes():
    m = matcher_for(("N1", date(2024, 1, 1), "A-B", "temporary surcharge"))
    assert m.match("A-B", date(2024, 6, 3)).nearest_rejected == ()


def test_empty_notes_means_everything_unexplained():
    result = NoteMatcher([]).match("A-B", date(2025, 1, 6))
    assert result.verdict == "unexplained" and result.nearest_rejected == ()


def test_matching_is_repeatable():
    m1, m2 = NoteMatcher.from_csv(NOTES), NoteMatcher.from_csv(NOTES)
    weeks = pd.date_range("2024-01-01", "2025-12-22", freq="7D").date
    for route in ["Chennai-Bangalore", "Mumbai-Pune", "Ahmedabad-Mumbai"]:
        for week in weeks:
            assert m1.match(route, week) == m2.match(route, week)


# ---------------------------------------------------------------- real notes file

def test_real_notes_only_three_can_justify():
    notes = load_notes(NOTES)
    assert [n.note_id for n in notes if n.effect == "increase"] == ["N001", "N002", "N003"]


@pytest.mark.parametrize("route, week, expected", [
    ("Ahmedabad-Mumbai", date(2025, 1, 20), "N002"),
    ("Chennai-Bangalore", date(2025, 2, 24), "N001"),
    ("Chennai-Bangalore", date(2025, 3, 3), "N001"),
    ("Chennai-Bangalore", date(2025, 3, 10), None),   # N001 ends Mar 8
    ("Chennai-Bangalore", date(2025, 3, 17), None),   # N009 says back to normal
    ("Mumbai-Delhi", date(2024, 7, 29), None),        # N005 says not affected
    ("Mumbai-Pune", date(2025, 10, 27), None),        # N010 says no rate change
    ("Mumbai-Pune", date(2025, 9, 15), None),         # N006 says stable
])
def test_real_notes_matching(route, week, expected):
    assert NoteMatcher.from_csv(NOTES).match(route, week).matched_note_id == expected


def test_duplicate_note_ids_rejected(tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("note_id,date,applies_to,note\nN1,2025-01-01,A-B,x\nN1,2025-01-02,A-B,y\n")
    with pytest.raises(NotesValidationError):
        load_notes(path)


def test_bad_note_date_rejected(tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("note_id,date,applies_to,note\nN1,01/02/2025,A-B,x\n")
    with pytest.raises(NotesValidationError):
        load_notes(path)


# ---------------------------------------------------------------- vector index

def test_index_ranks_the_relevant_note_first():
    index = TfidfIndex(["heavy flooding forced detours", "festival surcharge on the corridor", "diesel prices rose"])
    scores = index.scores("why was there flood damage")
    assert scores.argmax() == 0


def test_index_scores_are_zero_for_unknown_words():
    index = TfidfIndex(["heavy flooding forced detours"])
    assert index.scores("zzz qqq").tolist() == [0.0]
