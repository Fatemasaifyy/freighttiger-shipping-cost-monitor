import math

import pandas as pd
import pytest

from baselines import compute_baselines, own_history_baseline, peer_baseline, pct_change
from tests.helpers import mondays, weekly_frame


def series_for(route, route_type, values, start="2024-01-01"):
    return [(route, route_type, w, v) for w, v in zip(mondays(start, len(values)), values)]


# ---------- own history ----------

def test_first_week_has_no_history():
    out = own_history_baseline(weekly_frame(series_for("A-B", "Short", [1.0, 2.0])))
    assert math.isnan(out.loc[0, "own_history_avg"])
    assert out.loc[0, "own_history_weeks"] == 0


@pytest.mark.parametrize("n_prior", range(1, 8))
def test_fewer_than_eight_prior_weeks_uses_all_available(n_prior):
    values = [float(i + 1) for i in range(n_prior + 1)]
    out = own_history_baseline(weekly_frame(series_for("A-B", "Short", values)))
    row = out.iloc[n_prior]
    assert row["own_history_weeks"] == n_prior
    assert row["own_history_avg"] == pytest.approx(sum(values[:n_prior]) / n_prior)


def test_window_is_capped_at_eight_strictly_prior_weeks():
    values = [float(i) for i in range(1, 13)]  # weeks 1..12
    out = own_history_baseline(weekly_frame(series_for("A-B", "Short", values)))
    row = out.iloc[11]  # week 12: prior window is weeks 4..11
    assert row["own_history_weeks"] == 8
    assert row["own_history_avg"] == pytest.approx(sum(range(4, 12)) / 8)


def test_current_week_never_leaks_into_its_own_baseline():
    values = [1.0] * 8 + [1000.0]
    out = own_history_baseline(weekly_frame(series_for("A-B", "Short", values)))
    assert out.iloc[8]["own_history_avg"] == pytest.approx(1.0)


def test_future_weeks_do_not_affect_past_baselines():
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    short = own_history_baseline(weekly_frame(series_for("A-B", "Short", base)))
    longer = own_history_baseline(weekly_frame(series_for("A-B", "Short", base + [999.0, 999.0])))
    pd.testing.assert_series_equal(
        short["own_history_avg"], longer["own_history_avg"].iloc[: len(base)], check_names=False
    )


def test_gap_weeks_shrink_the_window_instead_of_reaching_further_back():
    # data in weeks 1..3, gap in weeks 4..9, then week 10.
    weeks = mondays("2024-01-01", 10)
    rows = [("A-B", "Short", w, v) for w, v in zip(weeks[:3], [1.0, 2.0, 3.0])]
    rows.append(("A-B", "Short", weeks[9], 10.0))
    out = own_history_baseline(weekly_frame(rows))
    row = out[out["week_of"] == pd.Timestamp(weeks[9])].iloc[0]
    # window for week 10 is weeks 2..9 -> only weeks 2 and 3 have data
    assert row["own_history_weeks"] == 2
    assert row["own_history_avg"] == pytest.approx(2.5)


def test_route_with_long_gap_has_no_history():
    weeks = mondays("2024-01-01", 12)
    rows = [("A-B", "Short", weeks[0], 1.0), ("A-B", "Short", weeks[11], 5.0)]
    out = own_history_baseline(weekly_frame(rows))
    assert out.iloc[1]["own_history_weeks"] == 0
    assert math.isnan(out.iloc[1]["own_history_avg"])


def test_history_is_per_route():
    rows = series_for("A-B", "Short", [1.0, 1.0, 1.0]) + series_for("C-D", "Short", [9.0, 9.0, 9.0])
    out = own_history_baseline(weekly_frame(rows))
    assert out[out["route"] == "A-B"]["own_history_avg"].dropna().eq(1.0).all()
    assert out[out["route"] == "C-D"]["own_history_avg"].dropna().eq(9.0).all()


def test_route_starting_later_gets_its_own_short_history():
    rows = series_for("A-B", "Short", [1.0] * 10) + series_for("C-D", "Short", [4.0, 6.0], start="2024-02-26")
    out = own_history_baseline(weekly_frame(rows))
    cd = out[out["route"] == "C-D"].reset_index(drop=True)
    assert cd.loc[0, "own_history_weeks"] == 0
    assert cd.loc[1, "own_history_weeks"] == 1
    assert cd.loc[1, "own_history_avg"] == pytest.approx(4.0)


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError):
        own_history_baseline(weekly_frame(series_for("A-B", "Short", [1.0])), window_weeks=0)


# ---------- peers ----------

def test_peer_average_excludes_self():
    w = "2024-01-01"
    out = peer_baseline(weekly_frame([
        ("A-B", "Medium", w, 1.0),
        ("C-D", "Medium", w, 2.0),
        ("E-F", "Medium", w, 6.0),
    ])).set_index("route")
    assert out.loc["A-B", "peer_avg"] == pytest.approx(4.0)
    assert out.loc["C-D", "peer_avg"] == pytest.approx(3.5)
    assert out.loc["E-F", "peer_avg"] == pytest.approx(1.5)
    assert (out["peer_routes"] == 2).all()


def test_two_route_type_peer_is_the_other_route():
    w = "2024-01-01"
    out = peer_baseline(weekly_frame([("A-B", "Short", w, 1.0), ("C-D", "Short", w, 3.0)])).set_index("route")
    assert out.loc["A-B", "peer_avg"] == 3.0
    assert out.loc["C-D", "peer_avg"] == 1.0


def test_single_route_in_route_type_has_no_peer():
    w = "2024-01-01"
    out = peer_baseline(weekly_frame([
        ("A-B", "Long", w, 5.0),
        ("C-D", "Short", w, 1.0),
        ("E-F", "Short", w, 2.0),
    ])).set_index("route")
    assert math.isnan(out.loc["A-B", "peer_avg"])
    assert out.loc["A-B", "peer_routes"] == 0


def test_peers_only_from_same_week_and_same_route_type():
    out = peer_baseline(weekly_frame([
        ("A-B", "Short", "2024-01-01", 1.0),
        ("C-D", "Short", "2024-01-08", 100.0),  # different week
        ("E-F", "Long", "2024-01-01", 50.0),    # different route_type
        ("G-H", "Short", "2024-01-01", 3.0),
    ])).set_index("route")
    assert out.loc["A-B", "peer_avg"] == 3.0
    assert out.loc["A-B", "peer_routes"] == 1
    assert math.isnan(out.loc["C-D", "peer_avg"])


def test_peer_missing_that_week_reduces_peer_count():
    out = peer_baseline(weekly_frame([
        ("A-B", "Medium", "2024-01-08", 1.0),
        ("C-D", "Medium", "2024-01-08", 2.0),
        ("E-F", "Medium", "2024-01-01", 9.0),  # E-F shipped only in the previous week
    ])).set_index("route")
    assert out.loc["A-B", "peer_avg"] == 2.0
    assert out.loc["A-B", "peer_routes"] == 1


# ---------- combined ----------

def test_pct_change_handles_missing_and_zero_baselines():
    out = pct_change(pd.Series([2.0, 2.0, 2.0]), pd.Series([1.0, float("nan"), 0.0]))
    assert out.iloc[0] == pytest.approx(100.0)
    assert math.isnan(out.iloc[1]) and math.isnan(out.iloc[2])


def test_compute_baselines_keeps_every_row_and_is_deterministic():
    rows = series_for("A-B", "Short", [1.0, 2.0, 3.0]) + series_for("C-D", "Short", [2.0, 2.0, 2.0])
    weekly = weekly_frame(rows)
    first = compute_baselines(weekly)
    second = compute_baselines(weekly.sample(frac=1.0, random_state=1))
    assert len(first) == len(weekly)
    pd.testing.assert_frame_equal(first, second)
    last_ab = first[first["route"] == "A-B"].iloc[-1]
    assert last_ab["pct_vs_own_history"] == pytest.approx((3.0 / 1.5 - 1) * 100)
    assert last_ab["pct_vs_peers"] == pytest.approx(50.0)
