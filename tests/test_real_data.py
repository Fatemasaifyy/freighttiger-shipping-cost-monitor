"""Sanity checks against the real shipment file and the rows in sample_output_format.csv.

Skipped automatically when the data files are not next to the code.
"""

from pathlib import Path

import pandas as pd
import pytest

from baselines import compute_baselines
from preprocessing import build_weekly_table

ROOT = Path(__file__).resolve().parents[1]
SHIPMENTS = ROOT / "data" / "shipment_records.csv"

pytestmark = pytest.mark.skipif(not SHIPMENTS.exists(), reason="shipment_records.csv not present")


@pytest.fixture(scope="module")
def table():
    return compute_baselines(build_weekly_table(SHIPMENTS))


# (route, week_of, cost_per_tonne_km, % vs own history, % vs similar routes) from the sample file
SAMPLE_ROWS = [
    ("Delhi-Jaipur", "2024-11-11", 4.17, 35.5, 21.0),
    ("Ahmedabad-Mumbai", "2025-01-20", 3.29, 29.5, 22.5),
    ("Mumbai-Pune", "2025-09-15", 3.98, 9.2, 23.6),
]


@pytest.mark.parametrize("route, week, metric, own_pct, peer_pct", SAMPLE_ROWS)
def test_reproduces_sample_output_numbers(table, route, week, metric, own_pct, peer_pct):
    row = table[(table["route"] == route) & (table["week_of"] == pd.Timestamp(week))].iloc[0]
    assert round(row["cost_per_tonne_km"], 2) == metric
    assert round(row["pct_vs_own_history"], 1) == own_pct
    assert round(row["pct_vs_peers"], 1) == peer_pct


def test_every_week_of_is_a_monday(table):
    assert (table["week_of"].dt.weekday == 0).all()


def test_one_row_per_route_week(table):
    assert not table.duplicated(["route", "week_of"]).any()


def test_shipment_counts_add_up(table):
    raw = pd.read_csv(SHIPMENTS)
    assert table["shipment_count"].sum() == len(raw)


def test_only_first_eight_weeks_have_short_history(table):
    for _, route_rows in table.groupby("route"):
        weeks = route_rows.sort_values("week_of")["own_history_weeks"].tolist()
        assert weeks[:9] == list(range(9))
        assert all(w == 8 for w in weeks[8:])


def test_all_rows_have_peers(table):
    # every route_type has 2+ routes and the data has no gaps, so every row has a peer
    assert (table["peer_routes"] >= 1).all()
    assert table.groupby("route_type")["peer_routes"].max().to_dict() == {"Long": 1, "Medium": 2, "Short": 1}
