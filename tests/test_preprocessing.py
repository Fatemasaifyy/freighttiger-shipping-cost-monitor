import pandas as pd
import pytest

from preprocessing import DataValidationError, add_week_of, clean_shipments, weekly_route_metrics
from tests.helpers import make_raw, make_shipments


# ---------- week bucketing ----------

@pytest.mark.parametrize(
    "date, expected_monday",
    [
        ("2024-01-01", "2024-01-01"),  # Monday maps to itself
        ("2024-01-07", "2024-01-01"),  # Sunday belongs to the week that started the previous Monday
        ("2024-01-08", "2024-01-08"),  # next Monday starts a new week
        ("2024-03-02", "2024-02-26"),  # week spanning a month boundary
        ("2024-02-29", "2024-02-26"),  # leap day
        ("2025-01-01", "2024-12-30"),  # week spanning a year boundary
        ("2025-12-28", "2025-12-22"),  # last date in the real file (a Sunday)
    ],
)
def test_week_of_is_monday_of_monday_to_sunday_week(date, expected_monday):
    df = add_week_of(make_shipments([("A", "B", "Short", 10, 100, 1000, date)]))
    assert df.loc[0, "week_of"] == pd.Timestamp(expected_monday)
    assert df.loc[0, "week_of"].weekday() == 0


def test_sunday_and_next_monday_land_in_different_weeks():
    df = weekly_route_metrics(add_week_of(make_shipments([
        ("A", "B", "Short", 10, 100, 1000, "2024-01-07"),
        ("A", "B", "Short", 10, 100, 3000, "2024-01-08"),
    ])))
    assert list(df["week_of"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")]
    assert list(df["cost_per_tonne_km"]) == [1.0, 3.0]


# ---------- metric ----------

def test_weekly_metric_is_pooled_total_cost_over_total_tonne_km():
    # per-shipment ratios: 1000 / 1000 = 1.0 and 9000 / 3000 = 3.0
    df = weekly_route_metrics(make_shipments([
        ("A", "B", "Short", 10, 100, 1000, "2024-01-01"),
        ("A", "B", "Short", 30, 100, 9000, "2024-01-03"),
    ]))
    # pooled: 10000 / 4000 = 2.5; mean of ratios would be (1 + 3) / 2 = 2.0
    assert df.loc[0, "cost_per_tonne_km"] == pytest.approx(2.5)
    assert df.loc[0, "shipment_count"] == 2
    assert df.loc[0, "total_tonne_km"] == pytest.approx(4000)


def test_routes_are_directional_and_kept_separate():
    df = weekly_route_metrics(make_shipments([
        ("A", "B", "Short", 10, 100, 1000, "2024-01-01"),
        ("B", "A", "Short", 10, 100, 2000, "2024-01-01"),
    ]))
    assert set(df["route"]) == {"A-B", "B-A"}


def test_missing_weeks_are_not_padded():
    df = weekly_route_metrics(make_shipments([
        ("A", "B", "Short", 10, 100, 1000, "2024-01-01"),
        ("A", "B", "Short", 10, 100, 1000, "2024-01-22"),
    ]))
    assert len(df) == 2


def test_output_is_sorted_and_deterministic():
    rows = [
        ("C", "D", "Long", 10, 1000, 20000, "2024-01-09"),
        ("A", "B", "Short", 10, 100, 1000, "2024-01-02"),
        ("A", "B", "Short", 12, 100, 1500, "2024-01-01"),
    ]
    first = weekly_route_metrics(make_shipments(rows))
    second = weekly_route_metrics(make_shipments(list(reversed(rows))))
    pd.testing.assert_frame_equal(first, second)
    assert list(first["route"]) == ["A-B", "C-D"]


# ---------- cleaning and validation ----------

def test_whitespace_and_route_type_case_are_normalised():
    df = make_shipments([(" Mumbai ", "Pune ", "short", 10, 100, 1000, "2024-01-01")])
    assert df.loc[0, "route"] == "Mumbai-Pune"
    assert df.loc[0, "route_type"] == "Short"


def test_missing_column_is_rejected():
    raw = make_raw([("A", "B", "Short", 10, 100, 1000, "2024-01-01")]).drop(columns="distance_km")
    with pytest.raises(DataValidationError, match="distance_km"):
        clean_shipments(raw)


@pytest.mark.parametrize(
    "row, message",
    [
        (("A", "B", "Short", 0, 100, 1000, "2024-01-01"), "Non-positive"),
        (("A", "B", "Short", 10, -5, 1000, "2024-01-01"), "Non-positive"),
        (("A", "B", "Short", 10, 100, -1, "2024-01-01"), "Negative"),
        (("A", "B", "Short", "ten", 100, 1000, "2024-01-01"), "Non-numeric"),
        (("A", "B", "Short", 10, 100, 1000, "01/02/2024"), "shipment_date"),
        (("A", "B", "Short", 10, 100, 1000, "2024-02-30"), "shipment_date"),
        (("A", "B", "Tiny", 10, 100, 1000, "2024-01-01"), "route_type"),
        (("", "B", "Short", 10, 100, 1000, "2024-01-01"), "Blank"),
    ],
)
def test_bad_rows_fail_loudly(row, message):
    with pytest.raises(DataValidationError, match=message):
        make_shipments([row])


def test_duplicate_shipment_ids_are_rejected():
    raw = make_raw([
        ("A", "B", "Short", 10, 100, 1000, "2024-01-01"),
        ("A", "B", "Short", 10, 100, 1000, "2024-01-02"),
    ])
    raw["shipment_id"] = "S1"
    with pytest.raises(DataValidationError, match="Duplicate"):
        clean_shipments(raw)


def test_route_with_two_route_types_is_rejected():
    with pytest.raises(DataValidationError, match="more than one route_type"):
        make_shipments([
            ("A", "B", "Short", 10, 100, 1000, "2024-01-01"),
            ("A", "B", "Medium", 10, 100, 1000, "2024-01-08"),
        ])


def test_zero_cost_is_allowed():
    df = weekly_route_metrics(make_shipments([("A", "B", "Short", 10, 100, 0, "2024-01-01")]))
    assert df.loc[0, "cost_per_tonne_km"] == 0
