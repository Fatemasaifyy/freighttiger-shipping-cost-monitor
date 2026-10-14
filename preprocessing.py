"""Module 1: ingestion, validation, Monday-to-Sunday bucketing and weekly cost per tonne-km.

The output of this module is one row per (route, week_of) with the pooled metric

    cost_per_tonne_km = sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)

computed over every shipment on that route in that week. Pooling (rather than averaging
per-shipment ratios) follows the brief's "total freight cost / (quantity x distance)" and
reproduces the numbers in sample_output_format.csv exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

VALID_ROUTE_TYPES = ("Short", "Medium", "Long")

REQUIRED_COLUMNS = (
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
)

TEXT_COLUMNS = ("shipment_id", "origin", "destination", "route_type")
NUMERIC_COLUMNS = ("quantity_tonnes", "distance_km", "freight_cost_inr")

WEEKLY_COLUMNS = [
    "route",
    "origin",
    "destination",
    "route_type",
    "week_of",
    "shipment_count",
    "total_cost_inr",
    "total_tonne_km",
    "cost_per_tonne_km",
]


class DataValidationError(ValueError):
    """Raised when the shipment file breaks the input contract."""


def load_shipments(path: str | Path) -> pd.DataFrame:
    """Read shipment_records.csv and return a validated, cleaned DataFrame."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    return clean_shipments(raw)


def clean_shipments(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise raw shipment rows.

    Fails loudly instead of silently dropping rows: a dropped row would change the
    weekly numbers without anyone noticing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    df = raw.copy()
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df["route_type"] = df["route_type"].str.capitalize()

    blank = df[list(TEXT_COLUMNS)].eq("").any(axis=1)
    _raise_if(df, blank, "Blank origin/destination/route_type/shipment_id")

    bad_type = ~df["route_type"].isin(VALID_ROUTE_TYPES)
    _raise_if(df, bad_type, f"route_type not in {VALID_ROUTE_TYPES}")

    dupes = df["shipment_id"].duplicated(keep=False)
    _raise_if(df, dupes, "Duplicate shipment_id")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    non_numeric = df[list(NUMERIC_COLUMNS)].isna().any(axis=1)
    _raise_if(df, non_numeric, "Non-numeric quantity/distance/cost")

    # quantity and distance are the metric's denominator, so they must be strictly positive.
    bad_denominator = (df["quantity_tonnes"] <= 0) | (df["distance_km"] <= 0)
    _raise_if(df, bad_denominator, "Non-positive quantity_tonnes or distance_km")
    _raise_if(df, df["freight_cost_inr"] < 0, "Negative freight_cost_inr")

    df["shipment_date"] = pd.to_datetime(df["shipment_date"], format="%Y-%m-%d", errors="coerce")
    _raise_if(df, df["shipment_date"].isna(), "Unparseable shipment_date (expected YYYY-MM-DD)")

    df["route"] = df["origin"] + "-" + df["destination"]

    # A route must carry one route_type, otherwise "exclude self from peers" is ambiguous.
    types_per_route = df.groupby("route")["route_type"].nunique()
    conflicting = types_per_route[types_per_route > 1].index
    _raise_if(df, df["route"].isin(conflicting), "Route appears with more than one route_type")

    return df


def add_week_of(df: pd.DataFrame) -> pd.DataFrame:
    """Add week_of: the Monday of the Monday-to-Sunday week containing shipment_date."""
    out = df.copy()
    dates = out["shipment_date"].dt.normalize()
    out["week_of"] = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    return out


def weekly_route_metrics(shipments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shipments into one row per (route, week_of) with pooled cost per tonne-km.

    Weeks with no shipments on a route produce no row; nothing is padded or interpolated.
    """
    df = shipments if "week_of" in shipments.columns else add_week_of(shipments)
    df = df.assign(tonne_km=df["quantity_tonnes"] * df["distance_km"])

    weekly = (
        df.groupby(["route", "origin", "destination", "route_type", "week_of"], as_index=False)
        .agg(
            shipment_count=("shipment_id", "size"),
            total_cost_inr=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
        )
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost_inr"] / weekly["total_tonne_km"]
    return weekly[WEEKLY_COLUMNS].sort_values(["route", "week_of"]).reset_index(drop=True)


def build_weekly_table(path: str | Path) -> pd.DataFrame:
    """Convenience entry point: CSV path -> weekly route metrics."""
    return weekly_route_metrics(add_week_of(load_shipments(path)))


def _raise_if(df: pd.DataFrame, mask: pd.Series, message: str) -> None:
    if mask.any():
        ids = df.loc[mask, "shipment_id"].astype(str).head(10).tolist()
        raise DataValidationError(f"{message}: {int(mask.sum())} row(s), e.g. {ids}")
