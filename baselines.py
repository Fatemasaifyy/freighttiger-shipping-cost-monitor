"""Module 2: the two graded comparison baselines, computed with zero look-ahead.

vs own history
    Mean of the route's weekly cost_per_tonne_km over the 8 calendar weeks strictly before
    week_of (week_of - 8 weeks ... week_of - 1 week). If fewer than 8 of those weeks have
    data (start of the series, or gaps), the mean uses only the weeks that exist and
    own_history_weeks says how many. No padding, no extrapolation. The very first week of a
    route has no baseline (NaN, 0 weeks).

vs similar routes
    Mean of cost_per_tonne_km in the same week across every *other* route with the same
    route_type. A route that is alone in its route_type that week has no peer baseline
    (NaN, 0 peers).

Both means are simple (unweighted) averages of weekly route values, which reproduces the
percentages in sample_output_format.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OWN_HISTORY_WEEKS = 8

BASELINE_COLUMNS = [
    "own_history_avg",
    "own_history_weeks",
    "pct_vs_own_history",
    "peer_avg",
    "peer_routes",
    "pct_vs_peers",
]


def own_history_baseline(weekly: pd.DataFrame, window_weeks: int = OWN_HISTORY_WEEKS) -> pd.DataFrame:
    """Return weekly with own_history_avg and own_history_weeks added.

    Pairs every route-week with the same route's earlier weeks that fall inside the
    trailing window, so the current week can never leak into its own baseline.
    """
    if window_weeks < 1:
        raise ValueError("window_weeks must be at least 1")

    keys = weekly[["route", "week_of"]]
    history = weekly[["route", "week_of", "cost_per_tonne_km"]].rename(
        columns={"week_of": "prior_week", "cost_per_tonne_km": "prior_value"}
    )
    pairs = keys.merge(history, on="route")
    weeks_back = (pairs["week_of"] - pairs["prior_week"]).dt.days / 7
    in_window = (weeks_back >= 1) & (weeks_back <= window_weeks)

    stats = (
        pairs[in_window]
        .groupby(["route", "week_of"])["prior_value"]
        .agg(own_history_avg="mean", own_history_weeks="size")
        .reset_index()
    )
    out = weekly.merge(stats, on=["route", "week_of"], how="left", validate="one_to_one")
    out["own_history_weeks"] = out["own_history_weeks"].fillna(0).astype(int)
    return out


def peer_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return weekly with peer_avg and peer_routes added (same route_type, same week, excluding self)."""
    keys = weekly[["route", "route_type", "week_of"]]
    peers = weekly[["route", "route_type", "week_of", "cost_per_tonne_km"]].rename(
        columns={"route": "peer_route", "cost_per_tonne_km": "peer_value"}
    )
    pairs = keys.merge(peers, on=["route_type", "week_of"])
    pairs = pairs[pairs["route"] != pairs["peer_route"]]

    stats = (
        pairs.groupby(["route", "week_of"])["peer_value"]
        .agg(peer_avg="mean", peer_routes="size")
        .reset_index()
    )
    out = weekly.merge(stats, on=["route", "week_of"], how="left", validate="one_to_one")
    out["peer_routes"] = out["peer_routes"].fillna(0).astype(int)
    return out


def pct_change(value: pd.Series, baseline: pd.Series) -> pd.Series:
    """Percent difference of value over baseline; NaN where the baseline is missing or zero."""
    safe = baseline.where(baseline != 0)
    return (value / safe - 1.0) * 100.0


def compute_baselines(weekly: pd.DataFrame, window_weeks: int = OWN_HISTORY_WEEKS) -> pd.DataFrame:
    """Add both baselines and their percent deltas to the weekly route table."""
    out = own_history_baseline(weekly, window_weeks)
    out = peer_baseline(out)
    out["pct_vs_own_history"] = pct_change(out["cost_per_tonne_km"], out["own_history_avg"])
    out["pct_vs_peers"] = pct_change(out["cost_per_tonne_km"], out["peer_avg"])
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["route", "week_of"]).reset_index(drop=True)
