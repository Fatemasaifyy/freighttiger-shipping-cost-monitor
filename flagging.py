"""Module 4a: the flag rule.

A route-week is flagged when its cost is rising AND it looks out of the ordinary:

    rising          pct_vs_own_history > 0        (above the route's own trailing average)
    out of ordinary pct_vs_own_history >= 15%     (unusual for this route)
                    or pct_vs_peers >= 20%        (unusual next to similar-length routes)

Why these numbers (see README, "Choosing the flag rule"):
- Week-to-week noise vs own history is small (standard deviation 3-6% per route), so +15% is
  well outside normal movement, while a 5-7% nationwide diesel rise (note N003) stays below it.
- Peer gaps are partly structural (Mumbai-Pune normally runs ~11% above Delhi-Jaipur), so the
  peer threshold sits higher, at +20%, to avoid flagging a route for simply being itself.
- Requiring "rising" means a route that is merely expensive next to its peers, but cheaper
  than its own recent past, is not flagged.

A route's first week has no own history, so it can never count as rising and is not flagged.
"""

from __future__ import annotations

import pandas as pd

OWN_HISTORY_THRESHOLD_PCT = 15.0
PEER_THRESHOLD_PCT = 20.0


def apply_flags(
    table: pd.DataFrame,
    own_threshold: float = OWN_HISTORY_THRESHOLD_PCT,
    peer_threshold: float = PEER_THRESHOLD_PCT,
) -> pd.DataFrame:
    """Add is_flagged and flag_basis columns to the baseline table."""
    out = table.copy()
    own = out["pct_vs_own_history"]
    peer = out["pct_vs_peers"]
    rising = own > 0  # NaN compares False, so weeks without history are never rising
    own_hit = own >= own_threshold
    peer_hit = peer >= peer_threshold

    out["is_flagged"] = rising & (own_hit | peer_hit)
    out["flag_basis"] = ""
    out.loc[out["is_flagged"] & own_hit & peer_hit, "flag_basis"] = "own history and similar routes"
    out.loc[out["is_flagged"] & own_hit & ~peer_hit, "flag_basis"] = "own history"
    out.loc[out["is_flagged"] & ~own_hit & peer_hit, "flag_basis"] = "similar routes"
    return out
