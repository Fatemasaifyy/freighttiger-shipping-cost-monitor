import pandas as pd

from preprocessing import clean_shipments


def make_raw(rows):
    """rows: (origin, destination, route_type, qty, dist, cost, date) tuples -> raw string frame."""
    records = [
        {
            "shipment_id": f"S{i:05d}",
            "origin": o,
            "destination": d,
            "route_type": rt,
            "material": "Steel Coils",
            "quantity_tonnes": str(q),
            "distance_km": str(km),
            "freight_cost_inr": str(c),
            "shipment_date": date,
            "transporter": "Test Co",
        }
        for i, (o, d, rt, q, km, c, date) in enumerate(rows, start=1)
    ]
    return pd.DataFrame(records)


def make_shipments(rows):
    return clean_shipments(make_raw(rows))


def weekly_frame(values):
    """values: (route, route_type, week_of, cost_per_tonne_km) tuples -> weekly table."""
    df = pd.DataFrame(values, columns=["route", "route_type", "week_of", "cost_per_tonne_km"])
    df["week_of"] = pd.to_datetime(df["week_of"])
    return df


def mondays(start, n):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, periods=n, freq="7D")]
