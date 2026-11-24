"""Distances, nearest customers and the day-route planner."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HQ = {"name": "HQ Stuttgart", "lat": 48.7758, "lon": 9.1829}

ROAD_FACTOR = 1.3      # straight line -> road distance
SPEED_KMH = 55         # mixed motorway and town traffic
VISIT_MIN = 45         # time on site per lead
DAY_START = 8 * 60     # 08:00


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    d = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 12742 * np.arcsin(np.sqrt(d))


def nearest_customer(leads: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    d = haversine_km(leads["lat"].to_numpy()[:, None], leads["lon"].to_numpy()[:, None],
                     customers["lat"].to_numpy()[None, :], customers["lon"].to_numpy()[None, :])
    idx = d.argmin(axis=1)
    return pd.DataFrame({
        "nearest_customer_id": customers["customer_id"].to_numpy()[idx],
        "cust_lat": customers["lat"].to_numpy()[idx],
        "cust_lon": customers["lon"].to_numpy()[idx],
        "nearest_customer_km_check": d.min(axis=1),
    }, index=leads.index)


def _matrix(lat, lon):
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) * ROAD_FACTOR


def _tour_len(order, dist):
    return sum(dist[a, b] for a, b in zip(order[:-1], order[1:]))


def _solve(dist: np.ndarray) -> list[int]:
    """Round trip from node 0: nearest neighbour, then 2-opt until no improvement."""
    n = len(dist)
    order, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: dist[order[-1], j])
        order.append(nxt)
        left.remove(nxt)
    order.append(0)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 2):
            for k in range(i + 1, len(order) - 1):
                a, b, c, d = order[i - 1], order[i], order[k], order[k + 1]
                if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d] - 1e-9:
                    order[i:k + 1] = order[i:k + 1][::-1]
                    improved = True
    return order


@dataclass
class Route:
    stops: pd.DataFrame          # ordered stops with arrival times
    path: list[list[float]]      # [lon, lat] incl. HQ at both ends
    km: float
    minutes: float
    dropped: int                 # candidates that did not fit into the day


def _fmt(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


def plan_day_route(candidates: pd.DataFrame, max_stops: int = 8, day_hours: float = 9.0) -> Route | None:
    """Best leads first; drop the weakest until the round trip fits into the working day."""
    pool = candidates.sort_values("score", ascending=False).head(max_stops)
    dropped = len(candidates) - len(pool)
    while len(pool):
        lat = np.r_[HQ["lat"], pool["lat"].to_numpy()]
        lon = np.r_[HQ["lon"], pool["lon"].to_numpy()]
        dist = _matrix(lat, lon)
        order = _solve(dist)
        km = _tour_len(order, dist)
        minutes = km / SPEED_KMH * 60 + VISIT_MIN * len(pool)
        if minutes <= day_hours * 60 or len(pool) == 1:
            break
        pool = pool.iloc[:-1]
        dropped += 1
    if not len(pool):
        return None

    stops = pool.iloc[[i - 1 for i in order[1:-1]]].copy()
    t, arrive, leg = DAY_START, [], []
    for a, b in zip(order[:-2], order[1:-1]):
        t += dist[a, b] / SPEED_KMH * 60
        arrive.append(_fmt(t))
        leg.append(dist[a, b])
        t += VISIT_MIN
    stops["stop"] = range(1, len(stops) + 1)
    stops["arrival"] = arrive
    stops["leg_km"] = np.round(leg, 1)
    path = [[lon[i], lat[i]] for i in order]
    return Route(stops=stops.reset_index(drop=True), path=path, km=float(km),
                 minutes=float(minutes), dropped=dropped)
