from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightConfig:
    old: float
    mid: float
    l5: float

    @classmethod
    def stable(cls):
        return cls(0.55, 0.20, 0.25)

    @classmethod
    def role_change(cls):
        return cls(0.35, 0.20, 0.45)


def split_non_overlapping(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Old season, games 6-10 from the end, last 5. Never overlapping."""
    x = df.sort_values("GAME_DATE").reset_index(drop=True)
    n = len(x)
    if n <= 5:
        return {"old": x.iloc[0:0], "mid": x.iloc[0:0], "l5": x.copy()}
    if n <= 10:
        return {"old": x.iloc[0:0], "mid": x.iloc[:-5].copy(), "l5": x.iloc[-5:].copy()}
    return {
        "old": x.iloc[:-10].copy(),
        "mid": x.iloc[-10:-5].copy(),
        "l5": x.iloc[-5:].copy(),
    }


def active_weights(buckets: Dict[str, pd.DataFrame], cfg: WeightConfig):
    raw = {"old": cfg.old, "mid": cfg.mid, "l5": cfg.l5}
    active = {k: w for k, w in raw.items() if not buckets[k].empty}
    total = sum(active.values())
    return {k: (active.get(k, 0.0) / total if total else 0.0) for k in raw}


def weighted_average_feature(features: dict, weights: dict, key: str, default=np.nan):
    vals, ws = [], []
    for bucket in ("old", "mid", "l5"):
        value = features.get(bucket, {}).get(key, np.nan)
        if np.isfinite(value) and weights.get(bucket, 0) > 0:
            vals.append(value)
            ws.append(weights[bucket])
    return float(np.average(vals, weights=ws)) if vals else default
