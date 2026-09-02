from __future__ import annotations

"""Exposure normalization helpers.

NBA regulation is 48 minutes. Overtime adds exposure, which is useful for
within-game rate estimation but should not make a historical game look like a
faster 48-minute game or a larger normal-rotation minutes game.

The helpers below therefore expose a *regulation-equivalent* factor for
count/exposure quantities. Rates such as 3PA/FGA, TOV/POSS, OREB/miss and
DREB/chance should continue to be estimated from their natural event
opportunities and are not mechanically scaled by this module.
"""

import numpy as np
import pandas as pd

REGULATION_MINUTES = 48.0
OVERTIME_MINUTES = 5.0


def game_length_minutes(df: pd.DataFrame) -> pd.Series:
    """Best available scheduled game length for each row.

    Priority:
      1) explicit GAME_LENGTH_MIN,
      2) OT_COUNT,
      3) OT_FLAG (one-OT fallback),
      4) regulation 48.
    """
    idx = df.index
    length = pd.Series(REGULATION_MINUTES, index=idx, dtype=float)

    if "GAME_LENGTH_MIN" in df.columns:
        explicit = pd.to_numeric(df["GAME_LENGTH_MIN"], errors="coerce")
        good = explicit >= REGULATION_MINUTES
        length.loc[good] = explicit.loc[good]

    if "OT_COUNT" in df.columns:
        otc = pd.to_numeric(df["OT_COUNT"], errors="coerce").fillna(0.0).clip(lower=0.0)
        inferred = REGULATION_MINUTES + OVERTIME_MINUTES * otc
        # Prefer the longer coherent exposure if explicit metadata and OT_COUNT
        # disagree; this avoids silently treating a flagged OT as regulation.
        length = pd.concat([length, inferred.astype(float)], axis=1).max(axis=1)

    if "OT_FLAG" in df.columns:
        raw = df["OT_FLAG"]
        if raw.dtype == bool:
            flag = raw.fillna(False)
        else:
            flag = raw.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "ot"})
        # Only a fallback. Never overwrite an explicit/multi-OT length.
        length.loc[flag & (length <= REGULATION_MINUTES + 1e-9)] = REGULATION_MINUTES + OVERTIME_MINUTES

    return length.clip(lower=REGULATION_MINUTES)


def regulation_equivalent_factor(df: pd.DataFrame) -> pd.Series:
    """48 / actual game length, bounded to (0, 1]."""
    length = game_length_minutes(df)
    return (REGULATION_MINUTES / length).clip(lower=0.50, upper=1.0)


def regulation_equivalent_values(df: pd.DataFrame, values) -> pd.Series:
    """Scale a row-level count/exposure quantity to a 48-minute equivalent."""
    v = pd.to_numeric(pd.Series(values, index=df.index), errors="coerce")
    return v * regulation_equivalent_factor(df)


def regulation_equivalent_minutes_row(row) -> float:
    """Normalize one player's historical minutes to a 48-minute game."""
    try:
        m = float(pd.to_numeric(pd.Series([row.get("MIN", np.nan)]), errors="coerce").iloc[0])
    except Exception:
        return np.nan
    if not np.isfinite(m):
        return np.nan

    # Construct a one-row frame so the precedence rules stay identical.
    payload = {}
    for c in ("GAME_LENGTH_MIN", "OT_COUNT", "OT_FLAG"):
        try:
            payload[c] = [row.get(c, np.nan)]
        except Exception:
            pass
    factor = float(regulation_equivalent_factor(pd.DataFrame(payload or {"OT_FLAG": [False]})).iloc[0])
    return m * factor
