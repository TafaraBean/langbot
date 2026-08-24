"""Order Block Detector, pandas/numpy only.

A faithful port of LuxAlgo's "Order Block Detector [LuxAlgo]" Pine Script v5
indicator: order blocks anchored at volume pivots (a local peak in traded
volume over a symmetric left/right window), filtered by the prevailing
price-swing direction, and boxed as half a candle (wick to midpoint) rather
than the full candle. Distinct from smc.py's structure-break-anchored order
blocks - a separate LuxAlgo indicator with its own algorithm.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smc import BEARISH_LEG, BULLISH_LEG, compute_legs


@dataclass
class VolumeOrderBlock:
    top: float
    bottom: float
    avg: float
    left_time: pd.Timestamp        # anchor candle's own time (for drawing the box)
    confirmed_time: pd.Timestamp   # when the volume pivot actually confirmed (length bars after
                                    # left_time - it takes that long to know no later bar had more
                                    # volume) - the earliest a strategy could actually know about it
    mitigated_time: "pd.Timestamp | None" = None


@dataclass
class VolumeOBResult:
    bullish: list = field(default_factory=list)   # newest first, active only (capped)
    bearish: list = field(default_factory=list)
    bullish_history: list = field(default_factory=list)   # every block ever formed, newest first
    bearish_history: list = field(default_factory=list)


def _pivot_high(series: np.ndarray, left: int, right: int) -> np.ndarray:
    """Port of `ta.pivothigh(series, left, right)`: True at index `p` iff
    series[p] is strictly greater than every value in the `left` bars
    before it and the `right` bars after it (checked as two separate
    windows, so a tie on either side disqualifies the pivot)."""
    n = len(series)
    is_pivot = np.zeros(n, dtype=bool)
    for p in range(left, n - right):
        if series[p] > series[p - left:p].max() and series[p] > series[p + 1:p + right + 1].max():
            is_pivot[p] = True
    return is_pivot


def compute_order_blocks(df: pd.DataFrame, config: dict) -> VolumeOBResult:
    result = VolumeOBResult()
    length = config.get("length", 5)
    n = len(df)
    if n <= 2 * length:
        return result

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    hl2 = (high + low) / 2
    time = df["time"].to_numpy()
    volume = df["tick_volume"].to_numpy(float)

    legs = compute_legs(df["high"], df["low"], length)
    is_vol_pivot = _pivot_high(volume, length, length)

    if config.get("mitigation", "Wick") == "Close":
        target_bull_series = df["close"].rolling(length).min().to_numpy(float)
        target_bear_series = df["close"].rolling(length).max().to_numpy(float)
    else:
        target_bull_series = df["low"].rolling(length).min().to_numpy(float)
        target_bear_series = df["high"].rolling(length).max().to_numpy(float)

    bull_ext = config.get("bull_ext_last", 3)
    bear_ext = config.get("bear_ext_last", 3)

    for i in range(length, n):
        p = i - length
        if is_vol_pivot[p]:
            if legs[i] == BULLISH_LEG:
                result.bullish.insert(0, VolumeOrderBlock(top=hl2[p], bottom=low[p],
                                                            avg=(hl2[p] + low[p]) / 2, left_time=time[p],
                                                            confirmed_time=time[i]))
            elif legs[i] == BEARISH_LEG:
                result.bearish.insert(0, VolumeOrderBlock(top=high[p], bottom=hl2[p],
                                                            avg=(high[p] + hl2[p]) / 2, left_time=time[p],
                                                            confirmed_time=time[i]))

        target_bull, target_bear = target_bull_series[i], target_bear_series[i]
        if not np.isnan(target_bull):
            for ob in result.bullish:
                if ob.mitigated_time is None and target_bull < ob.bottom:
                    ob.mitigated_time = time[i]
        if not np.isnan(target_bear):
            for ob in result.bearish:
                if ob.mitigated_time is None and target_bear > ob.top:
                    ob.mitigated_time = time[i]

    result.bullish_history = result.bullish
    result.bearish_history = result.bearish
    result.bullish = [ob for ob in result.bullish if ob.mitigated_time is None][:bull_ext]
    result.bearish = [ob for ob in result.bearish if ob.mitigated_time is None][:bear_ext]
    return result
