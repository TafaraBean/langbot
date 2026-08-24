"""Smart Money Concepts indicator, pandas/numpy only.

A faithful port of LuxAlgo's "Smart Money Concepts [LuxAlgo]" Pine Script v5
indicator: swing/internal market structure (BOS/CHoCH), order blocks, fair
value gaps, equal highs/lows, and premium/discount zones. Function names and
the per-bar step order intentionally track the original script so the two
can be compared line-by-line.

Computed once over a fixed set of historical bars (Pine's "Historical" mode)
- there is no live tick stream here to replicate "Present" mode's repainting
of the most recent, still-forming pivot.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import mt5_data

BULLISH_LEG = 1
BEARISH_LEG = 0
BULLISH = 1
BEARISH = -1

TIMEFRAME_MINUTES = {
    "M1": 1, "M3": 3, "M5": 5, "M15": 15, "H1": 60, "H4": 240,
    "D1": 1440, "W1": 10080, "MN1": 43200,
}


@dataclass
class Pivot:
    current_level: float = np.nan
    last_level: float = np.nan
    crossed: bool = False
    bar_time: "pd.Timestamp | None" = None
    bar_index: "int | None" = None


@dataclass
class TrailingExtremes:
    top: float = np.nan
    bottom: float = np.nan
    bar_time: "pd.Timestamp | None" = None
    bar_index: "int | None" = None
    last_top_time: "pd.Timestamp | None" = None
    last_bottom_time: "pd.Timestamp | None" = None


@dataclass
class StructureEvent:
    kind: str            # "BOS" or "CHoCH"
    direction: str        # "bullish" or "bearish"
    price: float
    pivot_time: pd.Timestamp
    break_time: pd.Timestamp
    internal: bool


@dataclass
class OrderBlock:
    high: float
    low: float
    bar_time: pd.Timestamp        # anchor candle's own time (for drawing the box)
    bias: int                     # BULLISH or BEARISH
    internal: bool
    confirmed_time: pd.Timestamp  # when the structure break that created this block occurred -
                                   # the earliest a strategy could actually know about it; always
                                   # >= bar_time, since the anchor candle precedes its own break
    mitigated_time: "pd.Timestamp | None" = None


@dataclass
class FairValueGap:
    # top/bottom are assigned exactly as the source script does, which for a
    # bearish gap means top < bottom numerically (it names the field holding
    # the lower price "top"). Rendering code should use min()/max() for the
    # box edges; mitigation must compare against these fields as-is, since
    # that asymmetry is also where the two gap directions get filled from.
    top: float
    bottom: float
    bias: int
    left_time: pd.Timestamp
    right_time: pd.Timestamp


@dataclass
class EqualLevel:
    kind: str              # "EQH" or "EQL"
    price: float
    from_time: pd.Timestamp
    to_time: pd.Timestamp


@dataclass
class MtfLevel:
    top: float
    bottom: float
    left_time: pd.Timestamp


@dataclass
class SMCResult:
    swing_structure: list = field(default_factory=list)
    internal_structure: list = field(default_factory=list)
    swing_order_blocks: list = field(default_factory=list)
    internal_order_blocks: list = field(default_factory=list)
    swing_order_blocks_history: list = field(default_factory=list)
    internal_order_blocks_history: list = field(default_factory=list)
    fair_value_gaps: list = field(default_factory=list)
    equal_highs: list = field(default_factory=list)
    equal_lows: list = field(default_factory=list)
    trailing: TrailingExtremes = field(default_factory=TrailingExtremes)
    swing_trend_bias: int = 0
    premium_zone: "tuple[float, float] | None" = None
    equilibrium_zone: "tuple[float, float] | None" = None
    discount_zone: "tuple[float, float] | None" = None
    zone_left_time: "pd.Timestamp | None" = None
    strong_weak_top_label: "str | None" = None
    strong_weak_bottom_label: "str | None" = None
    daily_levels: "MtfLevel | None" = None
    weekly_levels: "MtfLevel | None" = None
    monthly_levels: "MtfLevel | None" = None


def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=1).mean()


def compute_legs(high: pd.Series, low: pd.Series, size: int) -> np.ndarray:
    """Port of `leg(size)`: a persistent leg value that flips to BEARISH_LEG
    once the high `size` bars back exceeds the highest high of the last
    `size` bars (a confirmed swing high `size` bars back), or to BULLISH_LEG
    on the mirrored low condition."""
    highest = high.rolling(size).max().to_numpy()
    lowest = low.rolling(size).min().to_numpy()
    high_shifted = high.shift(size).to_numpy()
    low_shifted = low.shift(size).to_numpy()

    new_leg_high = high_shifted > highest
    new_leg_low = low_shifted < lowest

    legs = np.zeros(len(high), dtype=int)
    current = BEARISH_LEG
    for i in range(len(high)):
        if new_leg_high[i]:
            current = BEARISH_LEG
        elif new_leg_low[i]:
            current = BULLISH_LEG
        legs[i] = current
    return legs


def _advance_pivot(i, size, legs, high, low, time, high_pivot: Pivot, low_pivot: Pivot):
    """Port of the pivot-recording half of `getCurrentStructure()`. Returns
    ("high"|"low", pivot_index) when a new pivot is confirmed this bar, else
    None."""
    if i < size or legs[i] == legs[i - 1]:
        return None

    pivot_idx = i - size
    if legs[i] == BULLISH_LEG:
        low_pivot.last_level = low_pivot.current_level
        low_pivot.current_level = low[pivot_idx]
        low_pivot.crossed = False
        low_pivot.bar_time = time[pivot_idx]
        low_pivot.bar_index = pivot_idx
        return ("low", pivot_idx)

    high_pivot.last_level = high_pivot.current_level
    high_pivot.current_level = high[pivot_idx]
    high_pivot.crossed = False
    high_pivot.bar_time = time[pivot_idx]
    high_pivot.bar_index = pivot_idx
    return ("high", pivot_idx)


def _store_order_block(order_blocks: list, pivot: Pivot, break_i: int, parsed_high, parsed_low,
                        times, bias: int, internal: bool) -> None:
    """Port of `storeOrdeBlock`: the anchor candle is the most extreme parsed
    high (bearish break) or parsed low (bullish break) between the broken
    pivot's bar and the break bar.

    `order_blocks` is kept as a full, append-only history (mitigated blocks
    are tagged via `mitigated_time`, never removed - the source script's
    100-block cap was an internal Pine memory limit, not something that
    should also throw away history a backtest needs). Callers derive the
    "currently active, most recent N" view via `_active_order_blocks`."""
    start, end = pivot.bar_index, break_i
    if start is None or end <= start:
        return

    if bias == BEARISH:
        offset = int(np.argmax(parsed_high[start:end]))
    else:
        offset = int(np.argmin(parsed_low[start:end]))
    idx = start + offset

    order_blocks.insert(0, OrderBlock(high=float(parsed_high[idx]), low=float(parsed_low[idx]),
                                       bar_time=times[idx], bias=bias, internal=internal,
                                       confirmed_time=times[break_i]))


def _mitigate_order_blocks(order_blocks: list, close_i, high_i, low_i, mitigation_mode: str, time_i) -> None:
    """Port of `deleteOrderBlocks`, adapted to tag rather than remove (see
    `_store_order_block`)."""
    bearish_source = close_i if mitigation_mode == "Close" else high_i
    bullish_source = close_i if mitigation_mode == "Close" else low_i

    for ob in order_blocks:
        if ob.mitigated_time is not None:
            continue
        if (ob.bias == BEARISH and bearish_source > ob.high) or (ob.bias == BULLISH and bullish_source < ob.low):
            ob.mitigated_time = time_i


def _active_order_blocks(history: list, count: int) -> list:
    return [ob for ob in history if ob.mitigated_time is None][:count]


def _display_structure(i, o, h, l, c, time, pivot_high: Pivot, pivot_low: Pivot, trend_bias: int,
                        reference_high: Pivot, reference_low: Pivot, internal: bool,
                        confluence_filter: bool, events: list, order_blocks: "list | None",
                        parsed_high, parsed_low) -> int:
    """Port of `displayStructure()`: crossover/crossunder of close against
    the last unbroken pivot level, tagged BOS/CHoCH from the running trend,
    plus order block creation on internal/swing breaks."""
    bullish_bar = bearish_bar = True
    if confluence_filter:
        bullish_bar = (h[i] - max(c[i], o[i])) > min(c[i], o[i] - l[i])
        bearish_bar = (h[i] - max(c[i], o[i])) < min(c[i], o[i] - l[i])

    prev_close = c[i - 1] if i > 0 else np.nan

    if not pivot_high.crossed and not np.isnan(pivot_high.current_level):
        crossover = (not np.isnan(prev_close) and prev_close <= pivot_high.current_level
                     and c[i] > pivot_high.current_level)
        extra = (pivot_high.current_level != reference_high.current_level and bullish_bar) if internal else True
        if crossover and extra:
            kind = "CHoCH" if trend_bias == BEARISH else "BOS"
            pivot_high.crossed = True
            trend_bias = BULLISH
            events.append(StructureEvent(kind=kind, direction="bullish", price=pivot_high.current_level,
                                          pivot_time=pivot_high.bar_time, break_time=time[i], internal=internal))
            if order_blocks is not None:
                _store_order_block(order_blocks, pivot_high, i, parsed_high, parsed_low, time, BULLISH, internal)

    if not pivot_low.crossed and not np.isnan(pivot_low.current_level):
        crossunder = (not np.isnan(prev_close) and prev_close >= pivot_low.current_level
                      and c[i] < pivot_low.current_level)
        extra = (pivot_low.current_level != reference_low.current_level and bearish_bar) if internal else True
        if crossunder and extra:
            kind = "CHoCH" if trend_bias == BULLISH else "BOS"
            pivot_low.crossed = True
            trend_bias = BEARISH
            events.append(StructureEvent(kind=kind, direction="bearish", price=pivot_low.current_level,
                                          pivot_time=pivot_low.bar_time, break_time=time[i], internal=internal))
            if order_blocks is not None:
                _store_order_block(order_blocks, pivot_low, i, parsed_high, parsed_low, time, BEARISH, internal)

    return trend_bias


def _update_trailing_extremes(trailing: TrailingExtremes, high_i, low_i, time_i) -> None:
    """Port of `updateTrailingExtremes()`."""
    new_top = high_i if np.isnan(trailing.top) else max(trailing.top, high_i)
    if new_top == high_i:
        trailing.last_top_time = time_i
    trailing.top = new_top

    new_bottom = low_i if np.isnan(trailing.bottom) else min(trailing.bottom, low_i)
    if new_bottom == low_i:
        trailing.last_bottom_time = time_i
    trailing.bottom = new_bottom


def compute_smc(df: pd.DataFrame, config: dict) -> SMCResult:
    """Orchestrator mirroring the main execution block's per-bar call order.
    `config` gates each feature so disabled pieces cost nothing; see
    dashboard.py for the exact keys it assembles."""
    result = SMCResult()
    n = len(df)
    if n == 0:
        return result

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    time = df["time"].to_numpy()
    bar_index = np.arange(n)

    atr_measure = _atr(df, 200).to_numpy(float)
    tr = _true_range(df).to_numpy(float)
    cum_tr = np.nancumsum(tr)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_mean_range = cum_tr / np.maximum(bar_index, 1)
    volatility = atr_measure if config.get("order_block_filter", "Atr") == "Atr" else cum_mean_range

    high_vol_bar = (h - l) >= (2 * volatility)
    parsed_high = np.where(high_vol_bar, l, h)
    parsed_low = np.where(high_vol_bar, h, l)

    open_shift = np.roll(o, 1)
    close_shift = np.roll(c, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_delta_percent = (close_shift - open_shift) / (open_shift * 100)
    bar_delta_percent[0] = np.nan
    cum_abs_bdp = np.nancumsum(np.abs(bar_delta_percent))

    swing_size = config.get("swing_length", 50)
    internal_size = 5
    equal_size = config.get("equal_length", 3)
    equal_threshold = config.get("equal_threshold", 0.1)
    mitigation_mode = config.get("order_block_mitigation", "High/Low")

    show_structure = config.get("show_swing_structure", False)
    show_internal = config.get("show_internal_structure", False)
    show_swing_obs = config.get("show_swing_order_blocks", False)
    show_internal_obs = config.get("show_internal_order_blocks", False)
    show_equal = config.get("show_equal_highs_lows", False)
    show_fvg = config.get("show_fair_value_gaps", False)
    show_trailing = config.get("show_high_low_swings", False) or config.get("show_premium_discount_zones", False)

    # Swing and internal pivot tracking run unconditionally every bar in the
    # source script (not gated by any display toggle), because internal
    # structure's confluence check compares live against the swing pivots
    # even when swing structure itself isn't shown.
    swing_legs = compute_legs(df["high"], df["low"], swing_size)
    internal_legs = compute_legs(df["high"], df["low"], internal_size)
    equal_legs = compute_legs(df["high"], df["low"], equal_size) if show_equal else None

    swing_high, swing_low = Pivot(), Pivot()
    internal_high, internal_low = Pivot(), Pivot()
    equal_high, equal_low = Pivot(), Pivot()
    trailing = result.trailing

    swing_trend = internal_trend = 0
    internal_ob_count = config.get("internal_order_blocks_count", 5)
    swing_ob_count = config.get("swing_order_blocks_count", 5)
    fvg_auto_threshold = config.get("fvg_auto_threshold", True)

    for i in range(n):
        if show_trailing:
            _update_trailing_extremes(trailing, h[i], l[i], time[i])

        if show_fvg and i >= 2:
            result.fair_value_gaps[:] = [
                fvg for fvg in result.fair_value_gaps
                if not ((l[i] < fvg.bottom and fvg.bias == BULLISH) or (h[i] > fvg.top and fvg.bias == BEARISH))
            ]

        pivot_event = _advance_pivot(i, swing_size, swing_legs, h, l, time, swing_high, swing_low)
        if pivot_event is not None:
            direction, _ = pivot_event
            if direction == "low":
                trailing.bottom = swing_low.current_level
                trailing.bar_time = swing_low.bar_time
                trailing.bar_index = swing_low.bar_index
                trailing.last_bottom_time = swing_low.bar_time
            else:
                trailing.top = swing_high.current_level
                trailing.bar_time = swing_high.bar_time
                trailing.bar_index = swing_high.bar_index
                trailing.last_top_time = swing_high.bar_time

        _advance_pivot(i, internal_size, internal_legs, h, l, time, internal_high, internal_low)

        if equal_legs is not None and i >= equal_size and equal_legs[i] != equal_legs[i - 1]:
            pivot_idx = i - equal_size
            if equal_legs[i] == BULLISH_LEG:
                if not np.isnan(equal_low.current_level) and abs(equal_low.current_level - l[pivot_idx]) < equal_threshold * atr_measure[i]:
                    result.equal_lows.append(EqualLevel(kind="EQL", price=l[pivot_idx],
                                                          from_time=equal_low.bar_time, to_time=time[pivot_idx]))
                equal_low.last_level = equal_low.current_level
                equal_low.current_level = l[pivot_idx]
                equal_low.bar_time = time[pivot_idx]
                equal_low.bar_index = pivot_idx
            else:
                if not np.isnan(equal_high.current_level) and abs(equal_high.current_level - h[pivot_idx]) < equal_threshold * atr_measure[i]:
                    result.equal_highs.append(EqualLevel(kind="EQH", price=h[pivot_idx],
                                                           from_time=equal_high.bar_time, to_time=time[pivot_idx]))
                equal_high.last_level = equal_high.current_level
                equal_high.current_level = h[pivot_idx]
                equal_high.bar_time = time[pivot_idx]
                equal_high.bar_index = pivot_idx

        if show_internal or show_internal_obs:
            internal_trend = _display_structure(
                i, o, h, l, c, time, internal_high, internal_low, internal_trend,
                swing_high, swing_low, True, config.get("internal_confluence_filter", False),
                result.internal_structure, result.internal_order_blocks if show_internal_obs else None,
                parsed_high, parsed_low)

        if show_structure or show_swing_obs:
            swing_trend = _display_structure(
                i, o, h, l, c, time, swing_high, swing_low, swing_trend,
                swing_high, swing_low, False, False,
                result.swing_structure, result.swing_order_blocks if show_swing_obs else None,
                parsed_high, parsed_low)

        if show_internal_obs:
            _mitigate_order_blocks(result.internal_order_blocks, c[i], h[i], l[i], mitigation_mode, time[i])
        if show_swing_obs:
            _mitigate_order_blocks(result.swing_order_blocks, c[i], h[i], l[i], mitigation_mode, time[i])

        if show_fvg and i >= 2:
            threshold = (cum_abs_bdp[i] / max(i, 1) * 2) if fvg_auto_threshold else 0.0
            bdp = bar_delta_percent[i] if not np.isnan(bar_delta_percent[i]) else 0.0
            if l[i] > h[i - 2] and c[i - 1] > h[i - 2] and bdp > threshold:
                result.fair_value_gaps.append(FairValueGap(top=l[i], bottom=h[i - 2], bias=BULLISH,
                                                             left_time=time[i - 1], right_time=time[i]))
            elif h[i] < l[i - 2] and c[i - 1] < l[i - 2] and -bdp > threshold:
                result.fair_value_gaps.append(FairValueGap(top=h[i], bottom=l[i - 2], bias=BEARISH,
                                                             left_time=time[i - 1], right_time=time[i]))

    result.swing_trend_bias = swing_trend

    if show_internal_obs:
        result.internal_order_blocks_history = result.internal_order_blocks
        result.internal_order_blocks = _active_order_blocks(result.internal_order_blocks, internal_ob_count)
    if show_swing_obs:
        result.swing_order_blocks_history = result.swing_order_blocks
        result.swing_order_blocks = _active_order_blocks(result.swing_order_blocks, swing_ob_count)

    if config.get("show_premium_discount_zones") and not np.isnan(trailing.top) and not np.isnan(trailing.bottom):
        result.premium_zone = (trailing.top, 0.95 * trailing.top + 0.05 * trailing.bottom)
        result.equilibrium_zone = (0.525 * trailing.top + 0.475 * trailing.bottom,
                                    0.525 * trailing.bottom + 0.475 * trailing.top)
        result.discount_zone = (0.95 * trailing.bottom + 0.05 * trailing.top, trailing.bottom)
        result.zone_left_time = trailing.bar_time

    if config.get("show_high_low_swings") and not np.isnan(trailing.top) and not np.isnan(trailing.bottom):
        result.strong_weak_top_label = "Strong High" if swing_trend == BEARISH else "Weak High"
        result.strong_weak_bottom_label = "Strong Low" if swing_trend == BULLISH else "Weak Low"

    return result


def mtf_levels(symbol: str, period_key: str) -> "MtfLevel | None":
    """Prior completed period's high/low for a Daily/Weekly/Monthly period,
    replacing Pine's `request.security(..., [high[1], low[1], time[1]])`
    with an explicit higher-timeframe fetch: the most recent 3 bars from MT5
    include the still-forming current period at the end, so the previous
    (fully closed) period is the second-to-last row."""
    rates = mt5_data.get_rates(symbol, period_key, bars=3)
    if len(rates) < 2:
        return None
    prior = rates.iloc[-2]
    return MtfLevel(top=float(prior["high"]), bottom=float(prior["low"]), left_time=prior["time"])


def higher_timeframe(chart_timeframe: str, period_key: str) -> bool:
    """True if the chart's timeframe is already coarser than `period_key`
    (matching Pine's `higherTimeframe()` guard that hides e.g. Daily levels
    on a Weekly chart)."""
    return TIMEFRAME_MINUTES[chart_timeframe] > TIMEFRAME_MINUTES[period_key]
