"""No-lookahead backtest: long setups off bullish order blocks only, sourced
exclusively from smc.py's internal (structure-break-anchored) order blocks
and order_blocks.py's volume-pivot-anchored detector - no other POI types,
no confirmation-timeframe split, single timeframe throughout.

Entry: immediate tap. Each order block is only tradeable from its
`confirmed_time` onward (the bar it actually became knowable - see the
`confirmed_time` field on smc.OrderBlock / order_blocks.VolumeOrderBlock,
which is later than the anchor candle's own time). A resting long limit
order at the zone's top edge fills on the first bar at or after that point
whose [low, high] range overlaps the zone; the fill price is fixed at
formation time, so waiting to see whether/when it gets tapped is not
lookahead.

Stop/target: fixed R:R off the zone edge - stop just below the block's low
(a small buffer, same convention as structure.compute_stop), target a fixed
multiple of that risk above entry.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import mt5_data
import order_blocks
import smc

DEFAULT_R_MULTIPLE = 2.0
STOP_BUFFER_PCT = 0.1
STOP_BUFFER_FLOOR_PCT = 0.02


@dataclass
class Trade:
    source: str                    # "internal_ob" or "detected_ob"
    zone_low: float
    zone_high: float
    confirmed_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    outcome: str                   # "Win", "Loss", "Open" (still running past the data's end)
    exit_time: "pd.Timestamp | None"
    exit_price: "float | None"
    r_multiple: float


def stop_buffer(zone_low: float, zone_high: float) -> float:
    height = max(zone_high - zone_low, 1e-9)
    return max(height * STOP_BUFFER_PCT, height * STOP_BUFFER_FLOOR_PCT)


def first_tap_index(low: np.ndarray, high: np.ndarray, start_idx: int, zone_low: float, zone_high: float) -> "int | None":
    """First bar index at/after `start_idx` whose [low, high] range overlaps
    the zone - the earliest a resting limit buy there could have filled."""
    for i in range(start_idx, len(low)):
        if low[i] <= zone_high and high[i] >= zone_low:
            return i
    return None


def simulate_long(high: np.ndarray, low: np.ndarray, time: np.ndarray, entry_idx: int, entry_price: float,
                   stop_price: float, target_price: float) -> dict:
    """Forward walk from entry_idx. Stop-first if a single bar's range
    contains both stop and target - the standard conservative convention
    already used by structure.simulate_trade."""
    risk = entry_price - stop_price
    for j in range(entry_idx, len(high)):
        if low[j] <= stop_price:
            return {"outcome": "Loss", "exit_time": time[j], "exit_price": stop_price, "r_multiple": -1.0}
        if high[j] >= target_price:
            reward = target_price - entry_price
            return {"outcome": "Win", "exit_time": time[j], "exit_price": target_price,
                    "r_multiple": reward / risk if risk else 0.0}
    return {"outcome": "Open", "exit_time": None, "exit_price": None, "r_multiple": 0.0}


def _backtest_blocks(df: pd.DataFrame, blocks: list, zone_of, source: str, r_multiple: float) -> list[Trade]:
    time = df["time"].to_numpy()
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    trades = []

    for ob in blocks:
        zone_low, zone_high = zone_of(ob)
        # +1: the confirming bar has already fully elapsed by the time its own
        # close makes the block knowable, so a resting order couldn't
        # realistically have been live until the bar after it.
        # (`time` holds tz-aware pd.Timestamp objects, same as confirmed_time -
        # no datetime64 conversion needed, and wrapping in one would drop the
        # tz and break the comparison.)
        confirmed_idx = int(np.searchsorted(time, ob.confirmed_time))
        tap_idx = first_tap_index(low, high, confirmed_idx + 1, zone_low, zone_high)
        if tap_idx is None:
            continue

        entry_price = zone_high
        buffer = stop_buffer(zone_low, zone_high)
        stop_price = zone_low - buffer
        risk = entry_price - stop_price
        if risk <= 0:
            continue
        target_price = entry_price + r_multiple * risk

        sim = simulate_long(high, low, time, tap_idx, entry_price, stop_price, target_price)
        trades.append(Trade(
            source=source, zone_low=zone_low, zone_high=zone_high, confirmed_time=ob.confirmed_time,
            entry_time=time[tap_idx], entry_price=entry_price, stop_price=stop_price, target_price=target_price,
            outcome=sim["outcome"], exit_time=sim["exit_time"], exit_price=sim["exit_price"],
            r_multiple=sim["r_multiple"],
        ))

    return trades


def run_backtest(symbol: str, timeframe: str = "M1", bars: int = 50_000, r_multiple: float = DEFAULT_R_MULTIPLE,
                  internal_swing_length: int = 50) -> dict:
    """Fetches `bars` candles (MT5 returns however much history it actually
    retains), detects every bullish internal SMC order block and every
    bullish volume-detected order block across the full series, and
    backtests a long-only tap-entry / fixed-R:R exit on each. Returns a
    dict with `trades` (list[Trade]), `df` (the fetched data), and summary
    stats."""
    ok, msg = mt5_data.connect(symbol)
    if not ok:
        raise RuntimeError(f"MT5 connect failed for {symbol!r}: {msg}")

    df = mt5_data.get_rates(symbol, timeframe, bars=bars)
    if df.empty:
        raise RuntimeError(f"No {timeframe} data returned for {symbol!r}.")

    smc_config = {
        "swing_length": internal_swing_length,
        "show_internal_order_blocks": True,
        "internal_order_blocks_count": 10**9,   # keep the whole active view; history is unbounded regardless
        "order_block_filter": "Atr",
        "order_block_mitigation": "High/Low",
    }
    smc_result = smc.compute_smc(df, smc_config)
    internal_bullish = [ob for ob in smc_result.internal_order_blocks_history if ob.bias == smc.BULLISH]

    vob_config = {"length": 5, "mitigation": "Wick", "bull_ext_last": 10**9, "bear_ext_last": 10**9}
    vob_result = order_blocks.compute_order_blocks(df, vob_config)
    detected_bullish = vob_result.bullish_history

    trades = []
    trades += _backtest_blocks(df, internal_bullish, lambda ob: (ob.low, ob.high), "internal_ob", r_multiple)
    trades += _backtest_blocks(df, detected_bullish, lambda ob: (ob.bottom, ob.top), "detected_ob", r_multiple)
    trades.sort(key=lambda t: t.entry_time)

    closed = [t for t in trades if t.outcome in ("Win", "Loss")]
    wins = [t for t in closed if t.outcome == "Win"]
    r_values = [t.r_multiple for t in closed]

    equity = np.cumsum(r_values) if r_values else np.array([])
    if len(equity):
        running_peak = np.maximum.accumulate(equity)
        max_drawdown = float(np.max(running_peak - equity))
    else:
        max_drawdown = 0.0

    stats = {
        "symbol": symbol, "timeframe": timeframe, "bars_fetched": len(df),
        "data_from": df["time"].iloc[0], "data_to": df["time"].iloc[-1],
        "total_trades": len(trades), "closed_trades": len(closed), "open_trades": len(trades) - len(closed),
        "wins": len(wins), "losses": len(closed) - len(wins),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "avg_r": (sum(r_values) / len(r_values)) if r_values else None,
        "total_r": sum(r_values) if r_values else 0.0,
        "max_drawdown_r": max_drawdown,
        "r_multiple": r_multiple,
        "internal_ob_count": len(internal_bullish), "detected_ob_count": len(detected_bullish),
    }
    return {"trades": trades, "df": df, "stats": stats}


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Source": "Internal OB" if t.source == "internal_ob" else "Detected OB",
        "Confirmed": t.confirmed_time, "Entry time": t.entry_time,
        "Zone": f"{t.zone_low:.2f}-{t.zone_high:.2f}",
        "Entry": round(t.entry_price, 2), "Stop": round(t.stop_price, 2), "Target": round(t.target_price, 2),
        "Outcome": t.outcome, "Exit time": t.exit_time,
        "R multiple": round(t.r_multiple, 2),
    } for t in trades])


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSDm"
    result = run_backtest(symbol)
    stats = result["stats"]

    print(f"Symbol: {stats['symbol']}  Timeframe: {stats['timeframe']}")
    print(f"Data: {stats['bars_fetched']} bars, {stats['data_from']} -> {stats['data_to']}")
    print(f"Internal bullish OBs found: {stats['internal_ob_count']}  "
          f"Detected bullish OBs found: {stats['detected_ob_count']}")
    print(f"Trades: {stats['total_trades']} total, {stats['closed_trades']} closed, {stats['open_trades']} still open")
    if stats["closed_trades"]:
        print(f"Win rate: {stats['win_rate']*100:.1f}%  Avg R: {stats['avg_r']:.2f}  "
              f"Total R: {stats['total_r']:.2f}  Max drawdown: {stats['max_drawdown_r']:.2f}R")
    else:
        print("No closed trades.")

    frame = trades_to_frame(result["trades"])
    if not frame.empty:
        print()
        print(frame.to_string(index=False))
