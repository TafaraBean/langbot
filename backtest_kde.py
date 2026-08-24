"""No-lookahead backtest: long setups targeting the price levels where a
causal, expanding volume-weighted KDE peaks (see volume_profile.py) - same
tap-entry / fixed-R:R mechanics as backtest_orderblocks.py (reused from
there), applied to KDE peak zones instead of order-block zones.

A peak level only becomes a tradeable candidate the first time it appears in
a snapshot's top-N peaks (not already within `min_sep_bw` bandwidths of a
previously-seen level) - otherwise a single stable, long-lived peak would
re-trigger a "new" trade at every checkpoint it persists through. From the
checkpoint it's first seen, it stays a valid target until tapped once (or
never, if price never returns) - the same one-shot-per-formation convention
`backtest_orderblocks.py` uses for order blocks.
"""

import numpy as np
import pandas as pd

import mt5_data
import volume_profile
from backtest_orderblocks import DEFAULT_R_MULTIPLE, Trade, first_tap_index, simulate_long, stop_buffer


def _new_peaks(snapshot: "volume_profile.ProfileSnapshot", seen_levels: list, min_sep_bw: float) -> list:
    min_sep = snapshot.bandwidth * min_sep_bw
    return [p for p in snapshot.peaks if all(abs(p - lv) >= min_sep for lv in seen_levels)]


def run_kde_backtest(symbol: str, timeframe: str = "M1", bars: int = 50_000, r_multiple: float = DEFAULT_R_MULTIPLE,
                      recompute_every: int = 50, warmup_bars: int = 200, n_peaks: int = 6,
                      zone_half_width_bw: float = volume_profile.DEFAULT_ZONE_HALF_WIDTH_BW,
                      min_sep_bw: float = volume_profile.DEFAULT_MIN_SEP_BW) -> dict:
    """Fetches `bars` candles, walks a causal expanding KDE forward over
    them, and backtests a long-only tap-entry / fixed-R:R exit on every
    newly-appearing peak level. Returns a dict with `trades`, `df`,
    `snapshots`, and summary stats (including `score` = total_r/max_drawdown,
    for ranking timeframes - see `sweep_timeframes`)."""
    ok, msg = mt5_data.connect(symbol)
    if not ok:
        raise RuntimeError(f"MT5 connect failed for {symbol!r}: {msg}")

    df = mt5_data.get_rates(symbol, timeframe, bars=bars)
    if df.empty:
        raise RuntimeError(f"No {timeframe} data returned for {symbol!r}.")

    snapshots = volume_profile.walk_forward_profiles(
        df, recompute_every=recompute_every, warmup_bars=warmup_bars, n_peaks=n_peaks, min_sep_bw=min_sep_bw)

    time = df["time"].to_numpy()
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)

    seen_levels: list = []
    trades: list[Trade] = []
    for snap in snapshots:
        fresh = _new_peaks(snap, seen_levels, min_sep_bw)
        seen_levels.extend(fresh)

        for peak_price in fresh:
            half = snap.bandwidth * zone_half_width_bw
            zone_low, zone_high = peak_price - half, peak_price + half
            # +1: same reasoning as backtest_orderblocks - this checkpoint's
            # bar has already elapsed by the time its own snapshot is known.
            tap_idx = first_tap_index(low, high, snap.bar_index + 1, zone_low, zone_high)
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
                source="kde_peak", zone_low=zone_low, zone_high=zone_high, confirmed_time=snap.bar_time,
                entry_time=time[tap_idx], entry_price=entry_price, stop_price=stop_price, target_price=target_price,
                outcome=sim["outcome"], exit_time=sim["exit_time"], exit_price=sim["exit_price"],
                r_multiple=sim["r_multiple"],
            ))

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
    total_r = sum(r_values) if r_values else 0.0
    score = (total_r / max_drawdown) if max_drawdown > 1e-9 else total_r

    stats = {
        "symbol": symbol, "timeframe": timeframe, "bars_fetched": len(df),
        "data_from": df["time"].iloc[0], "data_to": df["time"].iloc[-1],
        "snapshots": len(snapshots), "levels_seen": len(seen_levels),
        "total_trades": len(trades), "closed_trades": len(closed), "open_trades": len(trades) - len(closed),
        "wins": len(wins), "losses": len(closed) - len(wins),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "avg_r": (total_r / len(closed)) if closed else None,
        "total_r": total_r, "max_drawdown_r": max_drawdown, "score": score,
        "r_multiple": r_multiple,
    }
    return {"trades": trades, "df": df, "snapshots": snapshots, "stats": stats}


def sweep_timeframes(symbol: str, timeframes: "list[str] | None" = None, bars: int = 20_000,
                      r_multiple: float = DEFAULT_R_MULTIPLE, recompute_every: int = 50, warmup_bars: int = 200,
                      n_peaks: int = 6, progress_cb=None) -> list:
    """Runs `run_kde_backtest` across every given timeframe (default: all of
    mt5_data.TIMEFRAME_MAP), ranked by score (total_r/max_drawdown, falling
    back to total_r when there's no drawdown - same convention as
    structure.risk_adjusted_score). Timeframes MT5 can't fetch data for, or
    that don't have enough bars for one checkpoint, get an `error` entry
    instead of stats and always sort last. `progress_cb(i, n, timeframe)`,
    if given, is called before each run - lets a caller drive a progress
    bar."""
    tfs = timeframes or list(mt5_data.TIMEFRAME_MAP.keys())
    results = []
    for i, tf in enumerate(tfs):
        if progress_cb:
            progress_cb(i, len(tfs), tf)
        try:
            result = run_kde_backtest(symbol, timeframe=tf, bars=bars, r_multiple=r_multiple,
                                       recompute_every=recompute_every, warmup_bars=warmup_bars, n_peaks=n_peaks)
        except Exception as exc:
            results.append({"timeframe": tf, "error": str(exc)})
            continue
        results.append({"timeframe": tf, **result["stats"]})

    results.sort(key=lambda r: r.get("score", float("-inf")) if "error" not in r else float("-inf"), reverse=True)
    return results


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSDm"
    result = run_kde_backtest(symbol)
    stats = result["stats"]

    print(f"Symbol: {stats['symbol']}  Timeframe: {stats['timeframe']}")
    print(f"Data: {stats['bars_fetched']} bars, {stats['data_from']} -> {stats['data_to']}")
    print(f"Snapshots: {stats['snapshots']}  Levels seen: {stats['levels_seen']}")
    print(f"Trades: {stats['total_trades']} total, {stats['closed_trades']} closed, {stats['open_trades']} still open")
    if stats["closed_trades"]:
        print(f"Win rate: {stats['win_rate']*100:.1f}%  Avg R: {stats['avg_r']:.2f}  "
              f"Total R: {stats['total_r']:.2f}  Max drawdown: {stats['max_drawdown_r']:.2f}R  Score: {stats['score']:.2f}")
    else:
        print("No closed trades.")
