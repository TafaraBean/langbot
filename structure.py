"""Heuristic SMC-style structure detection: swings, bias, BOS/CHoCH, order
blocks, FVG/IFVG, engulfing candles.

These are pattern-matching approximations of discretionary concepts, meant
as a starting suggestion for the trader to confirm visually - not ground
truth.
"""

import pandas as pd

TF_ORDER = ["D1", "H4", "H1", "M15"]
TF_MINUTES = {"D1": 1440, "H4": 240, "H1": 60, "M15": 15, "M5": 5, "M3": 3, "M1": 1}


def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> list[dict]:
    swings = []
    n = len(df)
    for i in range(left, n - right):
        window_high = df["high"].iloc[i - left : i + right + 1]
        window_low = df["low"].iloc[i - left : i + right + 1]
        if df["high"].iloc[i] == window_high.max() and df["high"].iloc[i] > window_high.drop(window_high.index[left]).max():
            swings.append({"idx": i, "time": df["time"].iloc[i], "price": df["high"].iloc[i], "type": "high"})
        if df["low"].iloc[i] == window_low.min() and df["low"].iloc[i] < window_low.drop(window_low.index[left]).min():
            swings.append({"idx": i, "time": df["time"].iloc[i], "price": df["low"].iloc[i], "type": "low"})
    return sorted(swings, key=lambda s: s["idx"])


def classify_bias(swings: list[dict]) -> str:
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "Unclear"

    higher_high = highs[-1]["price"] > highs[-2]["price"]
    higher_low = lows[-1]["price"] > lows[-2]["price"]
    lower_high = highs[-1]["price"] < highs[-2]["price"]
    lower_low = lows[-1]["price"] < lows[-2]["price"]

    if higher_high and higher_low:
        return "Bullish"
    if lower_high and lower_low:
        return "Bearish"
    return "Unclear"


def swings_with_confirm_times(df: pd.DataFrame, swings: list[dict], right: int = 2) -> list[dict]:
    """Attaches a `_confirm_time` to each swing - the time its confirming
    bar (idx + right) occurred. Precompute this once per (df, swings) pair
    and reuse across many `bias_as_of` calls, so each query is a plain
    Timestamp comparison instead of a `df.iloc` lookup."""
    n = len(df)
    times = df["time"]
    return [{**s, "_confirm_time": times.iloc[min(s["idx"] + right, n - 1)]} for s in swings]


def bias_as_of(df: pd.DataFrame, as_of, swings: list[dict] | None = None) -> str:
    """Point-in-time bias: classify_bias using only what was knowable by
    `as_of` - bars up to that time, and swings only once their confirming
    bar has occurred.

    Pass `swings_with_confirm_times(df, find_swings(df))` (computed once)
    when calling this many times against the same df, e.g. once per POI in
    a batch backtest - filtering already-known swings by confirmation time
    is O(len(swings)) with no df access, whereas re-running find_swings from
    scratch per call (the default when `swings` is omitted) is O(len(df))
    each time and gets very slow across hundreds/thousands of POIs. Both
    give the same result."""
    if swings is None:
        sub = df[df["time"] <= as_of]
        if len(sub) < 5:
            return "Unclear"
        return classify_bias(find_swings(sub))

    known = [s for s in swings if s["_confirm_time"] <= as_of]
    return classify_bias(known)


def reference_bias(biases: dict[str, str]) -> str:
    reference = biases.get("D1", "Unclear")
    if reference == "Unclear":
        counts = {"Bullish": 0, "Bearish": 0}
        for b in biases.values():
            if b in counts:
                counts[b] += 1
        reference = max(counts, key=counts.get) if max(counts.values()) > 0 else "Unclear"
    return reference


def narrative_score(biases: dict[str, str]) -> tuple[int, str, str]:
    reference = reference_bias(biases)

    if reference == "Unclear":
        return 0, "Ranging", "\"Ping pong\" trading or no trade"

    aligned = sum(1 for tf in TF_ORDER if biases.get(tf) == reference)

    if aligned == 4:
        return aligned, "Very Strong", "Trade strictly with the trend"
    if aligned == 3:
        return aligned, "Strong", "Use stronger confirmations/POIs"
    return aligned, "Ranging", "\"Ping pong\" trading or no trade"


def structure_breaks(df: pd.DataFrame, swings: list[dict], right: int = 2) -> list[dict]:
    """Chronological BOS/CHoCH events. BOS = break with the prevailing trend
    (continuation), CHoCH = break against it (first reversal signal)."""
    highs = sorted([s for s in swings if s["type"] == "high"], key=lambda s: s["idx"])
    lows = sorted([s for s in swings if s["type"] == "low"], key=lambda s: s["idx"])

    events = []
    trend = None
    broken_idxs = set()

    for i in range(len(df)):
        close = df["close"].iloc[i]

        candidate_highs = [h for h in highs if h["idx"] + right <= i and h["idx"] not in broken_idxs]
        if candidate_highs and close > candidate_highs[-1]["price"]:
            level = candidate_highs[-1]
            kind = "CHoCH" if trend == "bearish" else "BOS"
            events.append({"idx": i, "time": df["time"].iloc[i], "direction": "bullish", "level": level["price"], "kind": kind})
            trend = "bullish"
            broken_idxs.add(level["idx"])

        candidate_lows = [l for l in lows if l["idx"] + right <= i and l["idx"] not in broken_idxs]
        if candidate_lows and close < candidate_lows[-1]["price"]:
            level = candidate_lows[-1]
            kind = "CHoCH" if trend == "bullish" else "BOS"
            events.append({"idx": i, "time": df["time"].iloc[i], "direction": "bearish", "level": level["price"], "kind": kind})
            trend = "bearish"
            broken_idxs.add(level["idx"])

    return events


def detect_order_blocks(df: pd.DataFrame, breaks: list[dict], lookback: int = 15) -> list[dict]:
    """Last opposite-colour candle before each BOS/CHoCH impulse leg."""
    obs = []
    for ev in breaks:
        i = ev["idx"]
        start = max(0, i - lookback)
        if ev["direction"] == "bullish":
            for j in range(i - 1, start - 1, -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    obs.append({"idx": j, "time": df["time"].iloc[j], "direction": "bullish", "high": df["high"].iloc[j], "low": df["low"].iloc[j], "break_idx": i, "break_kind": ev["kind"]})
                    break
        else:
            for j in range(i - 1, start - 1, -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    obs.append({"idx": j, "time": df["time"].iloc[j], "direction": "bearish", "high": df["high"].iloc[j], "low": df["low"].iloc[j], "break_idx": i, "break_kind": ev["kind"]})
                    break
    return obs


def detect_fvg(df: pd.DataFrame) -> list[dict]:
    fvgs = []
    for i in range(2, len(df)):
        c0, c2 = i - 2, i
        if df["low"].iloc[c2] > df["high"].iloc[c0]:
            fvgs.append({"idx": i, "time": df["time"].iloc[i], "direction": "bullish", "top": df["low"].iloc[c2], "bottom": df["high"].iloc[c0]})
        elif df["high"].iloc[c2] < df["low"].iloc[c0]:
            fvgs.append({"idx": i, "time": df["time"].iloc[i], "direction": "bearish", "top": df["low"].iloc[c0], "bottom": df["high"].iloc[c2]})
    return fvgs


def detect_ifvg(df: pd.DataFrame, fvgs: list[dict]) -> list[dict]:
    ifvgs = []
    for fvg in fvgs:
        for j in range(fvg["idx"] + 1, len(df)):
            close = df["close"].iloc[j]
            if fvg["direction"] == "bullish" and close < fvg["bottom"]:
                ifvgs.append({"idx": j, "time": df["time"].iloc[j], "direction": "bearish", "origin_idx": fvg["idx"], "top": fvg["top"], "bottom": fvg["bottom"]})
                break
            if fvg["direction"] == "bearish" and close > fvg["top"]:
                ifvgs.append({"idx": j, "time": df["time"].iloc[j], "direction": "bullish", "origin_idx": fvg["idx"], "top": fvg["top"], "bottom": fvg["bottom"]})
                break
    return ifvgs


def detect_engulfing(df: pd.DataFrame) -> list[dict]:
    events = []
    for i in range(1, len(df)):
        o1, h1, l1, c1 = df[["open", "high", "low", "close"]].iloc[i - 1]
        o2, h2, l2, c2 = df[["open", "high", "low", "close"]].iloc[i]
        engulfs_range = h2 >= h1 and l2 <= l1
        if engulfs_range and c2 > o2 and c1 < o1:
            events.append({"idx": i, "time": df["time"].iloc[i], "direction": "bullish"})
        elif engulfs_range and c2 < o2 and c1 > o1:
            events.append({"idx": i, "time": df["time"].iloc[i], "direction": "bearish"})
    return events


def most_recent(events: list[dict], n: int = 3) -> list[dict]:
    return sorted(events, key=lambda e: e["idx"], reverse=True)[:n]


def filter_events_in_zone(df: pd.DataFrame, events: list[dict], zone_low: float, zone_high: float) -> list[dict]:
    return [e for e in events if df["low"].iloc[e["idx"]] <= zone_high and df["high"].iloc[e["idx"]] >= zone_low]


FX_CODES = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "XAU", "XAG"}


def pip_size(symbol: str) -> float | None:
    """Pip size for FX pairs and metals (XAU/XAG), matched on the symbol's
    first 6 characters so broker suffixes (XAUUSDm, EURUSD.a, ...) still
    resolve. Returns None for anything else (indices, crypto, equities) -
    callers use that to fall back to the R:R target model instead of the
    Callisto pip ladder, which is FX/metal-specific."""
    s = symbol.upper()
    if len(s) < 6:
        return None
    base, quote = s[0:3], s[3:6]
    if base not in FX_CODES or quote not in FX_CODES:
        return None
    if base == "XAU":
        return 0.1
    if base == "XAG":
        return 0.01
    if quote == "JPY":
        return 0.01
    return 0.0001


def detect_inducement(poi: dict, swings: list[dict]) -> bool:
    """True if a shallower swing formed after the POI, resting on the near
    side of its zone (not yet swept) - liquidity the market tends to grab
    before actually tapping the POI itself."""
    origin_idx = poi["idx"]
    zone_low, zone_high = poi["zone_low"], poi["zone_high"]
    kind = "low" if poi["direction"] == "bullish" else "high"
    later = [s for s in swings if s["type"] == kind and s["idx"] > origin_idx]
    if not later:
        return False
    if poi["direction"] == "bullish":
        return any(s["price"] > zone_high for s in later)
    return any(s["price"] < zone_low for s in later)


def evaluate_hp_criteria(poi: dict, poi_type: str, swings: list[dict],
                          fvgs: list[dict], desired_direction: str | None) -> dict:
    """The 4-criteria high-probability-orderblock check: liquidity sweep,
    FVG/displacement in the impulse leg, higher-timeframe confluence, and
    inducement. `poi` must already be tagged via `tag_poi` (needs zone_low/
    zone_high/liquidity_swept)."""
    sweep = bool(poi.get("liquidity_swept"))

    if poi_type == "FVG":
        fvg_present = True
    else:
        break_idx = poi.get("break_idx", poi["idx"])
        fvg_present = any(
            f["direction"] == poi["direction"] and poi["idx"] < f["idx"] <= break_idx
            for f in fvgs
        )

    htf_confluence = desired_direction is not None and poi["direction"] == desired_direction
    inducement = detect_inducement(poi, swings)

    return {
        "sweep": sweep, "fvg": fvg_present, "htf": htf_confluence, "inducement": inducement,
        "score": sum([sweep, fvg_present, htf_confluence, inducement]),
    }


def attach_hp_scores(obs: list[dict], fvgs: list[dict], swings: list[dict],
                      desired_direction: str | None) -> None:
    """Mutates every OB/FVG dict in place, adding an `hp` key with the
    high-probability-orderblock criteria breakdown and score (0-4)."""
    for ob in obs:
        ob["hp"] = evaluate_hp_criteria(ob, "Order Block", swings, fvgs, desired_direction)
    for f in fvgs:
        f["hp"] = evaluate_hp_criteria(f, "FVG", swings, fvgs, desired_direction)


def earliest_confirmation(chochs: list[dict], ifvgs: list[dict], engulf: list[dict]) -> dict | None:
    """First chronological confirmation across all three kinds - CHoCH > IFVG > Engulfing
    breaks ties on the same bar. Used to pick the real first-touch trade for backtesting,
    as opposed to `most_recent`'s "what's the current green light" use."""
    tagged = [(e["idx"], 0, "CHoCH", e) for e in chochs]
    tagged += [(e["idx"], 1, "IFVG", e) for e in ifvgs]
    tagged += [(e["idx"], 2, "Engulfing", e) for e in engulf]
    if not tagged:
        return None
    idx, _, kind, e = sorted(tagged, key=lambda t: (t[0], t[1]))[0]
    return {"idx": idx, "kind": kind, "direction": e["direction"], "time": e["time"]}


# ---------------------------------------------------------------------------
# OB/FVG quality tagging: Premium/Discount, liquidity sweeps, zone size.
# These feed the stop/target rules below and the POI picker's quality badges.
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    value = tr.rolling(period).mean().iloc[-1]
    return float(value) if pd.notna(value) else float(tr.mean())


def impulse_leg(origin_idx: int, direction: str, df: pd.DataFrame, swings: list[dict]) -> tuple[float, float]:
    """Price range of the impulsive move a POI originated from - used to find
    its Premium/Discount half. For a bullish leg: low = the origin candle's
    low, high = the next confirmed swing high (or the running max if the leg
    is still extending). Mirrored for bearish."""
    highs = sorted([s for s in swings if s["type"] == "high"], key=lambda s: s["idx"])
    lows = sorted([s for s in swings if s["type"] == "low"], key=lambda s: s["idx"])

    if direction == "bullish":
        low = df["low"].iloc[origin_idx]
        later_highs = [h for h in highs if h["idx"] > origin_idx]
        high = later_highs[0]["price"] if later_highs else df["high"].iloc[origin_idx:].max()
    else:
        high = df["high"].iloc[origin_idx]
        later_lows = [l for l in lows if l["idx"] > origin_idx]
        low = later_lows[0]["price"] if later_lows else df["low"].iloc[origin_idx:].min()

    return float(low), float(high)


def premium_discount(zone_low: float, zone_high: float, leg_low: float, leg_high: float) -> str:
    if leg_high <= leg_low:
        return "Discount"
    midpoint = (zone_low + zone_high) / 2
    fifty_pct = (leg_low + leg_high) / 2
    return "Discount" if midpoint < fifty_pct else "Premium"


def detect_equal_levels(swings: list[dict], tolerance_pct: float = 0.05) -> list[dict]:
    """Clusters nearby swing highs (and separately lows) into 'equal highs/lows'
    liquidity pools - a recognized liquidity area alongside session extremes."""
    levels = []
    for kind in ("high", "low"):
        pts = sorted([s for s in swings if s["type"] == kind], key=lambda s: s["price"])
        cluster = []
        for s in pts:
            if cluster and abs(s["price"] - cluster[-1]["price"]) / cluster[-1]["price"] * 100 <= tolerance_pct:
                cluster.append(s)
            else:
                if len(cluster) >= 2:
                    levels.append({"type": kind, "price": sum(c["price"] for c in cluster) / len(cluster)})
                cluster = [s]
        if len(cluster) >= 2:
            levels.append({"type": kind, "price": sum(c["price"] for c in cluster) / len(cluster)})
    return levels


def detect_liquidity_sweep_before(df: pd.DataFrame, idx: int, levels: list[float], direction: str, lookback: int = 10) -> bool:
    """True if, shortly before `idx`, price wicked through one of `levels` and
    closed back on the origin side - a stop-hunt/liquidity-grab pattern that
    the doc treats as a precondition for a high-probability POI."""
    start = max(0, idx - lookback)
    window = df.iloc[start:idx]
    if window.empty or not levels:
        return False
    if direction == "bullish":
        return any(((window["low"] < lvl) & (window["close"] > lvl)).any() for lvl in levels)
    return any(((window["high"] > lvl) & (window["close"] < lvl)).any() for lvl in levels)


def zone_size_class(zone_low: float, zone_high: float, atr_value: float) -> str:
    if atr_value <= 0:
        return "Small"
    return "Large" if (zone_high - zone_low) > atr_value else "Small"


def preceding_swing_price(origin_idx: int, direction: str, swings: list[dict]) -> float | None:
    kind = "low" if direction == "bullish" else "high"
    candidates = [s for s in swings if s["type"] == kind and s["idx"] <= origin_idx]
    return candidates[-1]["price"] if candidates else None


def tag_poi(poi: dict, poi_type: str, df: pd.DataFrame, swings: list[dict], atr_value: float, equal_levels: list[dict]) -> dict:
    """Annotates a raw OB or FVG dict (from detect_order_blocks/detect_fvg) with
    zone_low/zone_high (normalized), premium_discount, liquidity_swept, and
    size_class. Mutates and returns the same dict."""
    is_ob = poi_type == "Order Block"
    zone_low = poi["low"] if is_ob else poi["bottom"]
    zone_high = poi["high"] if is_ob else poi["top"]
    origin_idx = poi["idx"]

    leg_low, leg_high = impulse_leg(origin_idx, poi["direction"], df, swings)
    level_prices = [lvl["price"] for lvl in equal_levels] + [s["price"] for s in swings]

    poi["poi_type"] = poi_type
    poi["zone_low"] = float(zone_low)
    poi["zone_high"] = float(zone_high)
    poi["premium_discount"] = premium_discount(zone_low, zone_high, leg_low, leg_high)
    poi["liquidity_swept"] = detect_liquidity_sweep_before(df, origin_idx, level_prices, poi["direction"])
    poi["size_class"] = zone_size_class(zone_low, zone_high, atr_value)
    poi["preceding_swing"] = preceding_swing_price(origin_idx, poi["direction"], swings)
    return poi


def build_target_candidates(direction: str, swings: list[dict], equal_levels: list[dict], liq_levels: dict, obs: list[dict], fvgs: list[dict]) -> list[dict]:
    """Primary targets = opposing structure (swings/equal-levels/session
    liquidity); secondary = the near edge of an opposing-direction OB/FVG."""
    opp_kind = "high" if direction == "bullish" else "low"
    opposing_direction = "bearish" if direction == "bullish" else "bullish"
    candidates = []

    for s in swings:
        if s["type"] == opp_kind:
            candidates.append({"price": s["price"], "label": f"Swing {opp_kind}", "kind": "structure"})
    for lvl in equal_levels:
        if lvl["type"] == opp_kind:
            candidates.append({"price": lvl["price"], "label": f"Equal {opp_kind}s", "kind": "structure"})
    for name, lvl in liq_levels.items():
        price = lvl["high"] if direction == "bullish" else lvl["low"]
        candidates.append({"price": price, "label": f"{name} liquidity", "kind": "structure"})
    for ob in obs:
        if ob["direction"] == opposing_direction:
            near_edge = ob["zone_low"] if direction == "bullish" else ob["zone_high"]
            candidates.append({"price": near_edge, "label": "Opposing Order Block", "kind": "opposing_zone"})
    for f in fvgs:
        if f["direction"] == opposing_direction:
            near_edge = f["zone_low"] if direction == "bullish" else f["zone_high"]
            candidates.append({"price": near_edge, "label": "Opposing FVG", "kind": "opposing_zone"})

    return candidates


# ---------------------------------------------------------------------------
# Stop/target placement and the no-lookahead forward-walk backtest.
# ---------------------------------------------------------------------------

def compute_stop(poi: dict, ltf_confirmation_idx: int, ltf_df: pd.DataFrame, buffer_pct: float = 0.1) -> float:
    """Size/type-dependent stop placement: small OB -> zone edge + buffer;
    small FVG -> beyond the swing point that preceded the leg (tighter than
    the raw gap edge); large zone (either type) -> the LTF confirmation
    candle's own extreme + buffer (the 'zoom to LTF for a refined stop' case)."""
    zone_low, zone_high = poi["zone_low"], poi["zone_high"]
    height = max(zone_high - zone_low, 1e-9)
    buffer = max(height * buffer_pct, height * 0.02)

    if poi.get("size_class") == "Large":
        window = ltf_df.iloc[max(0, ltf_confirmation_idx - 5): ltf_confirmation_idx + 1]
        if poi["direction"] == "bullish":
            return float(window["low"].min() - buffer)
        return float(window["high"].max() + buffer)

    if poi.get("poi_type") == "Order Block":
        return (zone_low - buffer) if poi["direction"] == "bullish" else (zone_high + buffer)

    swing_price = poi.get("preceding_swing")
    if swing_price is not None:
        return (swing_price - buffer) if poi["direction"] == "bullish" else (swing_price + buffer)
    return (zone_low - buffer) if poi["direction"] == "bullish" else (zone_high + buffer)


def select_target(direction: str, entry_price: float, candidates: list[dict], risk: float, min_rr: float = 1.0, fallback_r: float = 2.0) -> dict:
    """Nearest opposing structure/zone candidate beyond entry that clears
    `min_rr` - walks past closer candidates that don't meet the framework's
    'at least 1:1' floor instead of blindly taking whatever is nearest.
    Falls back to a fixed R-multiple (at least min_rr) only if nothing
    beyond entry clears it, so a trade always has a target."""
    if direction == "bullish":
        beyond = sorted([c for c in candidates if c["price"] > entry_price], key=lambda c: c["price"])
    else:
        beyond = sorted([c for c in candidates if c["price"] < entry_price], key=lambda c: -c["price"])

    if risk:
        for c in beyond:
            reward = abs(c["price"] - entry_price)
            if reward / risk >= min_rr:
                return {"price": c["price"], "label": c["label"], "kind": c["kind"]}

    fb = max(fallback_r, min_rr)
    price = entry_price + fb * risk if direction == "bullish" else entry_price - fb * risk
    return {"price": price, "label": f"Fallback {fb}R", "kind": "fallback"}


def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str, entry_price: float, stop_price: float, target_price: float) -> dict:
    """Forward walk from entry_idx (the fill bar) onward. If a single bar's
    range contains both stop and target, the stop is assumed hit first - the
    standard conservative backtesting convention, since intrabar order is
    unknowable from OHLC alone."""
    risk = abs(entry_price - stop_price)

    for j in range(entry_idx, len(df)):
        high, low = df["high"].iloc[j], df["low"].iloc[j]
        hit_stop = (low <= stop_price) if direction == "bullish" else (high >= stop_price)
        hit_target = (high >= target_price) if direction == "bullish" else (low <= target_price)

        if hit_stop:
            return {"outcome": "Loss", "exit_idx": j, "exit_time": df["time"].iloc[j], "exit_price": stop_price, "r_multiple": -1.0}
        if hit_target:
            reward = abs(target_price - entry_price)
            r_multiple = reward / risk if risk else 0.0
            return {"outcome": "Win", "exit_idx": j, "exit_time": df["time"].iloc[j], "exit_price": target_price, "r_multiple": r_multiple}

    return {"outcome": "Open", "exit_idx": None, "exit_time": None, "exit_price": None, "r_multiple": 0.0}


# ---------------------------------------------------------------------------
# Callisto FX pre-trade checklist: FX/metal-only pip TP ladder (TP1-4 fixed
# pips, TP5 = HTF liquidity) with partial scale-outs, and a 15M/1H structural
# stop - used instead of the single-target R:R model when `pip_size` resolves.
# ---------------------------------------------------------------------------

TP_LADDER_PIPS = [20, 40, 60, 100]
DEFAULT_TP_FRACTIONS = [0.10, 0.10, 0.10, 0.30, 0.40]


def structural_stop(direction: str, zone_low: float, zone_high: float, m15_swings: list[dict],
                     h1_swings: list[dict], pip: float, buffer_pips: float = 5.0) -> float | None:
    """Nearest opposing 15M/1H swing beyond the POI zone, per the checklist's
    'SL placement: 15 MIN / 1 HR structural levels' rule - not the zone edge
    itself. Returns None if no such swing exists, so the caller can fall
    back to `compute_stop`."""
    kind = "low" if direction == "bullish" else "high"
    candidates = [s["price"] for s in (m15_swings + h1_swings) if s["type"] == kind]
    if direction == "bullish":
        candidates = [p for p in candidates if p < zone_low]
        return (max(candidates) - buffer_pips * pip) if candidates else None
    candidates = [p for p in candidates if p > zone_high]
    return (min(candidates) + buffer_pips * pip) if candidates else None


def build_tp_ladder(direction: str, entry_price: float, symbol: str, target_candidates: list[dict],
                     tp_pips: list[float] | None = None) -> list[dict] | None:
    """TP1-TP4 at fixed pip distances (default 20/40/60/100, override with
    `tp_pips` - must be 4 values), TP5 at the nearest higher-timeframe
    liquidity/structure target beyond TP4 (falling back to a fixed extension
    if none exists). Returns None for non-FX/metal symbols (see `pip_size`)."""
    pip = pip_size(symbol)
    if pip is None:
        return None

    pips_list = tp_pips if tp_pips is not None else TP_LADDER_PIPS
    sign = 1 if direction == "bullish" else -1
    ladder = [
        {"label": f"TP{i + 1}", "price": entry_price + sign * pips * pip, "pips": pips}
        for i, pips in enumerate(pips_list)
    ]

    tp4_price = ladder[-1]["price"]
    structure_targets = [c for c in target_candidates if c["kind"] == "structure"]
    if direction == "bullish":
        beyond = sorted([c for c in structure_targets if c["price"] > tp4_price], key=lambda c: c["price"])
    else:
        beyond = sorted([c for c in structure_targets if c["price"] < tp4_price], key=lambda c: -c["price"])

    if beyond:
        price = beyond[0]["price"]
        ladder.append({"label": f"TP5 (HTF liquidity: {beyond[0]['label']})", "price": price, "pips": abs(price - entry_price) / pip})
    else:
        fallback_pips = pips_list[-1] * 1.5
        ladder.append({"label": "TP5 (fallback)", "price": entry_price + sign * fallback_pips * pip, "pips": fallback_pips})

    return ladder


def simulate_partial_trade(df: pd.DataFrame, entry_idx: int, direction: str, entry_price: float,
                            stop_price: float, ladder: list[dict], fractions: list[float] | None = None) -> dict:
    """Forward walk applying the checklist's scale-out plan: closes a
    fraction of the position at each TP level in chronological order
    (no lookahead), stop-first within a single bar if it contains both a
    pending TP and the stop - same conservative convention as
    `simulate_trade`. Returns a result dict shape-compatible with
    `simulate_trade` (outcome/exit_idx/exit_time/exit_price/r_multiple) plus
    a "legs" list of every partial fill and "closed_fraction"."""
    fractions = list(fractions or DEFAULT_TP_FRACTIONS)
    risk = abs(entry_price - stop_price)
    pending = list(zip(ladder, fractions))
    legs: list[dict] = []
    r_realized = 0.0
    closed_fraction = 0.0

    for j in range(entry_idx, len(df)):
        high, low = df["high"].iloc[j], df["low"].iloc[j]
        hit_stop = (low <= stop_price) if direction == "bullish" else (high >= stop_price)

        if hit_stop:
            remaining = 1.0 - closed_fraction
            if remaining > 1e-9:
                r_realized += remaining * -1.0
                legs.append({"label": "Stop", "price": stop_price, "time": df["time"].iloc[j], "idx": j, "fraction": remaining})
                closed_fraction = 1.0
            break

        still_pending = []
        for tp, frac in pending:
            reached = (high >= tp["price"]) if direction == "bullish" else (low <= tp["price"])
            if reached:
                r_leg = (abs(tp["price"] - entry_price) / risk) if risk else 0.0
                r_realized += frac * r_leg
                closed_fraction += frac
                legs.append({"label": tp["label"], "price": tp["price"], "time": df["time"].iloc[j], "idx": j, "fraction": frac})
            else:
                still_pending.append((tp, frac))
        pending = still_pending

        if closed_fraction >= 1.0 - 1e-9 or not pending:
            break

    outcome = ("Win" if r_realized > 0 else "Loss") if closed_fraction >= 1.0 - 1e-9 else "Open"
    last = legs[-1] if legs else None

    return {
        "outcome": outcome,
        "exit_idx": last["idx"] if last else None,
        "exit_time": last["time"] if last else None,
        "exit_price": last["price"] if last else None,
        "r_multiple": r_realized,
        "legs": legs,
        "closed_fraction": closed_fraction,
    }


def risk_adjusted_score(r_multiples: list[float]) -> float:
    """Total R divided by the equity curve's max drawdown (largest
    peak-to-trough dip in cumulative R) - rewards a combination that both
    performs well AND stays smooth, not just whichever posts the highest
    total. Returns -inf for an empty list (no trades - can't compare), and
    total R directly when there's no drawdown (avoids a divide-by-zero
    blowup on a monotonically rising curve)."""
    if not r_multiples:
        return float("-inf")

    equity = []
    running = 0.0
    for r in r_multiples:
        running += r
        equity.append(running)

    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    total_r = equity[-1]
    return total_r if max_dd <= 1e-9 else total_r / max_dd
