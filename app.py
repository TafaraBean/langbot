"""75x Trading Framework dashboard - live MT5 data, top-down Bias -> Narrative
-> POI -> Confirmation workflow, plus a CSV trade journal."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import journal
import mt5_data
import sessions
import structure

st.set_page_config(page_title="75x Framework", layout="wide")

DEFAULT_SYMBOL = "XAUUSDm"
BULL = "#2ecc71"
BEAR = "#e74c3c"

# Tab 6 parameter optimizer presets - min_rr is deliberately not swept here,
# per the framework's "at least 1:1" floor being a fixed rule, not a knob.
TP_LADDER_PRESETS = {
    "Tight (10/20/30/50)": [10, 20, 30, 50],
    "Default (20/40/60/100)": [20, 40, 60, 100],
    "Wide (30/60/90/150)": [30, 60, 90, 150],
}
TP_FRACTION_PRESETS = {
    "Front-loaded (20/20/20/20/20)": [0.20, 0.20, 0.20, 0.20, 0.20],
    "Default (10/10/10/30/40)": [0.10, 0.10, 0.10, 0.30, 0.40],
    "Back-loaded (5/5/10/30/50)": [0.05, 0.05, 0.10, 0.30, 0.50],
}

for key, default in [
    ("biases", {}), ("narrative", (0, "Ranging", "No trade")),
    ("active_poi", None), ("confirmation", None),
]:
    st.session_state.setdefault(key, default)


def build_chart(df: pd.DataFrame, title: str, swings=None, obs=None, fvgs=None, hlines=None, zone=None, trade_markers=None) -> go.Figure:
    local = df.copy()
    local["time"] = local["time"].dt.tz_convert(sessions.local_tz())

    fig = go.Figure(data=[go.Candlestick(
        x=local["time"], open=local["open"], high=local["high"], low=local["low"], close=local["close"],
        increasing_line_color=BULL, decreasing_line_color=BEAR, name="price",
    )])

    if swings:
        highs = [s for s in swings if s["type"] == "high"]
        lows = [s for s in swings if s["type"] == "low"]
        fig.add_trace(go.Scatter(x=[local["time"].iloc[s["idx"]] for s in highs], y=[s["price"] for s in highs],
                                  mode="markers", marker=dict(color=BEAR, symbol="triangle-down", size=8), name="swing high"))
        fig.add_trace(go.Scatter(x=[local["time"].iloc[s["idx"]] for s in lows], y=[s["price"] for s in lows],
                                  mode="markers", marker=dict(color=BULL, symbol="triangle-up", size=8), name="swing low"))

    if obs:
        for ob in obs:
            color = BULL if ob["direction"] == "bullish" else BEAR
            fig.add_hrect(y0=ob["low"], y1=ob["high"], fillcolor=color, opacity=0.15, line_width=0)

    if fvgs:
        for f in fvgs:
            color = BULL if f["direction"] == "bullish" else BEAR
            fig.add_hrect(y0=f["bottom"], y1=f["top"], fillcolor=color, opacity=0.10, line_width=0)

    if hlines:
        for label, price in hlines:
            fig.add_hline(y=price, line_dash="dot", annotation_text=label, annotation_position="top left")

    if zone:
        color = BULL if zone["direction"] == "bullish" else BEAR
        fig.add_hrect(y0=zone["low"], y1=zone["high"], fillcolor=color, opacity=0.20, line_width=1.5,
                      line_color=color, annotation_text="Active POI", annotation_position="top left")

    if trade_markers:
        for m in trade_markers:
            fig.add_trace(go.Scatter(x=[m["time"]], y=[m["price"]], mode="markers+text",
                                      marker=dict(color=m["color"], size=13, symbol=m["symbol"], line=dict(width=2, color="white")),
                                      text=[m["label"]], textposition="top center", name=m["label"]))

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def bias_badge(bias: str) -> str:
    color = {"Bullish": BULL, "Bearish": BEAR}.get(bias, "#95a5a6")
    return f"<span style='background:{color};color:white;padding:2px 10px;border-radius:10px;font-weight:600'>{bias}</span>"


def to_naive_utc(ts) -> datetime:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


MAX_BACKTEST_SPAN_DAYS = {"M1": 2, "M3": 4, "M5": 8}


def passes_strict(poi: dict, desired_direction: str | None, desired_zone: str | None) -> bool:
    if not desired_direction:
        return True
    return poi["direction"] == desired_direction and poi["premium_discount"] == desired_zone


def poi_badge(poi: dict) -> str:
    swept = "Swept" if poi["liquidity_swept"] else "No sweep"
    return f"{poi['premium_discount']} · {swept} · {poi['size_class']}"


def poi_to_active_dict(poi: dict, poi_tf: str, swings, equal_levels, liq_levels, obs, fvgs) -> dict:
    direction = poi["direction"]
    zone_low, zone_high = poi["zone_low"], poi["zone_high"]
    target_candidates = structure.build_target_candidates(direction, swings, equal_levels, liq_levels, obs, fvgs)
    return {
        "poi_type": poi["poi_type"], "direction": direction,
        "zone_low": zone_low, "zone_high": zone_high,
        "poi_price": (zone_low + zone_high) / 2,
        "formed_time": poi["time"],
        "premium_discount": poi["premium_discount"], "liquidity_swept": poi["liquidity_swept"],
        "size_class": poi["size_class"], "preceding_swing": poi["preceding_swing"],
        "hp": poi.get("hp"),
        "target_candidates": target_candidates, "_source_tf": poi_tf, "_ltf": None,
    }


def run_mock_trade(symbol: str, ltf: str, poi_active: dict, min_rr: float, max_span_days: int | None = None,
                    tp_pips: list[float] | None = None, tp_fractions: list[float] | None = None) -> dict:
    """Runs the full no-lookahead backtest for one active-POI dict: fetches
    LTF data spanning from its formation time, finds the first in-zone
    confirmation, fills at the next candle's open, and simulates forward.
    Returns a dict with status in {"no_signal", "awaiting_next_candle", "ok"}.

    For FX/metal symbols (structure.pip_size resolves), stop and target
    follow the Callisto checklist: a 15M/1H structural stop and a fixed-pip
    TP1-TP5 ladder with partial scale-outs (result["partial"] is True and
    result["ladder"] holds the levels). Everything else keeps the original
    single R:R-target model. `tp_pips`/`tp_fractions` override the ladder's
    default TP1-4 pip distances / scale-out fractions (used by Tab 6's
    parameter optimizer; min_rr itself is never swept there).

    `max_span_days` overrides MAX_BACKTEST_SPAN_DAYS's fixed 2/4/8-day cap on
    how far past the POI's formation to search for a confirmation - Tab 6
    passes a wider value so older POIs found by a longer lookback get a real
    chance to reach a confirmation instead of being clipped to recent days."""
    max_days = max_span_days if max_span_days is not None else MAX_BACKTEST_SPAN_DAYS.get(ltf, 8)
    date_to = to_naive_utc(datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    date_from = to_naive_utc(poi_active["formed_time"])
    earliest_allowed = date_to - timedelta(days=max_days)
    if date_from < earliest_allowed:
        date_from = earliest_allowed

    df = mt5_data.get_rates_range(symbol, ltf, date_from, date_to)
    if df.empty:
        df = mt5_data.get_rates(symbol, ltf, bars=300)

    swings = structure.find_swings(df, left=1, right=1)
    breaks = structure.structure_breaks(df, swings, right=1)
    chochs = [b for b in breaks if b["kind"] == "CHoCH"]
    fvgs = structure.detect_fvg(df)
    ifvgs = structure.detect_ifvg(df, fvgs)
    engulf = structure.detect_engulfing(df)

    zone_low, zone_high = poi_active["zone_low"], poi_active["zone_high"]
    chochs_zone = structure.filter_events_in_zone(df, chochs, zone_low, zone_high)
    ifvgs_zone = structure.filter_events_in_zone(df, ifvgs, zone_low, zone_high)
    engulf_zone = structure.filter_events_in_zone(df, engulf, zone_low, zone_high)

    tapped = bool(((df["low"] <= zone_high) & (df["high"] >= zone_low)).any())

    base = {
        "df": df, "swings": swings, "zone": {"low": zone_low, "high": zone_high, "direction": poi_active["direction"]},
        "chochs_zone": chochs_zone, "ifvgs_zone": ifvgs_zone, "engulf_zone": engulf_zone,
        "date_from": date_from, "date_to": date_to, "max_days": max_days, "tapped": tapped,
    }

    sig = structure.earliest_confirmation(chochs_zone, ifvgs_zone, engulf_zone)
    if not sig:
        return {**base, "status": "no_signal"}

    entry_idx = sig["idx"] + 1
    if entry_idx >= len(df):
        return {**base, "status": "awaiting_next_candle", "sig": sig}

    entry_price = float(df["open"].iloc[entry_idx])
    direction = poi_active["direction"]

    pip = structure.pip_size(symbol)
    if pip is not None:
        m15_df = mt5_data.get_rates(symbol, "M15", bars=200)
        h1_df = mt5_data.get_rates(symbol, "H1", bars=200)
        m15_swings = structure.find_swings(m15_df)
        h1_swings = structure.find_swings(h1_df)
        stop_price = structure.structural_stop(direction, zone_low, zone_high, m15_swings, h1_swings, pip)
        if stop_price is None:
            stop_price = structure.compute_stop(poi_active, sig["idx"], df)

        ladder = structure.build_tp_ladder(direction, entry_price, symbol, poi_active["target_candidates"], tp_pips=tp_pips)
        result = structure.simulate_partial_trade(df, entry_idx, direction, entry_price, stop_price, ladder, fractions=tp_fractions)
        risk = abs(entry_price - stop_price)
        target = {"price": ladder[-1]["price"], "label": ladder[-1]["label"], "kind": "tp_ladder"}
        rr = (abs(target["price"] - entry_price) / risk) if risk else None

        return {
            **base, "status": "ok", "sig": sig, "entry_idx": entry_idx, "entry_price": entry_price,
            "direction": direction, "stop_price": stop_price, "target": target, "result": result, "rr": rr,
            "partial": True, "ladder": ladder,
        }

    stop_price = structure.compute_stop(poi_active, sig["idx"], df)
    risk = abs(entry_price - stop_price)
    target = structure.select_target(direction, entry_price, poi_active["target_candidates"], risk=risk, min_rr=min_rr)
    result = structure.simulate_trade(df, entry_idx, direction, entry_price, stop_price, target["price"])
    rr = (abs(target["price"] - entry_price) / risk) if risk else None

    return {
        **base, "status": "ok", "sig": sig, "entry_idx": entry_idx, "entry_price": entry_price,
        "direction": direction, "stop_price": stop_price, "target": target, "result": result, "rr": rr,
        "partial": False,
    }


def prepare_symbol_data(symbol: str, poi_tf: str, batch_ltf: str, max_candidates: int = 10,
                         poi_bars: int = 300, point_in_time_bias: bool = False,
                         diagnostics: dict | None = None) -> dict:
    """Phase 1 (expensive, per symbol): fetch data, detect OB/FVG, compute
    bias (current or point-in-time), and filter+cap to quality candidates.
    Returns everything `backtest_candidates` (phase 2) needs to run many
    parameter combinations - e.g. a TP-ladder/HP-threshold sweep - without
    re-fetching MT5 data or re-detecting structure each time.

    `point_in_time_bias=False` (default) computes one "current" D1/H4/H1/M15
    bias snapshot and applies it to every POI found, regardless of when the
    POI formed - fine for a short/recent poi_bars window, but a lookahead
    bias over a long one (a POI from months ago shouldn't be judged against
    today's bias). `point_in_time_bias=True` instead judges each POI against
    the bias that existed as of its own formation time (structure.bias_as_of),
    fetching bias history sized to cover the whole poi_bars span.

    If `diagnostics` is passed a dict, it's populated in place with
    raw_candidates/quality_candidates counts."""
    df = mt5_data.get_rates(symbol, poi_tf, bars=poi_bars)
    swings = structure.find_swings(df)
    breaks = structure.structure_breaks(df, swings)
    obs = structure.detect_order_blocks(df, breaks)
    fvgs = structure.detect_fvg(df)
    atr_value = structure.atr(df)
    equal_levels = structure.detect_equal_levels(swings)
    for ob in obs:
        structure.tag_poi(ob, "Order Block", df, swings, atr_value, equal_levels)
    for f in fvgs:
        structure.tag_poi(f, "FVG", df, swings, atr_value, equal_levels)

    df_m15 = mt5_data.get_rates(symbol, "M15", bars=500)
    liq_levels = sessions.session_liquidity_levels(df_m15)

    if point_in_time_bias:
        span_minutes = poi_bars * structure.TF_MINUTES[poi_tf]
        bias_dfs = {
            tf: mt5_data.get_rates(symbol, tf, bars=int(span_minutes / structure.TF_MINUTES[tf]) + 250)
            for tf in structure.TF_ORDER
        }
        # Precompute swings (with confirm times) once per bias timeframe
        # instead of letting bias_as_of re-scan the whole df from scratch for
        # every POI - with hundreds/thousands of candidates that per-call
        # rescan is what was making this take forever.
        bias_swings = {
            tf: structure.swings_with_confirm_times(bdf, structure.find_swings(bdf))
            for tf, bdf in bias_dfs.items()
        }
        quality_candidates = []
        for poi in obs + fvgs:
            biases_poi = {tf: structure.bias_as_of(bdf, poi["time"], swings=bias_swings[tf]) for tf, bdf in bias_dfs.items()}
            ref_bias_poi = structure.reference_bias(biases_poi)
            desired_zone_poi = {"Bullish": "Discount", "Bearish": "Premium"}.get(ref_bias_poi)
            desired_direction_poi = {"Bullish": "bullish", "Bearish": "bearish"}.get(ref_bias_poi)
            poi["hp"] = structure.evaluate_hp_criteria(poi, poi["poi_type"], swings, fvgs, desired_direction_poi)
            if passes_strict(poi, desired_direction_poi, desired_zone_poi):
                quality_candidates.append(poi)
    else:
        biases = {}
        for tf in structure.TF_ORDER:
            bdf = mt5_data.get_rates(symbol, tf, bars=200)
            biases[tf] = structure.classify_bias(structure.find_swings(bdf))
        ref_bias = structure.reference_bias(biases)
        desired_zone = {"Bullish": "Discount", "Bearish": "Premium"}.get(ref_bias)
        desired_direction = {"Bullish": "bullish", "Bearish": "bearish"}.get(ref_bias)
        structure.attach_hp_scores(obs, fvgs, swings, desired_direction)
        quality_candidates = [c for c in (obs + fvgs) if passes_strict(c, desired_direction, desired_zone)]

    # obs and fvgs are each chronological oldest->newest, and obs all sort
    # before fvgs in the concatenation above - without this, capping to
    # max_candidates would systematically keep the OLDEST matches (mostly
    # order blocks) instead of a recency-weighted spread. Recent POIs are far
    # more likely to still be near current price and have a real chance of
    # being tapped within any reasonable confirmation window.
    quality_candidates = sorted(quality_candidates, key=lambda c: c["idx"], reverse=True)[:max_candidates]

    if diagnostics is not None:
        diagnostics["raw_candidates"] = len(obs) + len(fvgs)
        diagnostics["quality_candidates"] = len(quality_candidates)

    return {
        "symbol": symbol, "poi_tf": poi_tf, "batch_ltf": batch_ltf,
        "quality_candidates": quality_candidates,
        "swings": swings, "equal_levels": equal_levels, "liq_levels": liq_levels,
        "obs": obs, "fvgs": fvgs,
    }


def backtest_candidates(prepared: dict, min_rr: float, max_span_days: int | None = None,
                         tp_pips: list[float] | None = None, tp_fractions: list[float] | None = None,
                         min_hp_score: int = 0, diagnostics: dict | None = None) -> list[dict]:
    """Phase 2 (cheap, repeatable): given `prepared` per-symbol data from
    `prepare_symbol_data`, filters by `min_hp_score` and runs the
    no-lookahead mock trade for each remaining candidate with the given
    TP-ladder parameters. This is the part a parameter sweep repeats many
    times per combination, without touching MT5 again. Returns one row dict
    per POI whose mock trade reached status "ok", each carrying
    "_poi_active"/"_mt5_result" (the raw objects needed for per-row
    charts/journal logging) - strip keys starting with "_" before building a
    DataFrame for display.

    If `diagnostics` is passed a dict, it's populated in place with
    status_counts/tapped_no_signal to help pinpoint why a run finds few/no
    trades."""
    symbol, poi_tf, batch_ltf = prepared["symbol"], prepared["poi_tf"], prepared["batch_ltf"]
    swings, equal_levels, liq_levels = prepared["swings"], prepared["equal_levels"], prepared["liq_levels"]
    obs, fvgs = prepared["obs"], prepared["fvgs"]

    if diagnostics is not None:
        diagnostics["status_counts"] = {"ok": 0, "no_signal": 0, "awaiting_next_candle": 0}
        diagnostics["tapped_no_signal"] = 0

    rows = []
    for poi in prepared["quality_candidates"]:
        if poi.get("hp") and poi["hp"]["score"] < min_hp_score:
            continue
        poi_active = poi_to_active_dict(poi, poi_tf, swings, equal_levels, liq_levels, obs, fvgs)
        mt5_result = run_mock_trade(symbol, batch_ltf, poi_active, min_rr, max_span_days=max_span_days,
                                     tp_pips=tp_pips, tp_fractions=tp_fractions)
        if diagnostics is not None:
            diagnostics["status_counts"][mt5_result["status"]] = diagnostics["status_counts"].get(mt5_result["status"], 0) + 1
            if mt5_result["status"] == "no_signal" and mt5_result.get("tapped"):
                diagnostics["tapped_no_signal"] += 1
        if mt5_result["status"] != "ok":
            continue
        result = mt5_result["result"]
        rows.append({
            "Symbol": symbol,
            "POI Type": poi_active["poi_type"], "Direction": poi_active["direction"],
            "Zone": f"{poi_active['zone_low']:.2f}-{poi_active['zone_high']:.2f}",
            "Premium/Discount": poi_active["premium_discount"],
            "Swept": poi_active["liquidity_swept"], "Size": poi_active["size_class"],
            "HP Score": f"{poi_active['hp']['score']}/4" if poi_active.get("hp") else "-",
            "Confirmation": mt5_result["sig"]["kind"], "Confirmation time": mt5_result["sig"]["time"],
            "Entry": round(mt5_result["entry_price"], 2), "Stop": round(mt5_result["stop_price"], 2),
            "Target": round(mt5_result["target"]["price"], 2),
            "R:R": round(mt5_result["rr"], 2) if mt5_result["rr"] else None,
            "Outcome": result["outcome"], "R multiple": round(result["r_multiple"], 2),
            "Partial TPs": mt5_result.get("partial", False),
            "_poi_active": poi_active, "_mt5_result": mt5_result,
        })
    return rows


def backtest_symbol(symbol: str, poi_tf: str, batch_ltf: str, min_rr: float, max_candidates: int = 10,
                     poi_bars: int = 300, max_span_days: int | None = None,
                     point_in_time_bias: bool = False, diagnostics: dict | None = None,
                     tp_pips: list[float] | None = None, tp_fractions: list[float] | None = None,
                     min_hp_score: int = 0) -> list[dict]:
    """Thin wrapper combining prepare_symbol_data + backtest_candidates for
    callers that just want one full run (Tabs 2/5, and Tab 6's main
    "Run portfolio backtest" button) - see those two functions for what each
    parameter does. Tab 6's parameter optimizer calls them separately so it
    can reuse one prepare_symbol_data() call across many backtest_candidates()
    calls instead of re-fetching/re-detecting per combination."""
    prepared = prepare_symbol_data(symbol, poi_tf, batch_ltf, max_candidates, poi_bars,
                                    point_in_time_bias, diagnostics=diagnostics)
    return backtest_candidates(prepared, min_rr, max_span_days, tp_pips, tp_fractions, min_hp_score,
                                diagnostics=diagnostics)


# ---------------------------------------------------------------- Sidebar --
st.sidebar.title("75x Framework")
symbol = st.sidebar.text_input("Symbol", value=DEFAULT_SYMBOL)
min_rr = st.sidebar.number_input("Minimum R:R", value=1.0, min_value=0.1, step=0.1,
                                  help="Mock trades walk past closer targets that don't clear this ratio (framework floor: at least 1:1).")

if st.sidebar.button("Reconnect / Refresh data"):
    st.cache_data.clear()

ok, msg = mt5_data.connect(symbol)
if ok:
    st.sidebar.success(f"MT5 connected · {symbol}")
else:
    st.sidebar.error(msg)
    st.error("Could not connect to MetaTrader 5. Make sure the terminal is open, logged in, and the symbol is correct.")
    st.stop()

now_local = datetime.now().astimezone()
active = sessions.active_sessions_now()
st.sidebar.markdown(f"**Local time:** {now_local.strftime('%H:%M:%S %Z')}")
st.sidebar.markdown(f"**Active session(s):** {', '.join(active) if active else 'None'}")

with st.sidebar.expander("Session windows (local time)"):
    for name, (start, end) in sessions.local_session_windows().items():
        st.markdown(f"**{name}**: {start.strftime('%H:%M')} - {end.strftime('%H:%M')}")

with st.sidebar.expander("Kalisto FX Trader Standard"):
    st.markdown("""
- Clear Bias: never open a chart without a defined directional bias
- Multi-Timeframe Analysis: always establish a plan before execution
- Confirmation-Only Entry: no entries without specific triggers
- Pre-Defined Invalidations: know where the trade is wrong
- Capital Protection: take partials, protect initial investment
- Behavioral Discipline: control actions and emotions, not just strategy
- Comprehensive Journaling: document every trade, win or loss
""")

with st.sidebar.expander("Effectiveness killers"):
    st.markdown("""
- Over-complication: more than 5 indicators = conflicting information
- Strategy hopping: changing systems after a loss destroys the data pool
- Pair dilution: focus on one pair to build intuition
- Fear-based delays: stacking too many confirmations misses the move
""")

st.title("75x Trading Framework")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1 · Bias & Narrative", "2 · Point of Interest", "3 · Confirmation", "4 · Setup & Journal",
    "5 · POI Backtest Results", "6 · Portfolio",
])

# ---------------------------------------------------------- Tab 1: Bias ----
with tab1:
    st.caption("Bias is 100% chart data, not feelings. Higher timeframes override lower ones.")
    biases = {}
    cols = st.columns(2)
    for i, tf in enumerate(structure.TF_ORDER):
        df = mt5_data.get_rates(symbol, tf, bars=200)
        swings = structure.find_swings(df)
        bias = structure.classify_bias(swings)
        biases[tf] = bias
        with cols[i % 2]:
            st.markdown(f"#### {tf} &nbsp; {bias_badge(bias)}", unsafe_allow_html=True)
            st.plotly_chart(build_chart(df, tf, swings=swings), width='stretch', key=f"chart_bias_{tf}")

    st.session_state["biases"] = biases
    count, label, strategy = structure.narrative_score(biases)
    st.session_state["narrative"] = (count, label, strategy)

    st.divider()
    st.subheader("Narrative")
    c1, c2, c3 = st.columns(3)
    c1.metric("Timeframes aligned", f"{count}/4")
    c2.metric("Trend strength", label)
    c3.metric("Strategy", strategy)

# ------------------------------------------------------ Tab 2: POI ---------
with tab2:
    st.caption("Session liquidity, order blocks, and Fair Value Gaps. Auto-detected - verify visually.")
    poi_tf = st.selectbox("Timeframe", ["H4", "H1", "M15"], index=2)

    ref_bias = structure.reference_bias(st.session_state.get("biases", {}))
    desired_zone = {"Bullish": "Discount", "Bearish": "Premium"}.get(ref_bias)
    desired_direction = {"Bullish": "bullish", "Bearish": "bearish"}.get(ref_bias)

    df = mt5_data.get_rates(symbol, poi_tf, bars=300)
    swings = structure.find_swings(df)
    breaks = structure.structure_breaks(df, swings)
    obs = structure.detect_order_blocks(df, breaks)
    fvgs = structure.detect_fvg(df)
    atr_value = structure.atr(df)
    equal_levels = structure.detect_equal_levels(swings)

    for ob in obs:
        structure.tag_poi(ob, "Order Block", df, swings, atr_value, equal_levels)
    for f in fvgs:
        structure.tag_poi(f, "FVG", df, swings, atr_value, equal_levels)
    structure.attach_hp_scores(obs, fvgs, swings, desired_direction)

    df_m15 = mt5_data.get_rates(symbol, "M15", bars=500)
    liq_levels = sessions.session_liquidity_levels(df_m15)

    def _sort_key(poi):
        type_rank = 0 if poi["poi_type"] == "Order Block" else 1
        zone_rank = 0 if desired_zone and poi["premium_discount"] == desired_zone else 1
        hp_rank = -poi.get("hp", {}).get("score", 0)
        return (type_rank, zone_rank, hp_rank)

    candidates = sorted(obs + fvgs, key=_sort_key)

    fc1, fc2 = st.columns(2)
    with fc1:
        strict = st.checkbox("Strict quality filter (OB-preferred, Discount/Premium-correct, direction-aligned)", value=True)
    with fc2:
        hp_only = st.checkbox("High-probability only (4/4 criteria: sweep, FVG, HTF confluence, inducement)", value=False)

    if strict:
        candidates = [c for c in candidates if passes_strict(c, desired_direction, desired_zone)]
    if hp_only:
        candidates = [c for c in candidates if c.get("hp", {}).get("score", 0) == 4]

    active_poi = st.session_state.get("active_poi")
    if active_poi and active_poi.get("_source_tf") == poi_tf:
        st.plotly_chart(
            build_chart(df, f"{poi_tf} - Active POI (focused)", swings=swings,
                        zone={"low": active_poi["zone_low"], "high": active_poi["zone_high"], "direction": active_poi["direction"]}),
            width='stretch', key="chart_poi",
        )
    else:
        hlines = []
        for name, lvl in liq_levels.items():
            hlines.append((f"{name} high", lvl["high"]))
            hlines.append((f"{name} low", lvl["low"]))
        st.plotly_chart(build_chart(df, f"{poi_tf} - POIs", swings=swings, obs=obs, fvgs=fvgs, hlines=hlines),
                         width='stretch', key="chart_poi")

    st.subheader("Select active POI")
    untested_only = st.checkbox("Untested killzones only", value=True,
                                 help="Checklist step 2: only show session-liquidity highs/lows price hasn't already traded through.")
    options = ["-- none --"]
    lookup = {}

    for ob in candidates:
        if ob["poi_type"] != "Order Block":
            continue
        hp = ob.get("hp", {})
        label = f"Order Block ({ob['direction']}) @ {ob['low']:.2f}-{ob['high']:.2f} [{ob['break_kind']}] · {poi_badge(ob)} · HP {hp.get('score', 0)}/4"
        options.append(label)
        lookup[label] = ("ob_fvg", ob)
    for f in candidates:
        if f["poi_type"] != "FVG":
            continue
        hp = f.get("hp", {})
        label = f"FVG ({f['direction']}) @ {f['bottom']:.2f}-{f['top']:.2f} · {poi_badge(f)} · HP {hp.get('score', 0)}/4"
        options.append(label)
        lookup[label] = ("ob_fvg", f)
    for name, lvl in liq_levels.items():
        for kind, price, direction, tested in [
            ("high", lvl["high"], "bearish", lvl["high_tested"]),
            ("low", lvl["low"], "bullish", lvl["low_tested"]),
        ]:
            if strict and desired_direction and direction != desired_direction:
                continue
            if untested_only and tested:
                continue
            label = f"{name} liquidity {kind} @ {price:.2f} · {'Tested' if tested else 'Untested'}"
            options.append(label)
            lookup[label] = ("liquidity", {"name": name, "price": price, "direction": direction, "date": lvl["date"]})

    choice = st.selectbox("Active POI for this setup", options)

    if choice == "-- none --":
        st.session_state["active_poi"] = None
    else:
        kind, raw = lookup[choice]
        direction = raw["direction"]

        if kind == "ob_fvg":
            new_active_poi = poi_to_active_dict(raw, poi_tf, swings, equal_levels, liq_levels, obs, fvgs)
        else:
            buffer = atr_value * 0.15
            zone_low, zone_high = raw["price"] - buffer, raw["price"] + buffer
            target_candidates = structure.build_target_candidates(direction, swings, equal_levels, liq_levels, obs, fvgs)

            formed_time = pd.Timestamp(sessions.session_end_datetime(raw["name"], raw["date"]))

            new_active_poi = {
                "poi_type": f"{raw['name']} Session Liquidity", "direction": direction,
                "zone_low": zone_low, "zone_high": zone_high,
                "poi_price": raw["price"],
                "formed_time": formed_time,
                "premium_discount": "N/A", "liquidity_swept": "N/A",
                "size_class": "Small", "preceding_swing": None, "hp": None,
                "target_candidates": target_candidates, "_source_tf": poi_tf, "_ltf": None,
            }

        st.session_state["active_poi"] = new_active_poi
        st.success(f"Active POI set: {choice}")

# --------------------------------------------------- Tab 3: Confirmation --
with tab3:
    st.caption("Green light on the LTF once price is inside the active POI. CHoCH > IFVG > Engulfing.")
    active_poi = st.session_state.get("active_poi")
    if not active_poi:
        st.warning("Set an active POI in Tab 2 first.")
    else:
        st.info(f"Active POI: {active_poi['poi_type']} @ {active_poi['zone_low']:.2f}-{active_poi['zone_high']:.2f} ({active_poi['direction']})")

    ltf = st.selectbox("Confirmation timeframe", ["M1", "M3", "M5"], index=2)

    # NOTE: uses conf_-prefixed / tab3_-prefixed locals throughout, deliberately not
    # reusing Tab 2's `df`/`swings`/`fvgs` names - Streamlit's `with tabN:` blocks share
    # one script-level namespace (no real scoping), and Tab 5 later needs Tab 2's
    # POI-timeframe `obs`/`fvgs`/`swings` untouched to build its own candidate pool.
    if active_poi:
        mt = run_mock_trade(symbol, ltf, active_poi, min_rr)
        tab3_df, conf_swings, zone = mt["df"], mt["swings"], mt["zone"]
        chochs_zone, ifvgs_zone, engulf_zone = mt["chochs_zone"], mt["ifvgs_zone"], mt["engulf_zone"]
        st.caption(f"Backtest window: {mt['date_from']} - {mt['date_to']} UTC (capped at {mt['max_days']}d for {ltf}), {len(tab3_df)} candles.")
    else:
        tab3_df = mt5_data.get_rates(symbol, ltf, bars=300)
        conf_swings = structure.find_swings(tab3_df, left=1, right=1)
        breaks = structure.structure_breaks(tab3_df, conf_swings, right=1)
        chochs_zone = [b for b in breaks if b["kind"] == "CHoCH"]
        conf_fvgs = structure.detect_fvg(tab3_df)
        ifvgs_zone = structure.detect_ifvg(tab3_df, conf_fvgs)
        engulf_zone = structure.detect_engulfing(tab3_df)
        zone = None

    st.plotly_chart(build_chart(tab3_df, f"{ltf} - Confirmation", swings=conf_swings, zone=zone), width='stretch', key="chart_conf")

    ranked = []
    if structure.most_recent(chochs_zone, 1):
        e = structure.most_recent(chochs_zone, 1)[0]
        ranked.append(("CHoCH", e["direction"], e["time"]))
    if structure.most_recent(ifvgs_zone, 1):
        e = structure.most_recent(ifvgs_zone, 1)[0]
        ranked.append(("IFVG", e["direction"], e["time"]))
    if structure.most_recent(engulf_zone, 1):
        e = structure.most_recent(engulf_zone, 1)[0]
        ranked.append(("Engulfing", e["direction"], e["time"]))

    if ranked:
        kind, direction, t = ranked[0]
        st.session_state["confirmation"] = {"type": kind, "direction": direction, "time": str(t)}
        st.success(f"Strongest recent in-zone confirmation: **{kind}** ({direction}) at {t}")
    else:
        st.session_state["confirmation"] = None
        st.warning("No in-zone confirmation detected yet on this timeframe.")

    with st.expander("All recent signals"):
        st.write("CHoCH:", structure.most_recent(chochs_zone, 3))
        st.write("IFVG:", structure.most_recent(ifvgs_zone, 3))
        st.write("Engulfing:", structure.most_recent(engulf_zone, 3))

    # ------------------------------------------------- Mock Trade backtest --
    if active_poi:
        st.divider()
        st.subheader("Mock Trade (no-lookahead backtest)")
        swept = {"True": "Swept", "False": "No sweep"}.get(str(active_poi["liquidity_swept"]), "N/A")
        st.caption(f"Active POI quality: {active_poi['premium_discount']} · {swept} · {active_poi['size_class']} · Min R:R {min_rr}")

        if not mt["tapped"]:
            st.info("POI not tapped yet - price hasn't reached the zone. Wait (checklist step 3).")
        elif mt["status"] == "no_signal":
            st.info("POI tapped - waiting for a confirmation (CHoCH > IFVG > Engulfing) in the zone.")
        elif mt["status"] == "awaiting_next_candle":
            st.info(f"In-zone {mt['sig']['kind']} confirmation detected at {mt['sig']['time']}, but there's no next candle yet to fill an entry.")
        else:
            entry_idx, entry_price, direction = mt["entry_idx"], mt["entry_price"], mt["direction"]
            stop_price, target, result, rr = mt["stop_price"], mt["target"], mt["result"], mt["rr"]
            partial = mt.get("partial", False)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Direction", direction.capitalize())
            mc2.metric("Entry", f"{entry_price:.2f}")
            mc3.metric("Stop", f"{stop_price:.2f}", "15M/1H structural" if partial else None)
            mc4.metric("Target", f"{target['price']:.2f}", target["label"])

            mc5, mc6, mc7 = st.columns(3)
            mc5.metric("R:R", f"{rr:.2f}" if rr else "-")
            mc6.metric("Outcome", result["outcome"])
            mc7.metric("R multiple", f"{result['r_multiple']:.2f}" if result["outcome"] != "Open" else "-")

            if not partial and rr is not None and rr < min_rr:
                st.warning(f"Computed R:R is below the {min_rr} minimum - no candidate cleared it, so this fell back to a fixed-R target.")

            local_tz = sessions.local_tz()
            markers = [{"time": tab3_df["time"].iloc[entry_idx].tz_convert(local_tz), "price": entry_price,
                        "label": "Entry", "color": "#3498db", "symbol": "triangle-right"}]

            if partial:
                st.caption("Callisto pip ladder - TP1: 20p, TP2: 40p, TP3: 60p, TP4: 100p, TP5: HTF liquidity. "
                            "Secures 10% at TP1-3, 30% at TP4, 40% at TP5 (or the stop, for whatever's left).")
                hit_by_label = {leg["label"]: leg for leg in result["legs"]}
                ladder_rows = [{
                    "Level": tp["label"], "Price": round(tp["price"], 2), "Pips": round(tp["pips"], 1),
                    "Hit": "Yes" if tp["label"] in hit_by_label else "No",
                    "Hit time": hit_by_label[tp["label"]]["time"] if tp["label"] in hit_by_label else "",
                } for tp in mt["ladder"]]
                st.dataframe(pd.DataFrame(ladder_rows), width='stretch')
                st.caption(f"Closed fraction: {result['closed_fraction']*100:.0f}%")

                for leg in result["legs"]:
                    is_stop = leg["label"] == "Stop"
                    markers.append({"time": tab3_df["time"].iloc[leg["idx"]].tz_convert(local_tz), "price": leg["price"],
                                     "label": f"{leg['label']} ({leg['fraction']*100:.0f}%)",
                                     "color": BEAR if is_stop else BULL, "symbol": "x" if is_stop else "circle"})
            elif result["exit_idx"] is not None:
                exit_color = BULL if result["outcome"] == "Win" else BEAR
                markers.append({"time": tab3_df["time"].iloc[result["exit_idx"]].tz_convert(local_tz), "price": result["exit_price"],
                                 "label": result["outcome"], "color": exit_color, "symbol": "x"})

            st.plotly_chart(build_chart(tab3_df, f"{ltf} - Mock trade", swings=conf_swings, zone=zone, trade_markers=markers),
                             width='stretch', key="chart_mock_trade")

            if st.button("Log this backtested trade to journal"):
                biases_now = st.session_state.get("biases", {})
                journal.append_trade({
                    "symbol": symbol, "direction": "Buy" if direction == "bullish" else "Sell",
                    "bias_D1": biases_now.get("D1", ""), "bias_H4": biases_now.get("H4", ""),
                    "bias_H1": biases_now.get("H1", ""), "bias_M15": biases_now.get("M15", ""),
                    "narrative_score": st.session_state.get("narrative", (0, "", ""))[0],
                    "poi_type": active_poi["poi_type"], "poi_price": active_poi["poi_price"],
                    "confirmation_type": mt["sig"]["kind"],
                    "entry": entry_price, "stop_loss": stop_price, "take_profit": target["price"],
                    "risk_reward": round(rr, 2) if rr else "",
                    "outcome": result["outcome"], "pnl_r": round(result["r_multiple"], 2),
                    "sl_basis": "15M/1H structural" if partial else "OB/FVG zone or LTF confirmation candle",
                    "tp_ladder": json.dumps(mt["ladder"]) if partial else "",
                    "partial_legs": json.dumps(result["legs"], default=str) if partial else "",
                    "notes": "Auto-backtest (no-lookahead mock trade)",
                })
                st.success("Backtested trade logged to journal.")

# ----------------------------------------------- Tab 4: Setup & Journal ---
with tab4:
    biases = st.session_state.get("biases", {})
    count, label, strategy = st.session_state.get("narrative", (0, "Ranging", ""))
    active_poi = st.session_state.get("active_poi")
    confirmation = st.session_state.get("confirmation")

    bias_ok = bool(biases) and biases.get("D1", "Unclear") != "Unclear"
    narrative_ok = count >= 2
    poi_ok = active_poi is not None
    confirmation_ok = confirmation is not None

    if biases.get("D1", "Unclear") == "Unclear":
        st.warning("Bias unclear on D1 - per the framework's alignment rule, avoid this pair right now.")

    st.subheader("Four Pillars Checklist")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bias", "OK" if bias_ok else "Missing", biases.get("D1", "-"))
    c2.metric("Narrative", "OK" if narrative_ok else "Weak", f"{count}/4 · {label}")
    c3.metric("POI", "OK" if poi_ok else "Missing", active_poi["poi_type"] if active_poi else "-")
    c4.metric("Confirmation", "OK" if confirmation_ok else "Missing", confirmation["type"] if confirmation else "-")

    valid_setup = bias_ok and narrative_ok and poi_ok and confirmation_ok
    if valid_setup:
        st.success("VALID SETUP - all four pillars satisfied.")
    else:
        st.warning("Setup incomplete - not all pillars are satisfied yet.")

    st.divider()
    st.subheader("Log a trade")
    with st.form("trade_form", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns(3)
        direction = fc1.selectbox("Direction", ["Buy", "Sell"])
        entry = fc2.number_input("Entry", min_value=0.0, format="%.2f")
        stop_loss = fc3.number_input("Stop loss (invalidation)", min_value=0.0, format="%.2f")
        fc4, fc5, fc6 = st.columns(3)
        take_profit = fc4.number_input("Take profit", min_value=0.0, format="%.2f")
        outcome = fc5.selectbox("Outcome", ["Open", "Win", "Loss", "BE"])
        pnl_r = fc6.number_input("PnL (R multiple)", value=0.0, format="%.2f")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Save trade")

        if submitted:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            rr = round(reward / risk, 2) if risk else ""
            journal.append_trade({
                "symbol": symbol, "direction": direction,
                "bias_D1": biases.get("D1", ""), "bias_H4": biases.get("H4", ""),
                "bias_H1": biases.get("H1", ""), "bias_M15": biases.get("M15", ""),
                "narrative_score": count,
                "poi_type": active_poi["poi_type"] if active_poi else "",
                "poi_price": active_poi["poi_price"] if active_poi else "",
                "confirmation_type": confirmation["type"] if confirmation else "",
                "entry": entry, "stop_loss": stop_loss, "take_profit": take_profit,
                "risk_reward": rr, "outcome": outcome, "pnl_r": pnl_r, "notes": notes,
            })
            st.success("Trade logged.")

    st.divider()
    st.subheader("Journal")
    jdf = journal.load_journal()
    st.dataframe(jdf, width='stretch')

    s = journal.stats(jdf)
    if s["count"]:
        jc1, jc2, jc3 = st.columns(3)
        jc1.metric("Trades logged", s["count"])
        jc2.metric("Win rate", f"{s['win_rate']*100:.1f}%" if s["win_rate"] is not None else "-")
        jc3.metric("Avg R", f"{s['avg_r']:.2f}" if s["avg_r"] is not None else "-")
        if len(s["equity_curve"]) > 1:
            st.line_chart(s["equity_curve"].reset_index(drop=True))
    else:
        st.info("No trades logged yet.")

# ----------------------------------------------- Tab 5: Batch backtest -----
with tab5:
    st.caption("No-lookahead backtest run across every quality-passing POI with an in-zone confirmation - "
               "same OB-preferred, Discount/Premium-correct, direction-aligned rule as Tab 2's strict filter, "
               "regardless of whether that checkbox happens to be on right now.")
    batch_ltf = st.selectbox("Backtest timeframe", ["M1", "M3", "M5"], index=2, key="batch_ltf")

    quality_candidates = [c for c in (obs + fvgs) if passes_strict(c, desired_direction, desired_zone)][:10]

    with st.spinner(f"Backtesting {len(quality_candidates)} quality-passing POI(s)..."):
        rows = backtest_symbol(symbol, poi_tf, batch_ltf, min_rr)
    details = [(r["_poi_active"], r["_mt5_result"]) for r in rows]
    display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    if not rows:
        st.info("No quality-passing POIs with an in-zone confirmation found yet on this POI timeframe/backtest timeframe. "
                "Try a different Timeframe in Tab 2, or a different backtest timeframe above.")
    else:
        rows_df = pd.DataFrame(display_rows)
        st.dataframe(rows_df, width='stretch')

        closed = rows_df[rows_df["Outcome"].isin(["Win", "Loss"])]
        win_rate = (closed["Outcome"] == "Win").mean() if len(closed) else None
        avg_r = closed["R multiple"].mean() if len(closed) else None
        equity_curve = rows_df["R multiple"].cumsum()

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Trades", len(rows_df))
        bc2.metric("Win rate", f"{win_rate*100:.1f}%" if win_rate is not None else "-")
        bc3.metric("Avg R", f"{avg_r:.2f}" if avg_r is not None else "-")
        if len(equity_curve) > 1:
            st.line_chart(equity_curve.reset_index(drop=True))

        for i, (poi_active, mt5_result) in enumerate(details):
            result = mt5_result["result"]
            partial = mt5_result.get("partial", False)
            label = f"{poi_active['poi_type']} ({poi_active['direction']}) @ {poi_active['zone_low']:.2f}-{poi_active['zone_high']:.2f} - {result['outcome']}"
            with st.expander(label):
                if poi_active.get("hp"):
                    st.caption(f"HP score: {poi_active['hp']['score']}/4")

                local_tz = sessions.local_tz()
                markers = [{"time": mt5_result["df"]["time"].iloc[mt5_result["entry_idx"]].tz_convert(local_tz),
                            "price": mt5_result["entry_price"], "label": "Entry", "color": "#3498db", "symbol": "triangle-right"}]

                if partial:
                    st.dataframe(pd.DataFrame([{"Level": tp["label"], "Price": round(tp["price"], 2), "Pips": round(tp["pips"], 1)}
                                                for tp in mt5_result["ladder"]]), width='stretch')
                    for leg in result["legs"]:
                        is_stop = leg["label"] == "Stop"
                        markers.append({"time": mt5_result["df"]["time"].iloc[leg["idx"]].tz_convert(local_tz), "price": leg["price"],
                                         "label": f"{leg['label']} ({leg['fraction']*100:.0f}%)",
                                         "color": BEAR if is_stop else BULL, "symbol": "x" if is_stop else "circle"})
                elif result["exit_idx"] is not None:
                    exit_color = BULL if result["outcome"] == "Win" else BEAR
                    markers.append({"time": mt5_result["df"]["time"].iloc[result["exit_idx"]].tz_convert(local_tz),
                                     "price": result["exit_price"], "label": result["outcome"], "color": exit_color, "symbol": "x"})

                st.plotly_chart(
                    build_chart(mt5_result["df"], f"{batch_ltf} - {poi_active['poi_type']}", swings=mt5_result["swings"],
                                zone=mt5_result["zone"], trade_markers=markers),
                    width='stretch', key=f"chart_batch_{i}",
                )

                if st.button("Log to journal", key=f"log_batch_{i}"):
                    biases_now = st.session_state.get("biases", {})
                    journal.append_trade({
                        "symbol": symbol, "direction": "Buy" if poi_active["direction"] == "bullish" else "Sell",
                        "bias_D1": biases_now.get("D1", ""), "bias_H4": biases_now.get("H4", ""),
                        "bias_H1": biases_now.get("H1", ""), "bias_M15": biases_now.get("M15", ""),
                        "narrative_score": st.session_state.get("narrative", (0, "", ""))[0],
                        "poi_type": poi_active["poi_type"], "poi_price": poi_active["poi_price"],
                        "confirmation_type": mt5_result["sig"]["kind"],
                        "entry": mt5_result["entry_price"], "stop_loss": mt5_result["stop_price"],
                        "take_profit": mt5_result["target"]["price"],
                        "risk_reward": round(mt5_result["rr"], 2) if mt5_result["rr"] else "",
                        "outcome": result["outcome"], "pnl_r": round(result["r_multiple"], 2),
                        "sl_basis": "15M/1H structural" if partial else "OB/FVG zone or LTF confirmation candle",
                        "tp_ladder": json.dumps(mt5_result["ladder"]) if partial else "",
                        "partial_legs": json.dumps(result["legs"], default=str) if partial else "",
                        "notes": "Auto-backtest (batch scan, no-lookahead mock trade)",
                    })
                    st.success("Backtested trade logged to journal.")

# ------------------------------------------------------- Tab 6: Portfolio --
with tab6:
    st.caption("Same fresh bias -> POI -> quality filter -> no-lookahead mock-trade pipeline as Tabs 2/5, "
               "looped across a configurable multi-asset portfolio. Symbols are broker-specific - use your "
               "MT5 Market Watch's exact names (e.g. suffixed like XAUUSDm). The examples below are "
               "placeholders and likely won't match your broker.")

    pc1, pc2 = st.columns(2)
    with pc1:
        portfolio_poi_tf = st.selectbox("POI timeframe", ["H4", "H1", "M15"], index=2, key="portfolio_poi_tf")
    with pc2:
        portfolio_ltf = st.selectbox("Backtest timeframe", ["M1", "M3", "M5"], index=2, key="portfolio_ltf")

    pc3, pc4, pc5 = st.columns(3)
    with pc3:
        portfolio_poi_bars = st.number_input("POI lookback (bars)", min_value=300, max_value=20000, value=5000, step=500,
                                              help="How far back to scan for order blocks/FVGs on the POI timeframe.")
    with pc4:
        portfolio_span_days = st.number_input("Confirmation window (days)", min_value=2, max_value=180, value=30, step=1,
                                               help="How far past each POI's formation to search for a confirmation and simulate the trade.")
    with pc5:
        portfolio_max_candidates = st.number_input("Max POIs per symbol", min_value=1, max_value=500, value=75, step=5)

    portfolio_pit_bias = st.checkbox("Point-in-time bias (recommended)", value=True,
                                      help="Judge each POI against the D1/H4/H1/M15 bias as it was when the POI formed, "
                                           "instead of today's bias applied to the whole history. More correct, somewhat slower.")

    tf_minutes = {"H4": 240, "H1": 60, "M15": 15}[portfolio_poi_tf]
    st.caption(f"~ {portfolio_poi_bars * tf_minutes / 1440:.0f} calendar days of {portfolio_poi_tf} history per symbol. "
               "Larger values give a fairer sample but a slower run; very large M1/M3/M5 pulls may exceed your broker's "
               "stored history and simply return fewer bars.")

    tech_symbols = st.text_area("Tech stocks", value="AAPL, MSFT, NVDA, AMZN, GOOGL", key="portfolio_tech")
    fx_symbols = st.text_area("Currencies", value="EURUSDm, GBPUSDm, USDJPYm, AUDUSDm", key="portfolio_fx")
    crypto_symbols = st.text_area("Crypto", value="BTCUSDm, ETHUSDm", key="portfolio_crypto")

    run_portfolio = st.button("Run portfolio backtest")

    if run_portfolio:
        symbol_specs = []
        for asset_class, raw in [("Tech", tech_symbols), ("Currency", fx_symbols), ("Crypto", crypto_symbols)]:
            for sym in [s.strip() for s in raw.split(",") if s.strip()]:
                symbol_specs.append((asset_class, sym))

        if not symbol_specs:
            st.warning("No symbols entered - add at least one ticker to one of the lists above.")
            st.session_state["portfolio_results"] = None
            st.session_state["portfolio_skipped"] = []
        else:
            all_rows, skipped, diag_rows = [], [], []
            progress = st.progress(0.0)
            for i, (asset_class, sym) in enumerate(symbol_specs):
                progress.progress(i / len(symbol_specs), text=f"{sym} ({asset_class})...")
                ok, msg = mt5_data.connect(sym)
                if not ok:
                    skipped.append((sym, asset_class, msg))
                    continue
                diag = {}
                try:
                    sym_rows = backtest_symbol(sym, portfolio_poi_tf, portfolio_ltf, min_rr,
                                                max_candidates=portfolio_max_candidates,
                                                poi_bars=portfolio_poi_bars,
                                                max_span_days=portfolio_span_days,
                                                point_in_time_bias=portfolio_pit_bias,
                                                diagnostics=diag)
                except Exception as exc:
                    skipped.append((sym, asset_class, str(exc)))
                    continue
                for r in sym_rows:
                    r.pop("_poi_active", None)
                    r.pop("_mt5_result", None)
                    r["Asset class"] = asset_class
                all_rows.extend(sym_rows)
                status = diag.get("status_counts", {})
                diag_rows.append({
                    "Symbol": sym, "Asset class": asset_class,
                    "Raw POIs found": diag.get("raw_candidates", 0),
                    "Quality-passing": diag.get("quality_candidates", 0),
                    "Tapped, no confirmation": diag.get("tapped_no_signal", 0),
                    "No signal (untapped)": status.get("no_signal", 0) - diag.get("tapped_no_signal", 0),
                    "Awaiting next candle": status.get("awaiting_next_candle", 0),
                    "Completed trades": status.get("ok", 0),
                })
            progress.progress(1.0)
            progress.empty()
            st.session_state["portfolio_results"] = all_rows
            st.session_state["portfolio_skipped"] = skipped
            st.session_state["portfolio_diagnostics"] = diag_rows

    all_rows = st.session_state.get("portfolio_results")
    skipped = st.session_state.get("portfolio_skipped", [])
    diag_rows = st.session_state.get("portfolio_diagnostics", [])

    if skipped:
        with st.expander(f"{len(skipped)} symbol(s) skipped"):
            for sym, asset_class, msg in skipped:
                st.warning(f"{sym} ({asset_class}): {msg}")

    if diag_rows:
        with st.expander("Diagnostics: where each symbol's candidates ended up", expanded=not all_rows):
            st.caption("'Tapped, no confirmation' = price reached the POI zone but no CHoCH/IFVG/Engulfing fired there yet. "
                       "'No signal (untapped)' = price never revisited the zone within the confirmation window - "
                       "try a larger 'Confirmation window (days)' or a smaller 'POI lookback (bars)' if this dominates.")
            st.dataframe(pd.DataFrame(diag_rows), width='stretch')

    if all_rows is None:
        st.info("Configure your portfolio above and click 'Run portfolio backtest'.")
    elif not all_rows:
        st.info("No quality-passing POIs with an in-zone confirmation were found across the portfolio - see Diagnostics above.")
    else:
        results_df = pd.DataFrame(all_rows)
        lead_cols = ["Symbol", "Asset class"]
        results_df = results_df[lead_cols + [c for c in results_df.columns if c not in lead_cols]]
        results_df = results_df.sort_values("Confirmation time").reset_index(drop=True)

        st.subheader("Combined results")
        st.dataframe(results_df, width='stretch')

        closed = results_df[results_df["Outcome"].isin(["Win", "Loss"])]
        win_rate = (closed["Outcome"] == "Win").mean() if len(closed) else None
        avg_r = closed["R multiple"].mean() if len(closed) else None
        equity_curve = results_df["R multiple"].cumsum()

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total trades", len(results_df))
        mcol2.metric("Win rate", f"{win_rate*100:.1f}%" if win_rate is not None else "-")
        mcol3.metric("Avg R", f"{avg_r:.2f}" if avg_r is not None else "-")

        st.subheader("Portfolio equity curve")
        if len(equity_curve) > 1:
            st.line_chart(equity_curve.reset_index(drop=True))
        else:
            st.caption("Not enough trades yet for a curve.")

        st.subheader("Per asset-class breakdown")
        breakdown = []
        for asset_class in results_df["Asset class"].unique():
            sub = results_df[results_df["Asset class"] == asset_class]
            sub_closed = sub[sub["Outcome"].isin(["Win", "Loss"])]
            sub_win_rate = (sub_closed["Outcome"] == "Win").mean() if len(sub_closed) else None
            sub_avg_r = sub_closed["R multiple"].mean() if len(sub_closed) else None
            breakdown.append({
                "Asset class": asset_class, "Trades": len(sub),
                "Win rate": f"{sub_win_rate*100:.1f}%" if sub_win_rate is not None else "-",
                "Avg R": f"{sub_avg_r:.2f}" if sub_avg_r is not None else "-",
            })
        st.dataframe(pd.DataFrame(breakdown), width='stretch')

    # ------------------------------------------- Parameter optimization ----
    st.divider()
    st.subheader("Parameter optimization")
    st.caption("Sweeps the TP ladder / scale-out fractions and the HP-score minimum across the same portfolio "
               "and settings above - min_rr stays fixed at the sidebar's value throughout (the framework's "
               "'at least 1:1' floor is a rule, not something to tune away). Ranks combinations by total R "
               "divided by max drawdown, so a smoother, more consistent curve outranks a lucky-but-choppy one.")

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        opt_ladders = st.multiselect("TP ladder presets", list(TP_LADDER_PRESETS.keys()),
                                      default=list(TP_LADDER_PRESETS.keys()), key="opt_ladders")
    with oc2:
        opt_fractions = st.multiselect("Scale-out fraction presets", list(TP_FRACTION_PRESETS.keys()),
                                        default=list(TP_FRACTION_PRESETS.keys()), key="opt_fractions")
    with oc3:
        opt_hp_scores = st.multiselect("HP score minimum", [0, 1, 2, 3, 4], default=[0, 2, 4], key="opt_hp_scores")

    n_combos = len(opt_ladders) * len(opt_fractions) * len(opt_hp_scores)
    n_symbols = sum(1 for raw in (tech_symbols, fx_symbols, crypto_symbols) for s in raw.split(",") if s.strip())
    st.caption(f"{n_combos} combination(s) x {n_symbols} symbol(s) - each reuses the same fetched data per symbol, "
               "so this is much cheaper than re-running the full portfolio backtest once per combination.")

    run_optimizer = st.button("Run parameter optimization", disabled=n_combos == 0)

    if run_optimizer:
        opt_symbol_specs = []
        for asset_class, raw in [("Tech", tech_symbols), ("Currency", fx_symbols), ("Crypto", crypto_symbols)]:
            for sym in [s.strip() for s in raw.split(",") if s.strip()]:
                opt_symbol_specs.append((asset_class, sym))

        if not opt_symbol_specs:
            st.warning("No symbols entered - add at least one ticker to one of the lists above.")
            st.session_state["optimizer_results"] = None
        else:
            prepared_symbols = []
            skipped_opt = []
            progress = st.progress(0.0, text="Fetching data per symbol...")
            for i, (asset_class, sym) in enumerate(opt_symbol_specs):
                progress.progress(i / len(opt_symbol_specs), text=f"Preparing {sym} ({asset_class})...")
                ok, msg = mt5_data.connect(sym)
                if not ok:
                    skipped_opt.append((sym, asset_class, msg))
                    continue
                try:
                    prepared = prepare_symbol_data(sym, portfolio_poi_tf, portfolio_ltf,
                                                    max_candidates=portfolio_max_candidates,
                                                    poi_bars=portfolio_poi_bars,
                                                    point_in_time_bias=portfolio_pit_bias)
                except Exception as exc:
                    skipped_opt.append((sym, asset_class, str(exc)))
                    continue
                prepared_symbols.append((asset_class, prepared))

            leaderboard = []
            combos = [(l, f, h) for l in opt_ladders for f in opt_fractions for h in opt_hp_scores]
            for j, (ladder_name, fraction_name, hp_min) in enumerate(combos):
                progress.progress(j / max(len(combos), 1), text=f"Testing combo {j + 1}/{len(combos)}...")
                combo_rows = []
                for asset_class, prepared in prepared_symbols:
                    combo_rows.extend(backtest_candidates(
                        prepared, min_rr, max_span_days=portfolio_span_days,
                        tp_pips=TP_LADDER_PRESETS[ladder_name], tp_fractions=TP_FRACTION_PRESETS[fraction_name],
                        min_hp_score=hp_min,
                    ))
                combo_rows.sort(key=lambda r: r["Confirmation time"])
                r_multiples = [r["R multiple"] for r in combo_rows]
                closed = [r for r in combo_rows if r["Outcome"] in ("Win", "Loss")]
                win_rate = (sum(1 for r in closed if r["Outcome"] == "Win") / len(closed)) if closed else None
                avg_r = (sum(r["R multiple"] for r in closed) / len(closed)) if closed else None
                score = structure.risk_adjusted_score(r_multiples)
                leaderboard.append({
                    "TP ladder": ladder_name, "Scale-out": fraction_name, "HP min": hp_min,
                    "Trades": len(combo_rows),
                    "Win rate": f"{win_rate*100:.1f}%" if win_rate is not None else "-",
                    "Avg R": f"{avg_r:.2f}" if avg_r is not None else "-",
                    "Total R": round(sum(r_multiples), 2) if r_multiples else 0.0,
                    "Risk-adjusted score": round(score, 2) if score != float("-inf") else "-",
                    "_score": score, "_equity": r_multiples,
                })
            progress.progress(1.0)
            progress.empty()
            leaderboard.sort(key=lambda row: row["_score"], reverse=True)
            st.session_state["optimizer_results"] = leaderboard
            st.session_state["optimizer_skipped"] = skipped_opt

    optimizer_results = st.session_state.get("optimizer_results")
    optimizer_skipped = st.session_state.get("optimizer_skipped", [])

    if optimizer_skipped:
        with st.expander(f"{len(optimizer_skipped)} symbol(s) skipped in optimizer run"):
            for sym, asset_class, msg in optimizer_skipped:
                st.warning(f"{sym} ({asset_class}): {msg}")

    if optimizer_results is None:
        st.info("Pick your parameter ranges above and click 'Run parameter optimization'.")
    elif not optimizer_results or all(row["Trades"] == 0 for row in optimizer_results):
        st.info("No combination produced any trades - try widening the confirmation window or POI lookback above first.")
    else:
        st.subheader("Leaderboard (best risk-adjusted score first)")
        display_cols = ["TP ladder", "Scale-out", "HP min", "Trades", "Win rate", "Avg R", "Total R", "Risk-adjusted score"]
        st.dataframe(pd.DataFrame([{k: row[k] for k in display_cols} for row in optimizer_results]), width='stretch')

        best = optimizer_results[0]
        if best["_equity"]:
            st.caption(f"Best combination: {best['TP ladder']} ladder, {best['Scale-out']} scale-out, HP min {best['HP min']}.")
            st.line_chart(pd.Series(best["_equity"]).cumsum().reset_index(drop=True))

