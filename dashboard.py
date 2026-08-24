"""Indicator-constraint market dashboard.

Marks mock entries directly on a candlestick chart wherever a live,
slider-controlled set of indicator constraints (Bulls/Bears Power, RSI,
MACD, Moving Average, ATR) is satisfied. Nothing is logged or persisted -
the chart is the entire output.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest_kde
import indicators
import mt5_data
import order_blocks
import sessions
import smc
import volume_profile

st.set_page_config(page_title="Indicator dashboard", layout="wide")

DEFAULT_SYMBOL = "XAUUSDm"
BULL = "#26a69a"
BEAR = "#ef5350"
VOLUME_PROFILE = "#9b59b6"


def preview_range(series: pd.Series, pad_ratio: float = 0.5, floor_zero: bool = False) -> tuple[float, float]:
    """Sensible slider bounds for a symbol-scale-dependent indicator, derived
    from its own recent history rather than a fixed range."""
    clean = series.dropna()
    if clean.empty:
        return (0.0, 1.0)
    lo, hi = float(clean.min()), float(clean.max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * pad_ratio
    lo, hi = lo - pad, hi + pad
    if floor_zero:
        lo = max(0.0, lo)
    return (lo, hi)


def indicator_card(col, title: str, key: str, range_bounds: tuple[float, float], default_period: int,
                    period_bounds: tuple[int, int] = (2, 100), default_ratio: float = 0.5):
    """Renders one bordered indicator card: enable toggle, period slider,
    above/below operator, and a threshold slider scaled to range_bounds."""
    lo, hi = range_bounds
    default_threshold = lo + (hi - lo) * default_ratio
    with col, st.container(border=True):
        enabled = st.toggle(title, key=f"{key}_enabled")
        period = st.slider("Period", period_bounds[0], period_bounds[1], default_period,
                            key=f"{key}_period", disabled=not enabled)
        operator = st.segmented_control("Condition", ["above", "below"], default="above", required=True,
                                         key=f"{key}_operator", disabled=not enabled)
        threshold = st.slider("Threshold", lo, hi, default_threshold, key=f"{key}_threshold", disabled=not enabled)
    return {"enabled": enabled, "period": period, "operator": operator, "threshold": threshold}


def _tz(ts, tz):
    return pd.Timestamp(ts).tz_convert(tz)


def _add_structure_events(fig, events, tz, internal: bool):
    dash = "dot" if internal else "solid"
    width = 1 if internal else 1.5
    for ev in events:
        color = BULL if ev.direction == "bullish" else BEAR
        fig.add_trace(go.Scatter(
            x=[_tz(ev.pivot_time, tz), _tz(ev.break_time, tz)], y=[ev.price, ev.price], mode="lines",
            line=dict(color=color, width=width, dash=dash), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_annotation(x=_tz(ev.break_time, tz), y=ev.price, text=ev.kind, showarrow=False,
                            font=dict(color=color, size=9 if internal else 11),
                            yshift=9 if ev.direction == "bullish" else -9, row=1, col=1)


def _add_order_blocks(fig, blocks, tz, right_edge, internal: bool):
    opacity = 0.15 if internal else 0.25
    for ob in blocks:
        color = BULL if ob.bias == smc.BULLISH else BEAR
        fig.add_shape(type="rect", x0=_tz(ob.bar_time, tz), x1=right_edge, y0=ob.low, y1=ob.high,
                      fillcolor=color, opacity=opacity, line=dict(width=0), layer="below", row=1, col=1)


def _add_fair_value_gaps(fig, gaps, tz, extend_delta):
    for fvg in gaps:
        top, bottom = max(fvg.top, fvg.bottom), min(fvg.top, fvg.bottom)
        color = BULL if fvg.bias == smc.BULLISH else BEAR
        fig.add_shape(type="rect", x0=_tz(fvg.left_time, tz), x1=_tz(fvg.right_time, tz) + extend_delta,
                      y0=bottom, y1=top, fillcolor=color, opacity=0.2, line=dict(width=0),
                      layer="below", row=1, col=1)


def _add_equal_levels(fig, levels, tz):
    for lvl in levels:
        color = BEAR if lvl.kind == "EQH" else BULL
        fig.add_trace(go.Scatter(
            x=[_tz(lvl.from_time, tz), _tz(lvl.to_time, tz)], y=[lvl.price, lvl.price], mode="lines",
            line=dict(color=color, width=1, dash="dot"), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_annotation(x=_tz(lvl.to_time, tz), y=lvl.price, text=lvl.kind, showarrow=False,
                            font=dict(color=color, size=9), yshift=9 if lvl.kind == "EQH" else -9, row=1, col=1)


def _add_zones(fig, result, tz, right_edge):
    if not result.premium_zone:
        return
    x0 = _tz(result.zone_left_time, tz)
    bands = [(result.premium_zone, "#F23645"), (result.equilibrium_zone, "#878b94"), (result.discount_zone, "#089981")]
    for (top, bottom), color in bands:
        fig.add_shape(type="rect", x0=x0, x1=right_edge, y0=bottom, y1=top, fillcolor=color, opacity=0.1,
                      line=dict(width=0), layer="below", row=1, col=1)


def _add_strong_weak(fig, result, tz, right_edge):
    if not result.strong_weak_top_label:
        return
    trailing = result.trailing
    for level, last_time, label, color, shift in (
        (trailing.top, trailing.last_top_time, result.strong_weak_top_label, BEAR, 9),
        (trailing.bottom, trailing.last_bottom_time, result.strong_weak_bottom_label, BULL, -9),
    ):
        fig.add_trace(go.Scatter(x=[_tz(last_time, tz), right_edge], y=[level, level], mode="lines",
                                  line=dict(color=color, width=1), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_annotation(x=right_edge, y=level, text=label, showarrow=False, xanchor="right",
                            font=dict(color=color, size=9), yshift=shift, row=1, col=1)


def _add_mtf_level(fig, level, tz, right_edge, label, color):
    if level is None:
        return
    x0 = _tz(level.left_time, tz)
    for y in (level.top, level.bottom):
        fig.add_shape(type="line", x0=x0, x1=right_edge, y0=y, y1=y, line=dict(color=color, width=1),
                      row=1, col=1)
    fig.add_annotation(x=right_edge, y=level.top, text=f"P{label}H", showarrow=False, xanchor="left",
                        font=dict(color=color, size=9), row=1, col=1)
    fig.add_annotation(x=right_edge, y=level.bottom, text=f"P{label}L", showarrow=False, xanchor="left",
                        font=dict(color=color, size=9), row=1, col=1)


def _add_volume_order_blocks(fig, result: "order_blocks.VolumeOBResult", tz, right_edge):
    for blocks, color in ((result.bullish, BULL), (result.bearish, BEAR)):
        for ob in blocks:
            x0 = _tz(ob.left_time, tz)
            fig.add_shape(type="rect", x0=x0, x1=right_edge, y0=ob.bottom, y1=ob.top,
                          fillcolor=color, opacity=0.2, line=dict(width=0), layer="below", row=1, col=1)
            fig.add_trace(go.Scatter(x=[x0, right_edge], y=[ob.avg, ob.avg], mode="lines",
                                      line=dict(color="#9598a1", width=1), showlegend=False,
                                      hoverinfo="skip"), row=1, col=1)


def _add_volume_profile_density(fig, profile: "volume_profile.ProfileResult", time_min, time_max, right_edge):
    span_seconds = max((time_max - time_min).total_seconds(), 1)
    width = pd.Timedelta(seconds=span_seconds * 0.15)
    xs = [right_edge] + [right_edge - width * d for d in profile.density] + [right_edge]
    ys = [profile.grid[0]] + list(profile.grid) + [profile.grid[-1]]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, fill="toself", mode="lines", line=dict(color=VOLUME_PROFILE, width=1),
        fillcolor="rgba(155,89,182,0.15)", showlegend=False, hoverinfo="skip", name="Volume profile",
    ), row=1, col=1)


def _add_volume_profile_peaks(fig, profile: "volume_profile.ProfileResult", left_edge, right_edge):
    for rank, price in enumerate(profile.peaks, start=1):
        fig.add_shape(type="line", x0=left_edge, x1=right_edge, y0=price, y1=price,
                      line=dict(color=VOLUME_PROFILE, width=1, dash="dot"), row=1, col=1)
        fig.add_annotation(x=left_edge, y=price, text=f"VP{rank} {price:,.2f}", showarrow=False, xanchor="left",
                            yshift=8, font=dict(color=VOLUME_PROFILE, size=9), row=1, col=1)


def build_chart(df: pd.DataFrame, config: dict, mask: pd.Series, direction: str,
                 smc_config: dict | None = None, smc_result: "smc.SMCResult | None" = None,
                 vob_result: "order_blocks.VolumeOBResult | None" = None,
                 vp_result: "volume_profile.ProfileResult | None" = None,
                 show_vp_density: bool = False, show_vp_peaks: bool = False) -> go.Figure:
    panels = ["price"]
    if config["rsi"]["enabled"]:
        panels.append("rsi")
    if config["macd"]["enabled"]:
        panels.append("macd")
    if config["bulls_power"]["enabled"] or config["bears_power"]["enabled"]:
        panels.append("power")
    if config["atr"]["enabled"]:
        panels.append("atr")

    n = len(panels)
    row_heights = [0.5] + [0.5 / (n - 1)] * (n - 1) if n > 1 else [1.0]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

    local = df.copy()
    local["time"] = local["time"].dt.tz_convert(sessions.local_tz())

    fig.add_trace(go.Candlestick(
        x=local["time"], open=local["open"], high=local["high"], low=local["low"], close=local["close"],
        increasing_line_color=BULL, decreasing_line_color=BEAR, name="price",
    ), row=1, col=1)

    if config["ma"]["enabled"]:
        label = f"{config['ma']['kind']} {config['ma']['period']}"
        fig.add_trace(go.Scatter(x=local["time"], y=local["ma"], mode="lines",
                                  line=dict(color="#f2a900", width=1.5), name=label), row=1, col=1)

    if smc_config and smc_result is not None:
        tz = sessions.local_tz()
        right_edge = local["time"].iloc[-1]
        interval = local["time"].diff().dropna().median()

        if smc_config.get("show_premium_discount_zones"):
            _add_zones(fig, smc_result, tz, right_edge)
        if smc_config.get("show_internal_order_blocks"):
            _add_order_blocks(fig, smc_result.internal_order_blocks, tz, right_edge, internal=True)
        if smc_config.get("show_swing_order_blocks"):
            _add_order_blocks(fig, smc_result.swing_order_blocks, tz, right_edge, internal=False)
        if smc_config.get("show_fair_value_gaps"):
            extend = interval * smc_config.get("fvg_extend_bars", 1) if pd.notna(interval) else pd.Timedelta(0)
            _add_fair_value_gaps(fig, smc_result.fair_value_gaps, tz, extend)
        if smc_config.get("show_internal_structure"):
            _add_structure_events(fig, smc_result.internal_structure, tz, internal=True)
        if smc_config.get("show_swing_structure"):
            _add_structure_events(fig, smc_result.swing_structure, tz, internal=False)
        if smc_config.get("show_equal_highs_lows"):
            _add_equal_levels(fig, smc_result.equal_highs + smc_result.equal_lows, tz)
        if smc_config.get("show_high_low_swings"):
            _add_strong_weak(fig, smc_result, tz, right_edge)
        if smc_config.get("show_daily_levels"):
            _add_mtf_level(fig, smc_result.daily_levels, tz, right_edge, "D", "#2157f3")
        if smc_config.get("show_weekly_levels"):
            _add_mtf_level(fig, smc_result.weekly_levels, tz, right_edge, "W", "#2157f3")
        if smc_config.get("show_monthly_levels"):
            _add_mtf_level(fig, smc_result.monthly_levels, tz, right_edge, "M", "#2157f3")

    if vob_result is not None:
        _add_volume_order_blocks(fig, vob_result, sessions.local_tz(), local["time"].iloc[-1])

    if vp_result is not None:
        left_edge, right_edge = local["time"].iloc[0], local["time"].iloc[-1]
        if show_vp_density:
            _add_volume_profile_density(fig, vp_result, left_edge, right_edge, right_edge)
        if show_vp_peaks:
            _add_volume_profile_peaks(fig, vp_result, left_edge, right_edge)

    qualifying = local[mask.values]
    if not qualifying.empty:
        is_long = direction == "Long setup"
        color = BULL if is_long else BEAR
        symbol = "triangle-up" if is_long else "triangle-down"
        y = qualifying["low"] * 0.999 if is_long else qualifying["high"] * 1.001
        fig.add_trace(go.Scatter(
            x=qualifying["time"], y=y, mode="markers",
            marker=dict(color=color, symbol=symbol, size=11, line=dict(width=1, color="white")),
            name="mock entry",
        ), row=1, col=1)

    row = 2
    if "rsi" in panels:
        fig.add_trace(go.Scatter(x=local["time"], y=local["rsi"], line=dict(color="#7e57c2"), name="RSI"), row=row, col=1)
        fig.add_hline(y=config["rsi"]["threshold"], line_dash="dot", row=row, col=1)
        fig.update_yaxes(range=[0, 100], row=row, col=1)
        row += 1

    if "macd" in panels:
        fig.add_trace(go.Scatter(x=local["time"], y=local["macd_line"], line=dict(color="#42a5f5"), name="MACD"), row=row, col=1)
        fig.add_trace(go.Scatter(x=local["time"], y=local["macd_signal"], line=dict(color="#ff7043"), name="Signal"), row=row, col=1)
        fig.add_trace(go.Bar(x=local["time"], y=local["macd_hist"], marker_color="#90a4ae", name="Histogram"), row=row, col=1)
        if config["macd"]["mode"] == "Histogram vs threshold":
            fig.add_hline(y=config["macd"]["threshold"], line_dash="dot", row=row, col=1)
        row += 1

    if "power" in panels:
        if config["bulls_power"]["enabled"]:
            fig.add_trace(go.Scatter(x=local["time"], y=local["bulls_power"], line=dict(color=BULL), name="Bulls power"), row=row, col=1)
            fig.add_hline(y=config["bulls_power"]["threshold"], line_dash="dot", line_color=BULL, row=row, col=1)
        if config["bears_power"]["enabled"]:
            fig.add_trace(go.Scatter(x=local["time"], y=local["bears_power"], line=dict(color=BEAR), name="Bears power"), row=row, col=1)
            fig.add_hline(y=config["bears_power"]["threshold"], line_dash="dot", line_color=BEAR, row=row, col=1)
        fig.add_hline(y=0, line_color="gray", row=row, col=1)
        row += 1

    if "atr" in panels:
        fig.add_trace(go.Scatter(x=local["time"], y=local["atr"], line=dict(color="#8d6e63"), name="ATR"), row=row, col=1)
        fig.add_hline(y=config["atr"]["threshold"], line_dash="dot", row=row, col=1)
        row += 1

    fig.update_layout(
        height=360 + 170 * (n - 1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ----------------------------------------------------------------- Sidebar --
st.sidebar.title("Indicator dashboard")

st.session_state.setdefault("recent_symbols", [DEFAULT_SYMBOL])

symbol = st.sidebar.selectbox(
    "Symbol", options=st.session_state["recent_symbols"], accept_new_options=True,
    placeholder="Enter a symbol, e.g. XAUUSDm", key="symbol_select",
)
if not symbol:
    st.info("Enter a symbol in the sidebar to get started.")
    st.stop()
if symbol not in st.session_state["recent_symbols"]:
    st.session_state["recent_symbols"].append(symbol)

timeframe = st.sidebar.selectbox("Timeframe", options=list(mt5_data.TIMEFRAME_MAP.keys()),
                                  index=list(mt5_data.TIMEFRAME_MAP.keys()).index("H1"))
bars = st.sidebar.slider("Bars of history", 100, 1000, 300, step=50)
direction = st.sidebar.segmented_control("Signal direction", ["Long setup", "Short setup"],
                                          default="Long setup", required=True)

ok, msg = mt5_data.connect(symbol)
if ok:
    st.sidebar.success(f"MT5 connected · {symbol}")
else:
    st.sidebar.error(msg)
    st.stop()

raw_df = mt5_data.get_rates(symbol, timeframe, bars=bars)
if raw_df.empty:
    st.warning(f"No data returned for {symbol} on {timeframe}.")
    st.stop()

# ---------------------------------------------------------- Constraint panel --
st.subheader("Entry constraints", divider="gray")

bp_range = preview_range(indicators.bulls_power(raw_df, 13))
brp_range = preview_range(indicators.bears_power(raw_df, 13))
_, _, hist_preview = indicators.macd(raw_df["close"])
macd_range = preview_range(hist_preview)
atr_range = preview_range(indicators.atr(raw_df, 14), floor_zero=True)

config = {}
row1 = st.columns(3)
config["bulls_power"] = indicator_card(row1[0], "Bulls power", "bp", bp_range, 13)
config["bears_power"] = indicator_card(row1[1], "Bears power", "brp", brp_range, 13)
config["rsi"] = indicator_card(row1[2], "RSI", "rsi", (0.0, 100.0), 14, default_ratio=0.5)

row2 = st.columns(3)
with row2[0], st.container(border=True):
    macd_enabled = st.toggle("MACD", key="macd_enabled")
    macd_mode = st.selectbox("Constraint mode", ["Histogram vs threshold", "MACD vs signal line"],
                              key="macd_mode", disabled=not macd_enabled)
    fast = st.slider("Fast EMA", 2, 50, 12, key="macd_fast", disabled=not macd_enabled)
    slow = st.slider("Slow EMA", 2, 100, 26, key="macd_slow", disabled=not macd_enabled)
    signal_period = st.slider("Signal EMA", 2, 50, 9, key="macd_signal_period", disabled=not macd_enabled)
    macd_operator = st.segmented_control("Condition", ["above", "below"], default="above", required=True,
                                          key="macd_operator", disabled=not macd_enabled)
    macd_threshold = st.slider("Histogram threshold", macd_range[0], macd_range[1],
                                (macd_range[0] + macd_range[1]) / 2, key="macd_threshold",
                                disabled=not macd_enabled or macd_mode != "Histogram vs threshold")
config["macd"] = {"enabled": macd_enabled, "mode": macd_mode, "fast": fast, "slow": slow,
                   "signal": signal_period, "operator": macd_operator, "threshold": macd_threshold}

with row2[1], st.container(border=True):
    ma_enabled = st.toggle("Moving average", key="ma_enabled")
    ma_kind = st.segmented_control("Type", ["EMA", "SMA"], default="EMA", required=True,
                                    key="ma_kind", disabled=not ma_enabled)
    ma_period = st.slider("Period", 2, 200, 50, key="ma_period", disabled=not ma_enabled)
    ma_operator = st.segmented_control("Price is", ["above", "below"], default="above", required=True,
                                        key="ma_operator", disabled=not ma_enabled)
config["ma"] = {"enabled": ma_enabled, "kind": ma_kind, "period": ma_period, "operator": ma_operator}

config["atr"] = indicator_card(row2[2], "ATR", "atr", atr_range, 14, default_ratio=0.5)

active = [k for k, cfg in config.items() if cfg["enabled"]]
if not active:
    st.info("Enable at least one indicator above to mark mock entries on the chart.")

# ------------------------------------------------------- Smart Money Concepts --
st.subheader("Smart Money Concepts", divider="gray")

smc_row1 = st.columns(3)
with smc_row1[0], st.container(border=True):
    show_swing_structure = st.toggle("Swing Structure", key="smc_swing_structure")
    show_internal_structure = st.toggle("Internal Structure", key="smc_internal_structure")
    swing_length = st.slider("Swing length", 10, 300, 50, key="smc_swing_length",
                              help="Also feeds swing order blocks and internal structure's reference pivots.")
    internal_confluence_filter = st.checkbox("Internal confluence filter", key="smc_confluence",
                                              disabled=not show_internal_structure)

with smc_row1[1], st.container(border=True):
    show_swing_obs = st.toggle("Swing Order Blocks", key="smc_swing_ob")
    show_internal_obs = st.toggle("Internal Order Blocks", key="smc_internal_ob")
    ob_filter = st.segmented_control("Filter", ["Atr", "Range"], default="Atr", required=True,
                                      key="smc_ob_filter", disabled=not (show_swing_obs or show_internal_obs))
    ob_mitigation = st.segmented_control("Mitigation", ["Close", "High/Low"], default="High/Low", required=True,
                                          key="smc_ob_mitigation", disabled=not (show_swing_obs or show_internal_obs))
    swing_ob_count = st.slider("Swing OB count", 1, 20, 5, key="smc_swing_ob_count", disabled=not show_swing_obs)
    internal_ob_count = st.slider("Internal OB count", 1, 20, 5, key="smc_internal_ob_count",
                                   disabled=not show_internal_obs)

with smc_row1[2], st.container(border=True):
    show_equal = st.toggle("Equal High/Low", key="smc_equal")
    equal_length = st.slider("Bars confirmation", 1, 10, 3, key="smc_equal_length", disabled=not show_equal)
    equal_threshold = st.slider("Threshold", 0.0, 0.5, 0.1, step=0.1, key="smc_equal_threshold",
                                 disabled=not show_equal)

smc_row2 = st.columns(3)
with smc_row2[0], st.container(border=True):
    show_fvg = st.toggle("Fair Value Gaps", key="smc_fvg")
    fvg_auto_threshold = st.checkbox("Auto threshold", value=True, key="smc_fvg_auto", disabled=not show_fvg)
    fvg_extend_bars = st.slider("Extend (bars)", 0, 20, 1, key="smc_fvg_extend", disabled=not show_fvg)

with smc_row2[1], st.container(border=True):
    show_zones = st.toggle("Premium/Discount Zones", key="smc_zones")

with smc_row2[2], st.container(border=True):
    show_high_low_swings = st.toggle("Strong/Weak High/Low", key="smc_strong_weak")

daily_available = not smc.higher_timeframe(timeframe, "D1")
weekly_available = not smc.higher_timeframe(timeframe, "W1")
monthly_available = not smc.higher_timeframe(timeframe, "MN1")

smc_row3 = st.columns(3)
with smc_row3[0], st.container(border=True):
    show_daily = st.toggle("Daily levels", key="smc_daily", disabled=not daily_available,
                            help=None if daily_available else "Chart timeframe is coarser than Daily")
with smc_row3[1], st.container(border=True):
    show_weekly = st.toggle("Weekly levels", key="smc_weekly", disabled=not weekly_available,
                             help=None if weekly_available else "Chart timeframe is coarser than Weekly")
with smc_row3[2], st.container(border=True):
    show_monthly = st.toggle("Monthly levels", key="smc_monthly", disabled=not monthly_available,
                              help=None if monthly_available else "Chart timeframe is coarser than Monthly")

smc_config = {
    "show_swing_structure": show_swing_structure,
    "show_internal_structure": show_internal_structure,
    "swing_length": swing_length,
    "internal_confluence_filter": internal_confluence_filter,
    "show_swing_order_blocks": show_swing_obs,
    "show_internal_order_blocks": show_internal_obs,
    "order_block_filter": ob_filter,
    "order_block_mitigation": ob_mitigation,
    "swing_order_blocks_count": swing_ob_count,
    "internal_order_blocks_count": internal_ob_count,
    "show_equal_highs_lows": show_equal,
    "equal_length": equal_length,
    "equal_threshold": equal_threshold,
    "show_fair_value_gaps": show_fvg,
    "fvg_auto_threshold": fvg_auto_threshold,
    "fvg_extend_bars": fvg_extend_bars,
    "show_premium_discount_zones": show_zones,
    "show_high_low_swings": show_high_low_swings,
    "show_daily_levels": show_daily,
    "show_weekly_levels": show_weekly,
    "show_monthly_levels": show_monthly,
}

smc_active = any(smc_config[k] for k in (
    "show_swing_structure", "show_internal_structure", "show_swing_order_blocks",
    "show_internal_order_blocks", "show_equal_highs_lows", "show_fair_value_gaps",
    "show_premium_discount_zones", "show_high_low_swings", "show_daily_levels",
    "show_weekly_levels", "show_monthly_levels",
))

# ----------------------------------------------------------- Order blocks --
st.subheader("Order Block Detector", divider="gray")

vob_row = st.columns(2)
with vob_row[0], st.container(border=True):
    show_vol_obs = st.toggle("Show Order Blocks", key="vob_enabled")
    vob_length = st.slider("Volume pivot length", 1, 30, 5, key="vob_length", disabled=not show_vol_obs)
    vob_mitigation = st.segmented_control("Mitigation", ["Wick", "Close"], default="Wick", required=True,
                                           key="vob_mitigation", disabled=not show_vol_obs)
with vob_row[1], st.container(border=True):
    vob_bull_count = st.slider("Bullish OB count", 1, 10, 3, key="vob_bull_count", disabled=not show_vol_obs)
    vob_bear_count = st.slider("Bearish OB count", 1, 10, 3, key="vob_bear_count", disabled=not show_vol_obs)

vob_config = {"length": vob_length, "mitigation": vob_mitigation,
              "bull_ext_last": vob_bull_count, "bear_ext_last": vob_bear_count}

# -------------------------------------------------------------- Volume profile --
st.subheader("Volume Profile", divider="gray")

vp_row = st.columns(2)
with vp_row[0], st.container(border=True):
    show_vp_density = st.toggle("Show Volume Profile", key="vp_density",
                                 help="Sideways KDE of tick_volume by price, pinned to the chart's right edge.")
    vp_n_peaks = st.slider("Peaks to track", 1, 10, 6, key="vp_n_peaks")
with vp_row[1], st.container(border=True):
    show_vp_peaks = st.toggle("Show KDE Peak Levels", key="vp_peak_levels",
                               help="Horizontal lines at the highest-activity price levels.")

# ------------------------------------------------- KDE peak strategy sweep --
st.subheader("KDE Peak Strategy — Timeframe Comparison", divider="gray")
st.caption("No-lookahead backtest: long-only tap entries at the price levels where a causal, expanding "
           "volume-weighted KDE peaks, run across every timeframe to find which one performs best. "
           "This is expensive (can take a few minutes) and runs only when you click the button below - "
           "it is not part of the 15s auto-refresh.")

run_sweep = st.button("Run timeframe comparison")
if run_sweep:
    progress = st.progress(0.0)

    def _sweep_progress(i, n, tf):
        progress.progress(i / n, text=f"Backtesting {tf}...")

    with st.spinner(f"Running the KDE-peak backtest across every timeframe for {symbol}..."):
        st.session_state["kde_sweep_results"] = backtest_kde.sweep_timeframes(symbol, progress_cb=_sweep_progress)
        st.session_state["kde_sweep_symbol"] = symbol
    progress.empty()

sweep_results = st.session_state.get("kde_sweep_results")
if sweep_results and st.session_state.get("kde_sweep_symbol") == symbol:
    sweep_rows = []
    for r in sweep_results:
        if "error" in r:
            sweep_rows.append({"Timeframe": r["timeframe"], "Trades": "-", "Win rate": "-", "Avg R": "-",
                                "Total R": "-", "Max DD": "-", "Score": "-", "Note": r["error"]})
        else:
            sweep_rows.append({
                "Timeframe": r["timeframe"], "Trades": r["total_trades"],
                "Win rate": f'{r["win_rate"] * 100:.1f}%' if r["win_rate"] is not None else "-",
                "Avg R": f'{r["avg_r"]:.2f}' if r["avg_r"] is not None else "-",
                "Total R": f'{r["total_r"]:+.1f}', "Max DD": f'{r["max_drawdown_r"]:.1f}',
                "Score": f'{r["score"]:.2f}', "Note": "",
            })
    st.dataframe(pd.DataFrame(sweep_rows), width="stretch")

    best = next((r for r in sweep_results if "error" not in r), None)
    if best:
        wr = f'{best["win_rate"] * 100:.1f}% win rate, ' if best["win_rate"] is not None else ""
        st.success(f"Best timeframe: **{best['timeframe']}** — score {best['score']:.2f} "
                   f"({best['total_trades']} trades, {wr}{best['total_r']:+.1f}R total)")

# ------------------------------------------------------------------- Chart --
# Re-fetches fresh candles and rebuilds the chart every 15s on its own,
# without rerunning (and losing scroll/zoom state on) the rest of the page.


@st.fragment(run_every="15s")
def render_chart():
    raw_df = mt5_data.get_rates(symbol, timeframe, bars=bars)
    if raw_df.empty:
        st.warning(f"No data returned for {symbol} on {timeframe}.")
        return

    smc_result = None
    if smc_active:
        smc_result = smc.compute_smc(raw_df, smc_config)
        if show_daily:
            smc_result.daily_levels = smc.mtf_levels(symbol, "D1")
        if show_weekly:
            smc_result.weekly_levels = smc.mtf_levels(symbol, "W1")
        if show_monthly:
            smc_result.monthly_levels = smc.mtf_levels(symbol, "MN1")

    vob_result = order_blocks.compute_order_blocks(raw_df, vob_config) if show_vol_obs else None

    vp_result = None
    if show_vp_density or show_vp_peaks:
        vp_result = volume_profile.compute_profile(raw_df, n_peaks=vp_n_peaks)

    df = indicators.add_indicators(raw_df, config)
    mask = indicators.evaluate_constraints(df, config)

    now = pd.Timestamp.now(tz=sessions.local_tz()).strftime("%H:%M:%S")
    st.caption(f"{int(mask.sum())} of {len(df)} bars qualify as a {direction.split()[0].lower()} mock entry "
               f"under the current constraints. Auto-refreshing every 15s - last update {now}.")
    st.plotly_chart(build_chart(df, config, mask, direction, smc_config, smc_result, vob_result,
                                 vp_result, show_vp_density, show_vp_peaks), width="stretch")


render_chart()
