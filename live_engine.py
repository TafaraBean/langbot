"""Per-cycle live-trading orchestration for Tab 7.

Reconciles persisted state against MT5's actual open positions, detects
fresh confirmations using the exact same Bias -> Narrative -> POI ->
Confirmation pipeline the backtest tabs use, sizes and places real orders,
monitors/partial-closes open positions against the TP ladder, and journals
completed trades to a separate live journal.

`prepare_symbol_data`/`run_mock_trade`/`poi_to_active_dict` are defined in
app.py and passed in by the caller rather than imported here, to avoid a
circular import (app.py is the one that imports this module)."""

import json
from datetime import datetime, timezone

import journal
import live_state
import mt5_data
import mt5_trade
import structure

LIVE_JOURNAL_PATH = "live_journal.csv"
SYMBOLS = ["XAUUSDm", "XAGUSDm", "USTECm"]
POI_TF, CONF_TF = "M15", "M5"

# Best combination from the user's Tab 6 parameter-optimizer run (risk-adjusted
# score 7.32) - the actual pip/fraction lists stay owned by app.py's
# TP_LADDER_PRESETS/TP_FRACTION_PRESETS dicts (single source of truth, same
# ones Tab 6 sweeps), these are just the default preset names/values Tab 7's
# UI seeds itself with.
DEFAULT_MIN_RR = 2.0
DEFAULT_MIN_HP_SCORE = 2
DEFAULT_TP_LADDER_PRESET = "Tight (10/20/30/50)"
DEFAULT_TP_FRACTION_PRESET = "Back-loaded (5/5/10/30/50)"


def last_closed_bar(symbol: str, tf_key: str = CONF_TF):
    """Position 0 from copy_rates_from_pos (mt5_data.get_rates) is the still-
    forming candle, so the last FULLY CLOSED bar is row 1, not row 0."""
    df = mt5_data.get_rates(symbol, tf_key, bars=2)
    if len(df) < 2:
        return None
    return df["time"].iloc[-2]


def current_bias_snapshot(symbol: str) -> dict:
    """Same per-timeframe classify_bias/narrative_score pattern as Tab 1,
    used for the Tab 7 status row - cheap, mt5_data.get_rates is a 30s cache
    hit most cycles."""
    biases = {tf: structure.classify_bias(structure.find_swings(mt5_data.get_rates(symbol, tf, bars=200)))
              for tf in structure.TF_ORDER}
    count, label, _strategy = structure.narrative_score(biases)
    return {"biases": biases, "narrative_count": count, "narrative_label": label}


def detect_signal(symbol: str, state: dict, prepare_symbol_data, run_mock_trade, poi_to_active_dict,
                   min_rr: float, min_hp_score: int, tp_pips: list[float] | None,
                   tp_fractions: list[float] | None):
    """Runs the exact backtest signal-detection pipeline against CURRENT
    market state (point_in_time_bias=False - the live-bias path Tabs 1-3
    use, not Tab 6's historical point-in-time replay). Accepts only a
    confirmation on the LATEST candle in the fetched window - not any older
    signal inside run_mock_trade's search range - that hasn't already been
    acted on. Returns (poi_active, mt_result, dedupe_key) or None."""
    prepared = prepare_symbol_data(symbol, POI_TF, CONF_TF, point_in_time_bias=False)
    for poi in prepared["quality_candidates"]:
        if min_hp_score and poi.get("hp") and poi["hp"]["score"] < min_hp_score:
            continue
        poi_active = poi_to_active_dict(poi, POI_TF, prepared["swings"], prepared["equal_levels"],
                                         prepared["liq_levels"], prepared["obs"], prepared["fvgs"])
        mt = run_mock_trade(symbol, CONF_TF, poi_active, min_rr, tp_pips=tp_pips, tp_fractions=tp_fractions)
        if mt["status"] != "ok" or mt["entry_idx"] != len(mt["df"]) - 1:
            continue
        key = live_state.signal_key(symbol, poi_active, mt["sig"])
        if live_state.signal_already_acted(state, symbol, key):
            continue
        return poi_active, mt, key
    return None


def size_and_enter(symbol: str, poi_active: dict, mt_result: dict, risk_pct: float,
                    armed: bool, state: dict, tp_fractions: list[float] | None) -> dict:
    """Sizes off the REAL fillable price (current tick), not
    mt_result['entry_price'] (that's the historical next-candle-open used
    for backtest fidelity). Places a real order when armed; otherwise
    returns a dry-run preview only, without touching live_state."""
    tick = mt5_data.symbol_info_tick(symbol)
    if tick is None:
        return {"ok": False, "reason": "no live tick data for symbol"}

    direction = mt_result["direction"]
    live_entry = tick.ask if direction == "bullish" else tick.bid
    stop_price = mt_result["stop_price"]
    sl_distance = abs(live_entry - stop_price)

    account = mt5_trade.get_account_info()
    if account is None:
        return {"ok": False, "reason": "no account info available"}

    risk_amount = account["balance"] * risk_pct / 100
    spec = mt5_trade.get_symbol_spec(symbol)
    volume = mt5_trade.calc_lot_size(symbol, risk_amount, sl_distance, spec)
    if volume is None:
        return {"ok": False, "reason": "could not size a valid lot for this risk/SL/symbol combination"}

    partial = structure.pip_size(symbol) is not None
    tp_price = None if partial else mt_result["target"]["price"]

    if not armed:
        return {"ok": True, "dry_run": True, "direction": direction, "volume": volume,
                "entry": live_entry, "stop": stop_price, "tp": tp_price, "partial": partial}

    comment = f"75x live {symbol[:12]}"
    result = mt5_trade.place_market_order(symbol, direction, volume, stop_price, tp_price, comment)
    if not result["ok"]:
        return {"ok": False, "reason": result["retcode_desc"]}

    position = {
        "ticket": result["ticket"], "direction": direction,
        "entry_price": result["fill_price"] or live_entry, "stop_price": stop_price,
        "volume": result["fill_volume"] or volume,
        "ladder": mt_result["ladder"] if partial else None, "closed_legs": [],
        "tp_fractions": tp_fractions if partial else None,
        "poi_type": poi_active["poi_type"], "poi_price": poi_active["poi_price"],
        "confirmation_type": mt_result["sig"]["kind"],
        "target_price": tp_price,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    live_state.set_position(state, symbol, position)
    return {"ok": True, "dry_run": False, "ticket": result["ticket"], "direction": direction,
            "volume": position["volume"], "entry": position["entry_price"], "stop": stop_price}


def monitor_position(symbol: str, state: dict, armed: bool) -> dict | None:
    """Checks ALL not-yet-closed TP-ladder legs against the current tick
    every cycle (not just "the next one") - after a restart or a slow cycle,
    price may have already crossed more than one level, and closing every
    newly-crossed leg in one pass avoids permanently stranding a leg. A
    symbol with no ladder (USTECm) just reports floating P/L; its exit is
    governed entirely by the broker-side SL/TP already on the order."""
    sym_state = state["symbols"].get(symbol) or {}
    position = sym_state.get("position")
    if not position:
        return None

    live_positions = {p["ticket"]: p for p in mt5_trade.get_open_positions(symbol)}
    live_pos = live_positions.get(position["ticket"])
    if live_pos is None:
        return {"status": "closed_pending_reconcile"}

    status = {"status": "open", "ticket": position["ticket"], "direction": position["direction"],
              "volume_open": live_pos["volume"], "entry": position["entry_price"],
              "floating_pl": live_pos["profit"]}

    ladder = position.get("ladder")
    tick = mt5_data.symbol_info_tick(symbol) if ladder else None
    if not ladder or tick is None:
        status["next_leg"] = None
        return status

    closed_legs = set(position.get("closed_legs") or [])
    direction = position["direction"]
    exit_price = tick.bid if direction == "bullish" else tick.ask

    # Use the fractions actually in effect when THIS position's ladder was
    # built, not whatever preset Tab 7's UI happens to show right now - the
    # two can diverge if the user changes the preset while a position is open.
    fractions = position.get("tp_fractions") or structure.DEFAULT_TP_FRACTIONS
    label_to_frac = {leg["label"]: (fractions[i] if i < len(fractions) else 0.0) for i, leg in enumerate(ladder)}
    remaining_fraction = 1.0 - sum(label_to_frac.get(l, 0.0) for l in closed_legs)
    remaining_volume = live_pos["volume"]

    would_close = []
    for leg in ladder:
        if leg["label"] in closed_legs or remaining_fraction <= 1e-9:
            continue
        reached = (exit_price >= leg["price"]) if direction == "bullish" else (exit_price <= leg["price"])
        if not reached:
            continue

        frac = label_to_frac.get(leg["label"], 0.0)
        vol_to_close = remaining_volume * min(1.0, frac / remaining_fraction) if remaining_fraction else remaining_volume

        if not armed:
            would_close.append(leg["label"])
            continue

        result = mt5_trade.close_partial(position["ticket"], symbol, vol_to_close, comment=leg["label"][:31])
        if result["ok"]:
            live_state.mark_legs_closed(state, symbol, [leg["label"]])
            closed_legs.add(leg["label"])
            remaining_fraction -= frac
            remaining_volume = max(0.0, remaining_volume - result["closed_volume"])

    if would_close:
        status["would_close"] = would_close
    status["next_leg"] = next((leg["label"] for leg in ladder if leg["label"] not in closed_legs), None)
    return status


def reconcile_and_journal(symbol: str, state: dict) -> dict | None:
    """For a position tracked in state but no longer open in MT5, prices the
    realized outcome from the actual closing deal(s) and logs it to the live
    journal. MT5's positions_get() stays the source of truth for "is it
    still open" throughout - this never guesses."""
    sym_state = state["symbols"].get(symbol) or {}
    position = sym_state.get("position")
    if not position:
        return None

    live_tickets = {p["ticket"] for p in mt5_trade.get_open_positions(symbol)}
    if position["ticket"] in live_tickets:
        return None

    deals = mt5_trade.get_closing_deals(position["ticket"])
    if not deals:
        return {"status": "pending_history"}

    entry_price, stop_price = position["entry_price"], position["stop_price"]
    direction = position["direction"]
    risk = abs(entry_price - stop_price)
    original_volume = position.get("volume") or sum(d["volume"] for d in deals)

    total_r = 0.0
    if risk and original_volume:
        for d in deals:
            vol_fraction = d["volume"] / original_volume
            signed_move = (d["price"] - entry_price) if direction == "bullish" else (entry_price - d["price"])
            total_r += vol_fraction * (signed_move / risk)

    outcome = "Win" if total_r > 1e-9 else ("Loss" if total_r < -1e-9 else "BE")
    ladder = position.get("ladder")

    journal.append_trade({
        "symbol": symbol, "direction": "Buy" if direction == "bullish" else "Sell",
        "poi_type": position.get("poi_type", ""), "poi_price": position.get("poi_price", ""),
        "confirmation_type": position.get("confirmation_type", ""),
        "entry": entry_price, "stop_loss": stop_price,
        "take_profit": position.get("target_price") or (ladder[-1]["price"] if ladder else ""),
        "outcome": outcome, "pnl_r": round(total_r, 2),
        "sl_basis": "15M/1H structural" if ladder else "OB/FVG zone or LTF confirmation candle",
        "tp_ladder": json.dumps(ladder) if ladder else "",
        "partial_legs": json.dumps(deals, default=str),
        "notes": f"Live trade (MT5 ticket {position['ticket']})",
    }, path=LIVE_JOURNAL_PATH)

    live_state.clear_position(state, symbol)
    return {"status": "journaled", "outcome": outcome, "pnl_r": round(total_r, 2)}


def run_cycle(state: dict, risk_pct: float, armed: bool, prepare_symbol_data, run_mock_trade,
              poi_to_active_dict, min_rr: float, min_hp_score: int, tp_pips: list[float] | None,
              tp_fractions: list[float] | None) -> dict:
    """Top-level per-fragment-tick entry point. Reconciles first (always,
    even unarmed, since a position opened in an earlier armed session must
    still be watched), then either monitors an open position or - gated on a
    real new M5 candle close - runs detection for a fresh signal."""
    demo_ok, demo_msg = mt5_trade.is_demo_account()
    if armed and not demo_ok:
        armed = False

    results = {}
    for symbol in SYMBOLS:
        sym_result = {}

        recon = reconcile_and_journal(symbol, state)
        if recon:
            sym_result["reconcile"] = recon

        sym_state = state["symbols"].get(symbol) or {}
        if sym_state.get("position"):
            sym_result["position_status"] = monitor_position(symbol, state, armed)
        else:
            bar_time = last_closed_bar(symbol, CONF_TF)
            sym_result["last_bar"] = str(bar_time) if bar_time is not None else None
            sym_result["signal"] = None
            if bar_time is not None and not live_state.bar_already_processed(state, symbol, bar_time):
                live_state.mark_bar_processed(state, symbol, bar_time)
                found = detect_signal(symbol, state, prepare_symbol_data, run_mock_trade, poi_to_active_dict,
                                       min_rr, min_hp_score, tp_pips, tp_fractions)
                if found:
                    poi_active, mt_result, key = found
                    # Mark acted regardless of `armed`, so an unarmed dry-run
                    # cycle doesn't re-log the same signal forever once the
                    # user arms later mid-candle.
                    live_state.mark_signal_acted(state, symbol, key)
                    sym_result["signal"] = size_and_enter(symbol, poi_active, mt_result, risk_pct, armed, state,
                                                           tp_fractions)

        sym_result["bias"] = current_bias_snapshot(symbol)
        results[symbol] = sym_result

    live_state.save_state(state)
    return {"per_symbol": results, "demo_ok": demo_ok, "demo_msg": demo_msg, "armed": armed}
