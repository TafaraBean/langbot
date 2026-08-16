"""JSON-backed persisted state for the live trading engine.

MT5's own `positions_get()` is always the source of truth for what's actually
open (see mt5_trade.py) - this file only tracks bookkeeping MT5 has no
concept of: which candle/signal was already acted on (dedupe) and which
TP-ladder legs have already been closed for a tracked position. It must
survive process restarts, so unlike mt5_data.get_rates it is never cached
and is written atomically.
"""

import hashlib
import json
import os
from copy import deepcopy

LIVE_STATE_PATH = "live_state.json"
MAX_ACTED_SIGNALS = 200


def _empty_symbol_state() -> dict:
    return {"last_closed_bar_time": None, "acted_signals": [], "position": None}


def default_state() -> dict:
    return {"armed": False, "risk_pct": 1.0, "symbols": {}}


def _deep_merge_defaults(state: dict) -> dict:
    base = default_state()
    base.update({k: v for k, v in state.items() if k != "symbols"})
    base["symbols"] = {}
    for symbol, sym_state in (state.get("symbols") or {}).items():
        merged = _empty_symbol_state()
        merged.update(sym_state)
        base["symbols"][symbol] = merged
    return base


def load_state(path: str = LIVE_STATE_PATH) -> dict:
    """Loads persisted state, deep-merged onto defaults for forward-compat
    with new keys. ALWAYS forces armed=False regardless of what was
    persisted - a restart or fragment rerun must never silently resume live
    order placement; the user must re-arm explicitly every session."""
    if not os.path.exists(path):
        return default_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default_state()

    state = _deep_merge_defaults(raw)
    state["armed"] = False
    return state


def save_state(state: dict, path: str = LIVE_STATE_PATH) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _ensure_symbol(state: dict, symbol: str) -> dict:
    return state["symbols"].setdefault(symbol, _empty_symbol_state())


def _iso(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def signal_key(symbol: str, poi_active: dict, sig: dict) -> str:
    """Stable dedupe key for one specific confirmation on one specific POI,
    so the same signal is never re-entered across cycles or after a restart."""
    payload = {
        "symbol": symbol,
        "zone_low": poi_active.get("zone_low"),
        "zone_high": poi_active.get("zone_high"),
        "formed_time": _iso(poi_active.get("formed_time")),
        "sig_kind": sig.get("kind"),
        "sig_time": _iso(sig.get("time")),
        "sig_idx": sig.get("idx"),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def bar_already_processed(state: dict, symbol: str, bar_time) -> bool:
    sym_state = state["symbols"].get(symbol) or {}
    return sym_state.get("last_closed_bar_time") == _iso(bar_time)


def mark_bar_processed(state: dict, symbol: str, bar_time) -> None:
    _ensure_symbol(state, symbol)["last_closed_bar_time"] = _iso(bar_time)


def signal_already_acted(state: dict, symbol: str, key: str) -> bool:
    sym_state = state["symbols"].get(symbol) or {}
    return key in (sym_state.get("acted_signals") or [])


def mark_signal_acted(state: dict, symbol: str, key: str) -> None:
    sym_state = _ensure_symbol(state, symbol)
    acted = sym_state.setdefault("acted_signals", [])
    if key not in acted:
        acted.append(key)
    if len(acted) > MAX_ACTED_SIGNALS:
        sym_state["acted_signals"] = acted[-MAX_ACTED_SIGNALS:]


def set_position(state: dict, symbol: str, position: dict) -> None:
    _ensure_symbol(state, symbol)["position"] = deepcopy(position)


def clear_position(state: dict, symbol: str) -> None:
    _ensure_symbol(state, symbol)["position"] = None


def mark_legs_closed(state: dict, symbol: str, labels: list[str]) -> None:
    sym_state = _ensure_symbol(state, symbol)
    position = sym_state.get("position")
    if not position:
        return
    closed = position.setdefault("closed_legs", [])
    for label in labels:
        if label not in closed:
            closed.append(label)
