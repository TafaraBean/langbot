"""MetaTrader 5 trading API wrapper: account info, symbol specs, position
sizing, order placement, partial/full close, and position listing.

Every order this module sends is tagged with MAGIC_NUMBER so the live engine
can always tell its own positions apart from anything the user places by hand
in the terminal."""

import math

import MetaTrader5 as mt5

MAGIC_NUMBER = 750007
DEVIATION_POINTS = 20

_RETRY_RETCODES = {
    mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_CHANGED,
    mt5.TRADE_RETCODE_PRICE_OFF, mt5.TRADE_RETCODE_TIMEOUT, mt5.TRADE_RETCODE_CONNECTION,
}
_RETCODE_NAMES = {v: k for k, v in vars(mt5).items() if k.startswith("TRADE_RETCODE_")}


def describe_retcode(retcode: int | None) -> str:
    if retcode is None:
        return "no retcode"
    return f"{retcode} {_RETCODE_NAMES.get(retcode, 'UNKNOWN')}"


def get_account_info() -> dict | None:
    info = mt5.account_info()
    if info is None:
        return None
    return {
        "login": info.login, "server": info.server, "balance": info.balance,
        "equity": info.equity, "margin_free": info.margin_free,
        "currency": info.currency, "trade_mode": info.trade_mode,
    }


def is_demo_account() -> tuple[bool, str]:
    acct = get_account_info()
    if acct is None:
        return False, "No account info available - is the MT5 terminal connected and logged in?"
    if acct["trade_mode"] != mt5.ACCOUNT_TRADE_MODE_DEMO:
        return False, (f"Connected account #{acct['login']} on {acct['server']} is NOT a demo "
                        "account - live trading is disabled.")
    return True, f"Demo account #{acct['login']} on {acct['server']} - balance {acct['balance']:.2f} {acct['currency']}"


def get_symbol_spec(symbol: str) -> dict | None:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return {
        "volume_min": info.volume_min, "volume_max": info.volume_max, "volume_step": info.volume_step,
        "trade_contract_size": info.trade_contract_size, "trade_tick_value": info.trade_tick_value,
        "trade_tick_size": info.trade_tick_size, "digits": info.digits, "point": info.point,
        "filling_mode": info.filling_mode,
    }


def _round_volume(volume: float, spec: dict) -> float:
    step = spec["volume_step"] or 0.01
    steps = math.floor(volume / step + 1e-9)
    return round(steps * step, 8)


def calc_lot_size(symbol: str, risk_amount: float, sl_distance_price: float,
                   spec: dict | None = None) -> float | None:
    """lot = risk_amount / (loss per 1.0 lot at sl_distance_price), rounded down
    to volume_step and clamped to [volume_min, volume_max]. Returns None (not
    0.0) when the trade can't be sized at all, so callers can tell "unsizeable"
    apart from "sized down to zero"."""
    if risk_amount is None or risk_amount <= 0 or sl_distance_price is None or sl_distance_price <= 0:
        return None
    spec = spec or get_symbol_spec(symbol)
    if not spec or not spec.get("trade_tick_size") or not spec.get("trade_tick_value"):
        return None

    ticks = sl_distance_price / spec["trade_tick_size"]
    loss_per_lot = ticks * spec["trade_tick_value"]
    if loss_per_lot <= 0:
        return None

    volume = _round_volume(risk_amount / loss_per_lot, spec)
    volume = max(spec["volume_min"], min(spec["volume_max"], volume))
    if volume < spec["volume_min"]:
        return None
    return volume


def _filling_mode(spec: dict | None) -> int:
    """SYMBOL_FILLING_MODE is a bitmask (bit0=FOK, bit1=IOC support). Falls
    back to IOC, the most broadly accepted mode, when spec is unavailable."""
    if not spec:
        return mt5.ORDER_FILLING_IOC
    mode = spec.get("filling_mode") or 0
    if mode & 1:
        return mt5.ORDER_FILLING_FOK
    if mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def _build_request(order_type: int, symbol: str, volume: float, price: float, sl: float | None,
                    tp: float | None, comment: str, filling: int, ticket: int | None = None) -> dict:
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION_POINTS,
        "magic": MAGIC_NUMBER,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    if sl:
        request["sl"] = sl
    if tp:
        request["tp"] = tp
    if ticket is not None:
        request["position"] = ticket
    return request


def _send_checked(request: dict, symbol: str, is_buy: bool):
    """order_check first (best-effort screen - its retcode semantics vary by
    broker, so this only blocks the send on a code we recognize as fatal;
    order_send's own retcode remains authoritative), then order_send with one
    retry on a stale-price/timeout retcode and one retry with the alternate
    filling mode on INVALID_FILL."""
    check = mt5.order_check(request)
    if check is not None and check.retcode not in (0, mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        return check

    result = mt5.order_send(request)
    if result is None:
        return None

    if result.retcode in _RETRY_RETCODES:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            request["price"] = tick.ask if is_buy else tick.bid
            retry = mt5.order_send(request)
            if retry is not None:
                result = retry
    elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
        request["type_filling"] = (mt5.ORDER_FILLING_IOC if request["type_filling"] == mt5.ORDER_FILLING_FOK
                                    else mt5.ORDER_FILLING_FOK)
        retry = mt5.order_send(request)
        if retry is not None:
            result = retry

    return result


def _format_result(result) -> dict:
    if result is None:
        return {"ok": False, "retcode": None,
                "retcode_desc": f"order_send returned None (last_error={mt5.last_error()})",
                "ticket": None, "fill_price": None, "fill_volume": None}
    ok = result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL)
    return {
        "ok": ok, "retcode": result.retcode, "retcode_desc": describe_retcode(result.retcode),
        "ticket": getattr(result, "order", None),
        "fill_price": getattr(result, "price", None), "fill_volume": getattr(result, "volume", None),
    }


def place_market_order(symbol: str, direction: str, volume: float, sl_price: float | None,
                        tp_price: float | None, comment: str) -> dict:
    """direction: 'bullish' -> BUY, 'bearish' -> SELL. Resolves the actual
    opened position's ticket via get_open_positions afterwards rather than
    trusting order_send's own 'order' ticket to equal the position ticket,
    since that mapping isn't guaranteed identical across netting/hedging
    account modes."""
    spec = get_symbol_spec(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"ok": False, "retcode": None, "retcode_desc": "no tick data for symbol",
                "ticket": None, "fill_price": None, "fill_volume": None}

    is_buy = direction == "bullish"
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = tick.ask if is_buy else tick.bid
    filling = _filling_mode(spec)

    request = _build_request(order_type, symbol, volume, price, sl_price, tp_price, comment, filling)
    formatted = _format_result(_send_checked(request, symbol, is_buy))

    if formatted["ok"]:
        positions = get_open_positions(symbol)
        if positions:
            formatted["ticket"] = max(positions, key=lambda p: p["time"])["ticket"]

    return formatted


def _position_by_ticket(ticket: int) -> dict | None:
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return None
    p = positions[0]
    if p.magic != MAGIC_NUMBER:
        return None
    return {"ticket": p.ticket, "symbol": p.symbol, "type": p.type, "volume": p.volume,
            "price_open": p.price_open, "sl": p.sl, "tp": p.tp, "profit": p.profit, "time": p.time}


def close_partial(ticket: int, symbol: str, volume_to_close: float, comment: str) -> dict:
    """Re-reads the position's live remaining volume immediately before
    sending, so a requested close can never exceed what's actually still
    open (protects against a stale ladder-progress caller asking to close
    more than remains after an earlier leg already fired)."""
    pos = _position_by_ticket(ticket)
    if pos is None:
        return {"ok": False, "retcode": None, "retcode_desc": "position not found (already closed?)",
                "ticket": ticket, "fill_price": None, "fill_volume": None, "closed_volume": 0.0}

    spec = get_symbol_spec(symbol)
    volume = min(volume_to_close, pos["volume"])
    if spec:
        volume = _round_volume(volume, spec)
    if volume <= 0:
        return {"ok": False, "retcode": None, "retcode_desc": "nothing left to close",
                "ticket": ticket, "fill_price": None, "fill_volume": None, "closed_volume": 0.0}

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"ok": False, "retcode": None, "retcode_desc": "no tick data for symbol",
                "ticket": ticket, "fill_price": None, "fill_volume": None, "closed_volume": 0.0}

    is_buy_position = pos["type"] == mt5.POSITION_TYPE_BUY
    order_type = mt5.ORDER_TYPE_SELL if is_buy_position else mt5.ORDER_TYPE_BUY
    price = tick.bid if is_buy_position else tick.ask
    filling = _filling_mode(spec)

    request = _build_request(order_type, symbol, volume, price, None, None, comment, filling, ticket=ticket)
    formatted = _format_result(_send_checked(request, symbol, not is_buy_position))
    formatted["closed_volume"] = formatted["fill_volume"] or 0.0
    return formatted


def close_full(ticket: int, symbol: str, comment: str) -> dict:
    pos = _position_by_ticket(ticket)
    if pos is None:
        return {"ok": False, "retcode": None, "retcode_desc": "position not found (already closed?)",
                "ticket": ticket, "fill_price": None, "fill_volume": None, "closed_volume": 0.0}
    return close_partial(ticket, symbol, pos["volume"], comment)


def get_open_positions(symbol: str | None = None) -> list[dict]:
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if not positions:
        return []
    return [
        {"ticket": p.ticket, "symbol": p.symbol, "type": p.type, "volume": p.volume,
         "price_open": p.price_open, "sl": p.sl, "tp": p.tp, "profit": p.profit,
         "time": p.time, "comment": p.comment}
        for p in positions if p.magic == MAGIC_NUMBER
    ]


def get_closing_deals(ticket: int) -> list[dict]:
    """Realized exit deal(s) for a position ticket - used by reconciliation
    to price the actual fills once a tracked position is no longer open."""
    deals = mt5.history_deals_get(position=ticket)
    if not deals:
        return []
    return [
        {"ticket": d.ticket, "order": d.order, "time": d.time, "type": d.type,
         "volume": d.volume, "price": d.price, "profit": d.profit,
         "commission": d.commission, "swap": d.swap}
        for d in deals if d.entry == mt5.DEAL_ENTRY_OUT
    ]
