"""MetaTrader 5 connection lifecycle and OHLC data helpers."""

import MetaTrader5 as mt5
import pandas as pd
import streamlit as st

TIMEFRAME_MAP = {
    "D1": mt5.TIMEFRAME_D1,
    "H4": mt5.TIMEFRAME_H4,
    "H1": mt5.TIMEFRAME_H1,
    "M15": mt5.TIMEFRAME_M15,
    "M5": mt5.TIMEFRAME_M5,
    "M3": mt5.TIMEFRAME_M3,
    "M1": mt5.TIMEFRAME_M1,
}


def connect(symbol: str) -> tuple[bool, str]:
    """Initialize the MT5 terminal connection and ensure the symbol is visible.

    Returns (ok, message).
    """
    if not mt5.initialize():
        return False, f"mt5.initialize() failed: {mt5.last_error()}"

    if not mt5.symbol_select(symbol, True):
        return False, f"Symbol '{symbol}' not found / could not be added to Market Watch: {mt5.last_error()}"

    return True, "connected"


def shutdown() -> None:
    mt5.shutdown()


@st.cache_data(ttl=30, show_spinner=False)
def get_rates(symbol: str, timeframe_key: str, bars: int = 300) -> pd.DataFrame:
    """Fetch the most recent `bars` candles for symbol/timeframe as a DataFrame."""
    timeframe = TIMEFRAME_MAP[timeframe_key]
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def symbol_info_tick(symbol: str):
    return mt5.symbol_info_tick(symbol)


@st.cache_data(ttl=30, show_spinner=False)
def get_rates_range(symbol: str, timeframe_key: str, date_from, date_to) -> pd.DataFrame:
    """Fetch candles spanning [date_from, date_to] - used to backtest a POI
    from its formation time forward, instead of a fixed bar count."""
    timeframe = TIMEFRAME_MAP[timeframe_key]
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "tick_volume"]]
