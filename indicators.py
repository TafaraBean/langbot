"""Indicator calculations and constraint evaluation, pandas/numpy only."""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.where(avg_loss != 0, 100)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bulls_power(df: pd.DataFrame, period: int = 13) -> pd.Series:
    return df["high"] - ema(df["close"], period)


def bears_power(df: pd.DataFrame, period: int = 13) -> pd.Series:
    return df["low"] - ema(df["close"], period)


def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute only the indicator columns enabled in `config` and return an
    augmented copy of df. `config` maps indicator key -> its settings dict
    (see dashboard.py's constraint panel for the exact shape)."""
    out = df.copy()

    if config.get("bulls_power", {}).get("enabled"):
        period = config["bulls_power"]["period"]
        out["bulls_power"] = bulls_power(out, period)

    if config.get("bears_power", {}).get("enabled"):
        period = config["bears_power"]["period"]
        out["bears_power"] = bears_power(out, period)

    if config.get("rsi", {}).get("enabled"):
        period = config["rsi"]["period"]
        out["rsi"] = rsi(out["close"], period)

    if config.get("macd", {}).get("enabled"):
        cfg = config["macd"]
        macd_line, signal_line, histogram = macd(out["close"], cfg["fast"], cfg["slow"], cfg["signal"])
        out["macd_line"] = macd_line
        out["macd_signal"] = signal_line
        out["macd_hist"] = histogram

    if config.get("ma", {}).get("enabled"):
        cfg = config["ma"]
        out["ma"] = ema(out["close"], cfg["period"]) if cfg["kind"] == "EMA" else sma(out["close"], cfg["period"])

    if config.get("atr", {}).get("enabled"):
        period = config["atr"]["period"]
        out["atr"] = atr(out, period)

    return out


def _compare(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    return series > threshold if operator == "above" else series < threshold


def evaluate_constraints(df: pd.DataFrame, config: dict) -> pd.Series:
    """AND-combine every enabled constraint's per-bar boolean mask. Returns
    an all-False series if no constraint is enabled (nothing qualifies)."""
    mask = pd.Series(True, index=df.index)
    any_enabled = False

    if config.get("bulls_power", {}).get("enabled"):
        cfg = config["bulls_power"]
        mask &= _compare(df["bulls_power"], cfg["operator"], cfg["threshold"])
        any_enabled = True

    if config.get("bears_power", {}).get("enabled"):
        cfg = config["bears_power"]
        mask &= _compare(df["bears_power"], cfg["operator"], cfg["threshold"])
        any_enabled = True

    if config.get("rsi", {}).get("enabled"):
        cfg = config["rsi"]
        mask &= _compare(df["rsi"], cfg["operator"], cfg["threshold"])
        any_enabled = True

    if config.get("macd", {}).get("enabled"):
        cfg = config["macd"]
        if cfg["mode"] == "Histogram vs threshold":
            mask &= _compare(df["macd_hist"], cfg["operator"], cfg["threshold"])
        else:
            crossed_up = df["macd_line"] > df["macd_signal"]
            mask &= crossed_up if cfg["operator"] == "above" else ~crossed_up
        any_enabled = True

    if config.get("ma", {}).get("enabled"):
        cfg = config["ma"]
        above_ma = df["close"] > df["ma"]
        mask &= above_ma if cfg["operator"] == "above" else ~above_ma
        any_enabled = True

    if config.get("atr", {}).get("enabled"):
        cfg = config["atr"]
        mask &= _compare(df["atr"], cfg["operator"], cfg["threshold"])
        any_enabled = True

    return mask & any_enabled if any_enabled else pd.Series(False, index=df.index)
