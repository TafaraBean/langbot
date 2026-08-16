"""CSV-backed trade journal: append rows, load history, compute stats."""

import os
from datetime import datetime

import pandas as pd

JOURNAL_PATH = "journal.csv"

COLUMNS = [
    "timestamp", "symbol", "direction",
    "bias_D1", "bias_H4", "bias_H1", "bias_M15", "narrative_score",
    "poi_type", "poi_price", "confirmation_type",
    "entry", "stop_loss", "take_profit", "risk_reward",
    "outcome", "pnl_r", "notes",
    "sl_basis", "tp_ladder", "partial_legs",
]


def ensure_journal(path: str = JOURNAL_PATH) -> None:
    if not os.path.exists(path):
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)


def append_trade(row: dict, path: str = JOURNAL_PATH) -> None:
    ensure_journal(path)
    row = dict(row)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    row = {**{c: "" for c in COLUMNS}, **row}
    pd.DataFrame([row], columns=COLUMNS).to_csv(path, mode="a", header=False, index=False)


def load_journal(path: str = JOURNAL_PATH) -> pd.DataFrame:
    ensure_journal(path)
    return pd.read_csv(path)


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0, "win_rate": None, "avg_r": None, "equity_curve": pd.Series(dtype=float)}

    closed = df[df["outcome"].isin(["Win", "Loss", "BE"])]
    wins = (closed["outcome"] == "Win").sum()
    total_closed = len(closed)
    win_rate = wins / total_closed if total_closed else None

    pnl_r = pd.to_numeric(closed["pnl_r"], errors="coerce").fillna(0)
    avg_r = pnl_r.mean() if total_closed else None
    equity_curve = pnl_r.cumsum()

    return {
        "count": len(df),
        "closed_count": total_closed,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "equity_curve": equity_curve,
    }
