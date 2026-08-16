"""Session liquidity windows (framework spec is GMT+8), converted to the
machine's local timezone, plus prior-session high/low levels from data."""

from datetime import datetime, timedelta, timezone

import pandas as pd

GMT8 = timezone(timedelta(hours=8))

# (start_hour, start_min, end_hour, end_min) in GMT+8. NY AM wraps past midnight.
SESSIONS_GMT8 = {
    "Asian": (8, 0, 12, 0),
    "London": (14, 0, 17, 0),
    "NY AM": (20, 0, 0, 0),
}


def local_tz():
    return datetime.now().astimezone().tzinfo


def local_session_windows(reference_date: datetime | None = None) -> dict[str, tuple[datetime, datetime]]:
    """Today's session windows (in GMT+8) converted to local start/end datetimes."""
    ref = reference_date or datetime.now(GMT8)
    base = datetime(ref.year, ref.month, ref.day, tzinfo=GMT8)
    local_tz_ = local_tz()

    windows = {}
    for name, (sh, sm, eh, em) in SESSIONS_GMT8.items():
        start = base.replace(hour=sh, minute=sm)
        end = base.replace(hour=eh, minute=em)
        if end <= start:
            end += timedelta(days=1)
        windows[name] = (start.astimezone(local_tz_), end.astimezone(local_tz_))
    return windows


def active_sessions_now() -> list[str]:
    now_gmt8 = datetime.now(GMT8)
    active = []
    for name, (sh, sm, eh, em) in SESSIONS_GMT8.items():
        base = datetime(now_gmt8.year, now_gmt8.month, now_gmt8.day, tzinfo=GMT8)
        start = base.replace(hour=sh, minute=sm)
        end = base.replace(hour=eh, minute=em)
        if end <= start:
            end += timedelta(days=1)
        if start <= now_gmt8 < end:
            active.append(name)
        # also check yesterday's window in case it wraps into today (NY AM)
        start_y, end_y = start - timedelta(days=1), end - timedelta(days=1)
        if start_y <= now_gmt8 < end_y:
            active.append(name)
    return sorted(set(active))


def session_end_datetime(name: str, date_iso: str) -> datetime:
    """UTC end datetime of the named GMT+8 session window on the given date."""
    sh, sm, eh, em = SESSIONS_GMT8[name]
    day = datetime.fromisoformat(date_iso)
    end = datetime(day.year, day.month, day.day, eh, em, tzinfo=GMT8)
    if (eh, em) <= (sh, sm):
        end += timedelta(days=1)
    return end.astimezone(timezone.utc)


def session_liquidity_levels(df_m15: pd.DataFrame) -> dict[str, dict]:
    """Prior session's high/low from M15 candles, bucketed by GMT+8 session
    windows. Each level also carries a tested flag - the checklist's 'any
    Killzones high/low (untested) nearby?' - true if price has traded
    through that side since the session closed."""
    if df_m15.empty:
        return {}

    df = df_m15.copy()
    df["time_gmt8"] = df["time"].dt.tz_convert(GMT8)
    levels = {}

    for name, (sh, sm, eh, em) in SESSIONS_GMT8.items():
        last_complete = None
        for day_offset in range(3, 0, -1):
            day = (datetime.now(GMT8) - timedelta(days=day_offset)).date()
            start = datetime(day.year, day.month, day.day, sh, sm, tzinfo=GMT8)
            end = datetime(day.year, day.month, day.day, eh, em, tzinfo=GMT8)
            if end <= start:
                end += timedelta(days=1)
            mask = (df["time_gmt8"] >= start) & (df["time_gmt8"] < end)
            bucket = df[mask]
            if not bucket.empty:
                last_complete = {"high": bucket["high"].max(), "low": bucket["low"].min(), "date": day.isoformat()}
        if last_complete:
            after = df[df["time_gmt8"] >= end]
            last_complete["high_tested"] = bool((after["high"] >= last_complete["high"]).any()) if not after.empty else False
            last_complete["low_tested"] = bool((after["low"] <= last_complete["low"]).any()) if not after.empty else False
            levels[name] = last_complete

    return levels
