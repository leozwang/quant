#!/usr/bin/env python3
"""NYSE trading calendar helpers.

Dependency-free (stdlib only) so the scheduling layer adds no new packages to
requirements.txt. Provides holiday-aware session arithmetic in US/Eastern,
which is the timezone every trading date in the pipeline is expressed in.

All public functions take/return `datetime.date` for sessions and timezone-aware
`datetime.datetime` for instants. Never pass a naive datetime.
"""

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

# yfinance needs a short grace period after the bell before the daily bar
# consolidates. Settling earlier than this risks capturing an intraday tick.
CONSOLIDATION_GRACE = timedelta(minutes=15)


def now_et() -> datetime:
    """Current instant in US/Eastern."""
    return datetime.now(ET)


# ------------------------------------------------------------------------------
# Holiday computation
# ------------------------------------------------------------------------------

def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth `weekday` (Mon=0) of the given month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The final `weekday` (Mon=0) of the given month."""
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """NYSE weekend observance: Saturday -> prior Friday, Sunday -> next Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=32)
def nyse_holidays(year: int) -> frozenset:
    """Full-day NYSE market closures for the given calendar year."""
    days = set()

    # New Year's Day. Special case: when Jan 1 falls on a Saturday the NYSE
    # does NOT close the preceding Friday (that Friday belongs to the prior year).
    new_years = date(year, 1, 1)
    if new_years.weekday() != 5:
        days.add(_observed(new_years))

    days.add(_nth_weekday(year, 1, 0, 3))            # MLK Jr. Day  - 3rd Monday Jan
    days.add(_nth_weekday(year, 2, 0, 3))            # Washington's - 3rd Monday Feb
    days.add(_easter(year) - timedelta(days=2))      # Good Friday
    days.add(_last_weekday(year, 5, 0))              # Memorial Day - last Monday May

    if year >= 2022:                                 # Juneteenth (observed from 2022)
        days.add(_observed(date(year, 6, 19)))

    days.add(_observed(date(year, 7, 4)))            # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))            # Labor Day    - 1st Monday Sep
    days.add(_nth_weekday(year, 11, 3, 4))           # Thanksgiving - 4th Thursday Nov
    days.add(_observed(date(year, 12, 25)))          # Christmas

    return frozenset(days)


@lru_cache(maxsize=32)
def nyse_early_closes(year: int) -> frozenset:
    """Half-sessions that close at 13:00 ET instead of 16:00 ET."""
    days = set()

    # Friday after Thanksgiving
    days.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))

    # July 3rd, when both it and July 4th are weekdays
    jul3 = date(year, 7, 3)
    if jul3.weekday() < 5 and date(year, 7, 4).weekday() < 5:
        days.add(jul3)

    # Christmas Eve, when it is a weekday
    dec24 = date(year, 12, 24)
    if dec24.weekday() < 5:
        days.add(dec24)

    return frozenset(days)


# ------------------------------------------------------------------------------
# Session arithmetic
# ------------------------------------------------------------------------------

def is_trading_day(d: date) -> bool:
    """True if `d` is a regular or half NYSE session."""
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def session_open(d: date) -> datetime:
    """Opening bell instant for session `d`."""
    return datetime.combine(d, REGULAR_OPEN, tzinfo=ET)


def session_close(d: date) -> datetime:
    """Closing bell instant for session `d`, accounting for half-days."""
    closing = EARLY_CLOSE if d in nyse_early_closes(d.year) else REGULAR_CLOSE
    return datetime.combine(d, closing, tzinfo=ET)


def next_trading_day(d: date) -> date:
    """First session strictly after `d`."""
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def previous_trading_day(d: date) -> date:
    """Last session strictly before `d`."""
    prev = d - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev


def is_market_open(now: datetime) -> bool:
    """True if `now` falls inside a live trading session."""
    d = now.date()
    return is_trading_day(d) and session_open(d) <= now < session_close(d)


def has_session_closed(d: date, now: datetime, grace: timedelta = CONSOLIDATION_GRACE) -> bool:
    """True if session `d` has finished and its daily bar should be consolidated.

    Non-trading dates can never close, so they always return False; this keeps
    rows whose target_date landed on a holiday from being settled against the
    wrong session.
    """
    if not is_trading_day(d):
        return False
    return now >= session_close(d) + grace


def last_completed_session(now: datetime) -> date:
    """Most recent session whose closing bell (plus grace) has already passed."""
    today = now.date()
    if has_session_closed(today, now):
        return today
    return previous_trading_day(today)


def is_first_session_of_week(d: date) -> bool:
    """True if `d` is the earliest session of its ISO week (normally Monday)."""
    prev = previous_trading_day(d)
    return prev.isocalendar()[:2] != d.isocalendar()[:2]


def describe(d: date) -> str:
    """Human-readable session label, e.g. '2026-09-21 (Mon)'."""
    return f"{d.isoformat()} ({d.strftime('%a')})"
