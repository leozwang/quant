#!/usr/bin/env python3
"""Master Quantitative Recommender & Supabase Pipeline.

Executes both the Daily Top 10 Recommender and Weekly Top 10 Recommender,
then safely synchronizes the results into Supabase:
  - Table: qlib_daily_top10_recommend
  - Table: qlib_weekly_top10_recommend

Idempotency / Single Upload Guarantee:
  Before uploading, this script queries Supabase to check if recommendations
  for the current trading date (or week) have already been stored. If so, it
  automatically SKIPS the upload to prevent duplicate submissions or unnecessary
  database writes. Use the `--force` flag if you explicitly wish to overwrite.
"""

import argparse
import base64
from datetime import datetime, timedelta
import errno
import fcntl
import http.client
import json
import os
import random
import socket
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

# NOTE: numpy / yfinance / the sister recommender modules are imported lazily at
# their point of use. They are heavy, and only the *execution* paths need them --
# keeping them out of module scope lets `--explain` render a plan on a machine
# where the data-science stack is not installed.


# Ensure current directory is on python path for importing sister scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# market_calendar is stdlib-only and is needed by the planner on every run,
# so it stays a module-level import.
from market_calendar import (
    ET,
    describe,
    has_session_closed,
    is_first_session_of_week,
    is_market_open,
    last_completed_session,
    next_trading_day,
    now_et,
)

DAILY_TABLE = "qlib_daily_top10_recommend"
WEEKLY_TABLE = "qlib_weekly_top10_recommend"

# Conflict targets matching the UNIQUE constraints declared in doc/supabase_schema.sql.
DAILY_CONFLICT_TARGET = "trade_date,symbol"
WEEKLY_CONFLICT_TARGET = "week_start_date,symbol"

# Columns owned by the settlement stage. The recommenders emit these as None, so
# they must never be included in an upsert payload or a re-run would overwrite
# already-settled rows back to NULL and destroy the accuracy history.
SETTLEMENT_FIELDS = ("target_date_close", "predicted_diff")

# Minimal columns required for batch settlement upsert:
# 1. Unique conflict key (trade_date, symbol)
# 2. Mandatory NOT NULL columns declared in doc/supabase_schema.sql without defaults
# 3. Settlement columns owned by this stage (target_date_close, predicted_diff)
# All 12 unowned columns (predicted_close, market_regime, ret_1d, ret_3d, vol_surge,
# clv, rsi_7, rr_ratio, date, target_date, created_at, id) are excluded.
SETTLEMENT_REQUIRED_COLUMNS = (
    "trade_date",
    "symbol",
    "rank",
    "name",
    "sector",
    "setup_type",
    "close_price",
    "alpha_score",
    "stop_loss",
    "target_t1",
    "target_date_close",
    "predicted_diff",
)


def _slim_settlement_payload(records: list) -> list:
    """Extracts only conflict keys, mandatory NOT NULL schema columns, and settlement values.

    Strips all unowned columns so settlement cannot touch or overwrite columns it doesn't own.
    """
    return [{k: r[k] for k in SETTLEMENT_REQUIRED_COLUMNS if k in r} for r in records]


# "Eve-of-session": only generate recommendations when the next trading session
# is at most this many calendar days away. Sunday -> Monday is 1 day (allowed);
# Friday -> Monday is 3 days (skipped, since Sunday's run covers it).
MAX_EVE_GAP_DAYS = 1

# Cap on the settlement backlog query. Generous next to a normal backlog (one
# session = ~10 rows) but bounded, so a pathological table cannot turn every
# scheduled run into an ever-growing download.
UNSETTLED_QUERY_LIMIT = 2000

LOCK_PATH = os.path.join(tempfile.gettempdir(), "qlib_recommend_pipeline.lock")

# ------------------------------------------------------------------------------
# Continuous-trading switch
# ------------------------------------------------------------------------------
# US equities are moving toward extended and eventually near-continuous sessions.
# If the market never "closes", is_market_open() is permanently True, the
# RECOMMEND predicate never fires, and this pipeline would quietly stop producing
# recommendations forever while still exiting 0 -- a silent failure.
#
# Set this to True (or export QLIB_CONTINUOUS_TRADING=1) when that happens. The
# clock-based gate is then dropped, and correctness rests entirely on the
# session-agreement check inside the recommender, which compares the provider's
# actual data baseline against the expected session. That check is data-driven
# and therefore stays valid no matter what the trading hours are.
#
# The invariant we depend on is not "the market is closed" but "the official
# closing print for session D has been published". Even under 24/5 there is still
# a primary-listing closing auction that defines the daily bar, with overnight
# activity attributed to the following session.
CONTINUOUS_TRADING = os.environ.get("QLIB_CONTINUOUS_TRADING", "").lower() in ("1", "true", "yes")


TRANSIENT_NETWORK_ERRORS = (
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    ConnectionResetError,
    BrokenPipeError,
    TimeoutError,
    socket.timeout,
    ssl.SSLError,
)


class SupabaseUnavailable(RuntimeError):
    """Raised when an existence check cannot be completed.

    Treated as fail-closed: if we cannot prove that records are absent, we must
    not upload, otherwise a transient network error silently produces duplicates.
    Includes an `is_transient` flag to differentiate network drops from non-transient 4xx errors.
    """

    def __init__(self, message: str, is_transient: bool = False):
        super().__init__(message)
        self.is_transient = is_transient



def load_dotenv(override: bool = True):
    """Loads environment variables from local .env files if present.

    override=True ensures that a local project .env file takes precedence over
    stray global shell variables (e.g. an old project key in ~/.bash_profile).
    If no .env file exists (such as in CI), exported shell variables remain in effect.
    """
    search_paths = [
        os.path.join(current_dir, "../.env"),
        os.path.join(current_dir, ".env"),
        os.path.abspath(".env")
    ]
    for env_path in search_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip('"\'')
                            if k and (override or k not in os.environ):
                                os.environ[k] = v
                return os.path.abspath(env_path)
            except Exception:
                pass
    return None


def get_jwt_project_ref(token: str) -> str:
    """Extracts project 'ref' from JWT payload without verifying signature."""
    try:
        parts = token.split(".")
        if len(parts) >= 2:
            padding = "=" * (4 - len(parts[1]) % 4)
            payload_data = base64.urlsafe_b64decode(parts[1] + padding)
            return json.loads(payload_data.decode("utf-8")).get("ref", "")
    except Exception:
        pass
    return ""


class SupabaseSync:
    """Handles querying and uploading recommendation records to Supabase."""

    def __init__(self, url: str = None, key: str = None):
        load_dotenv()
        project_id = os.environ.get("SUPABASE_PROJECT_ID")
        env_url = os.environ.get("SUPABASE_URL") or (f"https://{project_id}.supabase.co" if project_id else "")
        env_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

        self.url = (url or env_url or "").rstrip("/")
        self.key = key or env_key or ""

    def _headers(self, prefer_merge: bool = False, return_representation: bool = False) -> dict:
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
            "User-Agent": "Mozilla/5.0 (compatible; QuantPipeline/1.0)",
        }
        if prefer_merge:
            headers["Prefer"] = "resolution=merge-duplicates"
        elif return_representation:
            headers["Prefer"] = "return=representation"
        return headers

    @property
    def is_configured(self) -> bool:
        """True only when both a base URL and an API key are available."""
        return bool(self.url and self.key)

    def _request_with_retry(
        self,
        req: urllib.request.Request,
        timeout: int = 15,
        max_retries: int = 4,
        initial_backoff: float = 0.5,
        action_name: str = "Supabase request",
    ) -> tuple:
        """Executes an HTTP request with retries and exponential backoff on transient network drops.

        Returns (status_code, response_body_bytes).
        Raises SupabaseUnavailable if all retries are exhausted.
        """
        backoff = initial_backoff
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    if attempt > 1:
                        print(f"  ✓ [DEBUG/RETRY] {action_name} succeeded on attempt {attempt}/{max_retries}!")
                    return resp.status, body
            except urllib.error.HTTPError as e:
                body = e.read()
                # 429 (rate-limit) and 5xx gateway errors (502, 503, 504) are transient
                if e.code in (429, 502, 503, 504) and attempt < max_retries:
                    delay = backoff + random.uniform(0.1, 0.4)
                    print(
                        f"  ⚠️  [DEBUG/RETRY] {action_name} received HTTP {e.code} ({e.reason}). "
                        f"Retrying in {delay:.2f}s (attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(delay)
                    backoff *= 2
                    continue
                err_text = body.decode("utf-8", errors="replace")
                raise SupabaseUnavailable(f"HTTP {e.code}: {err_text}", is_transient=False) from e
            except urllib.error.URLError as e:
                err_str = str(e.reason).lower()
                is_transient = isinstance(e.reason, TRANSIENT_NETWORK_ERRORS) or any(
                    msg in err_str for msg in (
                        "remote end closed",
                        "incompleteread",
                        "connection reset",
                        "broken pipe",
                        "timed out",
                        "handshake",
                        "eof occurred",
                    )
                )
                if is_transient and attempt < max_retries:
                    delay = backoff + random.uniform(0.1, 0.4)
                    print(
                        f"  ⚠️  [DEBUG/RETRY] {action_name} network drop ({e.reason}). "
                        f"Retrying in {delay:.2f}s (attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(delay)
                    backoff *= 2
                    continue
                raise SupabaseUnavailable(f"{action_name} network error: {e.reason}", is_transient=is_transient) from e
            except TRANSIENT_NETWORK_ERRORS as e:
                if attempt < max_retries:
                    delay = backoff + random.uniform(0.1, 0.4)
                    print(
                        f"  ⚠️  [DEBUG/RETRY] {action_name} dropped: {type(e).__name__} ({e}). "
                        f"Retrying in {delay:.2f}s (attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(delay)
                    backoff *= 2
                    continue
                raise SupabaseUnavailable(f"{action_name} connection dropped after {max_retries} attempts: {e}", is_transient=True) from e
            except Exception as e:
                raise SupabaseUnavailable(f"{action_name} unexpected error: {e}", is_transient=False) from e

        raise SupabaseUnavailable(f"{action_name} exhausted all {max_retries} retry attempts", is_transient=True)

    def _get_json(self, path: str, timeout: int = 15, max_retries: int = 4):
        """Performs an authenticated GET and returns the decoded JSON body."""
        if not self.is_configured:
            raise SupabaseUnavailable("Supabase credentials are not configured")
        req = urllib.request.Request(
            f"{self.url}{path}", headers=self._headers(), method="GET"
        )
        _, body = self._request_with_retry(
            req, timeout=timeout, max_retries=max_retries, action_name=f"GET {path.split('?')[0]}"
        )
        return json.loads(body.decode("utf-8"))

    def test_connection(self) -> tuple:
        """Pings Supabase to verify authorization and accessibility of *both* tables.

        The weekly table is checked too: it is only written on Sundays, so a
        permissions or migration problem there would otherwise stay invisible
        for up to a week before failing at upload time.
        """
        if not self.is_configured:
            return False, "Supabase credentials are not configured"

        problems = []
        for label, table in (("daily", DAILY_TABLE), ("weekly", WEEKLY_TABLE)):
            try:
                query_url = f"{self.url}/rest/v1/{table}?limit=1&select=id"
                req = urllib.request.Request(query_url, headers=self._headers(), method="GET")
                status, _ = self._request_with_retry(
                    req, timeout=8, max_retries=4, action_name=f"Ping {label} table"
                )
                if status not in (200, 206):
                    problems.append(f"{label} table '{table}' → HTTP {status}")
            except Exception as e:
                problems.append(f"{label} table '{table}' → {e}")

        if problems:
            return False, "; ".join(problems)
        return True, "OK (daily + weekly tables reachable)"

    def has_daily_records(self, trade_date: str) -> bool:
        """Checks if daily recommendations already exist for the specified trade_date.

        Raises SupabaseUnavailable if the check cannot be completed, so callers
        fail closed rather than assuming "no records" and inserting duplicates.
        """
        value = urllib.parse.quote(str(trade_date), safe="")
        try:
            data = self._get_json(f"/rest/v1/{DAILY_TABLE}?trade_date=eq.{value}&limit=1&select=id", timeout=10)
        except SupabaseUnavailable as e:
            raise SupabaseUnavailable(f"could not verify existing daily records: {e}") from e
        return len(data) > 0

    def has_weekly_records(self, week_start_date: str) -> bool:
        """Checks if weekly recommendations already exist for the specified week_start_date.

        Raises SupabaseUnavailable if the check cannot be completed, so callers
        fail closed rather than assuming "no records" and inserting duplicates.
        """
        value = urllib.parse.quote(str(week_start_date), safe="")
        try:
            data = self._get_json(f"/rest/v1/{WEEKLY_TABLE}?week_start_date=eq.{value}&limit=1&select=id", timeout=10)
        except SupabaseUnavailable as e:
            raise SupabaseUnavailable(f"could not verify existing weekly records: {e}") from e
        return len(data) > 0

    def has_daily_records_for_target(self, target_date: str) -> bool:
        """Checks if daily recommendations already exist for a given target_date.

        The planner keys on target_date rather than trade_date because the target
        session is knowable up front, whereas trade_date is only derivable after
        the (expensive) market data download. Checking first is what lets us skip
        that download entirely on a repeat run.
        """
        value = urllib.parse.quote(str(target_date), safe="")
        try:
            data = self._get_json(f"/rest/v1/{DAILY_TABLE}?target_date=eq.{value}&limit=1&select=id", timeout=10)
        except SupabaseUnavailable as e:
            raise SupabaseUnavailable(f"could not verify rows for target_date {target_date}: {e}") from e
        return len(data) > 0

    @staticmethod
    def _sanitize(records: list) -> list:
        """Drops settlement-owned columns whose value is None.

        The recommenders emit `target_date_close`/`predicted_diff` as None. Sending
        those in an upsert would reset already-settled rows to NULL, so they are
        removed here and left exclusively to the settlement stage.
        """
        cleaned = []
        for r in records:
            cleaned.append({
                k: v for k, v in r.items()
                if not (k in SETTLEMENT_FIELDS and v is None)
            })
        return cleaned

    def upload_daily(self, records: list, force: bool = False, check_existing: bool = True) -> tuple:
        """Uploads daily recommendations to 'qlib_daily_top10_recommend'.

        `check_existing` may be disabled by callers (such as the auto-mode planner)
        that have already established absence, avoiding a redundant round-trip.
        """
        if not records:
            return False, "No daily records provided."

        trade_date = records[0].get("trade_date")
        if check_existing and not force and trade_date:
            try:
                if self.has_daily_records(trade_date):
                    return (
                        False,
                        f"Skipped daily upload: Records for trade_date '{trade_date}' already exist in Supabase (use --force to overwrite)."
                    )
            except SupabaseUnavailable as e:
                return False, f"Skipped daily upload (fail-closed): {e}"

        endpoint = f"{self.url}/rest/v1/{DAILY_TABLE}?on_conflict={DAILY_CONFLICT_TARGET}"
        payload = json.dumps(self._sanitize(records)).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers=self._headers(prefer_merge=True), method="POST")

        try:
            status, _ = self._request_with_retry(
                req, timeout=15, max_retries=4, action_name=f"Upload {len(records)} daily records"
            )
            if status in (200, 201, 204):
                return True, f"Successfully uploaded {len(records)} daily records for {trade_date} to '{DAILY_TABLE}'."
            return False, f"Supabase responded with status {status}."
        except Exception as e:
            return False, f"Failed to upload daily records: {e}"

    def upload_weekly(self, records: list, force: bool = False, check_existing: bool = True) -> tuple:
        """Uploads weekly recommendations to 'qlib_weekly_top10_recommend'.

        `check_existing` may be disabled by callers (such as the auto-mode planner)
        that have already established absence, avoiding a redundant round-trip.
        """
        if not records:
            return False, "No weekly records provided."

        week_start_date = records[0].get("week_start_date")
        if check_existing and not force and week_start_date:
            try:
                if self.has_weekly_records(week_start_date):
                    return (
                        False,
                        f"Skipped weekly upload: Records for week_start_date '{week_start_date}' already exist in Supabase (use --force to overwrite)."
                    )
            except SupabaseUnavailable as e:
                return False, f"Skipped weekly upload (fail-closed): {e}"

        endpoint = f"{self.url}/rest/v1/{WEEKLY_TABLE}?on_conflict={WEEKLY_CONFLICT_TARGET}"
        payload = json.dumps(self._sanitize(records)).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers=self._headers(prefer_merge=True), method="POST")

        try:
            status, _ = self._request_with_retry(
                req, timeout=15, max_retries=4, action_name=f"Upload {len(records)} weekly records"
            )
            if status in (200, 201, 204):
                return True, f"Successfully uploaded {len(records)} weekly records for {week_start_date} to '{WEEKLY_TABLE}'."
            return False, f"Supabase responded with status {status}."
        except Exception as e:
            return False, f"Failed to upload weekly records: {e}"

    def get_records_for_target_date(self, target_date: str) -> list:
        """Fetches rows from qlib_daily_top10_recommend where target_date matches.

        Raises SupabaseUnavailable rather than returning []. An empty list means
        "no such predictions exist", which is a completely different situation
        from "the database could not be reached" -- conflating them made an
        outage print a reassuring 'no recommendations found'.
        """
        value = urllib.parse.quote(str(target_date), safe="")
        return self._get_json(
            f"/rest/v1/{DAILY_TABLE}?target_date=eq.{value}&select=*&order=rank.asc,id.asc"
        )

    def get_unsettled_daily_records(self) -> list:
        """Fetches target_dates of rows awaiting settlement.

        Raises SupabaseUnavailable rather than returning [] on failure. Returning
        an empty list would be indistinguishable from "the backlog is clear", so
        a transient outage would silently skip settlement and still exit 0.

        Only `target_date` is projected -- the planner needs nothing else, and
        this table accumulates every unsettled row ever written (including any
        that can never settle), so selecting * was unbounded work on every run.
        """
        return self._get_json(
            f"/rest/v1/{DAILY_TABLE}?target_date_close=is.null"
            f"&select=target_date&order=target_date.asc&limit={UNSETTLED_QUERY_LIMIT}"
        )

    def batch_update_daily_settlement(self, records: list) -> tuple:
        """Batch-updates target_date_close and predicted_diff for multiple records in ONE atomic upsert.

        Uses PostgREST on_conflict=trade_date,symbol with resolution=merge-duplicates.
        Reduces N HTTP round-trips to 1, eliminating cumulative failure risk on edge proxies.
        Slims the payload to only conflict keys, required schema columns, and settlement values.

        Returns (ok: bool, message: str, is_transient: bool).
        """
        if not records:
            return True, "No records to update.", False

        endpoint = f"{self.url}/rest/v1/{DAILY_TABLE}?on_conflict={DAILY_CONFLICT_TARGET}"
        slim_payload = _slim_settlement_payload(records)
        payload = json.dumps(slim_payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers=self._headers(prefer_merge=True), method="POST")

        try:
            status, _ = self._request_with_retry(
                req, timeout=15, max_retries=4, action_name=f"Batch settlement update ({len(records)} records)"
            )
            if status in (200, 201, 204):
                return True, f"Successfully batch-updated {len(records)} records in '{DAILY_TABLE}'.", False
            return False, f"Supabase responded with status {status}.", False
        except SupabaseUnavailable as e:
            return False, f"Failed to batch-update records: {e}", getattr(e, "is_transient", False)
        except Exception as e:
            return False, f"Failed to batch-update records: {e}", False

    def update_daily_settlement(self, record_id: int, target_date_close: float, predicted_diff: float) -> tuple:
        """Updates target_date_close and predicted_diff for a specific daily recommendation row."""
        endpoint = f"{self.url}/rest/v1/{DAILY_TABLE}?id=eq.{record_id}"
        payload = json.dumps({
            "target_date_close": target_date_close,
            "predicted_diff": predicted_diff
        }).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers=self._headers(return_representation=True),
            method="PATCH"
        )
        try:
            status, _ = self._request_with_retry(
                req, timeout=15, max_retries=4, action_name=f"Update settlement row {record_id}"
            )
            if status in (200, 204):
                return True, f"Record {record_id} successfully updated"
            return False, f"Record {record_id} update returned HTTP status {status}"
        except Exception as e:
            return False, f"Failed to update record {record_id}: {e}"


def fetch_target_date_close(symbol: str, target_date_str: str, now: datetime = None) -> float:
    """Fetches the official closing price for a stock on target_date.

    Refuses to return anything until that session's closing bell (plus the
    consolidation grace period) has passed. Writing a live intraday quote into
    `target_date_close` would be permanent -- the row stops being NULL and is
    never revisited -- so every downstream MAE/MAPE figure would be wrong.

    The lookup is date-exact. If the requested session is genuinely absent from
    history (a holiday, or beyond the window) this returns None rather than
    silently substituting a different session's close.
    """
    now = now or now_et()

    try:
        target_d = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        print(f"  ⚠️ Unparseable target_date '{target_date_str}' for {symbol}")
        return None

    if not has_session_closed(target_d, now):
        return None

    try:
        import numpy as np
        import yfinance as yf

        t = yf.Ticker(symbol)

        # 3 months of daily bars covers same-day settlement as well as any
        # accumulated backlog from missed runs.
        #
        # auto_adjust=False is deliberate and important. yfinance defaults to
        # dividend/split-ADJUSTED closes, which are rewritten retroactively every
        # time a dividend goes ex. Measured on AAPL for 2025-01-10, the adjusted
        # close reads 235.14 today versus an actual traded close of 236.85 -- a
        # $1.71 (0.7%) drift that grows with every payout.
        #
        # Storing an adjusted figure would mean `target_date_close` no longer
        # matches the price anyone could have traded at, and no longer matches
        # the units `predicted_close` was expressed in. Settling late (from a
        # backlog) or re-settling would then yield a different number than
        # settling on time, making the accuracy history irreproducible.
        hist = t.history(period="3mo", auto_adjust=False)
        if not hist.empty and "Close" in hist.columns:
            hist_dates = [str(d.date()) for d in hist.index]
            if target_date_str in hist_dates:
                val = hist["Close"].iloc[hist_dates.index(target_date_str)]
                if not np.isnan(val) and val > 0:
                    return round(float(val), 2)

        # NOTE: there is deliberately no live-quote fallback here. `fast_info`
        # returns the last trade, which after the bell is an after-hours print,
        # not the official closing auction price -- and under extended or
        # continuous trading it is simply a different number. If the official
        # bar has not been published yet, the correct action is to wait: the row
        # stays NULL and the next hourly run settles it. Latency costs an hour;
        # writing the wrong close corrupts the accuracy history permanently,
        # because a non-NULL row is never revisited.
    except Exception as e:
        print(f"  ⚠️ Error fetching close price for {symbol}: {e}")
    return None


_GHOST_CACHE = {}


def is_ghost_session(target_date_str: str, reference_symbol: str = "SPY") -> bool:
    """True if `target_date_str` is a date the market never actually traded.

    A rules-based calendar cannot know about unscheduled closures -- national
    days of mourning, weather, outages. 2025-01-09 (President Carter) is a real
    example: every rule says it was a normal Thursday, and the market was shut.

    A prediction written for such a date can never settle, so it would sit in the
    backlog forever, re-downloading data on every run and (since settlement
    failures are now reported) raising an alert every hour indefinitely.

    The distinction is decidable from data alone, with no calendar involved:

        no bar ON the date, but bars exist AFTER it  =>  the session never happened
        no bar ON the date, and no bars after it     =>  provider lag, just wait

    That second case is why we cannot simply treat "missing" as "void".
    """
    if target_date_str in _GHOST_CACHE:
        return _GHOST_CACHE[target_date_str]

    verdict = False
    try:
        import yfinance as yf

        target_d = datetime.strptime(target_date_str, "%Y-%m-%d").date()

        # The window MUST be centred on the target date, not on today. Using a
        # relative period (e.g. period="3mo") means any target older than that
        # window contains no bars at all, which would look exactly like a void
        # session and would silently discard perfectly good old predictions.
        # +/- 10 days comfortably spans the longest market closure on record.
        start = (target_d - timedelta(days=10)).isoformat()
        end = (target_d + timedelta(days=10)).isoformat()

        hist = yf.Ticker(reference_symbol).history(start=start, end=end)
        if not hist.empty:
            dates = {str(d.date()) for d in hist.index}
            on_date = target_date_str in dates
            after = any(d > target_date_str for d in dates)
            verdict = (not on_date) and after
    except Exception as e:
        # Never assert voidness on an error -- that would discard a real
        # prediction. Fail toward "keep retrying".
        print(f"  ⚠️ Ghost-session check failed for {target_date_str}: {e}")
        verdict = False

    _GHOST_CACHE[target_date_str] = verdict
    return verdict


def settle_target_date_predictions(sync: SupabaseSync, target_date: str = None, dry_run: bool = False,
                                   force: bool = False, now: datetime = None) -> list:
    """Inspects Supabase table for records with target_date == today (or target_date),
    fetches actual closing prices, fills target_date_close, and computes predicted_diff.
    """
    now = now or now_et()
    if not target_date:
        target_date = now.strftime("%Y-%m-%d")

    print("=" * 115)
    print("      🎯 PREDICTION SETTLEMENT & ACCURACY EVALUATION")
    print(f"      Target Settlement Date: {describe(datetime.strptime(target_date, '%Y-%m-%d').date())}")
    print(f"      Execution Mode        : {'DRY RUN (Preview Only)' if dry_run else 'ACTIVE (Updating Supabase)'}")
    print("=" * 115)

    # Fetch records matching target_date
    try:
        records = sync.get_records_for_target_date(target_date)
    except SupabaseUnavailable as e:
        print(f"  ⚠️ Could not read predictions for '{target_date}': {e}")
        print("     Settlement skipped; the backlog will be retried on the next run.")
        print("-" * 115 + "\n")
        return [{"symbol": "-", "target_date": target_date, "target_date_close": None,
                 "status": "Unreachable", "unreachable": True}]

    if not records:
        print(f"  ℹ️ No daily recommendations found in Supabase with target_date = '{target_date}'.")
        print("-" * 115 + "\n")
        return []

    # ==========================================================================
    # Ghost-session guard
    # ==========================================================================
    # Only worth checking once the date is genuinely overdue -- that is, once a
    # *later* session has already closed and the data still shows nothing. Until
    # then "no bar" is just provider lag. Gating on overdue keeps the common
    # same-day settlement path free of any extra network calls.
    target_d = datetime.strptime(target_date, "%Y-%m-%d").date()
    overdue = has_session_closed(next_trading_day(target_d), now)

    if overdue and is_ghost_session(target_date):
        print(f"\n  🚫 [VOID] No session took place on {describe(target_d)}.")
        print("     The market was unexpectedly closed (the rules-based calendar cannot know this).")
        print(f"     {len(records)} prediction(s) for this date can never settle and are being")
        print("     dropped from the backlog rather than retried forever.")
        print("-" * 115 + "\n")
        # Reported as voided, not as unsettled, so this does not trigger an alert.
        return [
            {
                "id": r["id"],
                "symbol": r["symbol"],
                "rank": r.get("rank", "-"),
                "trade_date": r.get("trade_date", "-"),
                "target_date": target_date,
                "predicted_close": r.get("predicted_close"),
                "target_date_close": None,
                "predicted_diff": None,
                "status": "Void (no session)",
                "void": True,
            }
            for r in records
        ]

    print(f"  ✓ Found {len(records)} recommendation record(s) with target_date '{target_date}'.")
    print("  📊 Fetching actual market close prices and calculating prediction deltas...\n")

    header = f"{'Rank':<5} {'Symbol':<7} {'Trade Date':<11} {'Target Date':<12} {'Pred Close':<12} {'Actual Close':<14} {'Diff ($)':<11} {'Diff (%)':<10} {'Status':<18}"
    print("-" * 115)
    print(header)
    print("-" * 115)

    settled_results = []
    abs_diffs = []
    pct_diffs = []
    to_update_records = []
    row_meta = []

    for r in records:
        rec_id = r["id"]
        symbol = r["symbol"]
        rank = r.get("rank", "-")
        trade_date = r.get("trade_date", "-")
        rec_target_date = r.get("target_date", target_date)
        pred_close = r.get("predicted_close")
        existing_actual = r.get("target_date_close")

        # If already settled and not force, keep existing
        if existing_actual is not None and not force:
            actual_close = existing_actual
            pred_diff = r.get("predicted_diff")
            if pred_diff is None and pred_close is not None:
                pred_diff = round(actual_close - pred_close, 2)
            status_str = "Already Settled"
            needs_update = False
        else:
            actual_close = fetch_target_date_close(symbol, rec_target_date, now=now)
            if actual_close is not None:
                if pred_close is not None:
                    pred_diff = round(actual_close - pred_close, 2)
                else:
                    pred_diff = None

                if dry_run:
                    status_str = "Dry Run (Preview)"
                    needs_update = False
                else:
                    status_str = "Pending Upload"
                    needs_update = True
            else:
                pred_diff = None
                status_str = "⚠️ Price Missing"
                needs_update = False

        updated_row = dict(r)
        updated_row["target_date_close"] = actual_close
        updated_row["predicted_diff"] = pred_diff
        if needs_update:
            to_update_records.append(updated_row)

        row_meta.append({
            "rec_id": rec_id,
            "symbol": symbol,
            "rank": rank,
            "trade_date": trade_date,
            "rec_target_date": rec_target_date,
            "pred_close": pred_close,
            "actual_close": actual_close,
            "pred_diff": pred_diff,
            "status_str": status_str,
            "needs_update": needs_update,
        })

    # Execute batch update in a single atomic round-trip if updates are needed
    batch_ok = True
    batch_msg = ""
    is_transient = False
    if to_update_records and not dry_run:
        print(f"  🚀 Batch updating {len(to_update_records)} settled record(s) in a single atomic Supabase round-trip...")
        batch_ok, batch_msg, is_transient = sync.batch_update_daily_settlement(to_update_records)
        if batch_ok:
            print(f"  ✓ {batch_msg}\n")
        elif is_transient:
            print(f"  ⚠️ Batch update failed transiently ({batch_msg}), attempting fallback to individual row updates...\n")
        else:
            print(f"  ❌ Batch update failed with non-transient error ({batch_msg}). Skipping per-row fallback.\n")

    for item in row_meta:
        rec_id = item["rec_id"]
        symbol = item["symbol"]
        rank = item["rank"]
        trade_date = item["trade_date"]
        rec_target_date = item["rec_target_date"]
        pred_close = item["pred_close"]
        actual_close = item["actual_close"]
        pred_diff = item["pred_diff"]
        status_str = item["status_str"]

        if item["needs_update"]:
            if batch_ok:
                status_str = "✓ Settled"
            elif is_transient:
                ok, msg = sync.update_daily_settlement(rec_id, actual_close, pred_diff)
                status_str = "✓ Settled" if ok else f"⚠️ {msg}"
            else:
                status_str = f"⚠️ Batch Failed"

        pred_str = f"${pred_close:.2f}" if pred_close is not None else "N/A"
        act_str = f"${actual_close:.2f}" if actual_close is not None else "Pending"

        if pred_diff is not None:
            diff_str = f"{pred_diff:+.2f}"
            abs_diff = abs(pred_diff)
            abs_diffs.append(abs_diff)
            if pred_close and pred_close > 0:
                pct = (pred_diff / pred_close) * 100.0
                pct_str = f"{pct:+.2f}%"
                pct_diffs.append(abs(pct))
            else:
                pct_str = "N/A"
        else:
            diff_str = "N/A"
            pct_str = "N/A"

        print(
            f"#{rank:<4} {symbol:<7} {trade_date:<11} {rec_target_date:<12} "
            f"{pred_str:<12} {act_str:<14} {diff_str:<11} {pct_str:<10} {status_str:<18}"
        )

        settled_results.append({
            "id": rec_id,
            "symbol": symbol,
            "rank": rank,
            "trade_date": trade_date,
            "target_date": rec_target_date,
            "predicted_close": pred_close,
            "target_date_close": actual_close,
            "predicted_diff": pred_diff,
            "status": status_str
        })

    print("-" * 115)
    # Print aggregate performance statistics
    if abs_diffs:
        mae = sum(abs_diffs) / len(abs_diffs)
        mape = sum(pct_diffs) / len(pct_diffs) if pct_diffs else 0.0
        print(f"  📈 SETTLEMENT ACCURACY METRICS ({len(abs_diffs)} stocks settled):")
        print(f"     • Mean Absolute Error (MAE)            : ${mae:.2f}")
        print(f"     • Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("=" * 115 + "\n")

    return settled_results


def run_sanity_check(sync: SupabaseSync, dry_run: bool = False):
    """Verifies Supabase credentials, reports configuration status, and tests connectivity."""
    print("----------------------------------------------------------------------")
    print("🔍 [SANITY CHECK] Supabase Configuration Audit:")

    if not sync.url or not sync.key:
        print("  ⚠️  WARNING: Supabase credentials are NOT fully configured!")
        print(f"      • SUPABASE_URL : {sync.url or '[MISSING]'}")
        print(f"      • SUPABASE_KEY : {'[CONFIGURED]' if sync.key else '[MISSING]'}")
        print("      -> Set environment variables or create a .env file (see .env.example):")
        print("         export SUPABASE_URL=\"https://<your-project-id>.supabase.co\"")
        print("         export SUPABASE_KEY=\"<your-anon-or-service-key>\"")
    else:
        key_ref = get_jwt_project_ref(sync.key)
        print("  ✓ Supabase credentials configured:")
        print(f"      • SUPABASE_URL : {sync.url}")
        print(f"      • SUPABASE_KEY : Project Ref '{key_ref if key_ref else 'Custom'}'")

    if sync.url and sync.key and not dry_run:
        is_ok, msg = sync.test_connection()
        if is_ok:
            print(f"  ✓ Cloud Connectivity : Connected to Supabase PostgREST API (Status: {msg})")
        else:
            print(f"  ⚠️  Cloud Connectivity : {msg}")
    elif dry_run:
        print("  ℹ️  Dry-run mode: Cloud connectivity ping skipped.")
    print("----------------------------------------------------------------------\n")


def acquire_lock(path: str = LOCK_PATH):
    """Takes an exclusive, non-blocking lock so overlapping runs cannot collide.

    Returns the open file handle on success (the caller must keep a reference for
    the lifetime of the run) or None if another instance already holds it. This
    matters most on an hourly schedule, where a slow yfinance download could
    otherwise still be running when the next invocation fires.
    """
    handle = open(path, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        handle.close()
        if e.errno in (errno.EACCES, errno.EAGAIN):
            return None
        raise
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def build_execution_plan(sync: SupabaseSync, args, now: datetime = None) -> dict:
    """Decides which stages should run, from current state rather than the weekday.

    Two independent predicates:

      SETTLE     - there exists at least one row with target_date_close IS NULL
                   whose target session has already closed. Backlog-driven, so a
                   missed run is picked up automatically by the next one.

      RECOMMEND  - the next trading session is at most MAX_EVE_GAP_DAYS away
                   ("eve-of-session") and has no rows yet. Sunday->Monday is 1 day
                   so it fires; Friday->Monday is 3 days so it does not.

    Together these reproduce a Sun-Thu recommend / Mon-Fri settle rhythm without
    hardcoding any weekday, and stay correct across market holidays.
    """
    now = now or now_et()
    auto = args.mode == "auto"
    dry_run = getattr(args, "dry_run", False)
    force = getattr(args, "force", False)
    settle_date = getattr(args, "settle_date", None)
    daily_only = getattr(args, "daily_only", False)
    weekly_only = getattr(args, "weekly_only", False)
    sync_configured = getattr(sync, "is_configured", True)

    plan = {
        "now": now,
        "mode": args.mode,
        # Under continuous trading this gate is meaningless and would block every
        # run, so it is forced off and the data-driven session-agreement check in
        # the recommender becomes the sole authority.
        "market_open": (not CONTINUOUS_TRADING) and is_market_open(now),
        "settle": False,
        "settle_dates": [],
        "settle_reason": "",
        "daily": False,
        "daily_reason": "",
        "weekly": False,
        "weekly_reason": "",
        "target_session": None,
        # The session the downloaded data is required to represent. Passed down
        # to the recommender, which aborts if the provider disagrees.
        "baseline_session": last_completed_session(now),
        "gap_days": None,
        "blocked": False,
    }

    # ---------------------------------------------------------------- settle
    if args.mode in ("settle", "auto", "both"):
        if not sync_configured and dry_run:
            plan["settle_reason"] = "dry-run: Supabase credentials not configured"
        elif settle_date:
            plan["settle"] = True
            plan["settle_dates"] = [settle_date]
            plan["settle_reason"] = "explicit --settle-date"
        else:
            try:
                unsettled = sync.get_unsettled_daily_records()
            except SupabaseUnavailable as e:
                # Fail closed: an unreadable backlog is not an empty backlog.
                plan["settle_reason"] = f"fail-closed: could not read settlement backlog: {e}"
                plan["blocked"] = True
                unsettled = None

            if unsettled is not None:
                ready = sorted({
                    r["target_date"] for r in unsettled
                    if r.get("target_date")
                    and has_session_closed(datetime.strptime(r["target_date"], "%Y-%m-%d").date(), now)
                })
                pending = sorted({
                    r["target_date"] for r in unsettled
                    if r.get("target_date") and r["target_date"] not in ready
                })
                if ready:
                    plan["settle"] = True
                    plan["settle_dates"] = ready
                    plan["settle_reason"] = f"{len(ready)} session(s) closed with unsettled rows"
                elif pending:
                    plan["settle_reason"] = f"{len(pending)} row-date(s) awaiting their closing bell"
                else:
                    plan["settle_reason"] = "no unsettled rows with a closed session"
    else:
        plan["settle_reason"] = f"mode={args.mode}"

    # ------------------------------------------------------------- recommend
    recommend_modes = ("recommend", "auto", "both")
    want_daily = args.mode in recommend_modes and not weekly_only
    want_weekly = args.mode in recommend_modes and not daily_only

    # Explicit reasons for the "one half suppressed by a CLI flag" case, so
    # --explain never prints an empty justification.
    if args.mode in recommend_modes:
        if weekly_only:
            plan["daily_reason"] = "--weekly-only"
        if daily_only:
            plan["weekly_reason"] = "--daily-only"

    target = next_trading_day(last_completed_session(now))
    plan["target_session"] = target
    plan["gap_days"] = (target - now.date()).days

    if want_daily or want_weekly:
        if auto and plan["market_open"]:
            reason = "market is currently open; bars are incomplete"
            plan["daily_reason"] = plan["weekly_reason"] = reason
        elif auto and not (0 <= plan["gap_days"] <= MAX_EVE_GAP_DAYS):
            reason = f"next session is {plan['gap_days']}d away (eve-of-session allows <= {MAX_EVE_GAP_DAYS}d)"
            plan["daily_reason"] = plan["weekly_reason"] = reason
        else:
            target_str = target.isoformat()

            if want_daily:
                if not sync_configured and dry_run:
                    plan["daily"] = True
                    plan["daily_reason"] = f"dry-run preview for target_date {target_str} (Supabase unconfigured)"
                else:
                    try:
                        exists = (not force) and sync.has_daily_records_for_target(target_str)
                        plan["daily"] = not exists
                        plan["daily_reason"] = (
                            f"rows already present for target_date {target_str}" if exists
                            else f"no rows for target_date {target_str}"
                        )
                    except SupabaseUnavailable as e:
                        plan["daily_reason"] = f"fail-closed: {e}"
                        plan["blocked"] = True

            if want_weekly:
                if auto and not is_first_session_of_week(target):
                    plan["weekly_reason"] = f"{describe(target)} is not the first session of its week"
                elif not sync_configured and dry_run:
                    plan["weekly"] = True
                    plan["weekly_reason"] = f"dry-run preview for week_start_date {target_str} (Supabase unconfigured)"
                else:
                    try:
                        exists = (not force) and sync.has_weekly_records(target_str)
                        plan["weekly"] = not exists
                        plan["weekly_reason"] = (
                            f"rows already present for week_start_date {target_str}" if exists
                            else f"no rows for week_start_date {target_str}"
                        )
                    except SupabaseUnavailable as e:
                        plan["weekly_reason"] = f"fail-closed: {e}"
                        plan["blocked"] = True
    else:
        plan["daily_reason"] = plan["weekly_reason"] = f"mode={args.mode}"

    return plan


def render_plan(plan: dict):
    """Prints the resolved execution plan and why each stage was chosen."""
    now = plan["now"]
    def mark(flag):
        return "RUN " if flag else "SKIP"

    print("----------------------------------------------------------------------")
    print("🗓️  [EXECUTION PLAN]")
    print(f"      Now                  : {now.strftime('%Y-%m-%d %H:%M %Z (%a)')}")
    print(f"      Mode                 : {plan['mode']}")
    print(f"      Market currently open: {'yes' if plan['market_open'] else 'no'}")
    if plan["target_session"]:
        print(f"      Next trading session : {describe(plan['target_session'])}  ·  gap {plan['gap_days']}d")
    print("")
    settle_dates = ", ".join(plan["settle_dates"]) if plan["settle_dates"] else "-"
    print(f"      [{mark(plan['settle'])}] SETTLE          → {plan['settle_reason']}")
    if plan["settle_dates"]:
        print(f"                              dates: {settle_dates}")
    print(f"      [{mark(plan['daily'])}] RECOMMEND daily → {plan['daily_reason']}")
    print(f"      [{mark(plan['weekly'])}] RECOMMEND weekly→ {plan['weekly_reason']}")
    print("----------------------------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Daily & Weekly Quantitative Recommenders and Sync to Supabase."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "recommend", "settle", "both"],
        default="auto",
        help=(
            "auto (default): derive stages from market state -- settle any closed-session "
            "backlog, and recommend only on the eve of the next session. "
            "recommend/settle: force just that stage. both: legacy unconditional behaviour."
        )
    )
    parser.add_argument(
        "--explain", action="store_true", help="Print the resolved execution plan and exit without doing work"
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="Number of recommendations per report (default: 10)"
    )
    parser.add_argument(
        "--daily-only", action="store_true", help="Run only the daily recommender"
    )
    parser.add_argument(
        "--weekly-only", action="store_true", help="Run only the weekly recommender"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Calculate and print reports only; do not upload to Supabase"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate and re-upload PREDICTIONS even if rows already exist for the "
             "target session. Safe: settled columns are never overwritten."
    )
    parser.add_argument(
        "--resettle", action="store_true",
        help="Re-fetch and OVERWRITE closes on rows that are already settled. "
             "Destructive: yfinance returns dividend-adjusted prices that are revised "
             "retroactively, so a re-settled close will not match what was originally "
             "recorded. Use only to repair known-bad data, never in cron."
    )
    parser.add_argument(
        "--settle-date", type=str, default=None, help="Target date to settle in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--settle-only", action="store_true", help="Only run prediction settlement and scorecard update; skip new recommendations"
    )
    parser.add_argument(
        "--skip-settle", action="store_true", help="Skip prediction settlement step"
    )
    parser.add_argument(
        "--no-lock", action="store_true", help="Skip the single-instance lock (not recommended for scheduled runs)"
    )
    args = parser.parse_args()

    # Backwards compatibility: the legacy boolean flags map onto the new modes.
    # An explicit --mode always wins.
    if args.mode == "auto":
        if args.settle_only:
            args.mode = "settle"
        elif args.skip_settle:
            args.mode = "recommend"

    if args.settle_date:
        try:
            datetime.strptime(args.settle_date, "%Y-%m-%d")
        except ValueError:
            parser.error(f"--settle-date must be YYYY-MM-DD, got '{args.settle_date}'")

    # --resettle only reaches already-settled rows, and the automatic backlog
    # contains only UNsettled ones -- so on its own it would quietly do nothing.
    # Require the caller to name the date they mean to rewrite.
    if args.resettle and not args.settle_date:
        parser.error(
            "--resettle requires --settle-date YYYY-MM-DD. The automatic backlog only "
            "contains unsettled rows, so --resettle alone would have no effect."
        )

    lock_handle = None
    if not args.no_lock and not args.explain:
        lock_handle = acquire_lock()
        if lock_handle is None:
            print(f"\n⏭️  Another pipeline run already holds {LOCK_PATH}; exiting without duplicating work.\n")
            return 0

    sync = SupabaseSync()
    now = now_et()

    print("\n" + "=" * 70)
    print("      🚀 QLIB QUANTITATIVE RECOMMENDER & SUPABASE SYNC PIPELINE")
    print(f"      Execution Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 70 + "\n")

    # Run Sanity Check
    run_sanity_check(sync, dry_run=args.dry_run)

    plan = build_execution_plan(sync, args, now=now)
    render_plan(plan)

    if args.explain:
        return 0

    failures = []

    # --------------------------------------------------------------------------
    # 1. Prediction Settlement Step
    #    Backlog-driven: every closed session with outstanding rows is settled,
    #    so a missed run self-heals on the next invocation.
    # --------------------------------------------------------------------------
    if plan["settle"]:
        for settle_date in plan["settle_dates"]:
            settled = settle_target_date_predictions(
                sync=sync,
                target_date=settle_date,
                dry_run=args.dry_run,
                force=args.resettle,
                now=now
            )
            # A row that came back without a close is still outstanding. Surface
            # it rather than letting a silent gap accumulate in the accuracy
            # history -- the next run will retry it from the backlog.
            # Voided rows are a resolved outcome, not an outstanding one: the
            # session never happened, so there is nothing to wait for. Counting
            # them as failures would alert forever on a day that can never settle.
            voided = [r for r in settled if r.get("void")]
            unsettled = [
                r for r in settled
                if r.get("target_date_close") is None and not r.get("void")
            ]
            if voided:
                print(f"  🚫 [SETTLE] {settle_date}: {len(voided)} prediction(s) voided (no session).\n")
            if unsettled and not args.dry_run:
                symbols = ", ".join(sorted(r["symbol"] for r in unsettled))
                failures.append(
                    f"settlement {settle_date}: {len(unsettled)} row(s) still unsettled ({symbols})"
                )

    # --------------------------------------------------------------------------
    # 2. Daily Recommender (T+1 to T+2 Horizon)
    #    The existence check already happened during planning, so reaching here
    #    means the expensive universe download is genuinely needed.
    # --------------------------------------------------------------------------
    if plan["daily"]:
        from daily_top10_recommender import run_daily_recommender

        print("Executing Daily Top 10 Recommender...")
        baseline = plan["baseline_session"].isoformat()
        daily_records, _market_info = run_daily_recommender(
            top_n=args.top_n, expected_baseline=baseline
        )

        if not daily_records:
            # A session mismatch is by design and self-correcting, so it must not
            # be reported like a real fault -- otherwise ordinary provider lag
            # would page someone every hour until the provider caught up.
            skipped = isinstance(_market_info, dict) and _market_info.get("skipped")
            if skipped:
                print(f"  ⏭️  [DAILY] Skipped ({skipped}); will retry on the next run.\n")
            else:
                print("  ⚠️ [DAILY] Recommender produced no records; nothing to upload.\n")
                failures.append("daily recommender returned no records")
        elif args.dry_run:
            print("  ℹ️ [DAILY] Dry-run mode active. Skipped Supabase upload.\n")
        else:
            trade_date = daily_records[0]["trade_date"]
            print(f"  📤 [DAILY] Uploading {len(daily_records)} records for trade_date '{trade_date}' to '{DAILY_TABLE}'...")
            success, msg = sync.upload_daily(daily_records, force=args.force, check_existing=False)
            if success:
                print(f"  ✓ [DAILY] {msg}\n")
            else:
                print(f"  ⚠️ [DAILY] Upload notice: {msg}\n")
                failures.append(f"daily upload: {msg}")

    # --------------------------------------------------------------------------
    # 3. Weekly Recommender (1-Week Horizon)
    # --------------------------------------------------------------------------
    if plan["weekly"]:
        from weekly_top10_recommender import run_weekly_recommender

        print("Executing Weekly Top 10 Recommender...")
        weekly_records, week_start = run_weekly_recommender(top_n=args.top_n)

        if not weekly_records:
            print("  ⚠️ [WEEKLY] Recommender produced no records; nothing to upload.\n")
            failures.append("weekly recommender returned no records")
        elif args.dry_run:
            print("  ℹ️ [WEEKLY] Dry-run mode active. Skipped Supabase upload.\n")
        else:
            print(f"  📤 [WEEKLY] Uploading {len(weekly_records)} records for week '{week_start}' to '{WEEKLY_TABLE}'...")
            success, msg = sync.upload_weekly(weekly_records, force=args.force, check_existing=False)
            if success:
                print(f"  ✓ [WEEKLY] {msg}\n")
            else:
                print(f"  ⚠️ [WEEKLY] Upload notice: {msg}\n")
                failures.append(f"weekly upload: {msg}")

    print("=" * 70)
    if failures or plan["blocked"]:
        print("      ⚠️  PIPELINE COMPLETED WITH ISSUES")
        for f in failures:
            print(f"        • {f}")
        if plan["blocked"]:
            print("        • one or more existence checks could not be verified")
        print("=" * 70 + "\n")
        return 1

    print("      ✓ PIPELINE EXECUTION COMPLETE")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

