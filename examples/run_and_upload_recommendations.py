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
from datetime import datetime
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np
import yfinance as yf

# Ensure current directory is on python path for importing sister scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from daily_top10_recommender import run_daily_recommender
from weekly_top10_recommender import run_weekly_recommender

DAILY_TABLE = "qlib_daily_top10_recommend"
WEEKLY_TABLE = "qlib_weekly_top10_recommend"


def load_dotenv(override: bool = True):
    """Loads environment variables from local .env files if present."""
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
        }
        if prefer_merge:
            headers["Prefer"] = "resolution=merge-duplicates"
        elif return_representation:
            headers["Prefer"] = "return=representation"
        return headers

    def test_connection(self) -> tuple:
        """Pings Supabase REST endpoint to verify authorization and table accessibility."""
        query_url = f"{self.url}/rest/v1/{DAILY_TABLE}?limit=1&select=id"
        req = urllib.request.Request(query_url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                return True, f"HTTP {resp.status} OK"
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            return False, f"HTTP {e.code}: {err_body}"
        except Exception as e:
            return False, str(e)

    def has_daily_records(self, trade_date: str) -> bool:
        """Checks if daily recommendations already exist for the specified trade_date."""
        query_url = f"{self.url}/rest/v1/{DAILY_TABLE}?trade_date=eq.{trade_date}&limit=1&select=id"
        req = urllib.request.Request(query_url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return len(data) > 0
        except Exception as e:
            print(f"[SupabaseSync] Query notice: {e}")
            return False

    def has_weekly_records(self, week_start_date: str) -> bool:
        """Checks if weekly recommendations already exist for the specified week_start_date."""
        query_url = f"{self.url}/rest/v1/{WEEKLY_TABLE}?week_start_date=eq.{week_start_date}&limit=1&select=id"
        req = urllib.request.Request(query_url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return len(data) > 0
        except Exception as e:
            print(f"[SupabaseSync] Query notice: {e}")
            return False

    def upload_daily(self, records: list, force: bool = False) -> tuple:
        """Uploads daily recommendations to 'qlib_daily_top10_recommend'."""
        if not records:
            return False, "No daily records provided."

        trade_date = records[0].get("trade_date")
        if not force and trade_date and self.has_daily_records(trade_date):
            return (
                False,
                f"Skipped daily upload: Records for trade_date '{trade_date}' already exist in Supabase (use --force to overwrite)."
            )

        endpoint = f"{self.url}/rest/v1/{DAILY_TABLE}"
        payload = json.dumps(records).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers=self._headers(prefer_merge=True), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status in (200, 201, 204):
                    return True, f"Successfully uploaded {len(records)} daily records for {trade_date} to '{DAILY_TABLE}'."
                return False, f"Supabase responded with status {resp.status}."
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            return False, f"Supabase HTTP {e.code} Error: {err_body}"
        except Exception as e:
            return False, f"Failed to upload daily records: {e}"

    def upload_weekly(self, records: list, force: bool = False) -> tuple:
        """Uploads weekly recommendations to 'qlib_weekly_top10_recommend'."""
        if not records:
            return False, "No weekly records provided."

        week_start_date = records[0].get("week_start_date")
        if not force and week_start_date and self.has_weekly_records(week_start_date):
            return (
                False,
                f"Skipped weekly upload: Records for week_start_date '{week_start_date}' already exist in Supabase (use --force to overwrite)."
            )

        endpoint = f"{self.url}/rest/v1/{WEEKLY_TABLE}"
        payload = json.dumps(records).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers=self._headers(prefer_merge=True), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status in (200, 201, 204):
                    return True, f"Successfully uploaded {len(records)} weekly records for {week_start_date} to '{WEEKLY_TABLE}'."
                return False, f"Supabase responded with status {resp.status}."
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            return False, f"Supabase HTTP {e.code} Error: {err_body}"
        except Exception as e:
            return False, f"Failed to upload weekly records: {e}"

    def get_records_for_target_date(self, target_date: str) -> list:
        """Fetches rows from qlib_daily_top10_recommend where target_date matches."""
        query_url = f"{self.url}/rest/v1/{DAILY_TABLE}?target_date=eq.{target_date}&select=*&order=rank.asc,id.asc"
        req = urllib.request.Request(query_url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[SupabaseSync] Query target_date notice: {e}")
            return []

    def get_unsettled_daily_records(self) -> list:
        """Fetches rows from qlib_daily_top10_recommend where target_date_close is NULL."""
        query_url = f"{self.url}/rest/v1/{DAILY_TABLE}?target_date_close=is.null&select=*&order=target_date.asc,rank.asc,id.asc"
        req = urllib.request.Request(query_url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[SupabaseSync] Query unsettled records notice: {e}")
            return []

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
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status in (200, 204):
                    return True, f"Record {record_id} successfully updated"
                return False, f"Record {record_id} update returned HTTP status {resp.status}"
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            return False, f"Supabase HTTP {e.code} Error: {err_body}"
        except Exception as e:
            return False, f"Failed to update record {record_id}: {e}"


def fetch_target_date_close(symbol: str, target_date_str: str) -> float:
    """Fetches actual closing price for a stock on target_date.
    
    Handles today's live/closing price (including fast_info fallback) and historical sessions.
    """
    try:
        t = yf.Ticker(symbol)
        today_str = datetime.now().strftime("%Y-%m-%d")

        # If target_date is today or later, query fast_info or 1m interval for latest live/settled price
        if target_date_str >= today_str:
            if hasattr(t, "fast_info"):
                lp = getattr(t.fast_info, "last_price", None) or t.fast_info.get("lastPrice")
                if lp is not None and not np.isnan(lp) and lp > 0:
                    return round(float(lp), 2)
            m1 = t.history(period="1d", interval="1m")
            if not m1.empty and "Close" in m1.columns:
                last_c = m1["Close"].dropna()
                if not last_c.empty:
                    return round(float(last_c.iloc[-1]), 2)

        # Check 1-month daily history
        hist = t.history(period="1mo")
        if not hist.empty and "Close" in hist.columns:
            hist_dates = [str(d.date()) for d in hist.index]
            if target_date_str in hist_dates:
                idx = hist_dates.index(target_date_str)
                val = hist["Close"].iloc[idx]
                if not np.isnan(val) and val > 0:
                    return round(float(val), 2)

            # If today's bar had NaN Close in daily history, check fast_info
            if hasattr(t, "fast_info"):
                lp = getattr(t.fast_info, "last_price", None) or t.fast_info.get("lastPrice")
                if lp is not None and not np.isnan(lp) and lp > 0:
                    return round(float(lp), 2)

            # Fallback to the latest available non-NaN close
            valid_closes = hist["Close"].dropna()
            if not valid_closes.empty:
                return round(float(valid_closes.iloc[-1]), 2)
    except Exception as e:
        print(f"  ⚠️ Error fetching close price for {symbol}: {e}")
    return None


def settle_target_date_predictions(sync: SupabaseSync, target_date: str = None, dry_run: bool = False, force: bool = False) -> list:
    """Inspects Supabase table for records with target_date == today (or target_date),
    fetches actual closing prices, fills target_date_close, and computes predicted_diff.
    """
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 115)
    print("      🎯 STEP 1: PREDICTION SETTLEMENT & ACCURACY EVALUATION")
    print(f"      Target Settlement Date: {target_date}")
    print(f"      Execution Mode        : {'DRY RUN (Preview Only)' if dry_run else 'ACTIVE (Updating Supabase)'}")
    print("=" * 115)

    # Fetch records matching target_date
    records = sync.get_records_for_target_date(target_date)

    if not records:
        print(f"  ℹ️ No daily recommendations found in Supabase with target_date = '{target_date}'.")
        unsettled = sync.get_unsettled_daily_records()
        if unsettled:
            earlier_dates = sorted(list(set(r["target_date"] for r in unsettled if r.get("target_date") and r["target_date"] < target_date)))
            if earlier_dates:
                print(f"  💡 Notice: Found unsettled records for earlier dates: {earlier_dates}. (Run with --settle-date <DATE> to settle them)\n")
        print("-" * 115 + "\n")
        return []

    print(f"  ✓ Found {len(records)} recommendation record(s) with target_date '{target_date}'.")
    print("  📊 Fetching actual market close prices and calculating prediction deltas...\n")

    header = f"{'Rank':<5} {'Symbol':<7} {'Trade Date':<11} {'Target Date':<12} {'Pred Close':<12} {'Actual Close':<14} {'Diff ($)':<11} {'Diff (%)':<10} {'Status':<18}"
    print("-" * 115)
    print(header)
    print("-" * 115)

    settled_results = []
    abs_diffs = []
    pct_diffs = []

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
        else:
            actual_close = fetch_target_date_close(symbol, rec_target_date)
            if actual_close is not None:
                if pred_close is not None:
                    pred_diff = round(actual_close - pred_close, 2)
                else:
                    pred_diff = None

                if dry_run:
                    status_str = "Dry Run (Preview)"
                else:
                    ok, msg = sync.update_daily_settlement(rec_id, actual_close, pred_diff)
                    status_str = "✓ Settled" if ok else f"⚠️ {msg}"
            else:
                pred_diff = None
                status_str = "⚠️ Price Missing"

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


def main():
    parser = argparse.ArgumentParser(
        description="Run Daily & Weekly Quantitative Recommenders and Sync to Supabase."
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
        "--force", action="store_true", help="Force upload/overwrite even if today's records already exist"
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
    args = parser.parse_args()

    sync = SupabaseSync()

    print("\n" + "=" * 70)
    print("      🚀 QLIB QUANTITATIVE RECOMMENDER & SUPABASE SYNC PIPELINE")
    print(f"      Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Run Sanity Check
    run_sanity_check(sync, dry_run=args.dry_run)

    # --------------------------------------------------------------------------
    # 1. Prediction Settlement Step (The New Feature)
    # --------------------------------------------------------------------------
    if not args.skip_settle:
        settle_target_date_predictions(
            sync=sync,
            target_date=args.settle_date,
            dry_run=args.dry_run,
            force=args.force
        )

    if args.settle_only:
        print("=" * 70)
        print("      ✓ PREDICTION SETTLEMENT COMPLETE (--settle-only specified)")
        print("=" * 70 + "\n")
        return

    # --------------------------------------------------------------------------
    # 2. Daily Recommender (T+1 to T+2 Horizon)
    # --------------------------------------------------------------------------
    if not args.weekly_only:
        print("[2/3] Executing Daily Top 10 Recommender...")
        daily_records, market_info = run_daily_recommender(top_n=args.top_n)

        if not args.dry_run and daily_records:
            trade_date = daily_records[0]["trade_date"]
            print(f"[Supabase] Evaluating daily upload status for trade_date '{trade_date}'...")
            
            if sync.has_daily_records(trade_date) and not args.force:
                print(f"  ℹ️ [DAILY] Records for trade_date '{trade_date}' already exist in '{DAILY_TABLE}'.")
                print("  ⏭️ [DAILY] Skipping upload to ensure single-upload per day. (Pass --force to overwrite)\n")
            else:
                print(f"  📤 [DAILY] Uploading {len(daily_records)} records to '{DAILY_TABLE}'...")
                success, msg = sync.upload_daily(daily_records, force=args.force)
                if success:
                    print(f"  ✓ [DAILY] {msg}\n")
                else:
                    print(f"  ⚠️ [DAILY] Upload notice: {msg}\n")
        elif args.dry_run:
            print("  ℹ️ [DAILY] Dry-run mode active. Skipped Supabase upload.\n")

    # --------------------------------------------------------------------------
    # 3. Weekly Recommender (1-Week Horizon)
    # --------------------------------------------------------------------------
    if not args.daily_only:
        print("[3/3] Executing Weekly Top 10 Recommender...")
        weekly_records, week_start = run_weekly_recommender(top_n=args.top_n)

        if not args.dry_run and weekly_records:
            print(f"[Supabase] Evaluating weekly upload status for week_start_date '{week_start}'...")
            
            if sync.has_weekly_records(week_start) and not args.force:
                print(f"  ℹ️ [WEEKLY] Records for week '{week_start}' already exist in '{WEEKLY_TABLE}'.")
                print("  ⏭️ [WEEKLY] Skipping upload to ensure single-upload per week. (Pass --force to overwrite)\n")
            else:
                print(f"  📤 [WEEKLY] Uploading {len(weekly_records)} records to '{WEEKLY_TABLE}'...")
                success, msg = sync.upload_weekly(weekly_records, force=args.force)
                if success:
                    print(f"  ✓ [WEEKLY] {msg}\n")
                else:
                    print(f"  ⚠️ [WEEKLY] Upload notice: {msg}\n")
        elif args.dry_run:
            print("  ℹ️ [WEEKLY] Dry-run mode active. Skipped Supabase upload.\n")


    print("=" * 70)
    print("      ✓ PIPELINE EXECUTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
