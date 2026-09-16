import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# 1. Candidate Universe: S&P 500 / Nasdaq 100 leaders + High-Liquidity Sector & Asset ETFs
UNIVERSE = {
    # ETFs (Broad & Sector)
    "SPY": ("Broad Index", "S&P 500 ETF"),
    "QQQ": ("Tech Index", "Nasdaq 100 ETF"),
    "IWM": ("Small Cap", "Russell 2000 ETF"),
    "SMH": ("Semiconductors", "VanEck Semiconductor ETF"),
    "XLK": ("Technology", "Technology Select Sector SPDR"),
    "XLE": ("Energy", "Energy Select Sector SPDR"),
    "XLF": ("Financials", "Financial Select Sector SPDR"),
    "XLV": ("Healthcare", "Health Care Select Sector SPDR"),
    "XLI": ("Industrials", "Industrial Select Sector SPDR"),
    "GLD": ("Commodity/Hedge", "SPDR Gold Shares ETF"),
    
    # Megacap Tech & Growth Leaders
    "NVDA": ("Technology", "NVIDIA Corporation"),
    "AAPL": ("Technology", "Apple Inc."),
    "MSFT": ("Technology", "Microsoft Corporation"),
    "AMZN": ("Consumer Discretionary", "Amazon.com Inc."),
    "GOOGL": ("Communication", "Alphabet Inc."),
    "META": ("Communication", "Meta Platforms Inc."),
    "TSLA": ("Consumer Discretionary", "Tesla Inc."),
    "AVGO": ("Technology", "Broadcom Inc."),
    "AMD": ("Technology", "Advanced Micro Devices"),
    "ORCL": ("Technology", "Oracle Corporation"),
    "PLTR": ("Technology", "Palantir Technologies"),
    "NFLX": ("Communication", "Netflix Inc."),
    "CRM": ("Technology", "Salesforce Inc."),
    
    # Financials & Payments
    "JPM": ("Financials", "JPMorgan Chase & Co."),
    "GS": ("Financials", "Goldman Sachs Group"),
    "V": ("Financials", "Visa Inc."),
    "MA": ("Financials", "Mastercard Inc."),
    "BLK": ("Financials", "BlackRock Inc."),
    
    # Healthcare & Biotech
    "LLY": ("Healthcare", "Eli Lilly and Company"),
    "UNH": ("Healthcare", "UnitedHealth Group"),
    "ABBV": ("Healthcare", "AbbVie Inc."),
    "ISRG": ("Healthcare", "Intuitive Surgical"),
    
    # Industrials, Defense & Energy
    "GE": ("Industrials", "GE Aerospace"),
    "CAT": ("Industrials", "Caterpillar Inc."),
    "RTX": ("Industrials", "RTX Corporation"),
    "XOM": ("Energy", "Exxon Mobil Corporation"),
    "CVX": ("Energy", "Chevron Corporation"),
    
    # Consumer & Retail Staples
    "COST": ("Consumer Staples", "Costco Wholesale"),
    "WMT": ("Consumer Staples", "Walmart Inc."),
    "HD": ("Consumer Discretionary", "Home Depot Inc.")
}


def calculate_alpha_factors(df: pd.DataFrame, spy_df: pd.DataFrame) -> dict:
    """Calculate 1-week horizon alpha factors for a single ticker."""
    if len(df) < 55:
        return None
    
    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]
    
    current_price = close.iloc[-1]
    
    # 1. Momentum Factors
    ret_5d = (close.iloc[-1] / close.iloc[-6]) - 1.0 if len(close) >= 6 else 0.0
    ret_20d = (close.iloc[-1] / close.iloc[-21]) - 1.0 if len(close) >= 21 else 0.0
    
    # Benchmark Relative Strength vs SPY over 20 days
    spy_ret_20d = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-21]) - 1.0 if len(spy_df) >= 21 else 0.0
    rs_spy = ret_20d - spy_ret_20d
    
    # 2. Trend & Moving Average Alignment
    ema_10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
    ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    
    trend_score = 0
    if current_price > ema_10: trend_score += 1
    if ema_10 > ema_20: trend_score += 1
    if ema_20 > sma_50: trend_score += 1
    if current_price > sma_50: trend_score += 1
    
    # 3. Volatility & Risk-Adjusted Quality
    daily_rets = close.pct_change().dropna().iloc[-20:]
    vol_20d = daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0.001
    sharpe_proxy = (daily_rets.mean() * 252) / vol_20d if vol_20d > 0 else 0.0
    
    # 4. Volume Flow & Institutional Participation
    vol_ma20 = volume.iloc[-20:].mean() if len(volume) >= 20 else 1.0
    vol_ratio = (volume.iloc[-5:].mean() / vol_ma20) if vol_ma20 > 0 else 1.0
    
    # 5. Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14 = (100 - (100 / (1 + rs))).iloc[-1]
    if np.isnan(rsi_14):
        rsi_14 = 50.0

    # 6. Average True Range (ATR 14) for stop-loss recommendation
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean().iloc[-1]
    
    return {
        "price": current_price,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "rs_spy": rs_spy,
        "trend_score": trend_score,
        "vol_20d": vol_20d,
        "sharpe_proxy": sharpe_proxy,
        "vol_ratio": vol_ratio,
        "rsi_14": rsi_14,
        "atr_14": atr_14,
        "ema_10": ema_10,
        "ema_20": ema_20
    }


def run_weekly_recommender(top_n: int = 10, export_csv: str = None):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching market data for universe of {len(UNIVERSE)} US assets...")
    tickers_list = list(UNIVERSE.keys())
    
    # Download last 6 months data for all tickers
    data = yf.download(tickers_list, period="6mo", interval="1d", progress=False, group_by="ticker")
    
    if "SPY" not in data:
        print("Error: Could not retrieve SPY benchmark data.")
        return [], ""

    # Ensure latest session bar is complete if today's row exists with NaN close
    last_idx = data.index[-1]
    if ("SPY", "Close") in data.columns and pd.isna(data.loc[last_idx, ("SPY", "Close")]):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Daily bars for {last_idx.strftime('%Y-%m-%d')} pending consolidation; syncing session from fast_info...")
        for sym in tickers_list:
            if (sym, "Close") in data.columns:
                try:
                    t = yf.Ticker(sym)
                    fi = t.fast_info
                    lp = getattr(fi, "last_price", None) or fi.get("lastPrice")
                    if lp is not None and not np.isnan(lp) and lp > 0:
                        op = getattr(fi, "open", None) or fi.get("open") or lp
                        hi = getattr(fi, "day_high", None) or fi.get("dayHigh") or lp
                        lo = getattr(fi, "day_low", None) or fi.get("dayLow") or lp
                        vol = getattr(fi, "last_volume", None) or fi.get("lastVolume")
                        data.loc[last_idx, (sym, "Close")] = round(float(lp), 2)
                        data.loc[last_idx, (sym, "Open")] = round(float(op), 2)
                        data.loc[last_idx, (sym, "High")] = round(float(hi), 2)
                        data.loc[last_idx, (sym, "Low")] = round(float(lo), 2)
                        if vol is not None and not np.isnan(vol):
                            data.loc[last_idx, (sym, "Volume")] = vol
                except Exception:
                    pass

    spy_df = data["SPY"].dropna()
    last_dt = spy_df.index[-1]
    days_until_monday = (7 - last_dt.weekday()) % 7
    if days_until_monday == 0 and last_dt.weekday() != 0:
        days_until_monday = 7
    week_start = (last_dt + pd.Timedelta(days=days_until_monday)).strftime("%Y-%m-%d")

    results = []
    for symbol, (sector, name) in UNIVERSE.items():
        try:
            if symbol not in data:
                continue
            df = data[symbol].dropna()
            if len(df) < 50:
                continue
            
            metrics = calculate_alpha_factors(df, spy_df)
            if metrics is None:
                continue
                
            metrics["symbol"] = symbol
            metrics["sector"] = sector
            metrics["name"] = name
            results.append(metrics)
        except Exception:
            pass

    if not results:
        print("Error: No qualifying candidate data processed.")
        return [], week_start

    res_df = pd.DataFrame(results)
    
    # Standardize factors (Z-scores)
    z_mom = (res_df["ret_20d"] - res_df["ret_20d"].mean()) / (res_df["ret_20d"].std() or 1.0)
    z_rs = (res_df["rs_spy"] - res_df["rs_spy"].mean()) / (res_df["rs_spy"].std() or 1.0)
    z_sharpe = (res_df["sharpe_proxy"] - res_df["sharpe_proxy"].mean()) / (res_df["sharpe_proxy"].std() or 1.0)
    z_vol = (res_df["vol_ratio"] - res_df["vol_ratio"].mean()) / (res_df["vol_ratio"].std() or 1.0)
    z_trend = (res_df["trend_score"] - res_df["trend_score"].mean()) / (res_df["trend_score"].std() or 1.0)
    
    # RSI penalty for extreme overbought (>75) or weak (<40)
    rsi_penalty = np.where(res_df["rsi_14"] > 75, -0.5, 0.0) + np.where(res_df["rsi_14"] < 40, -1.0, 0.0)

    # Composite Alpha Score for 1-Week Holding Horizon:
    composite_score = (
        0.20 * z_mom +
        0.15 * z_rs +
        0.25 * z_trend +
        0.20 * z_sharpe +
        0.20 * z_vol +
        rsi_penalty
    )
    res_df["alpha_score"] = composite_score
    
    # Sort by alpha score
    sorted_df = res_df.sort_values(by="alpha_score", ascending=False).reset_index(drop=True)
    
    # Portfolio Construction with Sector Diversification Constraints:
    selected = []
    sector_counts = {}
    
    for _, row in sorted_df.iterrows():
        sec = row["sector"]
        if sector_counts.get(sec, 0) < 2 or sec in ["Broad Index", "Commodity/Hedge"]:
            selected.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected) == top_n:
            break
            
    top10_df = pd.DataFrame(selected).reset_index(drop=True)
    
    print("\n" + "=" * 85)
    print(f"      QUANTITATIVE TOP {top_n} RECOMMENDATIONS (1-WEEK HORIZON: WEEK OF {week_start})")
    print("=" * 85)
    print(f"{'Rank':<4} {'Symbol':<6} {'Sector':<16} {'Price':<8} {'5D%':<7} {'20D%':<8} {'RSI':<6} {'Score':<6} {'Stop-Loss':<10} {'Target':<10}")
    print("-" * 85)
    
    db_rows = []
    for i, r in top10_df.iterrows():
        p = r['price']
        atr = r['atr_14']
        stop_loss = round(max(p - 1.5 * atr, p * 0.965), 2)  # Max 3.5% risk or 1.5 ATR
        target = round(p + 2.5 * atr, 2)
        print(f"#{i+1:<3} {r['symbol']:<6} {r['sector']:<16} ${p:<7.2f} {r['ret_5d']*100:>+5.1f}% {r['ret_20d']*100:>+5.1f}% {r['rsi_14']:<5.1f} {r['alpha_score']:<6.2f} ${stop_loss:<9.2f} ${target:<9.2f}")

        db_rows.append({
            "week_start_date": week_start,
            "rank": i + 1,
            "symbol": r["symbol"],
            "name": r["name"],
            "sector": r["sector"],
            "close_price": round(p, 2),
            "ret_5d": round(r["ret_5d"], 4),
            "ret_20d": round(r["ret_20d"], 4),
            "vol_ratio": round(r["vol_ratio"], 2),
            "rsi_14": round(r["rsi_14"], 2),
            "alpha_score": round(r["alpha_score"], 2),
            "stop_loss": stop_loss,
            "target_price": target
        })
        
    print("=" * 85 + "\n")

    if export_csv:
        export_path = os.path.abspath(export_csv)
        pd.DataFrame(db_rows).to_csv(export_path, index=False)
        print(f"✓ Successfully exported recommendations to: {export_path}\n")

    return db_rows, week_start


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weekly Quantitative Top 10 Recommender")
    parser.add_argument("--top_n", type=int, default=10, help="Number of assets to recommend (default: 10)")
    parser.add_argument("--csv", type=str, default=None, help="Optional output CSV path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Explicit dry run: calculate and inspect results only, without writing to Supabase (default behavior)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        default=False,
        help="Upload results to Supabase table 'qlib_weekly_top10_recommend'"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force upload to Supabase even if records for this week already exist"
    )
    args = parser.parse_args()

    # Inform user of execution mode upfront
    if args.dry_run or not args.upload:
        print("\n🔍 [DRY-RUN MODE] Calculating and displaying recommendations to screen only.")
        print("   (No data will be written to Supabase database. Pass --upload to enable cloud sync)\n")

    records, week_start = run_weekly_recommender(top_n=args.top_n, export_csv=args.csv)

    if args.upload and not args.dry_run:
        try:
            from run_and_upload_recommendations import SupabaseSync, WEEKLY_TABLE
            sync = SupabaseSync()
            print(f"[Supabase] Uploading {len(records)} records for week '{week_start}' to '{WEEKLY_TABLE}'...")
            success, msg = sync.upload_weekly(records, force=args.force)
            if success:
                print(f"  ✓ {msg}\n")
            else:
                print(f"  ℹ️ {msg}\n")
        except Exception as e:
            print(f"  ⚠️ Supabase upload notice: {e}\n")
    else:
        print("✓ Dry run complete: Inspected output safely without writing to Supabase database.\n")
