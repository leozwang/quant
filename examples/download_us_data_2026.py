#!/usr/bin/env python3
"""Download US Stock and ETF Data from Yahoo Finance for 2026 Jan to Aug.

Fetches 1D OHLCV bars from 2026-01-01 to 2026-08-31 (~166 trading days)
and saves them in both CSV and Qlib high-performance binary (.bin) layout.
Reports exact row counts and storage footprints.
"""

import argparse
import os
import shutil
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

# Standard Core Universe (53 high-liquidity stocks and active ETFs)
CORE_UNIVERSE = [
    # Broad & Sector ETFs
    "SPY", "QQQ", "IWM", "SMH", "XLK", "XLE", "XLF", "XLV", "XLI", "GLD",
    # Megacap Tech & Growth
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    "ORCL", "PLTR", "NFLX", "CRM", "MU", "QCOM", "ARM", "TSM",
    # High-Beta & Growth Momentum
    "COIN", "HOOD", "SHOP", "APP", "MSTR",
    # Financials & Payments
    "JPM", "GS", "MS", "V", "MA", "BLK",
    # Healthcare & Biotech
    "LLY", "UNH", "ABBV", "ISRG", "NVO",
    # Industrials & Defense
    "GE", "CAT", "RTX", "LMT",
    # Energy & Commodities
    "XOM", "CVX", "CCJ",
    # Consumer Staples & Retail
    "COST", "WMT", "HD"
]

# S&P 500 Sample Universe (Representative Top 100 Holdings)
SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "AVGO", "JPM",
    "LLY", "UNH", "V", "XOM", "MA", "COST", "HD", "PG", "JNJ", "WMT",
    "ABBV", "BAC", "KO", "MRK", "CVX", "CRM", "NFLX", "AMD", "PEP", "ORCL",
    "TMO", "LIN", "WFC", "CSCO", "ACN", "MCD", "ABT", "ADBE", "IBM", "PM",
    "GE", "TXN", "ISRG", "CAT", "QCOM", "AMAT", "NOW", "INTU", "VZ", "GS",
    "CMCSA", "DIS", "PFE", "UBER", "AXP", "MS", "SPGI", "RTX", "BKNG", "HON",
    "LOW", "COP", "UNP", "AMGN", "T", "SYK", "PGR", "BLK", "ETN", "BSX",
    "TJX", "PANW", "DE", "VRTX", "LMT", "BA", "MDT", "CB", "PLTR", "ADI",
    "CI", "MMC", "GILD", "FI", "ADP", "SBUX", "C", "SO", "BMY", "DUK",
    "SCHW", "MDLZ", "CL", "ICE", "REGN", "SHW", "MO", "EOG", "ZTS", "WM"
]

START_DATE = "2026-01-01"
END_DATE = "2026-09-01"  # Inclusive of August 31, 2026


def download_and_size_data(tickers: list, output_dir: Path):
    print("\n" + "=" * 70)
    print(" 📥 YAHOO FINANCE US MARKET DATA DOWNLOAD (JAN 2026 - AUG 2026)")
    print(f" Target Horizon : {START_DATE} to 2026-08-31 (~166 Trading Days)")
    print(f" Asset Count    : {len(tickers)} symbols")
    print(f" Output Folder  : {output_dir}")
    print("=" * 70 + "\n")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    qlib_dir = output_dir / "qlib_bin"
    feat_dir = qlib_dir / "features"
    cal_dir = qlib_dir / "calendars"
    inst_dir = qlib_dir / "instruments"
    feat_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading historical daily bars from Yahoo Finance for {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, group_by="ticker", progress=False)

    # Establish common trading calendar (using SPY or first valid ticker)
    benchmark_key = "SPY" if "SPY" in data else list(tickers)[0]
    bench_df = data[benchmark_key].dropna()
    calendars = [d.strftime("%Y-%m-%d") for d in bench_df.index]

    with open(cal_dir / "day.txt", "w", encoding="utf-8") as f:
        for d in calendars:
            f.write(f"{d}\n")

    valid_tickers = []
    total_rows = 0
    fields = ["open", "high", "low", "close", "volume", "factor"]
    col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}

    inst_lines = []
    for s in tickers:
        if s not in data:
            continue
        df = data[s].dropna()
        if df.empty or len(df) < 20:
            continue

        valid_tickers.append(s)
        total_rows += len(df)

        # 1. Save raw CSV
        df.to_csv(csv_dir / f"{s}.csv")

        # 2. Serialize into Qlib high-performance binary (.bin) layout
        s_feat_dir = feat_dir / s.lower()
        s_feat_dir.mkdir(parents=True, exist_ok=True)

        s_df = df.copy()
        s_df.index = s_df.index.strftime("%Y-%m-%d")
        s_df = s_df.reindex(calendars)

        start_date = s_df.dropna(subset=["Close"]).index.min()
        date_idx = calendars.index(start_date)
        s_slice = s_df.loc[start_date:]

        for fld in fields:
            if fld in col_map and col_map[fld] in s_slice.columns:
                vals = s_slice[col_map[fld]].fillna(0).values.astype("<f")
            elif fld == "factor":
                vals = np.ones(len(s_slice), dtype="<f")
            else:
                vals = np.zeros(len(s_slice), dtype="<f")

            bin_data = np.hstack([[float(date_idx)], vals]).astype("<f")
            (s_feat_dir / f"{fld}.day.bin").write_bytes(bin_data.tobytes())

        inst_lines.append(f"{s}\t{df.index[0].strftime('%Y-%m-%d')}\t{df.index[-1].strftime('%Y-%m-%d')}\n")

    with open(inst_dir / "all.txt", "w", encoding="utf-8") as f:
        f.writelines(inst_lines)

    # Calculate exact directory sizes
    csv_bytes = sum(f.stat().st_size for f in csv_dir.glob("*.csv"))
    qlib_bytes = sum(f.stat().st_size for f in qlib_dir.rglob("*") if f.is_file())

    print("\n" + "=" * 70)
    print("                  DATA SIZE & STORAGE SUMMARY")
    print("=" * 70)
    print(f" • Successfully Downloaded : {len(valid_tickers)} of {len(tickers)} tickers")
    print(f" • Trading Session Span    : {calendars[0]} to {calendars[-1]} ({len(calendars)} trading days)")
    print(f" • Total OHLCV Data Rows   : {total_rows:,} records")
    print("-" * 70)
    print(f" • Raw CSV Files Total     : {csv_bytes / 1024:.1f} KB  ({csv_bytes / (1024*1024):.2f} MB)")
    print(f"   ↳ Average per ticker    : {csv_bytes / len(valid_tickers) / 1024:.1f} KB")
    print(f" • Qlib Binary Format (.bin): {qlib_bytes / 1024:.1f} KB  ({qlib_bytes / (1024*1024):.2f} MB)")
    print(f"   ↳ Average per ticker    : {qlib_bytes / len(valid_tickers) / 1024:.1f} KB")
    print("-" * 70)
    print(f" 📂 Data Saved To:")
    print(f"    • CSV  : {csv_dir}")
    print(f"    • Qlib : {qlib_dir}")
    print("=" * 70 + "\n")

    return {
        "ticker_count": len(valid_tickers),
        "trading_days": len(calendars),
        "total_rows": total_rows,
        "csv_bytes": csv_bytes,
        "qlib_bytes": qlib_bytes
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Yahoo Finance US Data for 2026 Jan to Aug")
    parser.add_argument(
        "--universe",
        choices=["core50", "sp100", "all"],
        default="core50",
        help="Asset universe to download: core50 (53 stocks/ETFs) or sp100 (top 100 S&P)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/yahoo_data_2026_jan_aug",
        help="Target output directory"
    )
    args = parser.parse_args()

    selected_tickers = CORE_UNIVERSE if args.universe == "core50" else SP500_SAMPLE
    download_and_size_data(selected_tickers, Path(args.output))
