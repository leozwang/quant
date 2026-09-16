"""Live Crawl US market data from Yahoo Finance and dump into Qlib format.

Fetches the last 30 days of daily trading data from Yahoo Finance for
selected US equities, converts/dumps the data into Qlib's native binary
storage format in (/tmp/qlib_data_us_live), and queries the features
using Qlib's D.features() API.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from yahooquery import Ticker
import qlib
from qlib.data import D

DATA_DIR = Path("/tmp/qlib_data_us_live")
SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
FIELDS = ["open", "close", "high", "low", "volume", "factor"]


def crawl_and_dump_yahoo_data(symbols: list, target_dir: Path, days: int = 30):
    """Crawl data from Yahoo Finance and serialize into Qlib binary format."""
    print(f"Fetching live data from Yahoo Finance for {symbols} (last {days} days)...")
    
    # 1. Fetch live data via yahooquery
    ticker = Ticker(symbols)
    raw_df = ticker.history(period=f"{days}d", interval="1d")
    
    if raw_df is None or raw_df.empty:
        raise RuntimeError("Failed to fetch live data from Yahoo Finance.")
    
    df = raw_df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # Calculate price adjustment factor
    if "adjclose" in df.columns and "close" in df.columns:
        df["factor"] = df["adjclose"] / df["close"]
    else:
        df["factor"] = 1.0

    # 2. Setup Qlib directory structure
    if target_dir.exists():
        shutil.rmtree(target_dir)
        
    cal_dir = target_dir / "calendars"
    inst_dir = target_dir / "instruments"
    feat_dir = target_dir / "features"
    cal_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # 3. Dump calendars
    calendars = sorted(df["date"].unique())
    cal_path = cal_dir / "day.txt"
    with open(cal_path, "w", encoding="utf-8") as f:
        for d in calendars:
            f.write(f"{d}\n")
    print(f"Dumped {len(calendars)} calendar days: {calendars[0]} to {calendars[-1]}")

    # 4. Dump instruments list
    inst_path = inst_dir / "all.txt"
    with open(inst_path, "w", encoding="utf-8") as f:
        for s in sorted(symbols):
            s_df = df[df["symbol"] == s]
            if not s_df.empty:
                f.write(f"{s}\t{s_df['date'].min()}\t{s_df['date'].max()}\n")

    # 5. Dump features in Qlib binary layout (<start_date_index: float32> + <values: float32...>)
    for s in symbols:
        s_df = df[df["symbol"] == s].copy()
        if s_df.empty:
            continue
        s_feat_dir = feat_dir / s.lower()
        s_feat_dir.mkdir(parents=True, exist_ok=True)

        s_df = s_df.set_index("date").reindex(calendars)
        s_valid = s_df.dropna(subset=["close"])
        if s_valid.empty:
            continue

        start_date = s_valid.index.min()
        date_index = calendars.index(start_date)
        s_slice = s_df.loc[start_date:s_valid.index.max()]

        for field in FIELDS:
            col = field if field in s_slice.columns else None
            if col:
                vals = s_slice[col].fillna(0).values.astype("<f")
            else:
                vals = np.zeros(len(s_slice), dtype="<f")
            bin_data = np.hstack([[float(date_index)], vals]).astype("<f")
            bin_path = s_feat_dir / f"{field}.day.bin"
            bin_data.tofile(str(bin_path))

    print(f"Successfully converted and dumped into Qlib format at {target_dir}")


def main():
    # Crawl last 30 days and dump into /tmp/qlib_data_us_live
    crawl_and_dump_yahoo_data(SYMBOLS, DATA_DIR, days=30)

    # Initialize Qlib with the freshly generated live dataset
    qlib.init(provider_uri=str(DATA_DIR), region="us")

    # Query the live features via Qlib's native API
    features = ["$close", "$volume", "$high", "$low"]
    print(f"\nQuerying Qlib native features for {SYMBOLS[:2]} across live calendar range:")
    df = D.features(SYMBOLS[:2], features)

    print(f"Total rows retrieved: {len(df)}")
    print("\n--- Recent Trading Days Sample ---")
    print(df.tail(10))


if __name__ == "__main__":
    main()
