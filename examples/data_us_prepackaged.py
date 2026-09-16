"""Download and inspect pre-packaged US market data for Qlib.

Downloads the official pre-packaged Qlib US dataset via curl and extracts
it into a temporary directory (/tmp/qlib_data_us_prepackaged).
Queries at least one full year of historical data.
"""

import os
import shutil
import subprocess
from pathlib import Path
import qlib
from qlib.data import D

DATA_DIR = Path("/tmp/qlib_data_us_prepackaged")
ZIP_URL = "https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_us_1d_latest.zip"
TMP_ZIP = Path("/tmp/qlib_data_us_1d_latest.zip")


def ensure_prepackaged_data(target_dir: Path):
    """Download and extract pre-packaged US dataset if not already present."""
    cal_file = target_dir / "calendars" / "day.txt"
    if cal_file.exists():
        print(f"Pre-packaged US dataset found at {target_dir}")
        return

    print(f"Downloading pre-packaged US dataset from {ZIP_URL}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using curl
    curl_cmd = ["curl", "-L", "-o", str(TMP_ZIP), ZIP_URL]
    print(f"Running: {' '.join(curl_cmd)}")
    subprocess.run(curl_cmd, check=True)

    # Unzip into target_dir
    print(f"Extracting dataset to {target_dir}...")
    unzip_cmd = ["unzip", "-q", "-o", str(TMP_ZIP), "-d", str(target_dir)]
    subprocess.run(unzip_cmd, check=True)

    # Clean up archive
    if TMP_ZIP.exists():
        TMP_ZIP.unlink()
    print("Dataset extracted successfully.")


def main():
    ensure_prepackaged_data(DATA_DIR)

    # Initialize Qlib with the pre-packaged US data
    qlib.init(provider_uri=str(DATA_DIR), region="us")

    # Query 1 full year of US market data (2019 calendar year)
    symbols = ["AAPL", "MSFT"]
    features = ["$close", "$volume", "$high", "$low"]
    start_date = "2019-01-01"
    end_date = "2019-12-31"

    print(f"\nQuerying Qlib for {symbols} from {start_date} to {end_date} (1 Year):")
    df = D.features(symbols, features, start_time=start_date, end_time=end_date)
    
    print(f"Total rows retrieved: {len(df)}")
    print("\n--- First 5 trading days ---")
    print(df.head())
    print("\n--- Last 5 trading days ---")
    print(df.tail())


if __name__ == "__main__":
    main()
