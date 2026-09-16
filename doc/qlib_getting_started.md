# Getting Started with Microsoft Qlib

Welcome to **Qlib**, an AI-oriented quantitative investment platform designed for financial market prediction and strategy analysis. This guide covers installation, data preparation, modeling, and running a complete backtesting pipeline.

---

## 📋 Table of Contents
1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Data Initialization](#2-data-initialization)
   - [Pre-Packaged Dump (Equities & ETFs)](#pre-packaged-dump-equities--etfs)
   - [Acquiring Data from 2020 to Today (2026)](#acquiring-data-from-2020-to-today-2026)
3. [Workflow Architecture](#3-workflow-architecture)
4. [Running with qrun](#4-running-with-qrun)
5. [Advanced Usage (API)](#5-advanced-usage-api)

---

## 1. Prerequisites & Installation

### Environment & Python Version

> [!IMPORTANT]
> Qlib natively supports **Python 3.8 to 3.12**. It relies on C/Cython extensions with Python C-API bindings that can fail to compile on Python 3.13+.
> If your default system or conda base environment is Python 3.13+, create and activate a dedicated Python 3.11 environment:
> ```bash
> conda create -n qlib python=3.11 -y
> conda activate qlib
> ```

### Upstream Repository Checkout & Workspace Layout

> [!NOTE]
> **Workspace Layout**: Do **not** clone Microsoft's `qlib` repository directly inside your Git repository (such as directly inside `src/qlib/`), as this creates a nested Git repository conflict and pollutes your project tree with upstream C/C++ compilation artifacts.
>
> Instead, clone the upstream Qlib source into an external folder such as `~/src/qlib-source`. Keep your quantitative strategy folder clean for your own strategies, configurations, models, and custom data scripts.

```bash
# 1. Clone upstream Qlib to an external directory outside your project repository
git clone https://github.com/microsoft/qlib.git ~/src/qlib-source
cd ~/src/qlib-source

# 2. Install critical pre-requisites
pip install numpy
pip install --upgrade cython

# 3. Build and install into your active Python environment
# Note: Editable mode (`pip install -e .`) is recommended so upstream scripts,
# data download utilities, and benchmark configs remain accessible.
pip install -e .
```

---

## 2. Data Initialization

Qlib stores financial datasets in a high-performance binary format (`.bin`). You need to download or convert historical data into this layout before initialization.

### Pre-Packaged Dump (Equities & ETFs)

Microsoft Qlib provides pre-packaged binary datasets for quick offline development and benchmark reproducibility.

- **Coverage**: Contains **8,994 US equities and ETFs** across NYSE, NASDAQ, and AMEX.
  - **Equities**: Apple, Microsoft, NVIDIA, Amazon, etc.
  - **ETFs**: Broad market (`SPY`, `QQQ`, `DIA`, `IWM`, `VTI`, `VOO`), Sector SPDRs (`XLK`, `XLF`, `XLE`, `XLV`), Fixed Income & Commodities (`TLT`, `GLD`, `SLV`), and Thematic (`ARKK`).
  - **Index Lists**: Predefined universe filters in `instruments/` (`sp500.txt`, `nasdaq100.txt`, `all.txt`).
- **Time Horizon**: Covers **December 31, 1999 through November 10, 2020** (5,250 trading days).
- **Features Included**: `$open`, `$high`, `$low`, `$close`, `$volume`, `$factor` (split/dividend adjustment), and `$change`.

To download and query the pre-packaged US dataset:
```bash
# Run the included automated download and 1-year query script:
python src/qlib/examples/data_us_prepackaged.py
```

Or manually via curl:
```bash
# US market data (~429MB zip, ~777MB uncompressed):
mkdir -p ~/.qlib/qlib_data/us_data
curl -L -o /tmp/qlib_data_us_1d_latest.zip https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_us_1d_latest.zip
unzip -q -o /tmp/qlib_data_us_1d_latest.zip -d ~/.qlib/qlib_data/us_data
rm /tmp/qlib_data_us_1d_latest.zip

# China A-share market data (~196MB zip):
mkdir -p ~/.qlib/qlib_data/cn_data
curl -L -o /tmp/qlib_data_cn_1d_latest.zip https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_cn_1d_latest.zip
unzip -q -o /tmp/qlib_data_cn_1d_latest.zip -d ~/.qlib/qlib_data/cn_data
rm /tmp/qlib_data_cn_1d_latest.zip
```

### Acquiring Data from 2020 to Today (2026)

> [!NOTE]
> **Is there a pre-packaged dump containing 2020 to 2026 data?**
> **No.** Official public Qlib `.zip` dumps are frozen snapshots ending in late 2020 / mid 2021 to ensure consistent academic benchmarks and respect financial data licensing. Microsoft does not provide ongoing public bulk dump archives.
>
> - **Official Recommendation**: Microsoft Qlib explicitly directs users who require data beyond 2020 to use its built-in data collectors (`scripts/data_collector/yahoo/collector.py` or tools like `yahooquery`/`yfinance`) to crawl recent data directly from the source.

To access data from November 2020 through today (2026), choose one of the following methods:

1. **Live Crawling via Yahoo Finance (Included Script)**:
   For your custom stock or ETF watchlist, crawl recent bars live and serialize them directly into Qlib's binary format:
   ```bash
   # Crawls the last 30 days (or customized range) and tests Qlib native loading:
   python src/qlib/examples/data_us_live.py
   ```

2. **Incremental Append (`DumpDataUpdate`)**:
   Keep the 1999–2020 pre-packaged base data and append only new trading days from 2020 to 2026 to `calendars/day.txt` and `features/<symbol>/*.day.bin` using Qlib's binary dumper:
   ```bash
   python scripts/data_collector/yahoo/collector.py update_data_to_bin \
       --qlib_data_1d_dir ~/.qlib/qlib_data/us_data
   ```

3. **External Quantitative Data APIs**:
   For large-scale, survivorship-bias-free institutional datasets covering 2020 to today:
   - **Alpaca Markets**: Free tier offering daily and minute US equity bars.
   - **Tiingo**: High-precision split/dividend adjusted prices and ETF distributions.
   - **Polygon.io**: Institutional real-time and historical aggregates.

### Testing Data Access in Python
```python
import qlib
from qlib.data import D

# Initialize workspace with US pre-packaged dataset
qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")

# Test querying 1 year of US market data for AAPL and S&P 500 ETF (SPY)
features = ["$close", "$volume", "$high", "$low"]
df = D.features(["AAPL", "SPY"], features, start_time="2019-01-01", end_time="2019-12-31")
print(df.head())
```

---

## 3. Workflow Architecture

A modular pipeline defines standard Qlib strategy configurations:

```
  ┌───────────────┐      ┌──────────────┐      ┌─────────────────┐      ┌─────────────────┐
  │  Data Handler │ ───> │ Model (Zoo)  │ ───> │ Strategy Module │ ───> │ Executor (Back) │
  └───────────────┘      └──────────────┘      └─────────────────┘      └─────────────────┘
     (Alpha158)             (LightGBM)            (Top-K Drop)            (Transaction)
```

- **Data Handler:** Performs rolling feature processing and indicator calculation (e.g., `Alpha158`).
- **Model:** Generates predictive raw risk/return alpha scores (e.g., `LightGBM`, `GRU`, `Transformer`).
- **Strategy:** Determines position rebalancing rules based on predictive alphas.
- **Executor:** Simulates order filling, market impact, and evaluates portfolios.

---

## 4. Running with `qrun`

The easiest way to execute a workflow is by calling `qrun` against a predefined descriptive YAML configuration file.

```bash
# Navigate to the cloned upstream Qlib directory
cd ~/src/qlib-source

# Run standard LightGBM + Alpha158 benchmark workflow
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```

This automates training, prediction, portfolio construction, backtesting, and performance report generation.

---

## 5. Advanced Usage (API)

For researcher flexibility, individual modules can be initialized via standard Python code snippets instead of configuration scripts:

```python
from qlib.utils import init_instance_by_config
from qlib.contrib.model.gbdt import LGBModel

model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "lr": 0.05,
        "max_depth": 6,
        "num_leaves": 31
    }
}

# Dynamically load the instance
model = init_instance_by_config(model_config)
print(f"Successfully configured: {type(model)}")
```

---

## 6. Quantitative Recommender Systems

In addition to deep benchmark models, high-performance rule-based and factor-ranked recommender engines are available under `src/qlib/examples/`:

### 📅 Daily Recommender (`daily_top10_recommender.py`)
Designed for next-session (T+1 to T+2) swing entries, momentum breakouts, and dip pullbacks across liquid US megacaps, growth momentum equities, and sector ETFs.

- **Horizon**: 1 to 2 sessions.
- **Factor Pipeline**:
  - **Intraday Buying Pressure (Close Location Value / CLV)**: Detects institutional closing accumulation.
  - **Volume Surge Factor**: Uncovers abnormal volume breakouts vs 20-day baseline.
  - **3-Day Relative Strength**: Compares short-term velocity directly against SPY.
  - **Micro-trend Moving Average Stack**: Evaluates EMA 3 > EMA 8 > EMA 20 alignment.
  - **Fast RSI (7-period)**: Real-time oscillation scoring with extreme penalty filters.
  - **Market Regime Gate**: Calibrates risk/exposure based on SPY benchmark EMA trend.
  - **Dynamic ATR Risk Brackets**: Automatically calculates entry, strict stop-loss, and 1.5+ R:R profit targets.
- **Usage**:
  ```bash
  # Standard Daily Top 10
  python src/qlib/examples/daily_top10_recommender.py

  # Custom Top N and CSV export
  python src/qlib/examples/daily_top10_recommender.py --top_n 5 --csv /path/to/daily.csv
  ```

### 📆 Weekly Recommender (`weekly_top10_recommender.py`)
Designed for 1-week swing rebalancing, combining medium-term momentum, 20-day Sharpe ratio proxies, and sector concentration limits.

- **Horizon**: 1-week holding period (rebalance each Monday).
- **Usage**:
  ```bash
  python src/qlib/examples/weekly_top10_recommender.py
  ```