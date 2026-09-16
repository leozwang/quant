# Microsoft Qlib Quantitative Trading & Supabase Pipeline

A production-grade algorithmic quantitative recommendation engine featuring next-session (**Daily Top 10**) and swing horizon (**Weekly Top 10**) alpha models, fully integrated with **Supabase Cloud Database** and automated prediction settlement tracking.

---

## 📋 Table of Contents
1. [Overview & Strategy](#1-overview--strategy)
2. [Installation & Setup](#2-installation--setup)
3. [Supabase Cloud Architecture](#3-supabase-cloud-architecture)
4. [Database Schema: `qlib_daily_top10_recommend`](#4-database-schema-qlib_daily_top10_recommend)
5. [Database Schema: `qlib_weekly_top10_recommend`](#5-database-schema-qlib_weekly_top10_recommend)
6. [Execution Guide](#6-execution-guide)

---

## 1. Overview & Strategy

- **Daily Top 10 Recommender** ([`examples/daily_top10_recommender.py`](examples/daily_top10_recommender.py)):
  - **Horizon**: Next session execution (T+1 to T+2).
  - **Schedule**: Run every day after the 4:00 PM EST market close (or Sunday evening for Monday open).
  - **Signals**: Close Location Value (CLV, intraday buying pressure), Volume Surge, 3-Day Relative Strength vs SPY, Micro-trend EMA alignment, and fast Wilder RSI(7).
  - **Key Dates**:
    - `trade_date`: Market session date of the underlying price bars (e.g. `2026-09-11`).
    - `date`: Executable timestamp when recommendations became live for trading (e.g. `2026-09-13 20:00:00`, Sep 13th 8:00 PM).
    - `target_date`: Target trading session date for execution (e.g. `2026-09-14`, Sep 14th).

- **Weekly Top 10 Recommender** ([`examples/weekly_top10_recommender.py`](examples/weekly_top10_recommender.py)):
  - **Horizon**: 1-week swing rebalancing (Monday open through Friday close).
  - **Schedule**: Run over the weekend or Friday post-close.
  - **Signals**: 20-day trend velocity, Sharpe ratio proxy, weekly volume flow ratio, and SMA-50 alignment.

---

## 2. Installation & Setup

### 1. Install Dependencies
Install all required libraries using [`requirements.txt`](requirements.txt):

```bash
# Optional: Create and activate a dedicated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r src/qlib/requirements.txt
```

### 2. Configure Environment
Copy the `.env.example` template to `.env` and fill in your Supabase credentials:

```bash
cp src/qlib/.env.example src/qlib/.env
```

Set your project variables inside `src/qlib/.env`:
```env
SUPABASE_PROJECT_ID=<your-project-id>
SUPABASE_URL=https://<your-project-id>.supabase.co
SUPABASE_KEY=<your-anon-or-service-role-key>
```
*(Alternatively, export them directly in your shell: `export SUPABASE_URL=...` and `export SUPABASE_KEY=...`)*

---

## 3. Supabase Cloud Architecture

- **Project ID**: `<your-project-id>`
- **Cloud Endpoint**: `https://<your-project-id>.supabase.co`
- **Tables**:
  - `qlib_daily_top10_recommend` (Daily recommendations)
  - `qlib_weekly_top10_recommend` (Weekly recommendations)

Credentials are configured via environment variables (`SUPABASE_URL`, `SUPABASE_KEY`) or a local `.env` file (see `.env.example`).

---

## 4. Database Schema: `qlib_daily_top10_recommend`

| Column | Type | Nullable | Example | Financial Meaning & Details |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `BIGINT` | No | `1` | Primary key identity sequence. |
| `trade_date` | `DATE` | No | `2026-09-11` | Market session close of historical data analyzed. |
| `date` | `TIMESTAMP` | Yes | `2026-09-13 20:00:00` | **Executable Date**: Date and time when the recommendation was generated and became executable (Sep 13th, 8:00 PM). |
| `target_date` | `DATE` | Yes | `2026-09-14` | **Target Date**: Target market session date for trade execution (Sep 14th). |
| `predicted_close` | `NUMERIC(10,2)` | Yes | `267.15` | **Predicted Close**: Statistical point forecast / expected closing price on the target date. |
| `target_date_close` | `NUMERIC(10,2)` | Yes | `239.01` | **Target Date Close**: Realized closing price on target date (used for P&L and outcome tracking). |
| `predicted_diff` | `NUMERIC(10,2)` | Yes | `-28.14` | **Predicted Price Difference**: Realized close minus predicted close ($TargetDateClose - PredictedClose$). Evaluates model point forecast accuracy. |
| `rank` | `INTEGER` | No | `1` | Rank order by conviction score (`#1` to `#10`). |
| `symbol` | `TEXT` | No | `ARM` | Asset ticker symbol. |
| `name` | `TEXT` | No | `Arm Holdings plc` | Company / fund name. |
| `sector` | `TEXT` | No | `Technology` | GICS classification (capped at max 2 equities per sector for risk management). |
| `setup_type` | `TEXT` | No | `🎯 SWING` | Pattern classification: `🚀 BREAKOUT`, `⚡ PULLBACK`, `📈 MOMENTUM`, `🛡️ OVERSOLD`, `🎯 SWING`. |
| `close_price` | `NUMERIC(10,2)` | No | `264.79` | Official closing price (USD) at market close. |
| `ret_1d` | `NUMERIC(7,4)` | Yes | `0.0417` | 1-Day return percentage (+4.17%). |
| `ret_3d` | `NUMERIC(7,4)` | Yes | `0.0125` | 3-Day return percentage (+1.25%). |
| `vol_surge` | `NUMERIC(6,2)` | Yes | `1.14` | Volume multiplier vs 20-day baseline (1.14x). |
| `clv` | `NUMERIC(4,2)` | Yes | `0.59` | Close Location Value: $(2 \cdot Close - High - Low)/(High - Low)$. |
| `rsi_7` | `NUMERIC(5,2)` | Yes | `79.90` | 7-period fast Wilder RSI momentum oscillator. |
| `alpha_score` | `NUMERIC(6,2)` | No | `1.32` | Multi-factor quantitative conviction composite score. |
| `stop_loss` | `NUMERIC(10,2)` | No | `258.17` | Daily stop-loss level (capped at 1.0 ATR or 2.5% max risk). |
| `target_t1` | `NUMERIC(10,2)` | No | `285.63` | Take-profit milestone based on 1.6x ATR expansion. |
| `rr_ratio` | `NUMERIC(4,1)` | Yes | `3.1` | Risk-to-reward mathematical expectancy ratio (e.g. 3.1:1). |
| `market_regime` | `TEXT` | Yes | `🔴 DEFENSIVE` | Macro market gatekeeper based on index trend alignment. |
| `created_at` | `TIMESTAMPTZ` | No | `2026-09-14 04:36` | Database record creation timestamp. |

---

## 5. Database Schema: `qlib_weekly_top10_recommend`

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `id` | `BIGINT` | `1` | Primary key identity. |
| `week_start_date` | `DATE` | `2026-09-14` | Monday start date of trading week. |
| `rank` | `INTEGER` | `1` | Weekly conviction rank (`#1` to `#10`). |
| `symbol` | `TEXT` | `CRM` | Asset ticker. |
| `name` | `TEXT` | `Salesforce Inc.` | Company name. |
| `sector` | `TEXT` | `Technology` | GICS sector. |
| `close_price` | `NUMERIC(10,2)` | `247.72` | Friday entry baseline close. |
| `ret_5d` | `NUMERIC(7,4)` | `-0.0632` | Prior 1-week return (-6.32%). |
| `ret_20d` | `NUMERIC(7,4)` | `0.2302` | Prior 1-month return (+23.02%). |
| `vol_ratio` | `NUMERIC(6,2)` | `0.73` | 5-day / 20-day volume accumulation ratio. |
| `rsi_14` | `NUMERIC(5,2)` | `71.20` | Classic 14-period RSI momentum indicator. |
| `alpha_score` | `NUMERIC(6,2)` | `1.58` | Weekly factor composite conviction score. |
| `stop_loss` | `NUMERIC(10,2)` | `239.05` | Weekly stop-loss (1.5x ATR, capped at 3.5%). |
| `target_price` | `NUMERIC(10,2)` | `277.55` | 1-week target price (2.5x ATR). |
| `created_at` | `TIMESTAMPTZ` | `2026-09-14` | Record creation timestamp. |

---

## 6. Execution Guide

### 1. Dry-Run / Inspect Calculations (Zero Supabase Writes)
To calculate signals and inspect recommendations in the terminal without touching the database:
```bash
# Daily Top 10 Dry-Run (safe calculation mode)
python src/qlib/examples/daily_top10_recommender.py --dry-run

# Weekly Top 10 Dry-Run
python src/qlib/examples/weekly_top10_recommender.py --dry-run
```
*(By default, running either standalone script without `--upload` always runs in safe dry-run mode).*

### 2. Run Master Pipeline & Upload to Supabase (Daily + Weekly)
```bash
python src/qlib/examples/run_and_upload_recommendations.py
```
> **Idempotent / Single-Upload Guarantee**: Automatically checks Supabase before uploading. If today's recommendations are already uploaded, it cleanly skips to prevent duplicates. Pass `--force` to overwrite.
