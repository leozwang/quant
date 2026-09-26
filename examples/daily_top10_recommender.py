#!/usr/bin/env python3
"""Daily Quantitative Top 10 Recommender for US Equities & ETFs.

A short-horizon (T+1 to T+2) quantitative recommendation system tailored for
daily trading sessions. Analyzes intraday buying pressure, short-term momentum,
volume breakouts, fast RSI, and market regime conditions to generate high-probability
daily setups with precise entry, stop-loss, and profit targets.
"""

import argparse
from datetime import datetime
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# Ensure sibling modules resolve when this script is executed directly.
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from market_calendar import last_completed_session, next_trading_day, now_et

# ==============================================================================
# 1. Candidate Universe: High-Liquidity US Equities, Growth Leaders & Active ETFs
# ==============================================================================
UNIVERSE = {
    # Broad & Key Sector ETFs
    "SPY": ("Broad Index", "SPDR S&P 500 ETF Trust"),
    "QQQ": ("Tech Index", "Invesco QQQ Trust (Nasdaq 100)"),
    "IWM": ("Small Cap", "iShares Russell 2000 ETF"),
    "SMH": ("Semiconductors", "VanEck Semiconductor ETF"),
    "XLK": ("Technology", "Technology Select Sector SPDR"),
    # Tagged "Technology" (not a bespoke label) so it competes under the same
    # 2-per-sector cap as XLK and the tech single-names. Giving it its own
    # sector would exempt it from that contest and quietly bias it into the
    # top-N regardless of signal strength.
    "VGT": ("Technology", "Vanguard Information Technology ETF"),
    "XLE": ("Energy", "Energy Select Sector SPDR"),
    "XLF": ("Financials", "Financial Select Sector SPDR"),
    "XLV": ("Healthcare", "Health Care Select Sector SPDR"),
    "XLI": ("Industrials", "Industrial Select Sector SPDR"),
    "GLD": ("Commodities", "SPDR Gold Shares ETF"),
    
    # Megacap & Tech Leaders
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
    "MU": ("Technology", "Micron Technology"),
    "QCOM": ("Technology", "QUALCOMM Inc."),
    "ARM": ("Technology", "Arm Holdings plc"),
    "TSM": ("Technology", "Taiwan Semiconductor Manufacturing"),

    # High-Beta & Growth Momentum
    "COIN": ("Financials", "Coinbase Global"),
    "HOOD": ("Financials", "Robinhood Markets"),
    "SHOP": ("Technology", "Shopify Inc."),
    "APP": ("Technology", "AppLovin Corporation"),
    "MSTR": ("Technology", "MicroStrategy Inc."),

    # Financials & Payments
    "JPM": ("Financials", "JPMorgan Chase & Co."),
    "GS": ("Financials", "Goldman Sachs Group"),
    "MS": ("Financials", "Morgan Stanley"),
    "V": ("Financials", "Visa Inc."),
    "MA": ("Financials", "Mastercard Inc."),
    "BLK": ("Financials", "BlackRock Inc."),

    # Healthcare & Biotech
    "LLY": ("Healthcare", "Eli Lilly and Company"),
    "UNH": ("Healthcare", "UnitedHealth Group"),
    "ABBV": ("Healthcare", "AbbVie Inc."),
    "ISRG": ("Healthcare", "Intuitive Surgical"),
    "NVO": ("Healthcare", "Novo Nordisk"),

    # Industrials, Defense & Aerospace
    "GE": ("Industrials", "GE Aerospace"),
    "CAT": ("Industrials", "Caterpillar Inc."),
    "RTX": ("Industrials", "RTX Corporation"),
    "LMT": ("Industrials", "Lockheed Martin"),

    # Energy & Commodities
    "XOM": ("Energy", "Exxon Mobil Corporation"),
    "CVX": ("Energy", "Chevron Corporation"),
    "CCJ": ("Energy", "Cameco Corporation"),

    # Consumer & Retail Leaders
    "COST": ("Consumer Staples", "Costco Wholesale"),
    "WMT": ("Consumer Staples", "Walmart Inc."),
    "HD": ("Consumer Discretionary", "Home Depot Inc.")
}


# ==============================================================================
# 2. Daily Factor Engineering (1-Day to 2-Day Horizon Alphas)
# ==============================================================================
def calculate_daily_factors(df: pd.DataFrame, spy_df: pd.DataFrame) -> dict:
    """Calculates short-horizon alpha factors for a single ticker."""
    if len(df) < 35:
        return None

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]
    open_p = df["Open"]

    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    today_high = float(high.iloc[-1])
    today_low = float(low.iloc[-1])
    today_open = float(open_p.iloc[-1])
    today_vol = float(volume.iloc[-1])

    # 1. Immediate & Medium-Term Return Velocity + Prior 4-Day Consolidation/Pullback
    ret_1d = (current_price / prev_price) - 1.0
    ret_3d = (current_price / float(close.iloc[-4])) - 1.0 if len(close) >= 4 else ret_1d
    ret_5d = (current_price / float(close.iloc[-6])) - 1.0 if len(close) >= 6 else ret_1d
    ret_20d = (current_price / float(close.iloc[-21])) - 1.0 if len(close) >= 21 else ret_5d
    prior_4d = (prev_price / float(close.iloc[-6])) - 1.0 if len(close) >= 6 else 0.0

    # 2. Daily & 20-Day Structural Relative Strength vs SPY Benchmark
    spy_ret_1d = (float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-2])) - 1.0
    spy_ret_3d = (float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-4])) - 1.0 if len(spy_df) >= 4 else spy_ret_1d
    spy_ret_20d = (float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-21])) - 1.0 if len(spy_df) >= 21 else spy_ret_3d
    rs_1d = ret_1d - spy_ret_1d
    rs_3d = ret_3d - spy_ret_3d
    rs_20d = ret_20d - spy_ret_20d

    # 3. Close Location Value (CLV) / Intraday Buying Pressure:
    # Ranges from -1.0 (closed at low) to +1.0 (closed at high).
    # Values > +0.50 indicate aggressive institutional accumulation into the bell.
    day_range = today_high - today_low
    if day_range > 0:
        clv = ((current_price - today_low) - (today_high - current_price)) / day_range
    else:
        clv = 0.0

    # 4. Volume Surge Factor
    vol_ma20 = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else today_vol
    vol_surge = (today_vol / vol_ma20) if vol_ma20 > 0 else 1.0

    # 5. Short-Term Exponential Moving Averages (EMA 3, 8, 20)
    ema_3 = float(close.ewm(span=3, adjust=False).mean().iloc[-1])
    ema_8 = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
    ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    
    # Micro Trend Alignment (Score 0 to 3)
    micro_trend = 0
    if current_price > ema_3: micro_trend += 1
    if ema_3 > ema_8: micro_trend += 1
    if ema_8 > ema_20: micro_trend += 1

    # 6. Fast RSI (7-period for responsive daily oscillations)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()

    # A window with no down days makes average loss 0. Dividing by NaN and
    # falling back to 50 would report such a stock as *neutral* when it is in
    # fact maximally overbought (RSI 100 by definition). That matters here
    # because rsi_7 > 74 carries a blow-off-top penalty and rsi_7 > 65 triggers
    # the mean-reversion dampener -- so the strongest, most extended names were
    # silently escaping both guards.
    _gain = gain.iloc[-1]
    _loss = loss.iloc[-1]
    if pd.isna(_gain) or pd.isna(_loss):
        rsi_7 = 50.0                      # not enough history to judge
    elif _loss == 0:
        rsi_7 = 100.0 if _gain > 0 else 50.0   # all up, or completely flat
    else:
        rsi_7 = float(100 - (100 / (1 + (_gain / _loss))))

    # 7. Bollinger Band Position (%B over 20 days, 2 std)
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    upper_band = float((sma_20 + 2 * std_20).iloc[-1])
    lower_band = float((sma_20 - 2 * std_20).iloc[-1])
    bb_range = upper_band - lower_band
    pct_b = (current_price - lower_band) / bb_range if bb_range > 0 else 0.5

    # 8. Average True Range (ATR 14) for precise daily risk brackets & vol-adjusted returns
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(window=14).mean().iloc[-1])
    atr_pct = (atr_14 / current_price) if current_price > 0 else 0.02
    norm_ret_1d = (ret_1d / atr_pct) if atr_pct > 0 else 0.0

    # 9. Rolling 30-day Beta relative to SPY Benchmark
    asset_rets = close.pct_change().dropna()
    spy_rets = spy_df["Close"].pct_change().dropna()
    common_idx = asset_rets.index.intersection(spy_rets.index)
    if len(common_idx) >= 15:
        a_vals = asset_rets.loc[common_idx].values[-30:]
        s_vals = spy_rets.loc[common_idx].values[-30:]
        var_s = np.var(s_vals)
        beta = float(np.cov(a_vals, s_vals)[0, 1] / var_s) if var_s > 1e-7 else 1.0
        beta = max(-1.2, min(2.5, beta))
    else:
        beta = 1.0

    # 10. Setup Archetype Classification for Daily Strategy
    # - BREAKOUT: Surge volume + strong close in upper quartile of day range
    # - PULLBACK: In 20D uptrend (above EMA20 or positive 20D RS), pulled back over prior 4 days, turning up today
    # - OVERSOLD: Fast RSI < 38 with intraday buying support (CLV > 0.30) and not in 20D freefall
    # - MOMENTUM: 20D relative leader with volume-supported continuation and non-exhausted RSI (<= 70)
    if vol_surge >= 1.25 and clv >= 0.45 and ret_1d > 0.004:
        setup_type = "🚀 BREAKOUT"
    elif (current_price >= ema_20 * 0.99 or rs_20d > 0) and prior_4d < -0.005 and (clv >= 0.25 or ret_1d > 0.002):
        setup_type = "⚡ PULLBACK"
    elif rsi_7 < 38 and clv > 0.30 and rs_20d > -0.05:
        setup_type = "🛡️ OVERSOLD"
    elif rs_20d > 0.015 and micro_trend >= 2 and clv >= 0.35 and vol_surge >= 1.05 and rsi_7 <= 70 and prior_4d <= 0.035:
        setup_type = "📈 MOMENTUM"
    else:
        setup_type = "🎯 SWING"

    return {
        "price": current_price,
        "ret_1d": ret_1d,
        "ret_3d": ret_3d,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "prior_4d": prior_4d,
        "norm_ret_1d": norm_ret_1d,
        "rs_1d": rs_1d,
        "rs_3d": rs_3d,
        "rs_20d": rs_20d,
        "clv": clv,
        "vol_surge": vol_surge,
        "micro_trend": micro_trend,
        "rsi_7": rsi_7,
        "pct_b": pct_b,
        "atr_14": atr_14,
        "atr_pct": atr_pct,
        "ema_3": ema_3,
        "ema_8": ema_8,
        "ema_20": ema_20,
        "beta": round(beta, 2),
        "setup_type": setup_type
    }


# ==============================================================================
# 3. Market Regime Analyzer (Macro Gatekeeper)
# ==============================================================================
def assess_market_regime(spy_df: pd.DataFrame, qqq_df: pd.DataFrame) -> dict:
    """Evaluates macro market state to calibrate exposure and risk parameters."""
    spy_close = spy_df["Close"]
    spy_price = float(spy_close.iloc[-1])
    spy_ema_8 = float(spy_close.ewm(span=8, adjust=False).mean().iloc[-1])
    spy_ema_20 = float(spy_close.ewm(span=20, adjust=False).mean().iloc[-1])
    spy_ret_1d = (spy_price / float(spy_close.iloc[-2])) - 1.0

    qqq_close = qqq_df["Close"]
    qqq_price = float(qqq_close.iloc[-1])
    qqq_ema_20 = float(qqq_close.ewm(span=20, adjust=False).mean().iloc[-1])

    if spy_price > spy_ema_8 and spy_ema_8 > spy_ema_20:
        regime = "🟢 BULLISH ACCELERATION"
        guidance = "Favorable for aggressive momentum breakouts and gap continuation."
        risk_multiplier = 1.0
    elif spy_price > spy_ema_20:
        regime = "🟡 CONSTRUCTIVE / UPTREND"
        guidance = "Good trend support. Focus on dip pullbacks to key moving averages."
        risk_multiplier = 0.85
    else:
        regime = "🔴 DEFENSIVE / CAUTION"
        guidance = "Index below 20-day EMA. Prioritize tight stop-losses and hedge assets."
        risk_multiplier = 0.65

    return {
        "regime": regime,
        "guidance": guidance,
        "spy_price": spy_price,
        "spy_ret_1d": spy_ret_1d,
        "risk_multiplier": risk_multiplier,
        "last_date": spy_df.index[-1].strftime("%Y-%m-%d")
    }


# ==============================================================================
# 4. Main Scoring & Recommendation Pipeline
# ==============================================================================
def run_daily_recommender(top_n: int = 10, export_csv: str = None,
                          expected_baseline: str = None):
    """Generate the daily top-N recommendations.

    expected_baseline: optional 'YYYY-MM-DD'. If supplied, the session that the
        downloaded data actually represents MUST equal this date, or the run is
        aborted. The caller (the orchestrator) derives this from the market
        calendar; the data derives it from the provider. Requiring the two to
        agree is what prevents acting on a stale or half-formed bar -- see the
        baseline check below for why that matters.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Downloading market data for universe of {len(UNIVERSE)} assets...")
    tickers_list = list(UNIVERSE.keys())

    # Download recent 3 months daily OHLCV bars
    data = yf.download(tickers_list, period="3mo", interval="1d", progress=False, group_by="ticker")

    if "SPY" not in data or "QQQ" not in data:
        print("Error: Could not retrieve SPY/QQQ benchmark data.")
        # Must match the success-path arity; callers unpack two values.
        return [], None

    # Ensure latest session bar is complete: if today's row exists with NaN close (intraday / near close),
    # populate OHLCV from fast_info so today's completed session is captured accurately.
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
    qqq_df = data["QQQ"].dropna()

    market_info = assess_market_regime(spy_df, qqq_df)
    trading_day = market_info["last_date"]

    # ==========================================================================
    # Session-agreement check (fail closed on stale or half-formed data)
    # ==========================================================================
    # The orchestrator derives the expected session from the market calendar;
    # `trading_day` is derived from whatever the data provider actually returned.
    # These agree only while the provider is current. When they disagree it means
    # one of:
    #
    #   * the provider has not yet published the session we expect (lag), or
    #   * the last row is *today* mid-session -- possibly a synthetic bar built
    #     from live quotes by the fast_info backfill above, which is NOT a close.
    #
    # Continuing in either case writes a prediction stamped with the wrong
    # target_date. Those rows then settle against an already-known close, which
    # silently flatters the accuracy statistics. Skipping and retrying on the
    # next invocation costs nothing, because the job is idempotent.
    if expected_baseline and trading_day != expected_baseline:
        print(
            f"\n  ⚠️ [DAILY] Session mismatch: data baseline is '{trading_day}' but the "
            f"calendar expects '{expected_baseline}'."
        )
        print("     Refusing to generate a prediction from a session that does not match.")
        print("     This is normally provider lag; the next scheduled run will retry.\n")
        # Distinguishable from a genuine failure (which returns None) so the
        # orchestrator does not raise an alert for an ordinary, self-healing skip.
        return [], {
            "skipped": "session_mismatch",
            "data_baseline": trading_day,
            "expected_baseline": expected_baseline,
        }

    results = []
    skipped = {"missing": [], "short": [], "stale": [], "error": []}
    benchmark_last = spy_df.index[-1]

    for symbol, (sector, name) in UNIVERSE.items():
        try:
            if symbol not in data:
                skipped["missing"].append(symbol)
                continue
            df = data[symbol].dropna()
            if len(df) < 30:
                skipped["short"].append(symbol)
                continue

            # dropna() removes today's row entirely if this ticker has any NaN in
            # it (halt, late print, provider gap). df.iloc[-1] would then silently
            # be YESTERDAY's bar, while spy_df.iloc[-1] is today's -- so every
            # relative-strength and beta figure would compare misaligned sessions,
            # and the row would still be stamped with today's trade_date.
            # Require the ticker to be on the same session as the benchmark.
            if df.index[-1] != benchmark_last:
                skipped["stale"].append(symbol)
                continue

            metrics = calculate_daily_factors(df, spy_df)
            if metrics is None:
                skipped["error"].append(symbol)
                continue

            metrics["symbol"] = symbol
            metrics["sector"] = sector
            metrics["name"] = name
            results.append(metrics)
        except Exception as e:
            # Previously a bare `continue`, which meant a systematic failure
            # could silently shrink the universe with no trace at all.
            skipped["error"].append(f"{symbol}({type(e).__name__})")
            continue

    dropped = sum(len(v) for v in skipped.values())
    if dropped:
        print(f"  ℹ️ {len(results)}/{len(UNIVERSE)} tickers usable; {dropped} skipped.")
        for reason, syms in skipped.items():
            if syms:
                shown = ", ".join(str(s) for s in syms[:8])
                more = f" (+{len(syms) - 8} more)" if len(syms) > 8 else ""
                print(f"       {reason:8}: {shown}{more}")


    if not results:
        print("Error: No qualifying candidate data processed.")
        # Must match the success-path arity; callers unpack two values.
        return [], None

    res_df = pd.DataFrame(results)

    def _zscore(series: pd.Series, clip_min: float = -2.5, clip_max: float = 2.5) -> pd.Series:
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return ((series - series.mean()) / std).clip(clip_min, clip_max)

    # Sector-level 1D participation & 20D structural relative strength
    res_df["sec_ret1d"] = res_df.groupby("sector")["ret_1d"].transform("mean")
    res_df["sec_rs20d"] = res_df.groupby("sector")["rs_20d"].transform("mean")

    # Standardize factors (winsorized across the daily candidate pool so single-day
    # parabolic outliers do not hijack Rank #1-#2 right at short-term exhaustion)
    z_norm_ret1d = _zscore(res_df["norm_ret_1d"], -2.0, 1.8)
    z_vol        = _zscore(res_df["vol_surge"], -1.5, 2.2)
    z_clv        = _zscore(res_df["clv"], -2.0, 2.0)
    z_rs20d      = _zscore(res_df["rs_20d"], -2.0, 2.0)
    z_sec_ret1d  = _zscore(res_df["sec_ret1d"], -2.0, 2.0)
    z_sec_rs20d  = _zscore(res_df["sec_rs20d"], -2.0, 2.0)
    z_prior4d    = _zscore(res_df["prior_4d"], -2.2, 2.2)

    # Exhaustion & Regime Penalties:
    # - Overbought fast RSI (> 74) or extended 4-day run-up prior to today (> +4.0%)
    # - Multi-day rally (> +2.5% over 3D) on below-average volume (< 1.15x)
    # - Persistent falling knife (RSI < 28)
    # - Sector in multi-week distribution (sec_rs20d < -2.5%) without a volume breakout
    exhaustion_penalty = (
        np.where(res_df["rsi_7"] > 74, -0.65, 0.0)
        + np.where(res_df["prior_4d"] > 0.04, -0.60, 0.0)
        + np.where((res_df["ret_3d"] > 0.025) & (res_df["vol_surge"] < 1.15), -0.55, 0.0)
        + np.where(res_df["rsi_7"] < 28, -0.45, 0.0)
        + np.where((res_df["sec_rs20d"] < -0.025) & (res_df["vol_surge"] < 1.35), -0.35, 0.0)
    )

    setup_bonus = np.select(
        [
            res_df["setup_type"] == "🚀 BREAKOUT",
            res_df["setup_type"] == "⚡ PULLBACK",
            res_df["setup_type"] == "🛡️ OVERSOLD",
            res_df["setup_type"] == "📈 MOMENTUM",
        ],
        [0.40, 0.35, 0.30, 0.20],
        default=-0.15,
    )

    # Daily Composite Alpha Score (Tailored for next-day conviction):
    # - 20% Volatility-adjusted 1-Day Ignition (ret_1d / atr_pct, winsorized at +1.8s)
    # - 20% Volume Surge participation
    # - 15% Intraday Buying Pressure (CLV: smart money accumulation into the close)
    # - 15% 20-Day Structural Relative Strength vs SPY
    # - 15% Sector Regime (10% Sector 1D Breadth + 5% Sector 20D Relative Trend)
    # - 15% Prior 4-Day Consolidation/Pullback Spring (-z_prior4d: penalizes multi-day chasers)
    composite_score = (
        0.20 * z_norm_ret1d
        + 0.20 * z_vol
        + 0.15 * z_clv
        + 0.15 * z_rs20d
        + 0.10 * z_sec_ret1d
        + 0.05 * z_sec_rs20d
        - 0.15 * z_prior4d
        + setup_bonus
        + exhaustion_penalty
    )
    res_df["alpha_score"] = composite_score

    # Compute expected 1D return BEFORE ranking so Ranking and Expected Value are unified
    exp_market_ret = -0.0025 if "DEFENSIVE" in market_info["regime"] else 0.0008
    beta_drag = res_df["beta"] * exp_market_ret
    shrunk_alpha = (res_df["alpha_score"] * 0.0035).clip(-0.008, 0.010)
    reversion_adj = np.where(
        (res_df["prior_4d"] > 0.03) & (res_df["rsi_7"] > 65),
        -0.25 * (res_df["prior_4d"] - 0.02),
        np.where(
            (res_df["prior_4d"] < -0.01) & (res_df["clv"] > 0.20),
            0.15 * np.minimum(0.02, -res_df["prior_4d"]),
            0.0,
        ),
    )
    res_df["predicted_ret_1d"] = beta_drag + shrunk_alpha + reversion_adj
    res_df["predicted_close"] = (res_df["price"] * (1.0 + res_df["predicted_ret_1d"])).round(2)

    # Gate out / demote any candidate whose expected close is not above current close
    res_df["alpha_score"] = np.where(
        res_df["predicted_close"] <= res_df["price"],
        res_df["alpha_score"] - 1.0,
        res_df["alpha_score"],
    )

    # Sort candidates by alpha score
    sorted_df = res_df.sort_values(by="alpha_score", ascending=False).reset_index(drop=True)

    # Portfolio Selection with Risk & Sector Diversification:
    # Max 2 equities per sector (excluding Broad Index ETFs & Gold Hedge)
    selected = []
    sector_counts = {}

    for _, row in sorted_df.iterrows():
        sec = row["sector"]
        if sector_counts.get(sec, 0) < 2 or sec in ["Broad Index", "Commodities"]:
            selected.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected) == top_n:
            break

    top_df = pd.DataFrame(selected).reset_index(drop=True)

    # ==============================================================================
    # 5. Formatted Output Presentation
    # ==============================================================================
    # Calculate executable date (published timestamp) and target trading date upfront.
    # next_trading_day() skips weekends AND exchange holidays; the previous weekday-offset
    # arithmetic could place target_date on a closed session, which then never settles.
    trade_dt = datetime.strptime(trading_day, "%Y-%m-%d")
    target_day_str = next_trading_day(trade_dt.date()).strftime("%Y-%m-%d")
    exec_dt_str = now_et().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 115)
    print(f"               DAILY QUANTITATIVE TOP {top_n} RECOMMENDATIONS (NEXT-SESSION FOCUS)")
    print(f"  • Trade Date (Data Baseline) : {trading_day}")
    print(f"  • Date (Executable Timestamp): {exec_dt_str}")
    print(f"  • Target Date (Next Session) : {target_day_str}")
    print("=" * 115)
    print(f" [MARKET REGIME] : {market_info['regime']} (SPY: ${market_info['spy_price']:.2f}, 1D: {market_info['spy_ret_1d']*100:+.2f}%)")
    print(f" [STRATEGY TACTIC]: {market_info['guidance']}")
    print("-" * 115)
    header_str = f"{'Rank':<4} {'Symbol':<6} {'Sector':<15} {'Setup':<13} {'Price':<8} {'Pred Close':<11} {'Target TP':<11} {'Stop-Loss':<11} {'1D%':<7} {'Vol':<6} {'Score':<6}"
    print(header_str)
    print("-" * 115)

    csv_rows = []
    for i, r in top_df.iterrows():
        p = r["price"]
        atr = r["atr_14"]
        predicted_close = float(r["predicted_close"])

        # Dynamic Tight Stop-Loss for Daily Trades:
        # Max risk is min(1.0 * ATR, 2.5% of price) to ensure capital preservation
        stop_loss_dist = min(1.0 * atr, p * 0.025)
        stop_loss = round(p - stop_loss_dist, 2)
        
        # Take-Profit Target (1.5 to 2.0x ATR for favorable 1.5+ risk-reward)
        target_dist = max(1.6 * atr, stop_loss_dist * 1.6)
        target = round(p + target_dist, 2)
        
        rr_ratio = round(target_dist / stop_loss_dist, 1) if stop_loss_dist > 0 else 1.5

        print(
            f"#{i+1:<3} {r['symbol']:<6} {r['sector']:<15} {r['setup_type']:<13} "
            f"${p:<7.2f} ${predicted_close:<10.2f} ${target:<10.2f} ${stop_loss:<10.2f} "
            f"{r['ret_1d']*100:>+5.1f}% {r['vol_surge']:>4.2f}x  {r['alpha_score']:<6.2f}"
        )

        csv_rows.append({
            "trade_date": trading_day,
            "date": exec_dt_str,
            "target_date": target_day_str,
            "predicted_close": predicted_close,
            "predicted_diff": None,
            "target_date_close": None,
            "rank": i + 1,
            "symbol": r["symbol"],
            "name": r["name"],
            "sector": r["sector"],
            "setup_type": r["setup_type"],
            "close_price": round(p, 2),
            "ret_1d": round(r["ret_1d"], 4),
            "ret_3d": round(r["ret_3d"], 4),
            "vol_surge": round(r["vol_surge"], 2),
            "clv": round(r["clv"], 2),
            "rsi_7": round(r["rsi_7"], 2),
            "alpha_score": round(r["alpha_score"], 2),
            "stop_loss": stop_loss,
            "target_t1": target,
            "rr_ratio": rr_ratio,
            "market_regime": market_info["regime"]
        })

    print("=" * 115)
    print(" Execution Guidance:")
    print(" • Recommended Entry : Market Open or Limit near 3-day EMA support.")
    print(" • Statistical Target : 'Pred Close' (predicted_close) is the statistical expected close on target_date.")
    print(" • Take-Profit Exit  : 'Target TP' (target_t1) is the upside profit-taking barrier (1.6x ATR).")
    print(" • Holding Duration  : 1 to 2 sessions. Take partial profits at Target, exit immediately on Stop-Loss.")
    print(" • Capital Allocation: Max 10% per position, equal-weighted among top selections.")
    print("=" * 115 + "\n")

    if export_csv:
        export_path = os.path.abspath(export_csv)
        pd.DataFrame(csv_rows).to_csv(export_path, index=False)
        print(f"✓ Successfully exported recommendations to: {export_path}\n")

    return csv_rows, market_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Quantitative Top 10 Recommender")
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
        help="Upload results to Supabase table 'qlib_daily_top10_recommend'"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force upload to Supabase even if records for today already exist"
    )
    parser.add_argument(
        "--no-session-check",
        action="store_true",
        default=False,
        help="Skip the session-agreement guard and compute on whatever the provider "
             "returns. Intended for inspecting signals while the market is open; the "
             "resulting numbers are based on an incomplete bar."
    )
    args = parser.parse_args()

    # Inform user of execution mode upfront
    if args.dry_run or not args.upload:
        print("\n🔍 [DRY-RUN MODE] Calculating and displaying recommendations to screen only.")
        print("   (No data will be written to Supabase database. Pass --upload to enable cloud sync)\n")

    # The orchestrator always supplies a baseline; a direct run must derive its
    # own, otherwise this script would happily compute on a half-formed intraday
    # bar (or one synthesised from live quotes) and -- with --upload -- write a
    # mis-dated prediction. Same protection, same default.
    expected = None if args.no_session_check else last_completed_session(now_et()).isoformat()
    if args.no_session_check:
        print("⚠️  [--no-session-check] Session guard disabled. If the market is open, the")
        print("    latest bar is incomplete and these numbers are NOT a tradable signal.\n")

    records, market = run_daily_recommender(
        top_n=args.top_n, export_csv=args.csv, expected_baseline=expected
    )

    if args.upload and not args.dry_run:
        if not records:
            print("  ⚠️ No records produced; nothing to upload.\n")
            sys.exit(1)
        try:
            from run_and_upload_recommendations import SupabaseSync, DAILY_TABLE
            sync = SupabaseSync()
            trade_date = records[0]["trade_date"]
            print(f"[Supabase] Uploading {len(records)} records for trade_date '{trade_date}' to '{DAILY_TABLE}'...")
            success, msg = sync.upload_daily(records, force=args.force)
            if success:
                print(f"  ✓ {msg}\n")
            else:
                print(f"  ℹ️ {msg}\n")
        except Exception as e:
            print(f"  ⚠️ Supabase upload notice: {e}\n")
    else:
        print("✓ Dry run complete: Inspected output safely without writing to Supabase database.\n")
