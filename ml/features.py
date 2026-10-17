"""
NEXUS ML Feature Extractor.

Extracts ~40 normalized features from raw candle data and layer results.
All math is numpy-only. Features are designed to be stable across symbols
by normalizing prices to ratios and volatility-adjusted returns.

Feature vector (length = 40):
  [0]  rsi_14              — RSI(14), normalized 0-1
  [1]  stochrsi_k          — StochRSI K, normalized 0-1
  [2]  stochrsi_d          — StochRSI D, normalized 0-1
  [3]  macd_hist_norm      — MACD histogram / ATR (sign-preserving)
  [4]  macd_line_norm      — MACD line / ATR
  [5]  ema20_ratio         — close / EMA20
  [6]  ema50_ratio         — close / EMA50
  [7]  ema200_ratio        — close / EMA200
  [8]  ema20_50_ratio      — EMA20 / EMA50 (stack alignment)
  [9]  ema50_200_ratio     — EMA50 / EMA200 (stack alignment)
  [10] adx_norm            — ADX / 100
  [11] plus_di_norm        — +DI / 100
  [12] minus_di_norm       — -DI / 100
  [13] atr_percentile      — ATR percentile, 0-1
  [14] vol_ratio           — volume / avg_volume_20, capped at 5
  [15] candle_body_ratio   — (close-open) / ATR (directional)
  [16] wick_ratio          — (high-low-|close-open|) / ATR (indecision)
  [17] close_position      — (close-low) / (high-low), 0-1
  [18] return_1            — (close / close[-1]) - 1, ATR-normalized
  [19] return_3            — (close / close[-3]) - 1, ATR-normalized
  [20] return_5            — (close / close[-5]) - 1, ATR-normalized
  [21] return_10           — (close / close[-10]) - 1, ATR-normalized
  [22] vol_1bar            — std of last 5 returns (micro-vol)
  [23] vol_10bar           — std of last 20 returns (macro-vol)
  [24] bid_ask_imbalance   — order book imbalance, -1 to 1
  [25] cvd_norm            — CVD signal encoded: BUL=1 BEAR=-1 NEU=0
  [26] oi_trend_norm       — OI trend encoded: RISING=1 FLAT=0 FALLING=-1
  [27] funding_rate_norm   — funding rate * 1000 (±0.1% range → ±100)
  [28] ls_ratio_norm       — (ls_ratio - 1) / 1, capped ±1
  [29] vwap_ratio          — close / VWAP
  [30] channel_pos_norm    — UPPER=1 LOWER=-1 MIDDLE=0 OUTSIDE=0
  [31] vp_zone_norm        — VAH=1 POC=0 VAL=-1 BETWEEN=0.5
  [32] session_quality     — OVERLAP=1 NY/LDN=0.6 ASIA=0.3 DEAD=0
  [33] btc_corr_ok         — 1 if BTC aligned else 0
  [34] supertrend_count    — aligned TF count / 3
  [35] l1_score_norm       — L1 score / 15
  [36] l2_score_norm       — L2 score / 25
  [37] l3_score_norm       — L3 score / 20
  [38] l4_score_norm       — L4 score / 20
  [39] l5_score_norm       — L5 score / 15
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_DIM = 40


# ──────────────────────────────────────────────────────────
# Internal indicator helpers (recomputed from raw candles)
# ──────────────────────────────────────────────────────────


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(values[:period])
    k = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    rs = avg_g / avg_l if avg_l != 0 else 1e9
    out[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l != 0 else 1e9
        out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _stochrsi(closes: np.ndarray, rsi_p=14, stoch_p=14, k_s=3, d_s=3):
    rsi = _rsi(closes, rsi_p)
    n = len(closes)
    raw_k = np.full(n, np.nan)
    for i in range(stoch_p - 1, n):
        w = rsi[i - stoch_p + 1: i + 1]
        v = w[~np.isnan(w)]
        if len(v) < stoch_p // 2:
            continue
        mn, mx = v.min(), v.max()
        raw_k[i] = 50.0 if mx == mn else 100.0 * (rsi[i] - mn) / (mx - mn)
    k = np.full(n, np.nan)
    for i in range(k_s - 1, n):
        w = raw_k[i - k_s + 1: i + 1]
        v = w[~np.isnan(w)]
        if len(v) == k_s:
            k[i] = v.mean()
    d = np.full(n, np.nan)
    for i in range(d_s - 1, n):
        w = k[i - d_s + 1: i + 1]
        v = w[~np.isnan(w)]
        if len(v) == d_s:
            d[i] = v.mean()
    return k, d


def _macd(closes: np.ndarray, fast=12, slow=26, sig=9):
    ef = _ema(closes, fast)
    es = _ema(closes, slow)
    line = ef - es
    signal = _ema(np.nan_to_num(line), sig)
    hist = line - signal
    return line, signal, hist


def _atr(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    k = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1 - k) + tr[i] * k
    return atr


def _safe(val, default=0.0):
    """Return val if finite, else default."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _clip(val, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(val)))


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────


def extract(
    candles: List[Dict[str, Any]],
    layer_results: Optional[Dict[str, Any]] = None,
    order_flow: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Extract a FEATURE_DIM-length feature vector from candles + optional layer data.

    Parameters
    ----------
    candles      : list of OHLCV dicts (at least 50 bars recommended)
    layer_results: dict with keys 'l1'-'l5' containing layer result objects (optional)
    order_flow   : dict with keys: bid_ask_imbalance, cvd_signal, oi_trend,
                   funding_rate, ls_ratio (optional)

    Returns
    -------
    np.ndarray of shape (FEATURE_DIM,), dtype float32
    Guaranteed to be finite (NaN/Inf replaced with 0).
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    if not candles or len(candles) < 30:
        logger.debug("features.extract: insufficient candles, returning zeros")
        return vec

    try:
        opens  = np.array([c["open"]   for c in candles], dtype=np.float64)
        highs  = np.array([c["high"]   for c in candles], dtype=np.float64)
        lows   = np.array([c["low"]    for c in candles], dtype=np.float64)
        closes = np.array([c["close"]  for c in candles], dtype=np.float64)
        vols   = np.array([c.get("volume", 0.0) for c in candles], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("features.extract: candle parse error: %s", exc)
        return vec

    n = len(closes)

    # ── Indicators ────────────────────────────────────────
    atr_arr  = _atr(highs, lows, closes, 14)
    last_atr = float(np.nanmean(atr_arr[-3:])) if not np.all(np.isnan(atr_arr)) else 1e-8
    if last_atr < 1e-10:
        last_atr = 1e-8

    rsi_arr   = _rsi(closes, 14)
    sk, sd    = _stochrsi(closes)
    ml, sl, mh = _macd(closes)

    ema20  = _ema(closes, 20)
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200)

    c  = closes[-1]
    e20  = float(np.nanmean(ema20[-3:]))  if not np.all(np.isnan(ema20))  else c
    e50  = float(np.nanmean(ema50[-3:]))  if not np.all(np.isnan(ema50))  else c
    e200 = float(np.nanmean(ema200[-3:])) if not np.all(np.isnan(ema200)) else c

    # safe denominators
    e20  = e20  if e20  > 0 else c
    e50  = e50  if e50  > 0 else c
    e200 = e200 if e200 > 0 else c

    rsi_val = _safe(rsi_arr[-1], 50.0)
    sk_val  = _safe(np.nanmean(sk[-3:]) if not np.all(np.isnan(sk)) else 50.0, 50.0)
    sd_val  = _safe(np.nanmean(sd[-3:]) if not np.all(np.isnan(sd)) else 50.0, 50.0)
    mh_val  = _safe(np.nanmean(mh[-3:]) if not np.all(np.isnan(mh)) else 0.0, 0.0)
    ml_val  = _safe(np.nanmean(ml[-3:]) if not np.all(np.isnan(ml)) else 0.0, 0.0)

    # Volume
    avg_vol = float(np.mean(vols[-20:])) if len(vols) >= 20 else float(np.mean(vols))
    vol_ratio = _clip(vols[-1] / avg_vol if avg_vol > 0 else 1.0, 0.0, 5.0) / 5.0

    # ATR percentile
    lookback = min(100, n)
    recent_atr = atr_arr[-lookback:]
    valid_atr = recent_atr[~np.isnan(recent_atr)]
    atr_pct = float(np.sum(valid_atr <= last_atr) / len(valid_atr)) if len(valid_atr) > 0 else 0.5

    # Returns (ATR-normalized)
    def _ret(lag):
        if n > lag:
            return _clip((closes[-1] / closes[-1 - lag] - 1) / (last_atr / closes[-1]), -3, 3) / 3
        return 0.0

    # Micro / macro volatility
    rets = np.diff(closes[-21:]) / closes[-21:-1] if n >= 21 else np.array([0.0])
    vol_5  = float(np.std(rets[-5:]))  / (last_atr / c) if c > 0 else 0.0
    vol_20 = float(np.std(rets[-20:])) / (last_atr / c) if c > 0 else 0.0

    # Candle shape
    body   = (closes[-1] - opens[-1]) / last_atr
    wick   = (highs[-1] - lows[-1] - abs(closes[-1] - opens[-1])) / last_atr
    cl_pos = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1]) if highs[-1] != lows[-1] else 0.5

    # ADX (reuse L1 values if available)
    adx_val    = 0.0
    plus_di    = 0.0
    minus_di   = 0.0
    if layer_results and "l1" in layer_results and layer_results["l1"] is not None:
        l1 = layer_results["l1"]
        adx_val  = _safe(getattr(l1, "adx", 0.0), 0.0) / 100.0
        plus_di  = _safe(l1.details.get("plus_di", 0.0), 0.0) / 100.0
        minus_di = _safe(l1.details.get("minus_di", 0.0), 0.0) / 100.0

    # VWAP ratio
    vwap = c  # default
    if layer_results and "l5" in layer_results and layer_results["l5"] is not None:
        l5 = layer_results["l5"]
        vwap_raw = l5.details.get("vwap", c)
        # layer5 stores vwap as {"value": ..., "upper_band": ..., ...}
        if isinstance(vwap_raw, dict):
            vwap_raw = vwap_raw.get("value", c)
        vwap_raw = _safe(vwap_raw, c)
        if vwap_raw and vwap_raw > 0:
            vwap = vwap_raw
    vwap_ratio = _clip(c / vwap - 1.0, -0.05, 0.05) / 0.05

    # Channel position
    channel_map = {"UPPER": 1.0, "LOWER": -1.0, "MIDDLE": 0.0, "OUTSIDE": 0.0}
    channel_pos = 0.0
    if layer_results and "l5" in layer_results and layer_results["l5"] is not None:
        channel_pos = channel_map.get(layer_results["l5"].channel_position, 0.0)

    # Volume profile zone
    vp_map = {"VAH": 1.0, "POC": 0.0, "VAL": -1.0, "BETWEEN": 0.5}
    vp_zone = 0.5
    if layer_results and "l5" in layer_results and layer_results["l5"] is not None:
        vp_zone = vp_map.get(layer_results["l5"].volume_profile_zone, 0.5)

    # Session quality
    session_quality_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
    session_q = 0.3
    btc_corr  = 0.0
    if layer_results and "l6" in layer_results and layer_results["l6"] is not None:
        l6 = layer_results["l6"]
        session_q = session_quality_map.get(l6.session_quality, 0.3)
        btc_corr  = 1.0 if l6.btc_correlation_ok else 0.0

    # Supertrend count
    st_count = 0.0
    if layer_results and "l2" in layer_results and layer_results["l2"] is not None:
        st_count = layer_results["l2"].aligned_count / 3.0

    # Order flow features
    bid_ask_imb = 0.0
    cvd_norm    = 0.0
    oi_norm     = 0.0
    fund_norm   = 0.0
    ls_norm     = 0.0
    if layer_results and "l4" in layer_results and layer_results["l4"] is not None:
        l4 = layer_results["l4"]
        bid_ask_imb = _clip(_safe(l4.bid_ask_imbalance, 0.0))
        cvd_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
        cvd_norm = cvd_map.get(l4.cvd_signal, 0.0)
        oi_map = {"RISING": 1.0, "FLAT": 0.0, "FALLING": -1.0}
        oi_norm = oi_map.get(l4.oi_trend, 0.0)
        fund_raw = _safe(l4.details.get("funding_rate", 0.0), 0.0) * 1000
        fund_norm = _clip(fund_raw, -2.0, 2.0) / 2.0
        ls_raw = _safe(l4.details.get("ls_ratio_value", 1.0), 1.0)
        ls_norm = _clip((ls_raw - 1.0), -1.0, 1.0)
    elif order_flow:
        bid_ask_imb = _clip(_safe(order_flow.get("bid_ask_imbalance", 0.0)))
        cvd_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
        cvd_norm = cvd_map.get(order_flow.get("cvd_signal", "NEUTRAL"), 0.0)
        oi_map = {"RISING": 1.0, "FLAT": 0.0, "FALLING": -1.0}
        oi_norm = oi_map.get(order_flow.get("oi_trend", "FLAT"), 0.0)
        fund_norm = _clip(_safe(order_flow.get("funding_rate", 0.0), 0.0) * 1000, -2.0, 2.0) / 2.0
        ls_norm = _clip(_safe(order_flow.get("ls_ratio", 1.0), 1.0) - 1.0, -1.0, 1.0)

    # Layer scores (normalized)
    def _lscore(key, mx):
        if layer_results and key in layer_results and layer_results[key] is not None:
            return _clip(_safe(layer_results[key].score, 0.0) / mx, 0.0, 1.0)
        return 0.0

    # ── Assemble vector ───────────────────────────────────
    vec[0]  = np.float32(_clip(rsi_val / 100.0, 0.0, 1.0))
    vec[1]  = np.float32(_clip(sk_val / 100.0, 0.0, 1.0))
    vec[2]  = np.float32(_clip(sd_val / 100.0, 0.0, 1.0))
    vec[3]  = np.float32(_clip(mh_val / last_atr, -3.0, 3.0) / 3.0)
    vec[4]  = np.float32(_clip(ml_val / last_atr, -3.0, 3.0) / 3.0)
    vec[5]  = np.float32(_clip(c / e20  - 1.0, -0.1, 0.1) / 0.1)
    vec[6]  = np.float32(_clip(c / e50  - 1.0, -0.1, 0.1) / 0.1)
    vec[7]  = np.float32(_clip(c / e200 - 1.0, -0.2, 0.2) / 0.2)
    vec[8]  = np.float32(_clip(e20 / e50  - 1.0, -0.05, 0.05) / 0.05)
    vec[9]  = np.float32(_clip(e50 / e200 - 1.0, -0.1, 0.1) / 0.1)
    vec[10] = np.float32(_clip(adx_val, 0.0, 1.0))
    vec[11] = np.float32(_clip(plus_di, 0.0, 1.0))
    vec[12] = np.float32(_clip(minus_di, 0.0, 1.0))
    vec[13] = np.float32(_clip(atr_pct, 0.0, 1.0))
    vec[14] = np.float32(_clip(vol_ratio, 0.0, 1.0))
    vec[15] = np.float32(_clip(body, -3.0, 3.0) / 3.0)
    vec[16] = np.float32(_clip(wick, 0.0, 3.0) / 3.0)
    vec[17] = np.float32(_clip(cl_pos, 0.0, 1.0))
    vec[18] = np.float32(_ret(1))
    vec[19] = np.float32(_ret(3))
    vec[20] = np.float32(_ret(5))
    vec[21] = np.float32(_ret(10))
    vec[22] = np.float32(_clip(vol_5,  0.0, 3.0) / 3.0)
    vec[23] = np.float32(_clip(vol_20, 0.0, 3.0) / 3.0)
    vec[24] = np.float32(bid_ask_imb)
    vec[25] = np.float32(cvd_norm)
    vec[26] = np.float32(oi_norm)
    vec[27] = np.float32(fund_norm)
    vec[28] = np.float32(ls_norm)
    vec[29] = np.float32(vwap_ratio)
    vec[30] = np.float32(channel_pos)
    vec[31] = np.float32(vp_zone)
    vec[32] = np.float32(session_q)
    vec[33] = np.float32(btc_corr)
    vec[34] = np.float32(_clip(st_count, 0.0, 1.0))
    vec[35] = np.float32(_lscore("l1", 15.0))
    vec[36] = np.float32(_lscore("l2", 25.0))
    vec[37] = np.float32(_lscore("l3", 20.0))
    vec[38] = np.float32(_lscore("l4", 20.0))
    vec[39] = np.float32(_lscore("l5", 15.0))

    # Replace any remaining NaN/Inf
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
    return vec


def extract_sequence(
    candles: List[Dict[str, Any]],
    seq_len: int = 50,
) -> np.ndarray:
    """
    Extract a sequence of feature vectors for LSTM input.

    Returns np.ndarray of shape (seq_len, FEATURE_DIM), dtype float32.
    Each row i is the feature vector computed from candles[:i+seq_len_offset+1].
    The last row corresponds to the most recent bar.
    """
    n = len(candles)
    if n < seq_len + 30:
        # Pad with zeros if not enough history
        result = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
        available = max(0, n - 30)
        for i in range(available):
            end = n - available + i + 1
            result[seq_len - available + i] = extract(candles[:end])
        return result

    result = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
    for i in range(seq_len):
        end = n - seq_len + i + 1
        result[i] = extract(candles[:end])
    return result
