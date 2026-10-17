"""
NEXUS ML Training Pipeline.

Fetches historical Binance kline data, engineers features and labels,
then trains and saves both models (RandomForest + LSTM).

Label strategy
──────────────
For each bar at index i we look FORWARD_BARS bars ahead:
  - If close[i + FORWARD_BARS] > close[i] + ATR_MULT * atr[i]  → LONG  (2)
  - If close[i + FORWARD_BARS] < close[i] - ATR_MULT * atr[i]  → SHORT (0)
  - Otherwise                                                    → NEUTRAL (1)

This produces a directional label that accounts for volatility, avoiding
labeling tiny moves as meaningful.

Usage
─────
  python -m ml.trainer --symbol ETHUSDT --tf 15m --bars 5000
  python -m ml.trainer --symbol BTCUSDT --bars 8000 --epochs 80
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import List, Tuple, Optional

import numpy as np

# Add parent to path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.features import extract, extract_sequence, FEATURE_DIM
from ml.models import RandomForestSignalClassifier, LSTMPricePredictor, SKLEARN_AVAILABLE, TORCH_AVAILABLE

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

FORWARD_BARS = 5     # how many bars ahead to measure outcome
ATR_MULT     = 0.8   # outcome threshold in ATR multiples


# ──────────────────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────────────────


async def _fetch_candles(symbol: str, interval: str, total_bars: int) -> List[dict]:
    """
    Fetch up to `total_bars` historical klines from Binance.
    Binance max per request = 1000, so we paginate if needed.
    """
    from data.binance_client import BinanceClient

    all_candles: List[dict] = []
    limit_per_req = 1000

    async with BinanceClient() as client:
        # First request — most recent
        batch = await client.get_klines(symbol, interval, limit=min(total_bars, limit_per_req))
        if not batch:
            logger.error("Failed to fetch candles for %s %s", symbol, interval)
            return []
        all_candles = batch

        # Paginate backwards using open_time of oldest fetched bar
        while len(all_candles) < total_bars:
            oldest_ts = all_candles[0]["open_time"]  # ms
            need = min(limit_per_req, total_bars - len(all_candles))
            # endTime must be strictly before oldest bar
            end_time = oldest_ts - 1
            batch = await client.get_klines(
                symbol, interval, limit=need, end_time=end_time
            )
            if not batch:
                break
            all_candles = batch + all_candles

    logger.info("Fetched %d candles for %s %s", len(all_candles), symbol, interval)
    return all_candles


# ──────────────────────────────────────────────────────────
# Label generation
# ──────────────────────────────────────────────────────────


def _compute_labels(
    candles: List[dict],
    forward_bars: int = FORWARD_BARS,
    atr_mult: float = ATR_MULT,
) -> np.ndarray:
    """
    Generate per-bar directional labels (0=SHORT, 1=NEUTRAL, 2=LONG).

    Returns array of length len(candles), with last forward_bars entries set to
    NEUTRAL (no future data available).
    """
    n = len(candles)
    labels = np.ones(n, dtype=np.int64)  # default NEUTRAL

    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    highs  = np.array([c["high"]  for c in candles], dtype=np.float64)
    lows   = np.array([c["low"]   for c in candles], dtype=np.float64)

    # ATR(14) Wilder
    period = 14
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1 - k) + tr[i] * k

    for i in range(n - forward_bars):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        c0 = closes[i]
        cf = closes[i + forward_bars]
        move = cf - c0
        threshold = atr_mult * a
        if move > threshold:
            labels[i] = 2   # LONG
        elif move < -threshold:
            labels[i] = 0   # SHORT
        # else stays NEUTRAL

    return labels


# ──────────────────────────────────────────────────────────
# Feature matrix construction
# ──────────────────────────────────────────────────────────


def build_feature_matrix(
    candles: List[dict],
    min_history: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, indices) where X[i] is the feature vector at candle index
    (min_history + i) using only data available up to that point.

    Returns
    -------
    X       : (N, FEATURE_DIM) float32
    indices : (N,) int — corresponding candle indices
    """
    n = len(candles)
    if n < min_history + FORWARD_BARS + 1:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.array([], dtype=np.int64)

    rows   = []
    idxs   = []

    for i in range(min_history, n - FORWARD_BARS):
        feat = extract(candles[max(0, i + 1 - config.ML_FEATURE_WINDOW): i + 1])
        rows.append(feat)
        idxs.append(i)

    if not rows:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.array([], dtype=np.int64)

    X = np.stack(rows, axis=0).astype(np.float32)
    return X, np.array(idxs, dtype=np.int64)


def build_sequence_matrix(
    candles: List[dict],
    seq_len: int,
    min_history: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X_seq, indices) where X_seq[i] has shape (seq_len, FEATURE_DIM).

    Returns
    -------
    X_seq   : (N, seq_len, FEATURE_DIM) float32
    indices : (N,) int
    """
    n = len(candles)
    start = max(min_history + seq_len, seq_len + 30)
    if n < start + FORWARD_BARS + 1:
        return np.empty((0, seq_len, FEATURE_DIM), dtype=np.float32), np.array([], dtype=np.int64)

    rows = []
    idxs = []
    for i in range(start, n - FORWARD_BARS):
        seq = extract_sequence(candles[max(0, i + 1 - config.ML_FEATURE_WINDOW): i + 1], seq_len=seq_len)
        rows.append(seq)
        idxs.append(i)

    if not rows:
        return np.empty((0, seq_len, FEATURE_DIM), dtype=np.float32), np.array([], dtype=np.int64)

    X_seq = np.stack(rows, axis=0).astype(np.float32)
    return X_seq, np.array(idxs, dtype=np.int64)


# ──────────────────────────────────────────────────────────
# Training orchestration
# ──────────────────────────────────────────────────────────


async def train(
    symbol: str = "ETHUSDT",
    interval: str = "15m",
    total_bars: int = 5000,
    rf_path: Optional[str] = None,
    lstm_path: Optional[str] = None,
    lstm_seq_len: int = 50,
    lstm_epochs: int = 50,
    forward_bars: int = FORWARD_BARS,
    atr_mult: float = ATR_MULT,
    skip_lstm: bool = False,
):
    """
    Full training pipeline.

    1. Fetch historical candles
    2. Generate labels
    3. Build feature matrices
    4. Train RandomForest
    5. Train LSTM (optional)
    6. Save models
    """
    rf_path   = rf_path   or os.path.join(config.ML_MODEL_DIR, "rf_signal.pkl")
    lstm_path = lstm_path or os.path.join(config.ML_MODEL_DIR, "lstm_price.pt")

    logger.info("=== NEXUS ML Training ===")
    logger.info("Symbol: %s  TF: %s  Bars: %d", symbol, interval, total_bars)

    # ── 1. Fetch ──────────────────────────────────────────
    candles = await _fetch_candles(symbol, interval, total_bars)
    if len(candles) < 200:
        logger.error("Not enough candle data (%d bars), aborting", len(candles))
        return

    # ── 2. Labels ─────────────────────────────────────────
    labels = _compute_labels(candles, forward_bars, atr_mult)
    dist = {0: int(np.sum(labels == 0)),
            1: int(np.sum(labels == 1)),
            2: int(np.sum(labels == 2))}
    logger.info("Label distribution: SHORT=%d NEUTRAL=%d LONG=%d", dist[0], dist[1], dist[2])

    # ── 3. RandomForest ───────────────────────────────────
    if SKLEARN_AVAILABLE:
        logger.info("Building feature matrix for RandomForest...")
        X, idxs = build_feature_matrix(candles)

        if len(X) == 0:
            logger.warning("Empty feature matrix, skipping RF training")
        else:
            y = labels[idxs]
            logger.info("RF training set: %d samples, %d features", len(X), X.shape[1])

            rf_model = RandomForestSignalClassifier()
            rf_model.build()
            rf_model.fit(X, y)
            rf_model.save(rf_path)
            logger.info("RandomForest saved to %s", rf_path)
    else:
        logger.warning("scikit-learn not available, skipping RF training")

    # ── 4. LSTM ───────────────────────────────────────────
    if not skip_lstm:
        if TORCH_AVAILABLE:
            logger.info("Building sequence matrix for LSTM (seq_len=%d)...", lstm_seq_len)
            X_seq, seq_idxs = build_sequence_matrix(candles, seq_len=lstm_seq_len)

            if len(X_seq) == 0:
                logger.warning("Empty sequence matrix, skipping LSTM training")
            else:
                y_seq = labels[seq_idxs]
                logger.info("LSTM training set: %d sequences", len(X_seq))

                lstm_model = LSTMPricePredictor(
                    seq_len=lstm_seq_len,
                    input_dim=FEATURE_DIM,
                )
                lstm_model.build()
                lstm_model.fit(X_seq, y_seq, epochs=lstm_epochs)
                lstm_model.save(lstm_path)
                logger.info("LSTM saved to %s", lstm_path)
        else:
            logger.warning("PyTorch not available, skipping LSTM training")
    else:
        logger.info("LSTM training skipped (--no-lstm flag)")

    logger.info("=== Training complete ===")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train NEXUS ML models")
    parser.add_argument("--symbol",   default="ETHUSDT")
    parser.add_argument("--tf",       default="15m", dest="interval")
    parser.add_argument("--bars",     default=5000,  type=int,
                        help="Total historical bars to fetch")
    parser.add_argument("--epochs",   default=50,    type=int,
                        help="LSTM training epochs")
    parser.add_argument("--seq-len",  default=50,    type=int,
                        help="LSTM sequence length")
    parser.add_argument("--forward",  default=FORWARD_BARS, type=int,
                        help="Forward bars for labeling")
    parser.add_argument("--atr-mult", default=ATR_MULT, type=float,
                        help="ATR multiplier for labeling threshold")
    parser.add_argument("--no-lstm",  action="store_true",
                        help="Skip LSTM training (RF only)")
    parser.add_argument("--rf-path",  default=None)
    parser.add_argument("--lstm-path", default=None)
    args = parser.parse_args()

    asyncio.run(train(
        symbol=args.symbol,
        interval=args.interval,
        total_bars=args.bars,
        rf_path=args.rf_path,
        lstm_path=args.lstm_path,
        lstm_seq_len=args.seq_len,
        lstm_epochs=args.epochs,
        forward_bars=args.forward,
        atr_mult=args.atr_mult,
        skip_lstm=args.no_lstm,
    ))
