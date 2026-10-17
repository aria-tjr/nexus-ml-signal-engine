"""
NEXUS Score Computation.

Aggregates all layer results into a final weighted score,
determines direction, grade, leverage, and ATR-based SL/TP levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from layers.layer1_regime import RegimeResult
from layers.layer2_trend import TrendResult
from layers.layer3_momentum import MomentumResult
from layers.layer4_orderflow import OrderFlowResult
from layers.layer5_structure import StructureResult
from layers.layer6_session import SessionResult
from layers.layer7_ml import MLLayerResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class NexusScore:
    total: float          # 0-100 (base layers) + ML delta
    grade: str            # "A+", "A", "B", "C", "SKIP"
    direction: str        # "LONG", "SHORT", "SKIP"
    leverage: int         # 200, 100, 50, 0
    layer_scores: dict
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward: float
    atr: float
    ml_score: float = 0.0       # L7 delta applied to total
    ml_direction: str = "N/A"   # ML predicted direction


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _atr14_from_candles(candles: list) -> float:
    """
    Compute ATR(14) from candle list using Wilder smoothing.
    Returns 0 if insufficient data.
    """
    if not candles or len(candles) < 15:
        return 0.0
    try:
        highs = np.array([c["high"] for c in candles], dtype=np.float64)
        lows = np.array([c["low"] for c in candles], dtype=np.float64)
        closes = np.array([c["close"] for c in candles], dtype=np.float64)

        n = len(closes)
        tr = np.empty(n, dtype=np.float64)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        atr = np.full(n, np.nan, dtype=np.float64)
        period = 14
        atr[period - 1] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha

        valid = atr[~np.isnan(atr)]
        return float(valid[-1]) if len(valid) > 0 else 0.0
    except Exception as exc:
        logger.warning("ATR computation error: %s", exc)
        return 0.0


def _get_leverage(score: float) -> int:
    """Return leverage from LEVERAGE_MAP based on score."""
    for low, high, lev in config.LEVERAGE_MAP:
        if low <= score <= high:
            return lev
    return 0


def _get_grade(score: float) -> str:
    if score >= 85:
        return "A+"
    elif score >= 70:
        return "A"
    elif score >= 55:
        return "B"
    elif score >= 40:
        return "C"
    return "SKIP"


# ──────────────────────────────────────────────────────────
# Main compute function
# ──────────────────────────────────────────────────────────


def compute(
    l1: RegimeResult,
    l2: TrendResult,
    l3: MomentumResult,
    l4: OrderFlowResult,
    l5: StructureResult,
    l6: SessionResult,
    candles_15m: list,
    regime: Optional[RegimeResult] = None,
    l7: Optional[MLLayerResult] = None,
) -> NexusScore:
    """
    Compute NEXUS score from all layer results.

    Uses adaptive weights based on market regime (trending vs volatile).
    Applies conflict penalties and generates ATR-based SL/TP levels.
    L7 (ML) applies a signed delta (-10..+10) on top of the base score.
    """
    # Regime determines weight set
    if regime is not None:
        use_l1 = regime
    else:
        use_l1 = l1

    if use_l1.volatility_state == "high":
        weights = config.LAYER_WEIGHTS_VOLATILE
    else:
        weights = config.LAYER_WEIGHTS_TRENDING

    # Raw layer scores (normalized to their max weights)
    # L1: max 15
    l1_norm = l1.score / 15.0 if 15.0 > 0 else 0
    # L2: max 25
    l2_norm = l2.score / 25.0 if 25.0 > 0 else 0
    # L3: max 20
    l3_norm = l3.score / 20.0 if 20.0 > 0 else 0
    # L4: max 20
    l4_norm = l4.score / 20.0 if 20.0 > 0 else 0
    # L5: max 15
    l5_norm = l5.score / 15.0 if 15.0 > 0 else 0
    # L6: max 5
    l6_norm = l6.score / 5.0 if 5.0 > 0 else 0

    # Weighted contributions
    weighted = (
        l1_norm * weights["l1"]
        + l2_norm * weights["l2"]
        + l3_norm * weights["l3"]
        + l4_norm * weights["l4"]
        + l5_norm * weights["l5"]
        + l6_norm * weights["l6"]
    )

    # ── Direction from L2 (primary) ──────────────────────
    direction = l2.direction  # "LONG", "SHORT", "MIXED"

    # Confirm/override with L3 and L4
    if direction == "MIXED":
        # Fall back to momentum
        if l3.macd_signal == "BUY" and l3.stochrsi_signal == "BUY":
            direction = "LONG"
        elif l3.macd_signal == "SELL" and l3.stochrsi_signal == "SELL":
            direction = "SHORT"

    # ── Conflict penalties ───────────────────────────────
    penalty = 0.0

    # L2 says LONG but L4 strongly says SHORT (or vice versa)
    l2_long = direction == "LONG"
    l4_bearish_strong = (
        l4.cvd_signal == "BEARISH"
        and l4.oi_trend == "FALLING"
        and l4.funding_bias == "SHORTS_PAYING"
    )
    l4_bullish_strong = (
        l4.cvd_signal == "BULLISH"
        and l4.oi_trend == "RISING"
        and l4.funding_bias == "LONGS_PAYING"
    )

    if l2_long and l4_bearish_strong:
        penalty += 10.0
        logger.debug("Applying L2/L4 conflict penalty (LONG vs bearish orderflow)")
    elif not l2_long and l4_bullish_strong:
        penalty += 10.0
        logger.debug("Applying L2/L4 conflict penalty (SHORT vs bullish orderflow)")

    # Fade risk penalty
    if l4.fade_risk:
        if (l4.crowd_positioning == "LONG_HEAVY" and direction == "LONG") or \
           (l4.crowd_positioning == "SHORT_HEAVY" and direction == "SHORT"):
            penalty += 5.0

    # Dead session penalty
    if l6.current_session == "DEAD":
        penalty += 3.0

    base_total = max(0.0, min(100.0, weighted - penalty))

    # ── L7 ML delta ──────────────────────────────────────
    ml_delta = 0.0
    ml_dir_out = "N/A"
    if l7 is not None and config.USE_ML:
        raw_delta = l7.score  # already in -10..+10
        # Only apply if ML confidence clears threshold
        if l7.confidence >= config.ML_MIN_CONFIDENCE:
            ml_delta = float(np.clip(raw_delta, config.ML_MAX_PENALTY, config.ML_MAX_BOOST))
        ml_dir_out = l7.ml_direction
        if ml_delta != 0:
            logger.debug("L7 ML delta: %.1f (direction=%s confidence=%.2f)",
                         ml_delta, l7.ml_direction, l7.confidence)

    total = max(0.0, min(100.0, base_total + ml_delta))

    # ── SL/TP using ATR ──────────────────────────────────
    # Use l1 ATR if available, else recompute
    atr = use_l1.atr_value if hasattr(use_l1, "atr_value") and use_l1.atr_value > 0 else 0.0

    if atr == 0.0:
        atr = _atr14_from_candles(candles_15m)

    entry_price = 0.0
    if candles_15m:
        try:
            entry_price = float(candles_15m[-1]["close"])
        except (KeyError, IndexError, TypeError):
            pass

    if direction == "LONG":
        stop_loss = entry_price - config.SL_ATR_MULT * atr
        tp1 = entry_price + 1.5 * atr
        tp2 = entry_price + config.TP_ATR_MULT * atr
        tp3 = entry_price + 4.5 * atr
    else:
        stop_loss = entry_price + config.SL_ATR_MULT * atr
        tp1 = entry_price - 1.5 * atr
        tp2 = entry_price - config.TP_ATR_MULT * atr
        tp3 = entry_price - 4.5 * atr

    sl_distance = abs(entry_price - stop_loss)
    tp1_distance = abs(tp1 - entry_price)
    risk_reward = tp1_distance / sl_distance if sl_distance > 0 else 0.0

    # ── Grade and leverage ────────────────────────────────
    grade = _get_grade(total)
    leverage = _get_leverage(total) if grade != "SKIP" and direction != "MIXED" else 0

    if grade == "SKIP" or direction == "MIXED":
        direction_out = "SKIP"
    else:
        direction_out = direction

    layer_scores = {
        "l1": round(l1.score, 2),
        "l2": round(l2.score, 2),
        "l3": round(l3.score, 2),
        "l4": round(l4.score, 2),
        "l5": round(l5.score, 2),
        "l6": round(l6.score, 2),
        "l7_ml_delta": round(ml_delta, 2),
        "penalty": round(penalty, 2),
        "base_total": round(base_total, 2),
        "weighted_raw": round(weighted, 2),
        "weights_used": "VOLATILE" if use_l1.volatility_state == "high" else "TRENDING",
    }

    return NexusScore(
        total=round(total, 2),
        grade=grade,
        direction=direction_out,
        leverage=leverage,
        layer_scores=layer_scores,
        entry_price=round(entry_price, 6),
        stop_loss=round(stop_loss, 6),
        take_profit_1=round(tp1, 6),
        take_profit_2=round(tp2, 6),
        take_profit_3=round(tp3, 6),
        risk_reward=round(risk_reward, 2),
        atr=round(atr, 6),
        ml_score=round(ml_delta, 2),
        ml_direction=ml_dir_out,
    )
