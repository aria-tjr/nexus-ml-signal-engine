"""
Layer 7 – ML Consensus.

Runs the NEXUS ML predictor (RandomForest + LSTM) and translates
model outputs into a score contribution for the NEXUS engine.

Score range: -10 to +10 (applied as a signed delta to the base score)
  +10 : Both models agree strongly with technical direction
  +5  : One model agrees moderately
   0  : Models neutral / not available
  -5  : One model disagrees moderately
  -10 : Both models disagree strongly

The layer is completely optional — if no trained models are found,
it returns a neutral result with score=0 and the rest of the system
operates as normal.
"""

from __future__ import annotations

import logging

import config
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class MLLayerResult:
    ml_direction:   str   = "NEUTRAL"   # "LONG", "SHORT", "NEUTRAL"
    confidence:     float = 0.0         # 0-1
    rf_available:   bool  = False
    lstm_available: bool  = False
    agrees_with_l2: bool  = False       # True if ML direction matches L2
    score:          float = 0.0         # -10 to +10
    details:        Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────────────────


def analyze(
    candles: List[Dict],
    direction_hint: str = "LONG",
    layer_results: Optional[Dict] = None,
    use_ml: bool = True,
) -> MLLayerResult:
    """
    Run ML consensus layer.

    Parameters
    ----------
    candles        : 15m OHLCV candle list (at least 50 bars)
    direction_hint : "LONG" or "SHORT" from L2 trend analysis
    layer_results  : dict of layer result objects (l1-l5) for feature enrichment
    use_ml         : if False, returns neutral result immediately (--no-ml flag)

    Returns
    -------
    MLLayerResult with score in range [-10, +10]
    """
    _neutral = MLLayerResult(
        details={"reason": "ml_disabled" if not use_ml else "no_models_loaded"}
    )

    if not use_ml:
        return _neutral

    if not candles or len(candles) < 30:
        logger.debug("Layer7: insufficient candles")
        return _neutral

    try:
        from ml.predictor import get_predictor
    except ImportError as exc:
        logger.debug("Layer7: ml.predictor import failed: %s", exc)
        return _neutral

    try:
        predictor = get_predictor()
    except Exception as exc:
        logger.warning("Layer7: predictor init failed: %s", exc)
        return _neutral

    if not predictor.is_ready:
        logger.debug("Layer7: no trained models available")
        _neutral.details["reason"] = "no_trained_models"
        return _neutral

    try:
        result = predictor.predict(
            candles,
            # Training (ml/trainer.py) only sees candles, so by default inference must too,
            # otherwise the models get feature values they were never trained on.
            layer_results=layer_results if getattr(config, "ML_USE_LAYER_FEATURES", False) else None,
            direction_hint=direction_hint,
        )
    except Exception as exc:
        logger.warning("Layer7: prediction failed: %s", exc)
        return _neutral

    agrees = result.direction == direction_hint

    return MLLayerResult(
        ml_direction=result.direction,
        confidence=round(result.confidence, 3),
        rf_available=result.rf_available,
        lstm_available=result.lstm_available,
        agrees_with_l2=agrees,
        score=round(result.ml_score, 2),
        details={
            **result.details,
            "agrees_with_l2": agrees,
            "rf_available":   result.rf_available,
            "lstm_available": result.lstm_available,
        },
    )
