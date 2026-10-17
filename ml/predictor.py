"""
NEXUS ML Inference Pipeline.

Loads saved models (if available) and runs predictions on live data.
Fully graceful — if models are missing or broken, returns neutral outputs.

Public API
──────────
  predictor = NexusPredictor()
  predictor.load()                     # loads from config.ML_MODEL_DIR
  result = predictor.predict(candles, layer_results)
  # result.rf_proba  : [p_short, p_neutral, p_long]
  # result.lstm_proba: [p_short, p_neutral, p_long]
  # result.ensemble  : weighted blend
  # result.direction : "LONG" | "SHORT" | "NEUTRAL"
  # result.confidence: 0.0 – 1.0
  # result.ml_score  : -10 to +10 contribution to NEXUS total
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import config
from ml.features import extract, extract_sequence, FEATURE_DIM
from ml.models import (
    RandomForestSignalClassifier,
    LSTMPricePredictor,
    SKLEARN_AVAILABLE,
    TORCH_AVAILABLE,
    LABEL_SHORT,
    LABEL_NEUTRAL,
    LABEL_LONG,
)

logger = logging.getLogger(__name__)

NEUTRAL_PROBA = np.array([1/3, 1/3, 1/3], dtype=np.float32)


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class MLResult:
    rf_proba:    np.ndarray = field(default_factory=lambda: NEUTRAL_PROBA.copy())
    lstm_proba:  np.ndarray = field(default_factory=lambda: NEUTRAL_PROBA.copy())
    ensemble:    np.ndarray = field(default_factory=lambda: NEUTRAL_PROBA.copy())
    direction:   str = "NEUTRAL"     # "LONG", "SHORT", "NEUTRAL"
    confidence:  float = 0.0         # 0-1, how strongly the model agrees
    ml_score:    float = 0.0         # -10 to +10 NEXUS score contribution
    rf_available:   bool = False
    lstm_available: bool = False
    details:     Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────


class NexusPredictor:
    """
    Loads and runs RF + LSTM models for NEXUS signal augmentation.

    Thread-safe for read (predict) after load(). Not thread-safe for load().
    """

    def __init__(
        self,
        rf_path:   Optional[str] = None,
        lstm_path: Optional[str] = None,
        seq_len:   int = 50,
    ):
        self.rf_path   = rf_path   or os.path.join(config.ML_MODEL_DIR, "rf_signal.pkl")
        self.lstm_path = lstm_path or os.path.join(config.ML_MODEL_DIR, "lstm_price.pt")
        self.seq_len   = seq_len

        self._rf:   Optional[RandomForestSignalClassifier] = None
        self._lstm: Optional[LSTMPricePredictor]           = None

    # ── Loading ───────────────────────────────────────────

    def load(self) -> "NexusPredictor":
        """Load models from disk. Silently skips missing files."""
        self._load_rf()
        self._load_lstm()
        return self

    def _load_rf(self):
        if not SKLEARN_AVAILABLE:
            logger.debug("sklearn not available, RF skipped")
            return
        if not os.path.exists(self.rf_path):
            logger.info("RF model not found at %s (run --train first)", self.rf_path)
            return
        try:
            self._rf = RandomForestSignalClassifier.load(self.rf_path)
            logger.info("RF model loaded")
        except Exception as exc:
            logger.warning("Failed to load RF model: %s", exc)

    def _load_lstm(self):
        if not TORCH_AVAILABLE:
            logger.debug("PyTorch not available, LSTM skipped")
            return
        if not os.path.exists(self.lstm_path):
            logger.info("LSTM model not found at %s (run --train first)", self.lstm_path)
            return
        try:
            self._lstm = LSTMPricePredictor.load(self.lstm_path)
            logger.info("LSTM model loaded (seq_len=%d)", self._lstm.seq_len)
        except Exception as exc:
            logger.warning("Failed to load LSTM model: %s", exc)

    # ── Inference ─────────────────────────────────────────

    def predict(
        self,
        candles: List[Dict[str, Any]],
        layer_results: Optional[Dict[str, Any]] = None,
        direction_hint: str = "LONG",
    ) -> MLResult:
        """
        Run ML inference on the most recent bar.

        Parameters
        ----------
        candles       : list of OHLCV dicts (at least 50 recommended)
        layer_results : dict with 'l1'-'l5' result objects (optional)
        direction_hint: "LONG" or "SHORT" from L2 trend analysis

        Returns
        -------
        MLResult with ensemble probabilities, direction, confidence, ml_score
        """
        if not candles or len(candles) < 30:
            return MLResult(details={"error": "insufficient candles"})

        rf_proba   = NEUTRAL_PROBA.copy()
        lstm_proba = NEUTRAL_PROBA.copy()
        rf_ok   = False
        lstm_ok = False

        # ── RandomForest prediction ───────────────────────
        if self._rf is not None and self._rf.is_trained:
            try:
                feat = extract(candles, layer_results=layer_results)
                rf_proba = self._rf.predict_proba(feat)
                rf_ok = True
            except Exception as exc:
                logger.warning("RF inference error: %s", exc)

        # ── LSTM prediction ───────────────────────────────
        if self._lstm is not None and self._lstm.is_trained:
            try:
                seq = extract_sequence(candles, seq_len=self._lstm.seq_len)
                lstm_proba = self._lstm.predict_proba(seq)
                lstm_ok = True
            except Exception as exc:
                logger.warning("LSTM inference error: %s", exc)

        # ── Ensemble blend ────────────────────────────────
        # Weights: RF=0.6, LSTM=0.4 if both available
        # Only RF: 1.0, Only LSTM: 1.0, Neither: uniform
        if rf_ok and lstm_ok:
            ensemble = 0.6 * rf_proba + 0.4 * lstm_proba
        elif rf_ok:
            ensemble = rf_proba.copy()
        elif lstm_ok:
            ensemble = lstm_proba.copy()
        else:
            ensemble = NEUTRAL_PROBA.copy()

        # Normalize
        s = ensemble.sum()
        if s > 0:
            ensemble /= s

        # ── Direction from ensemble ───────────────────────
        p_short, p_neutral, p_long = ensemble
        max_idx = int(np.argmax(ensemble))
        raw_direction = ["SHORT", "NEUTRAL", "LONG"][max_idx]

        # Confidence = max(p_short, p_long) - p_neutral
        # (how decisive is the directional call vs sitting on the fence)
        directional_confidence = max(p_short, p_long)
        confidence = float(directional_confidence)

        # ── ML Score contribution (-10 to +10) ───────────
        # +10: ML strongly agrees with direction_hint (confidence > 0.7)
        # +5 : ML moderately agrees (confidence 0.55-0.70)
        # 0  : ML is neutral or uncertain
        # -5 : ML mildly disagrees
        # -10: ML strongly disagrees (confidence > 0.7 opposite)
        ml_score = self._compute_ml_score(
            ensemble, direction_hint, confidence
        )

        result = MLResult(
            rf_proba=rf_proba,
            lstm_proba=lstm_proba,
            ensemble=ensemble,
            direction=raw_direction,
            confidence=confidence,
            ml_score=ml_score,
            rf_available=rf_ok,
            lstm_available=lstm_ok,
            details={
                "rf_proba":   rf_proba.tolist(),
                "lstm_proba": lstm_proba.tolist(),
                "ensemble":   ensemble.tolist(),
                "p_long":     float(p_long),
                "p_short":    float(p_short),
                "p_neutral":  float(p_neutral),
                "confidence": round(confidence, 3),
                "direction_hint": direction_hint,
            },
        )

        return result

    def _compute_ml_score(
        self,
        ensemble: np.ndarray,
        direction_hint: str,
        confidence: float,
    ) -> float:
        """
        Map ensemble probabilities to a -10..+10 NEXUS score delta.

        Rules:
          • Agreement at high confidence  → +10
          • Agreement at medium confidence → +5
          • Neutral / uncertain            → 0
          • Disagreement at medium         → -5
          • Disagreement at high           → -10
        """
        p_short, p_neutral, p_long = ensemble

        if direction_hint == "LONG":
            agree_p    = float(p_long)
            disagree_p = float(p_short)
        else:
            agree_p    = float(p_short)
            disagree_p = float(p_long)

        # Agreement
        weak = config.ML_MIN_CONFIDENCE
        strong = weak + 0.10
        if agree_p >= strong:
            return 10.0
        elif agree_p >= weak:
            return 5.0
        # Disagreement
        elif disagree_p >= strong:
            return -10.0
        elif disagree_p >= weak:
            return -5.0
        # Neutral / undecided
        else:
            return 0.0

    # ── Convenience ──────────────────────────────────────

    @property
    def has_rf(self) -> bool:
        return self._rf is not None and self._rf.is_trained

    @property
    def has_lstm(self) -> bool:
        return self._lstm is not None and self._lstm.is_trained

    @property
    def is_ready(self) -> bool:
        return self.has_rf or self.has_lstm


# ──────────────────────────────────────────────────────────
# Module-level singleton (lazy-loaded)
# ──────────────────────────────────────────────────────────

_predictor: Optional[NexusPredictor] = None


def get_predictor() -> NexusPredictor:
    """Return the module-level predictor, loading models on first call."""
    global _predictor
    if _predictor is None:
        _predictor = NexusPredictor()
        _predictor.load()
    return _predictor
