# All configuration, API keys, constants
BINANCE_API_KEY = ""
BINANCE_SECRET = ""
ALTFINS_API_KEY = ""
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

DEFAULT_SYMBOL = "ETHUSDT"
TIMEFRAMES = ["4h", "1h", "15m"]
PRIMARY_TF = "15m"

# Leverage gating
LEVERAGE_MAP = [
    (85, 100, 200),   # score 85-100 -> 200x
    (70, 84, 100),    # score 70-84 -> 100x
    (55, 69, 50),     # score 55-69 -> 50x
]
MIN_SCORE = 65

# Scanner
SCANNER_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                   "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
SCANNER_MIN_SCORE = 80
SCANNER_INTERVAL_SECONDS = 900  # 15 min

# Layer weights (sum to 100) - adaptive
# Note: L7 (ML) contributes a signed delta (-10..+10) on top, not a weighted layer.
LAYER_WEIGHTS_TRENDING = {
    "l1": 15,  # regime gate
    "l2": 25,  # trend - more important when trending
    "l3": 20,  # momentum
    "l4": 20,  # order flow
    "l5": 15,  # structure
    "l6": 5,   # session/correlation
}
LAYER_WEIGHTS_VOLATILE = {
    "l1": 15,
    "l2": 15,
    "l3": 15,
    "l4": 30,  # order flow more important in volatile
    "l5": 20,
    "l6": 5,
}

# ── ML / DL settings ──────────────────────────────────────
# Set USE_ML = False to bypass L7 entirely (e.g. before training)
USE_ML = True

# Directory where trained model files are saved/loaded
ML_MODEL_DIR = "models"

# Minimum ML confidence (0-1) to treat the ML signal as meaningful.
# Below this, the ml_score contribution is dampened to 0.
ML_MIN_CONFIDENCE = 0.55

# LSTM sequence length (must match what was used during training)
LSTM_SEQ_LEN = 50

# ML score caps: how much L7 can boost or penalize the base NEXUS score
ML_MAX_BOOST   =  10.0   # maximum positive delta
ML_MAX_PENALTY = -10.0   # maximum negative delta

# ── Trade filters / exits (chosen on tuning window Dec-2025..Mar-2026, see BACKTEST.md) ──
REQUIRE_FULL_TF_ALIGNMENT = True    # require L2 trend aligned on all of 4h/1h/15m
SIGNAL_COOLDOWN_BARS = 0            # bars to wait after a trade closes (scanner/backtest)
SL_ATR_MULT = 2.0
TP_ATR_MULT = 4.5                   # take_profit_2 distance
MAX_HOLD_BARS = 192

# ML features: candle-only by default so inference matches ml/trainer.py
ML_USE_LAYER_FEATURES = False
ML_FEATURE_WINDOW = 250             # bars per feature extraction (= live fetch size)
