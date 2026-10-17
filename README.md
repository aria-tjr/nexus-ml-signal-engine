# NEXUS — 7-Layer Crypto Signal Engine with ML Consensus

**Regime, trend, momentum, order flow, structure and session analysis plus an RF + BiLSTM model, combined into one 0–100 score per Binance futures pair.**

![Python](https://img.shields.io/badge/language-Python-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<!-- add demo GIF / screenshot of a terminal or Telegram signal here -->

---

## Why NEXUS?

- **No API key needed to start.** Market data comes from Binance's public Spot and Futures REST endpoints through a lightweight async `aiohttp` client. There is no SDK dependency.
- **Gated, not just averaged.** Layer 1 vetoes non-trending markets (ADX < 20) before any other layer runs.
- **Regime-adaptive weights.** Layer weights switch between *trending* and *volatile* profiles.
- **ML as a referee, not a dictator.** The ML layer can only nudge the score by ±10 points, and it is dampened when confidence is low.
- **Works without ML.** scikit-learn and PyTorch are optional. Use `--no-ml` or `USE_ML = False`.

## Features

- Multi-timeframe analysis (4h / 1h / 15m), fetched concurrently with `asyncio`
- Order-flow inputs: order book depth, aggregated trades (CVD), funding rate, open interest and its history, and the global long/short ratio
- Auto-scanner loop across a configurable symbol list (default: 10 majors, every 15 min)
- Built-in training pipeline (`--train`) for the ML layer
- Optional enrichment from the AltFins API (best-effort, non-fatal)
- Formatted signal output to the console, with optional Telegram delivery

## Architecture

```
            Binance Futures/Spot (public REST, aiohttp)      AltFins (optional)
                             │                                      │
                             ▼                                      ▼
  L1 Regime ── ADX gate ──► not trending? ──► SKIP
      │ trending
      ▼
  L2 Trend (4h/1h/15m alignment) ──► direction hint
      │
      ├─► L3 Momentum   (MACD, StochRSI, RSI)
      ├─► L4 Order Flow (CVD, OI trend, funding, L/S ratio, fade risk)
      ├─► L5 Structure  (channel position, VWAP, volume-profile zone)
      └─► L6 Session    (trading session quality, BTC correlation)
      │
      ▼
  L7 ML Consensus  (RandomForest+GradientBoosting voting  +  BiLSTM w/ attention)
      │            signed delta −10 … +10
      ▼
  NEXUS Score (0–100, regime-weighted) ──► grade / direction / leverage tier
      │
      ▼
  signal_formatter ──► stdout  ──► Telegram (optional)
```

| Layer | Module | What it looks at |
|---|---|---|
| L1 Regime | `layers/layer1_regime.py` | ADX trend strength, volatility state. Acts as the gate |
| L2 Trend | `layers/layer2_trend.py` | Trend alignment across 4h / 1h / 15m |
| L3 Momentum | `layers/layer3_momentum.py` | MACD, StochRSI, RSI |
| L4 Order Flow | `layers/layer4_orderflow.py` | CVD, open interest, funding, long/short ratio |
| L5 Structure | `layers/layer5_structure.py` | Channel position, VWAP, volume profile |
| L6 Session | `layers/layer6_session.py` | Market session, BTC correlation |
| L7 ML | `layers/layer7_ml.py`, `ml/` | RF+GB soft-voting ensemble and 2-layer bidirectional LSTM with attention |

```
├── analyze.py            # CLI entry point / pipeline orchestrator
├── config.py             # all settings
├── data/                 # binance_client.py, altfins_client.py
├── layers/               # L1–L7
├── ml/                   # features, models, trainer, predictor
├── models/               # trained model files (created by --train)
├── scoring/nexus_score.py
├── scanner/auto_scanner.py
└── output/               # signal_formatter.py, telegram_bot.py
```

## Quick start

```bash
git clone https://github.com/aria-tjr/nexus-ml-signal-engine.git
cd nexus-ml-signal-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CPU-only PyTorch is enough:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

python analyze.py                       # ETHUSDT on 15m (defaults)
```

## Usage

```bash
python analyze.py --symbol BTCUSDT                 # analyze one symbol
python analyze.py --symbol SOLUSDT --tf 1h         # different primary timeframe
python analyze.py --no-ml                          # skip the ML layer (L7)
python analyze.py --scanner                        # continuous multi-symbol scanner
python analyze.py --telegram                       # also push signals to Telegram

# Train the ML layer (saves to ./models)
python analyze.py --train --symbol ETHUSDT --train-bars 5000 --train-epochs 50
```

<!-- add example signal output (console or Telegram screenshot) here -->

## Configuration

All settings live in `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `DEFAULT_SYMBOL` / `PRIMARY_TF` | `ETHUSDT` / `15m` | Defaults for single analysis |
| `TIMEFRAMES` | `4h, 1h, 15m` | Timeframes used for alignment |
| `MIN_SCORE` | `65` | Below this score the signal is `SKIP` |
| `REQUIRE_FULL_TF_ALIGNMENT` | `True` | Only signal when 4h/1h/15m trends agree |
| `SL_ATR_MULT` / `TP_ATR_MULT` | `2.0` / `4.5` | Stop / target distance in ATR |
| `ML_USE_LAYER_FEATURES` | `False` | Candle-only ML features (matches training) |
| `SCANNER_SYMBOLS` | 10 majors | Symbols covered by `--scanner` |
| `SCANNER_MIN_SCORE` | `80` | Scanner alert threshold |
| `SCANNER_INTERVAL_SECONDS` | `900` | Scan interval |
| `LAYER_WEIGHTS_TRENDING` / `_VOLATILE` | see file | Regime-adaptive layer weights |
| `USE_ML` | `True` | Enable/disable L7 globally |
| `ML_MIN_CONFIDENCE` | `0.55` | Below this, the ML delta is dampened to 0 |
| `LSTM_SEQ_LEN` | `50` | LSTM input window |
| `ML_MAX_BOOST` / `ML_MAX_PENALTY` | `+10` / `-10` | Cap on the ML contribution |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | empty | Required only for `--telegram` |
| `ALTFINS_API_KEY` | empty | Optional AltFins enrichment |

> ⚠️ `LEVERAGE_MAP` suggests leverage tiers of up to 200x by score. These are output labels only, and the engine does **not** place orders. Treat them as extremely aggressive and lower them for any real use.

## Backtest results

Honest walk-forward test (`python3 backtest.py`) on Binance futures 15m data: 1x, 0.04% taker fee per side, entry at the next bar's open, no lookahead. Parameters were picked on Dec 2025–Mar 2026 (26 configs tried) and evaluated **once** on the untouched **Apr–Sep 2026** window. Details are in [BACKTEST.md](BACKTEST.md).

| Symbol | Strategy | Trades | Win rate | Net return | Max DD | PF | Sharpe |
|---|---|---|---|---|---|---|---|
| BTCUSDT | NEXUS v1 defaults | 579 | 33.9% | −30.9% | −31.0% | 0.83 | −2.76 |
| BTCUSDT | NEXUS v2 (tuned) | 125 | 32.0% | −0.3% | −15.1% | 1.01 | 0.06 |
| BTCUSDT | Buy & hold | – | – | +22.4% | −29.7% | – | 1.27 |
| ETHUSDT | NEXUS v1 defaults | 626 | 35.1% | −24.1% | −31.9% | 0.91 | −1.20 |
| ETHUSDT | NEXUS v2 (tuned) | 139 | 30.2% | −10.7% | −16.4% | 0.89 | −0.84 |
| ETHUSDT | Buy & hold | – | – | +27.5% | −38.2% | – | 1.11 |

![Backtest equity](backtest_equity.png)

**Honest takeaway:** fewer, higher-quality trades (score ≥ 65, full 4h/1h/15m trend alignment, wider 2/4.5 ATR exits) cut trading by about 78% and halved drawdowns, but the engine is **still not profitable out of sample after fees** and trails buy & hold. The ML layer now works but rarely clears its confidence threshold (non-zero on <0.4% of bars). The order-flow layer is neutral in backtests because historical order-book and OI data isn't available. Funding and slippage are not modelled. NEXUS is a research framework, not a money printer.

## Roadmap

- [x] Walk-forward backtest (`backtest.py`)
- [x] Fix ML feature bug and train/serve skew
- [x] Trade-frequency filters (score threshold, TF alignment, wider exits)
- [ ] Positive out-of-sample edge after fees (labels matched to trade horizon, real order-flow history)
- [ ] Load secrets from `.env` (python-dotenv is already in requirements)
- [ ] Unit tests and CI
- [ ] Dockerfile

## Contributing

Contributions are welcome! Ideas, bug reports and new layers are all appreciated.

1. Fork the repo and create a branch: `git checkout -b feature/my-layer`
2. Keep each layer self-contained in `layers/` and return a result object with a `score`
3. Open a pull request describing what changed and why

If you find NEXUS useful, please ⭐ the repo. It helps others discover it.

## Disclaimer

This project is for **research and educational purposes only** and is **not financial advice**. Crypto and leveraged derivatives trading carries a high risk of losing all of your capital. Signals can be wrong, and past behavior does not predict future results. The authors accept no liability for any losses. Never trade with money you cannot afford to lose.

## License

[MIT](LICENSE) © 2026 aria-tjr
