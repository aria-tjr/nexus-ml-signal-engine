"""
NEXUS Multi-Layer Signal Engine — Main Entry Point.

Usage:
  python analyze.py                          # analyze ETHUSDT 15m
  python analyze.py --symbol BTCUSDT        # analyze BTCUSDT 15m
  python analyze.py --symbol SOLUSDT --tf 1h
  python analyze.py --scanner               # run auto-scanner loop
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

import config
from data.binance_client import BinanceClient
from data.altfins_client import AltFinsClient
from layers import (
    layer1_regime,
    layer2_trend,
    layer3_momentum,
    layer4_orderflow,
    layer5_structure,
    layer6_session,
    layer7_ml,
)
from output import signal_formatter, telegram_bot
from scoring.nexus_score import NexusScore, compute

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def analyze(
    symbol: str = config.DEFAULT_SYMBOL,
    timeframe: str = config.PRIMARY_TF,
    send_telegram: bool = False,
    use_ml: bool = True,
) -> Optional[NexusScore]:
    """
    Full NEXUS analysis pipeline for a single symbol.

    Steps:
      1. Fetch all data concurrently
      2. Run L1 — if not trending, return skip signal
      3. Determine direction hint from L2
      4. Run L3-L6 with direction hint
      5. Compute NEXUS score
      6. Format and print output
      7. Return NexusScore

    Returns NexusScore or None on critical error.
    """
    logger.info("Analyzing %s [%s]...", symbol, timeframe)

    async with BinanceClient() as client:
        # ── 1. Fetch all data concurrently ──────────────────
        (
            candles_15m,
            candles_1h,
            candles_4h,
            order_book,
            agg_trades,
            funding_rate,
            open_interest,
            oi_hist,
            ls_ratio,
        ) = await asyncio.gather(
            client.get_klines(symbol, timeframe, limit=250),
            client.get_klines(symbol, "1h", limit=250),
            client.get_klines(symbol, "4h", limit=250),
            client.get_order_book(symbol, limit=20),
            client.get_agg_trades(symbol, limit=500),
            client.get_funding_rate(symbol),
            client.get_open_interest(symbol),
            client.get_open_interest_hist(symbol, period="15m", limit=20),
            client.get_long_short_ratio(symbol, period="15m", limit=20),
        )

        # BTC candles for correlation (skip if this IS BTC)
        if symbol.upper() not in ("BTCUSDT", "BTCPERP", "XBTUSDT"):
            btc_candles = await client.get_klines("BTCUSDT", timeframe, limit=100)
        else:
            btc_candles = candles_15m

    # AltFins data (optional, best-effort)
    altfins_data = None
    try:
        async with AltFinsClient() as af_client:
            altfins_data = await af_client.get_signals(symbol)
    except Exception as exc:
        logger.debug("AltFins fetch error (non-fatal): %s", exc)

    # Defensive check
    candles_15m = candles_15m or []
    candles_1h = candles_1h or []
    candles_4h = candles_4h or []
    agg_trades = agg_trades or []
    oi_hist = oi_hist or []
    ls_ratio = ls_ratio or []

    if len(candles_15m) < 50:
        logger.error("Insufficient candle data for %s (%d bars)", symbol, len(candles_15m))
        return None

    candles_by_tf = {
        "4h": candles_4h,
        "1h": candles_1h,
        "15m": candles_15m,
    }

    # ── 2. L1 Regime gate ────────────────────────────────
    l1 = layer1_regime.analyze(candles_15m)
    logger.info("L1 Regime: ADX=%.1f trending=%s vol=%s score=%.0f",
                l1.adx, l1.is_trending, l1.volatility_state, l1.score)

    if not l1.is_trending:
        logger.info("L1 gate: market not trending (ADX=%.1f < 20), issuing SKIP", l1.adx)
        # Build minimal score for formatting
        from layers.layer2_trend import TrendResult
        from layers.layer3_momentum import MomentumResult
        from layers.layer4_orderflow import OrderFlowResult
        from layers.layer5_structure import StructureResult
        from layers.layer6_session import SessionResult

        l2_empty = TrendResult(0, "MIXED", {}, {}, 0.0, {})
        l3_empty = MomentumResult("NEUTRAL", "NEUTRAL", "NONE", 50.0, 0.0, {})
        l4_empty = OrderFlowResult(0.0, "NEUTRAL", "FLAT", "NEUTRAL", "BALANCED", False, 0.0, {})
        l5_empty = StructureResult("MIDDLE", "AT", False, False, "BETWEEN", 5.0, {})
        l6_empty = SessionResult("DEAD", "LOW", False, 0.0, {})

        nexus = compute(l1, l2_empty, l3_empty, l4_empty, l5_empty, l6_empty, candles_15m, l1)
        formatted = signal_formatter.format_skip(
            symbol, timeframe, nexus, f"ADX={l1.adx:.1f} — no trending regime"
        )
        print(formatted)
        if send_telegram:
            await telegram_bot.send_signal(formatted)
        return nexus

    # ── 3. L2 Trend — determine direction hint ────────────
    l2 = layer2_trend.analyze(candles_by_tf)
    direction_hint = l2.direction if l2.direction in ("LONG", "SHORT") else "LONG"
    logger.info("L2 Trend: aligned=%d/%d direction=%s score=%.0f",
                l2.aligned_count, len(candles_by_tf), l2.direction, l2.score)

    # ── 4. Run L3-L6 with direction hint ─────────────────
    l3 = layer3_momentum.analyze(candles_15m, direction_hint)
    logger.info("L3 Momentum: MACD=%s StochRSI=%s RSI=%.1f score=%.0f",
                l3.macd_signal, l3.stochrsi_signal, l3.rsi_value, l3.score)

    l4 = layer4_orderflow.analyze(
        order_book,
        agg_trades,
        open_interest,
        oi_hist,
        funding_rate,
        ls_ratio,
        direction_hint,
    )
    logger.info("L4 OrderFlow: CVD=%s OI=%s Funding=%s FadeRisk=%s score=%.0f",
                l4.cvd_signal, l4.oi_trend, l4.funding_bias, l4.fade_risk, l4.score)

    l5 = layer5_structure.analyze(candles_15m, candles_1h)
    logger.info("L5 Structure: channel=%s VWAP=%s VPzone=%s score=%.0f",
                l5.channel_position, l5.vwap_position, l5.volume_profile_zone, l5.score)

    l6 = layer6_session.analyze(symbol, candles_15m, btc_candles, direction_hint)
    logger.info("L6 Session: %s quality=%s btc_ok=%s score=%.0f",
                l6.current_session, l6.session_quality, l6.btc_correlation_ok, l6.score)

    # ── 4b. L7 ML Consensus ───────────────────────────────
    effective_use_ml = use_ml and config.USE_ML
    layer_results_for_ml = {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "l5": l5, "l6": l6}
    l7 = layer7_ml.analyze(
        candles_15m,
        direction_hint=direction_hint,
        layer_results=layer_results_for_ml,
        use_ml=effective_use_ml,
    )
    if l7.rf_available or l7.lstm_available:
        logger.info("L7 ML: dir=%s conf=%.2f score=%+.1f RF=%s LSTM=%s",
                    l7.ml_direction, l7.confidence, l7.score,
                    l7.rf_available, l7.lstm_available)
    else:
        logger.info("L7 ML: no trained models available")

    # Optional: incorporate AltFins signals into details
    if altfins_data:
        logger.info("AltFins: momentum=%s trend=%s patterns=%d",
                    altfins_data.get("momentum_signal"),
                    altfins_data.get("trend_signal"),
                    len(altfins_data.get("patterns", [])))

    # ── 5. Compute NEXUS score ────────────────────────────
    nexus = compute(l1, l2, l3, l4, l5, l6, candles_15m, l1, l7=l7)
    logger.info("NEXUS: total=%.1f grade=%s direction=%s leverage=%dx",
                nexus.total, nexus.grade, nexus.direction, nexus.leverage)

    # ── 6. Format and print ───────────────────────────────
    # MIN_SCORE was previously not enforced (grade "C" 40-54 still produced a signal)
    if nexus.direction != "SKIP" and nexus.total < config.MIN_SCORE:
        nexus.direction = "SKIP"
    if (nexus.direction != "SKIP" and config.REQUIRE_FULL_TF_ALIGNMENT
            and l2.aligned_count < len(candles_by_tf)):
        nexus.direction = "SKIP"
    if nexus.grade == "SKIP" or nexus.direction == "SKIP":
        reason = f"Score {nexus.total:.0f} below threshold ({config.MIN_SCORE}) or filtered"
        formatted = signal_formatter.format_skip(symbol, timeframe, nexus, reason)
    else:
        formatted = signal_formatter.format_signal(
            symbol, timeframe, nexus, l1, l2, l3, l4, l5, l6, l7=l7
        )

    print(formatted)

    if send_telegram and nexus.direction != "SKIP":
        ok = await telegram_bot.send_signal(formatted)
        if ok:
            logger.info("Signal sent to Telegram")
        else:
            logger.warning("Failed to send signal to Telegram")

    # ── 7. Return score ───────────────────────────────────
    return nexus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NEXUS Multi-Layer Crypto Signal Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default=config.DEFAULT_SYMBOL,
        help=f"Trading symbol (default: {config.DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--tf",
        default=config.PRIMARY_TF,
        dest="timeframe",
        help=f"Primary timeframe (default: {config.PRIMARY_TF})",
    )
    parser.add_argument(
        "--scanner",
        action="store_true",
        help="Run auto-scanner loop across all configured symbols",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Send signals to Telegram",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train ML models on historical data then exit",
    )
    parser.add_argument(
        "--train-bars",
        default=5000,
        type=int,
        dest="train_bars",
        help="Number of historical bars to use for training (default: 5000)",
    )
    parser.add_argument(
        "--train-epochs",
        default=50,
        type=int,
        dest="train_epochs",
        help="LSTM training epochs (default: 50)",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        dest="no_ml",
        help="Disable ML layer (L7) for this run",
    )
    args = parser.parse_args()

    if args.train:
        from ml.trainer import train as ml_train
        asyncio.run(ml_train(
            symbol=args.symbol,
            interval=args.timeframe,
            total_bars=args.train_bars,
            lstm_epochs=args.train_epochs,
        ))
    elif args.scanner:
        from scanner.auto_scanner import run_scanner
        asyncio.run(run_scanner())
    else:
        asyncio.run(analyze(
            args.symbol,
            args.timeframe,
            send_telegram=args.telegram,
            use_ml=not args.no_ml,
        ))
