"""
NEXUS walk-forward backtest v2 (honest, no lookahead, tuning/out-of-sample split).

  python3 backtest.py            # tune on TUNE window, evaluate chosen config ONCE on OOS window

Protocol
  * Data: Binance USD-M futures monthly klines (data.binance.vision), 15m/1h/4h.
  * At each 15m close the engine sees only closed bars (last 250 15m bars; 1h/4h bars with
    close_time <= that close). Layer 4 (order flow) has no historical source and is neutral.
    Layer 6 uses the bar timestamp.
  * Entry at the next bar's open. Exit at SL / TP (ATR multiples) or after max-hold at close;
    if SL and TP are both touched in one bar, SL is assumed. One position per symbol,
    100% equity, 1x. Fees 0.04% taker per side. Funding and slippage NOT modelled.
  * ML (RandomForest+GB): for tuning, trained on the 4 months BEFORE the tuning window; for
    the OOS run, retrained on the tuning window (still strictly before OOS).
  * Parameters are chosen ONLY on the tuning window (objective: sum of BTC+ETH net return,
    >= 30 trades). The OOS window is evaluated once with the chosen config.
"""
from __future__ import annotations

import argparse, csv, datetime as dt, io, itertools, json, logging, math, os, pickle, urllib.request, zipfile
from typing import Dict, List

import numpy as np

import config
from layers import layer1_regime, layer2_trend, layer3_momentum, layer4_orderflow, layer5_structure, layer6_session
from scoring.nexus_score import compute

logging.basicConfig(level=logging.WARNING)
FEE = 0.0004
WINDOW = 250
DATA_DIR = ".bt_data"
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
ML_TRAIN_TUNE = ("2025-08", "2025-11")
TUNE = ("2025-12", "2026-03")
OOS = ("2026-04", "2026-09")

GRID = {"min_score": [55, 65, 75], "full_align": [False, True], "cooldown": [0, 32],
        "exit": [(1.5, 3.0, 96), (2.0, 4.5, 192)]}
ML_GRID = [0.55, 0.45]
BASELINE = {"min_score": 55, "full_align": False, "cooldown": 0, "exit": (1.5, 3.0, 96)}


def months(a, b):
    y, m = map(int, a.split("-")); ey, em = map(int, b.split("-")); out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}"); m += 1
        if m == 13: y, m = y + 1, 1
    return out


def shift(ym, k):
    y, m = map(int, ym.split("-")); t = y * 12 + m - 1 + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def ms(ym):
    y, m = map(int, ym.split("-"))
    return int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)


def load(symbol, tf, mlist):
    os.makedirs(DATA_DIR, exist_ok=True); rows = []
    for ym in mlist:
        name = f"{symbol}-{tf}-{ym}"; path = os.path.join(DATA_DIR, name + ".csv")
        if not os.path.exists(path):
            raw = urllib.request.urlopen(f"{BASE}/{symbol}/{tf}/{name}.zip", timeout=60).read()
            zipfile.ZipFile(io.BytesIO(raw)).extractall(DATA_DIR)
        with open(path) as f:
            for r in csv.reader(f):
                if r and r[0].isdigit():
                    rows.append({"open_time": int(r[0]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
                                 "close": float(r[4]), "volume": float(r[5]), "close_time": int(r[6])})
    rows.sort(key=lambda c: c["open_time"])
    return rows


# ── per-bar engine output (computed once, then reused by every config) ──
def precompute(symbol, win):
    cache = os.path.join(DATA_DIR, f"sig_{symbol}_{win[0]}_{win[1]}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    ml = months(shift(win[0], -1), win[1])
    c15, c1h, c4h = (load(symbol, tf, ml) for tf in ("15m", "1h", "4h"))
    btc = c15 if symbol == "BTCUSDT" else load("BTCUSDT", "15m", ml)
    bidx = {c["open_time"]: k for k, c in enumerate(btc)}
    t0, t1 = ms(win[0]), ms(shift(win[1], 1))
    rows = []; j1 = j4 = 0
    for k, c in enumerate(c15):
        ct = c["close_time"]
        while j1 < len(c1h) and c1h[j1]["close_time"] <= ct: j1 += 1
        while j4 < len(c4h) and c4h[j4]["close_time"] <= ct: j4 += 1
        if not (t0 <= c["open_time"] < t1): continue
        w = c15[k - WINDOW + 1:k + 1]
        rec = {"k": k, "sig": None}
        l1 = layer1_regime.analyze(w)
        if l1.is_trending:
            h1, h4 = c1h[max(0, j1 - WINDOW):j1], c4h[max(0, j4 - WINDOW):j4]
            l2 = layer2_trend.analyze({"4h": h4, "1h": h1, "15m": w})
            hint = l2.direction if l2.direction in ("LONG", "SHORT") else "LONG"
            l3 = layer3_momentum.analyze(w, hint)
            l4 = layer4_orderflow.analyze(None, None, None, None, None, None, hint)
            l5 = layer5_structure.analyze(w, h1)
            b = bidx.get(c["open_time"])
            l6 = layer6_session.analyze(symbol, w, btc[max(0, b - 99):b + 1] if b else w, hint,
                                        utc_now=dt.datetime.fromtimestamp((ct + 1) / 1000, dt.timezone.utc))
            n = compute(l1, l2, l3, l4, l5, l6, w, l1, l7=None)
            rec["sig"] = {"base": n.layer_scores["base_total"], "dir": n.direction, "hint": hint,
                          "atr": n.atr, "aligned": l2.aligned_count}
        rows.append(rec)
    out = {"c15": c15, "rows": rows}
    pickle.dump(out, open(cache, "wb"))
    return out


def ml_probs(symbol, train_win, pre):
    """Train RF+GB on train_win only, return per-row [p_short, p_neutral, p_long]."""
    from ml import trainer
    from ml.features import extract
    from ml.models import RandomForestSignalClassifier
    tr = load(symbol, "15m", months(shift(train_win[0], -1), train_win[1]))
    a, b = ms(train_win[0]), ms(shift(train_win[1], 1))
    lo = next(i for i, c in enumerate(tr) if c["open_time"] >= a)
    tr = tr[lo - WINDOW:] if lo >= WINDOW else tr
    tr = [c for c in tr if c["open_time"] < b]
    labels = trainer._compute_labels(tr)
    idx = np.arange(WINDOW, len(tr) - trainer.FORWARD_BARS)
    X = np.stack([extract(tr[i - WINDOW + 1:i + 1]) for i in idx])
    rf = RandomForestSignalClassifier().build().fit(X, labels[idx])
    rf.save(os.path.join(DATA_DIR, f"models_{symbol}_{train_win[0]}", "rf_signal.pkl"))
    c15 = pre["c15"]
    Xt = np.stack([extract(c15[r["k"] - WINDOW + 1:r["k"] + 1]) for r in pre["rows"]])
    return rf.predict_proba(Xt)


def ml_delta(p, hint, T):
    """Mirrors ml/predictor._compute_ml_score + the ML_MIN_CONFIDENCE gate in nexus_score."""
    ps, _, pl = p
    if max(ps, pl) < T: return 0.0
    agree, dis = (pl, ps) if hint == "LONG" else (ps, pl)
    if agree >= T + 0.10: return 10.0
    if agree >= T: return 5.0
    if dis >= T + 0.10: return -10.0
    if dis >= T: return -5.0
    return 0.0


def simulate(pre, cfg, probs=None, T=None):
    c15, rows = pre["c15"], pre["rows"]
    sl_m, tp_m, hold = cfg["exit"]
    end = rows[-1]["k"]; eq = 1.0; curve = []; trades = []
    pos = 0; next_ok = 0
    while pos < len(rows):
        r = rows[pos]; k = r["k"]; s = r["sig"]
        take = False
        if s and k >= next_ok and k < end and s["dir"] in ("LONG", "SHORT") and s["atr"] > 0:
            total = s["base"] + (ml_delta(probs[pos], s["hint"], T) if probs is not None else 0.0)
            total = max(0.0, min(100.0, total))
            take = total >= cfg["min_score"] and (not cfg["full_align"] or s["aligned"] == 3)
        if not take:
            curve.append((c15[k]["close_time"], eq)); pos += 1; continue
        long = s["dir"] == "LONG"; entry = c15[k + 1]["open"]; atr = s["atr"]
        sl = entry - sl_m * atr if long else entry + sl_m * atr
        tp = entry + tp_m * atr if long else entry - tp_m * atr
        x = k + 1
        while True:
            bar = c15[x]
            if (bar["low"] <= sl) if long else (bar["high"] >= sl): px = sl; break
            if (bar["high"] >= tp) if long else (bar["low"] <= tp): px = tp; break
            if x - k >= hold or x == end: px = bar["close"]; break
            x += 1
        for y in range(k, x):
            m = c15[y]["close"] / entry if y > k else 1.0
            curve.append((c15[y]["close_time"], eq * (m if long else 2 - m) * (1 - FEE)))
        g = px / entry if long else 2 - px / entry
        net = g * (1 - FEE) ** 2 - 1; eq *= 1 + net
        curve.append((c15[x]["close_time"], eq))
        trades.append({"t": c15[k + 1]["open_time"], "dir": s["dir"], "ret": net, "gross": g - 1})
        next_ok = x + 1 + cfg["cooldown"]
        pos += (x + 1 - k)
    return trades, curve


def metrics(trades, curve):
    eqs = np.array([e for _, e in curve]); peak = np.maximum.accumulate(eqs)
    rets = [t["ret"] for t in trades]; w = [x for x in rets if x > 0]; l = [x for x in rets if x <= 0]
    days = {}
    for t, e in curve: days[t // 86400000] = e
    d = np.array(list(days.values())); dr = np.diff(d) / d[:-1]
    return {"trades": len(rets), "win_rate": len(w) / len(rets) if rets else 0.0,
            "total_return": float(eqs[-1] - 1), "gross_return": float(np.prod([1 + t["gross"] for t in trades]) - 1),
            "max_drawdown": float(((eqs - peak) / peak).min()),
            "profit_factor": (sum(w) / -sum(l)) if l and sum(l) < 0 else (float("inf") if w else 0.0),
            "sharpe": float(dr.mean() / dr.std() * math.sqrt(365)) if dr.std() > 0 else 0.0}


def buy_hold(pre):
    c15 = pre["c15"]; ks = [r["k"] for r in pre["rows"]]
    e0 = c15[ks[0]]["open"]
    curve = [(c15[k]["close_time"], c15[k]["close"] / e0 * (1 - FEE)) for k in ks]
    curve[-1] = (curve[-1][0], curve[-1][1] * (1 - FEE))
    m = metrics([], curve); m.update(trades=1, win_rate=None, profit_factor=None, gross_return=None)
    return m, curve


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="backtest_results.json"); a = ap.parse_args()
    res = {"protocol": {"ml_train_for_tuning": ML_TRAIN_TUNE, "tune": TUNE, "oos": OOS, "fee_per_side": FEE,
                        "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in GRID.items()},
                        "ml_conf_grid": ML_GRID}}
    pre_t = {s: precompute(s, TUNE) for s in SYMBOLS}
    pre_o = {s: precompute(s, OOS) for s in SYMBOLS}
    print("precompute done", flush=True)

    # ── tuning (no ML) ──
    tuning = []
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID.keys(), vals))
        ms_ = {s: metrics(*simulate(pre_t[s], cfg)) for s in SYMBOLS}
        tuning.append({"cfg": cfg, "ret_sum": sum(m["total_return"] for m in ms_.values()),
                       "trades": sum(m["trades"] for m in ms_.values()), "per_symbol": ms_})
    ok = [t for t in tuning if t["trades"] >= 30] or tuning
    best = max(ok, key=lambda t: t["ret_sum"]); cfg = best["cfg"]
    print("best no-ML cfg on tuning:", cfg, round(best["ret_sum"], 4), best["trades"], flush=True)

    # ── tuning ML confidence threshold with chosen cfg ──
    probs_t = {s: ml_probs(s, ML_TRAIN_TUNE, pre_t[s]) for s in SYMBOLS}
    ml_tune = []
    for T in ML_GRID:
        ms_ = {s: metrics(*simulate(pre_t[s], cfg, probs_t[s], T)) for s in SYMBOLS}
        ml_tune.append({"T": T, "ret_sum": sum(m["total_return"] for m in ms_.values()), "per_symbol": ms_})
    T = max(ml_tune, key=lambda t: t["ret_sum"])["T"]
    print("best ML conf on tuning:", T, flush=True)
    res["tuning"] = {"configs_tried": len(tuning) + len(ml_tune), "no_ml_results": tuning, "ml_results": ml_tune,
                     "chosen_cfg": cfg, "chosen_ml_conf": T}

    # ── OOS: evaluated once ──
    res["oos"] = {}; curves = {}
    for s in SYMBOLS:
        probs = ml_probs(s, TUNE, pre_o[s])
        bh, bhc = buy_hold(pre_o[s])
        base_m = metrics(*simulate(pre_o[s], BASELINE))
        tr0, cv0 = simulate(pre_o[s], cfg); tr1, cv1 = simulate(pre_o[s], cfg, probs, T)
        res["oos"][s] = {"buy_and_hold": bh, "v1_baseline_no_ml": base_m,
                         "tuned_no_ml": metrics(tr0, cv0), "tuned_with_ml": metrics(tr1, cv1),
                         "ml_nonzero_delta_share": float(np.mean([ml_delta(p, r["sig"]["hint"], T) != 0
                                                                  for p, r in zip(probs, pre_o[s]["rows"]) if r["sig"]])),
                         "trades_tuned_with_ml": tr1}
        curves[s] = {"buy & hold": bhc, "tuned no-ML": cv0, "tuned + ML": cv1}
        print(s, json.dumps({k: v for k, v in res["oos"][s].items() if not k.startswith("trades")}, default=str), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(SYMBOLS), 1, figsize=(11, 4 * len(SYMBOLS)))
    for ax, s in zip(axes, SYMBOLS):
        for name, cv in curves[s].items():
            ax.plot([dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc) for t, _ in cv], [e for _, e in cv],
                    label=name, ls="--" if name == "tuned + ML" else "-")
        ax.axhline(1, color="grey", lw=0.5); ax.legend(); ax.set_ylabel("equity")
        ax.set_title(f"{s} out-of-sample {OOS[0]}..{OOS[1]} (1x, 0.04%/side)")
    fig.tight_layout(); fig.savefig("backtest_equity.png", dpi=110)


if __name__ == "__main__":
    main()
