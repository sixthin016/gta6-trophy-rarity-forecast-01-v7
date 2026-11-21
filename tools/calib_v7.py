#!/usr/bin/env python3
"""Calibration harness for v7 (own code, written fresh for the v7 build).

Question: why did v6 score 0.1487 on the real practice board when its
proxy said 0.0794? The proxy worlds (v4 worlds.py) spread the 103 new
trophies (4276 - 943 - 3230 for the 100% trophy) roughly proportional to
the old rate, alpha in [0.3, 1.0]. This harness searches truth families
with a different ALLOCATION SHAPE and finds which one reproduces the
four real-board anchors jointly:

  persistence 1.51, v5 1.85, v2 0.139, v6 0.148716
  (all four are our own entries' published practice losses)

Truth family: newcomer_rate_i = c * r_i**beta * exp(sigma * z_i) for the
54 non-100% trophies (z_i standard normal, fixed seed), c chosen so the
total new trophies over those 54 is exactly 103 (published aggregate).
Answer rate_i = (e0_i + newcomer_i * 3230 / 100) / 3868 * 100.
beta = 1 is proportional-to-old-rate (the old proxy's shape);
beta = 0 is equal per trophy; beta = 2 concentrates on common trophies.
sigma widens the per-trophy dispersion. Growth of the tracked cohort
inside the 90 min is set to 0 here (second-order; noted as a caveat).

Each model is run ONCE per case (forecasts do not depend on the truth),
then scored against every truth world. Usage:
  python3 tools/calib_v7.py
"""
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit-v27"
           "/gta6-trophy-rarity-forecast-01-kit/input")
V2 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v2/main.py")
V4 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v4/main.py")
V5 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v5/main.py")
V6 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v6/main.py")
V7 = ROOT / "main.py"

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
ANS_PLAYERS = 3868
ANS_EARNERS = 4276
ANCHORS = {"pers": 1.51, "v5": 1.85, "v2": 0.139, "v6": 0.148716}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(p):
    p = clamp_pct(p) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def pinball(y, qs):
    tot = 0.0
    for tau, q in zip(LEVELS, qs):
        d = logit_pct(y) - logit_pct(q)
        tot += 2.0 * max(tau * d, (tau - 1.0) * d)
    return tot / len(LEVELS)


def score_forecast(truth, forecast):
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    vals = []
    for i in range(len(truth)):
        vals.append(pinball(truth[i], by_id[i + 1]))
    return sum(vals) / len(vals)


def run_model(main_path, case):
    with tempfile.TemporaryDirectory() as td:
        ind = Path(td) / "in"
        outd = Path(td) / "out"
        ind.mkdir()
        outd.mkdir()
        (ind / (case["id"] + ".in")).write_text(json.dumps(case))
        import os
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(ind),
                   MAKEFUN_OUTPUT_DIR=str(outd))
        r = subprocess.run([sys.executable, str(main_path)], env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-1000:]
        return json.loads((outd / (case["id"] + ".out")).read_text())


def persistence_forecast(case):
    last = case["history"][-1]["rates"]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = last[i] if last[i] is not None else 0.0
        lg = logit_pct(r)
        rows.append({"id": tr["id"],
                     "q": [round(inv_logit_pct(lg + z * spread), 4)
                           for z in Z]})
    return {"trophies": rows}


def make_truth(case, beta, sigma, seed):
    """Answer rates for p00/p01 under the (beta, sigma, seed) family."""
    last = case["history"][-1]
    n0 = last["players_tracked"]
    dn = ANS_PLAYERS - n0
    rates = last["rates"]
    rng = random.Random(7919 + seed)
    zs = [rng.gauss(0.0, 1.0) for _ in rates]
    raw = []
    for i, r in enumerate(rates):
        if r >= 99.99:
            raw.append(0.0)
        else:
            raw.append((max(r, 0.01) ** beta) * math.exp(sigma * zs[i]))
    base = sum(rates[i] * n0 / 100.0 for i in range(len(rates)))
    fixed = sum(dn for i, r in enumerate(rates) if r >= 99.99)
    variable = sum(raw[i] * dn / 100.0 for i in range(len(rates)))
    target_new = ANS_EARNERS - base - fixed  # == 103
    scale = target_new / variable if variable > 0 else 0.0
    ans = []
    for i, r in enumerate(rates):
        e0 = r * n0 / 100.0
        if r >= 99.99:
            e1 = e0 + dn
        else:
            e1 = e0 + raw[i] * scale * dn / 100.0
        ans.append(100.0 * e1 / ANS_PLAYERS)
    return ans


def main():
    cases = {cid: json.loads((KIT / (cid + ".in")).read_text())
             for cid in ("p00", "p01")}
    print("running each model once per case ...", flush=True)
    fc = {}
    for name, path in (("v2", V2), ("v4", V4), ("v5", V5), ("v6", V6),
                       ("v7", V7)):
        fc[name] = {cid: run_model(path, cases[cid]) for cid in cases}
        print("  %s done" % name, flush=True)
    fc["pers"] = {cid: persistence_forecast(cases[cid]) for cid in cases}

    print("\n%-28s %7s %7s %7s %7s %7s  fit" %
          ("truth family", "pers", "v5", "v2", "v6", "v4"))
    print("real anchors:              %7.4f %7.4f %7.4f %7.4f" %
          (ANCHORS["pers"], ANCHORS["v5"], ANCHORS["v2"], ANCHORS["v6"]))
    results = []
    for beta in (0.0, 0.5, 1.0, 1.5, 2.0):
        for sigma in (0.0, 0.5, 1.0):
            for seed in (1, 2, 3):
                scores = {m: [] for m in fc}
                for cid in cases:
                    truth = make_truth(cases[cid], beta, sigma, seed)
                    for m in fc:
                        scores[m].append(
                            score_forecast(truth, fc[m][cid]))
                mean = {m: sum(v) / len(v) for m, v in scores.items()}
                # fit: squared log-error vs the four anchors
                fit = sum((math.log(max(mean[m], 1e-6) / ANCHORS[m])) ** 2
                          for m in ("pers", "v5", "v2", "v6"))
                results.append((fit, beta, sigma, seed, mean))
    results.sort(key=lambda r: r[0])
    for fit, beta, sigma, seed, mean in results:
        print("beta=%.1f sigma=%.1f seed=%d  %7.4f %7.4f %7.4f %7.4f %7.4f  %.4f"
              % (beta, sigma, seed, mean["pers"], mean["v5"], mean["v2"],
                 mean["v6"], mean["v4"], fit))
    print("\nBEST (smallest log-fit to the four real anchors):")
    fit, beta, sigma, seed, mean = results[0]
    print("beta=%.1f sigma=%.1f seed=%d fit=%.4f pers=%.4f v5=%.4f v2=%.4f "
          "v6=%.4f v4=%.4f" % (beta, sigma, seed, fit, mean["pers"],
                              mean["v5"], mean["v2"], mean["v6"],
                              mean["v4"]))


if __name__ == "__main__":
    main()
