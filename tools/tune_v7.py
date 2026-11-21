#!/usr/bin/env python3
"""Tune v7's flat-path fan on RECALIBRATED truth worlds (own code).

Why recalibrate: v6's proxy said 0.0794 but the real board says 0.1487.
The calibration (tools/calib_v7.py) shows the old proxy's dispersion is
far too narrow: on truth families with per-trophy newcomer-rate
dispersion sigma=1.0 (lognormal), v6 scores 0.12-0.14 (near the real
0.1487), while on sigma=0 worlds it scores 0.072-0.079 (the old proxy).
The real dispersion is sigma ~ 1.0-1.25. v6's fan (logit sigma ~0.23)
is far too narrow for that world; v7 widens it.

Suites scored here (all aggregate-consistent, total 4276 earners):
  (a) original proxy worlds (13: A/B/D + 10 valid-truth, v4 worlds.py)
  (b) recalibrated family beta=1, sigma in {0.5, 1.0, 1.25}, seeds 1-3
      (9 worlds; beta=1 is proportional-to-old-rate, the shape whose
      median v6's hazard-age median already matches)

Grid: stage 1 median knobs (NEW_AGE_H, GROWTH), stage 2 fan knobs
(S_MULT, S_ALPHA). N_MC reduced to 800 for speed; the winner is
re-verified at full N_MC by tools/backtest_v7.py (to be run after).
"""
import itertools
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v4/tools")
import worlds as W4
import main as v7

KIT_IN = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit-v27"
              "/gta6-trophy-rarity-forecast-01-kit/input")
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ANS_PLAYERS = 3868
ANS_EARNERS = 4276


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(p):
    p = clamp_pct(p) / 100.0
    return math.log(p / (1.0 - p))


def pinball(y, qs):
    tot = 0.0
    for tau, q in zip(LEVELS, qs):
        d = logit_pct(y) - logit_pct(q)
        tot += 2.0 * max(tau * d, (tau - 1.0) * d)
    return tot / len(LEVELS)


def score_forecast(truth, forecast):
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    vals = [pinball(truth[i], by_id[i + 1]) for i in range(len(truth))]
    return sum(vals) / len(vals)


def run_v7(case):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / (case["id"] + ".in")
        p.write_text(json.dumps(case))
        c, hist, series = v7.read_case(p)
        return v7.forecast_case(c, hist, series)


def make_truth(case, beta, sigma, seed):
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
    target_new = ANS_EARNERS - base - fixed
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
    cases, proxy = W4.proxy_worlds()
    p_cases = [json.loads((KIT_IN / (c + ".in")).read_text())
               for c in ("p00", "p01")]
    # (case, truth) pairs for suite (a) + suite (b)
    suite_a = []
    for wname, ans in sorted(proxy.items()):
        for cid in ("p00", "p01"):
            case = p_cases[0] if cid == "p00" else p_cases[1]
            suite_a.append((case, [ans[cid][i + 1] for i in range(55)]))
    suite_b = []
    for sigma in (0.5, 1.0, 1.25):
        for seed in (1, 2, 3):
            for case in p_cases:
                suite_b.append((case, make_truth(case, 1.0, sigma, seed)))
    v7.N_MC = 800

    def eval_cfg():
        fa = [score_forecast(t, run_v7(c)) for c, t in suite_a]
        fb = [score_forecast(t, run_v7(c)) for c, t in suite_b]
        ma = sum(fa) / len(fa)
        mb = sum(fb) / len(fb)
        return (ma + mb) / 2, ma, mb

    print("baseline v6 knobs (NEW_AGE=0.20 GROWTH=0.15 S_MULT=0.15 "
          "S_ALPHA=0.30):", flush=True)
    v7.NEW_AGE_H, v7.GROWTH, v7.S_MULT, v7.S_ALPHA = 0.20, 0.15, 0.15, 0.30
    print("  mean %.4f  proxy(a) %.4f  recalib(b) %.4f" % eval_cfg(),
          flush=True)

    print("stage 1: median knobs (S_MULT=0.35, S_ALPHA=0.80)", flush=True)
    stage1 = []
    for age, gr in itertools.product((0.15, 0.20, 0.30), (0.10, 0.15, 0.25)):
        v7.NEW_AGE_H = age
        v7.GROWTH = gr
        v7.S_MULT = 0.35
        v7.S_ALPHA = 0.80
        mean, ma, mb = eval_cfg()
        stage1.append((mean, ma, mb, age, gr))
        print("  mean %.4f proxy %.4f recalib %.4f  NEW_AGE=%.2f GROWTH=%.2f"
              % (mean, ma, mb, age, gr), flush=True)
    stage1.sort()
    _, _, _, best_age, best_gr = stage1[0]

    print("stage 2: fan knobs (NEW_AGE=%.2f, GROWTH=%.2f)"
          % (best_age, best_gr), flush=True)
    stage2 = []
    for sm, sa in itertools.product((0.15, 0.25, 0.35, 0.50),
                                    (0.30, 0.50, 0.80, 1.00)):
        v7.NEW_AGE_H = best_age
        v7.GROWTH = best_gr
        v7.S_MULT = sm
        v7.S_ALPHA = sa
        mean, ma, mb = eval_cfg()
        stage2.append((mean, ma, mb, sm, sa))
        print("  mean %.4f proxy %.4f recalib %.4f  S_MULT=%.2f S_ALPHA=%.2f"
              % (mean, ma, mb, sm, sa), flush=True)
    stage2.sort()
    print("BEST stage1: mean %.4f proxy %.4f recalib %.4f NEW_AGE=%.2f "
          "GROWTH=%.2f" % stage1[0])
    print("BEST stage2: mean %.4f proxy %.4f recalib %.4f S_MULT=%.2f "
          "S_ALPHA=%.2f" % stage2[0])


if __name__ == "__main__":
    main()
