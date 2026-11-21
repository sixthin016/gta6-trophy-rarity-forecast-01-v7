#!/usr/bin/env python3
"""Fan-shape experiment for v7 (own code, written fresh).

The tune (tools/tune_v7.py) says widening v6's fan symmetrically HURTS
on both the old proxy and the recalibrated worlds. This experiment asks
WHY, with a per-sigma breakdown, and tests a SKEWED fan: the truth's
upside is longer than its downside (newcomer counts are lognormal and
the answer rate is bounded below by pure dilution e0/3868), so widen
only the ABOVE-median quantiles in logit space.

Configs (all on v7's main, patched in-process; forecasts cached per
case, scored against every truth):
  A  v6 knobs, NEW_AGE=0.20 (baseline)
  B  A with NEW_AGE=0.15
  C  B with upper-widen U=1.5 (above-median logit gaps x1.5)
  D  B with upper-widen U=2.0
  E  B with symmetric wide fan S_MULT=0.35 S_ALPHA=0.80
Suites: original proxy (13 worlds) + recalibrated per sigma
(0.5 / 1.0 / 1.25 / 1.5, seeds 1-3). Prints per-sigma means.
"""
import json
import math
import random
import sys
import tempfile
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


def grade(truth, forecast):
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    vals = [pinball(truth[i], by_id[i + 1]) for i in range(len(truth))]
    return sum(vals) / len(vals)


def run_v7(case):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / (case["id"] + ".in")
        p.write_text(json.dumps(case))
        c, hist, series = v7.read_case(p)
        return v7.forecast_case(c, hist, series)


def widen_upper(forecast, u):
    """Scale above-median logit gaps by u, keep below-median as is."""
    out = {"trophies": []}
    for r in forecast["trophies"]:
        q = r["q"]
        med = logit_pct(q[2])
        nq = []
        for i, v in enumerate(q):
            lg = logit_pct(v)
            if i > 2:
                lg = med + u * (lg - med)
            nq.append(round(clamp_pct(inv_logit_pct(lg)), 4))
        for i in range(1, len(nq)):
            if nq[i] < nq[i - 1]:
                nq[i] = nq[i - 1]
        out["trophies"].append({"id": r["id"], "q": nq})
    return out


def recalibrated_truth(case, sigma, seed, beta=1.0):
    last = case["history"][-1]
    n0 = last["players_tracked"]
    dn = ANS_PLAYERS - n0
    rates = last["rates"]
    rng = random.Random(7919 + seed)
    zs = [rng.gauss(0.0, 1.0) for _ in rates]
    raw = [0.0 if r >= 99.99 else (max(r, 0.01) ** beta) * math.exp(sigma * zs[i])
           for i, r in enumerate(rates)]
    base = sum(rates[i] * n0 / 100.0 for i in range(len(rates)))
    fixed = sum(dn for r in rates if r >= 99.99)
    variable = sum(raw[i] * dn / 100.0 for i in range(len(rates)))
    scale = (ANS_EARNERS - base - fixed) / variable if variable > 0 else 0.0
    ans = []
    for i, r in enumerate(rates):
        e0 = r * n0 / 100.0
        e1 = e0 + dn if r >= 99.99 else e0 + raw[i] * scale * dn / 100.0
        ans.append(100.0 * e1 / ANS_PLAYERS)
    return ans


def main():
    cases, proxy = W4.proxy_worlds()
    p_cases = [json.loads((KIT_IN / (c + ".in")).read_text())
               for c in ("p00", "p01")]
    proxy_truths = []
    for wname, ans in sorted(proxy.items()):
        for cid in ("p00", "p01"):
            case = p_cases[0] if cid == "p00" else p_cases[1]
            proxy_truths.append((case, [ans[cid][i + 1] for i in range(55)]))
    recalib = {}
    for sigma in (0.5, 1.0, 1.25, 1.5):
        lst = []
        for seed in (1, 2, 3):
            for case in p_cases:
                lst.append((case, recalibrated_truth(case, sigma, seed)))
        recalib[sigma] = lst

    v7.N_MC = 800
    configs = [
        ("A v6knobs", dict(NEW_AGE_H=0.20, S_MULT=0.15, S_ALPHA=0.30, U=1.0)),
        ("B NEW_AGE.15", dict(NEW_AGE_H=0.15, S_MULT=0.15, S_ALPHA=0.30, U=1.0)),
        ("C B+U1.5", dict(NEW_AGE_H=0.15, S_MULT=0.15, S_ALPHA=0.30, U=1.5)),
        ("D B+U2.0", dict(NEW_AGE_H=0.15, S_MULT=0.15, S_ALPHA=0.30, U=2.0)),
        ("E B+wide", dict(NEW_AGE_H=0.15, S_MULT=0.35, S_ALPHA=0.80, U=1.0)),
    ]
    print("%-14s %7s %7s %7s %7s %7s %7s" %
          ("config", "proxy", "s=0.5", "s=1.0", "s=1.25", "s=1.5", "mean"))
    for name, cfg in configs:
        v7.NEW_AGE_H = cfg["NEW_AGE_H"]
        v7.S_MULT = cfg["S_MULT"]
        v7.S_ALPHA = cfg["S_ALPHA"]
        fcs = {cid: run_v7(p_cases[0] if cid == "p00" else p_cases[1])
               for cid in ("p00", "p01")}
        if cfg["U"] != 1.0:
            fcs = {cid: widen_upper(fc, cfg["U"]) for cid, fc in fcs.items()}
        pa = sum(grade(t, fcs["p00" if c is p_cases[0] else "p01"])
                 for c, t in proxy_truths) / len(proxy_truths)
        row = [pa]
        for sigma in (0.5, 1.0, 1.25, 1.5):
            s = sum(grade(t, fcs["p00" if c is p_cases[0] else "p01"])
                    for c, t in recalib[sigma]) / len(recalib[sigma])
            row.append(s)
        mean = sum(row) / len(row)
        print("%-14s " % name + " ".join("%7.4f" % v for v in row)
              + " %7.4f" % mean, flush=True)


if __name__ == "__main__":
    main()
