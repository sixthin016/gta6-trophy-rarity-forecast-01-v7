#!/usr/bin/env python3
"""Full grade for the v7 forecaster (own code, written fresh for v7).

Scores v7 head-to-head against v6, v4, v5 and the house persistence
baseline on five suites, all built only from the competition's published
numbers (never from any per-trophy answer, which is sealed):

  (a) practice-proxy worlds -- the real p00/p01 cases with answers built
      from the published aggregates (3,868 players, 4,276 trophies
      earned): 13 worlds (uniform / earliness / lognormal / 10 random
      valid truths). This is the OLD proxy: its per-trophy dispersion is
      far too narrow, which is exactly why v6 looked good here (0.0794)
      and scored 0.1487 on the real board.
  (b) recalibrated worlds -- same aggregates, but the 103 new trophies
      (over the 54 non-100% trophies) are spread with lognormal
      per-trophy dispersion sigma in {0.5, 1.0, 1.25} around the
      proportional-to-old-rate median. The calibration
      (tools/calib_v7.py) shows sigma ~ 1.0-1.25 is what reproduces the
      real board (v6 0.1487, persistence 1.51, v5 1.85). This is the
      suite v7 is tuned for; (a) is kept as a guard against overfitting
      to one dispersion.
  (c) v5-style final-like worlds -- 60 h history, 108 h horizon,
      Weibull / loglogistic / delayed / flat-refresh x 2 seeds.
  (d) v4-style synthetic worlds -- cohort simulation fitted to the
      launch-night snapshot: practice-like, final-like, expanding CV.
  (e) format / determinism / runtime checks on the real practice cases
      plus a moving-history case.

Usage: python3 tools/backtest_v7.py [proxy|recalib|final5|synth|checks|all]
"""
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v4/tools")
import worlds as W4

import main as v7

V4 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v4/main.py")
V5 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v5/main.py")
V6 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v6/main.py")
KIT_IN = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit-v27"
              "/gta6-trophy-rarity-forecast-01-kit/input")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
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
    """Mean pinball loss + typical miss (pp) + 90% coverage."""
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    losses, miss, cov = [], 0.0, 0
    for i in range(len(truth)):
        q = by_id[i + 1]
        losses.append(pinball(truth[i], q))
        miss += abs(truth[i] - q[2])
        if q[0] <= truth[i] <= q[4]:
            cov += 1
    n = len(truth)
    return sum(losses) / n, miss / n, cov / n


def run_in_subprocess(main_path, case):
    with tempfile.TemporaryDirectory() as td:
        ind = Path(td) / "in"
        outd = Path(td) / "out"
        ind.mkdir()
        outd.mkdir()
        (ind / (case["id"] + ".in")).write_text(json.dumps(case))
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(ind),
                   MAKEFUN_OUTPUT_DIR=str(outd))
        r = subprocess.run([sys.executable, str(main_path)], env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-1500:]
        return json.loads((outd / (case["id"] + ".out")).read_text())


def run_v7_in_process(case):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / (case["id"] + ".in")
        p.write_text(json.dumps(case))
        c, hist, series = v7.read_case(p)
        return v7.forecast_case(c, hist, series)


def persistence(case):
    last = case["history"][-1]["rates"]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = last[i] if last[i] is not None else 0.0
        lg = logit_pct(r)
        rows.append({"id": tr["id"],
                     "q": [round(inv_logit_pct(lg + z * spread), 4)
                           for z in Z]})
    return {"trophies": rows}


MODELS = {"v7": run_v7_in_process,
          "v6": lambda c: run_in_subprocess(V6, c),
          "v4": lambda c: run_in_subprocess(V4, c),
          "v5": lambda c: run_in_subprocess(V5, c),
          "pers": persistence}


def recalibrated_truth(case, sigma, seed, beta=1.0):
    """Answer rates under the recalibrated family: newcomer rate per
    trophy = c * r**beta * exp(sigma*z), rescaled so the total new
    trophies over the 54 non-100% trophies is exactly 103."""
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


def load_bt5():
    spec = importlib.util.spec_from_file_location(
        "bt5", "/home/hatch/workspace/makefun/"
        "gta6-trophy-rarity-forecast-01-v5/tools/backtest_v5.py")
    bt5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt5)
    return bt5


def suite_proxy(models=("v7", "v6", "v4", "v5", "pers")):
    print("== (a) practice-proxy worlds, old dispersion (13) ==")
    cases, proxy = W4.proxy_worlds()
    tot = {m: [] for m in models}
    for wname, ans in sorted(proxy.items()):
        for cid in ("p00", "p01"):
            truth = [ans[cid][i + 1] for i in range(55)]
            for m in models:
                tot[m].append(grade(truth, MODELS[m](cases[cid]))[0])
    for m in models:
        print("MEAN %-4s %.4f" % (m, sum(tot[m]) / len(tot[m])))
    return {m: sum(tot[m]) / len(tot[m]) for m in models}


def suite_recalib(models=("v7", "v6", "v4", "v5", "pers")):
    print("== (b) recalibrated worlds, sigma 0.5/1.0/1.25 (18) ==")
    p_cases = [json.loads((KIT_IN / (c + ".in")).read_text())
               for c in ("p00", "p01")]
    tot = {m: [] for m in models}
    sub = {}
    for sigma in (0.5, 1.0, 1.25):
        ssub = {m: [] for m in models}
        for seed in (1, 2, 3):
            for case in p_cases:
                truth = recalibrated_truth(case, sigma, seed)
                for m in models:
                    s = grade(truth, MODELS[m](case))[0]
                    tot[m].append(s)
                    ssub[m].append(s)
        sub[sigma] = {m: sum(v) / len(v) for m, v in ssub.items()}
        print("sigma=%.2f  " % sigma + "  ".join(
            "%s %.4f" % (m, sub[sigma][m]) for m in models))
    for m in models:
        print("MEAN %-4s %.4f" % (m, sum(tot[m]) / len(tot[m])))
    return {m: sum(tot[m]) / len(tot[m]) for m in models}


def suite_final5(models=("v7", "v6", "v4", "v5", "pers")):
    print("== (c) v5-style final-like worlds (60h hist, 108h horizon) ==")
    bt5 = load_bt5()
    tot = {m: [] for m in models}
    for world in ("weibull", "loglogistic", "delayed", "flat-refresh"):
        for seed in (101, 202):
            case, truth = bt5.build_case(world, seed)
            for m in models:
                tot[m].append(grade(truth, MODELS[m](case))[0])
    for m in models:
        print("MEAN %-4s %.4f" % (m, sum(tot[m]) / len(tot[m])))
    return {m: sum(tot[m]) / len(tot[m]) for m in models}


def suite_synth(models=("v7", "v6", "v4", "v5", "pers")):
    print("== (d) v4-style synthetic worlds (cohort simulation) ==")
    synth = W4.synthetic_worlds()
    tot = {m: [] for m in models}
    for prefix in ("synth-practice-", "synth-final-"):
        sel = {k: v for k, v in synth.items() if k.startswith(prefix)}
        sub = {m: [] for m in models}
        for name, (case, ans) in sorted(sel.items()):
            truth = [ans[i + 1] for i in range(55)]
            for m in models:
                s = grade(truth, MODELS[m](case))[0]
                tot[m].append(s)
                sub[m].append(s)
        print("%-16s " % prefix + "  ".join(
            "%s %.4f" % (m, sum(sub[m]) / len(sub[m])) for m in models))
    print("== expanding CV ==")
    for name, (case, ans) in sorted(W4.cv_worlds().items()):
        truth = [ans[i + 1] for i in range(55)]
        line = "%-12s " % name
        for m in models:
            s = grade(truth, MODELS[m](case))[0]
            tot[m].append(s)
            line += "%s %.4f  " % (m, s)
        print(line)
    for m in models:
        print("MEAN %-4s %.4f" % (m, sum(tot[m]) / len(tot[m])))
    return {m: sum(tot[m]) / len(tot[m]) for m in models}


def checks():
    print("== (e) format / determinism / runtime ==")
    for cid in ("p00", "p01"):
        case = json.loads((KIT_IN / (cid + ".in")).read_text())
        t0 = time.time()
        o1 = run_v7_in_process(case)
        dt = time.time() - t0
        o2 = run_v7_in_process(case)
        assert o1 == o2, "non-deterministic!"
        assert set(o1.keys()) == {"trophies"} and len(o1["trophies"]) == 55
        for r in o1["trophies"]:
            assert set(r.keys()) == {"id", "q"} and len(r["q"]) == 5
            assert all(0.0 <= v <= 100.0 for v in r["q"])
            assert all(r["q"][i] <= r["q"][i + 1] for i in range(4))
        blob = json.dumps(o1, separators=(",", ":"))
        assert len(blob) <= 65536, len(blob)
        print("%s: valid, %d bytes, deterministic, %.2fs"
              % (cid, len(blob), dt))
    bt5 = load_bt5()
    case, _ = bt5.build_case("weibull", 101)
    t0 = time.time()
    run_v7_in_process(case)
    print("moving-history case: %.2fs (cap 8s)" % (time.time() - t0))
    print("ALL CHECKS DONE")


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if "all" in which or "proxy" in which:
        suite_proxy()
    if "all" in which or "recalib" in which:
        suite_recalib()
    if "all" in which or "final5" in which:
        suite_final5()
    if "all" in which or "synth" in which:
        suite_synth()
    if "all" in which or "checks" in which:
        checks()
