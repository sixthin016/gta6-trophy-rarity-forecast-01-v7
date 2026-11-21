#!/usr/bin/env python3
"""Growth calibration for v7 (own code, written fresh).

Open question from the fan experiment: v6's real loss (0.1487) sits
above every truth family tried (best fit sigma~1.25 gives ~0.148 for
v6 but the fan/median knobs that win there are the narrow ones).
One term the families all set to ZERO is the tracked cohort's own
earning inside the horizon: v6's flat_median computes it (damped by
GROWTH=0.15) but the flat-path caller DISCARDS it -- count_fan is fed
earners0 (the last read's earners), not the grown earners. If the real
truth has substantial tracked growth, v6's median is too low for the
common trophies (where growth is biggest), and the fix is to feed the
grown earners into the fan.

This script searches (sigma, gamma): the truth is
  e1 = e0 + gamma * (n0 - e0) * (1 - exp(-hazard * horizon))
         + newcomers(sigma, beta=1)
  hazard = -ln(1 - r) / age_at_last_read, horizon = target - last_read.
gamma = 0 is no tracked growth; gamma = 1 is undamped hazard growth.
Scores v6 / persistence / v5 (forecasts cached once) and fits the
three solid real-board anchors: persistence 1.51, v5 1.85, v6 0.148716.
(v2's published 0.139 is impossible for its sigma=1.0 fan, whose floor
is ~0.197, so v2 is shown for reference only and not fitted.)
"""
import json
import math
import random
import subprocess
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
V2 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v2/main.py")
V4 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v4/main.py")
V5 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v5/main.py")
V6 = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-v6/main.py")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
ANS_PLAYERS = 3868
ANS_EARNERS = 4276
ANCHORS = {"pers": 1.51, "v5": 1.85, "v6": 0.148716}
RELEASE_H = 0.0  # 2026-11-19T00:00Z


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


def truth_with_growth(case, sigma, gamma, seed, beta=1.0):
    from datetime import datetime, timezone
    last = case["history"][-1]
    n0 = last["players_tracked"]
    dn = ANS_PLAYERS - n0
    rates = last["rates"]
    rel = datetime(2026, 11, 19, tzinfo=timezone.utc)
    t_last = datetime.fromisoformat(last["at"].replace("Z", "+00:00"))
    t_tgt = datetime.fromisoformat(case["target"].replace("Z", "+00:00"))
    age_h = max((t_last - rel).total_seconds() / 3600.0, 0.5)
    horizon_h = (t_tgt - t_last).total_seconds() / 3600.0
    rng = random.Random(7919 + seed)
    zs = [rng.gauss(0.0, 1.0) for _ in rates]
    raw = [0.0 if r >= 99.99 else (max(r, 0.01) ** beta) * math.exp(sigma * zs[i])
           for i, r in enumerate(rates)]
    growth = []
    for i, r in enumerate(rates):
        if r >= 99.99:
            growth.append(0.0)
            continue
        hazard = -math.log(max(1.0 - r / 100.0, 1e-6)) / age_h
        e0 = r * n0 / 100.0
        growth.append(gamma * (n0 - e0) * (1.0 - math.exp(-hazard * horizon_h)))
    base = sum(rates[i] * n0 / 100.0 for i in range(len(rates)))
    fixed = sum(dn for r in rates if r >= 99.99)
    variable = sum(raw[i] * dn / 100.0 for i in range(len(rates)))
    # newcomers must fill the remainder after base + growth + fixed
    target_new = ANS_EARNERS - base - fixed - sum(growth)
    scale = target_new / variable if variable > 0 else 0.0
    ans = []
    for i, r in enumerate(rates):
        e0 = r * n0 / 100.0
        if r >= 99.99:
            e1 = e0 + dn
        else:
            e1 = e0 + growth[i] + raw[i] * scale * dn / 100.0
        ans.append(100.0 * e1 / ANS_PLAYERS)
    return ans


def main():
    p_cases = [json.loads((KIT_IN / (c + ".in")).read_text())
               for c in ("p00", "p01")]
    print("running models once per case ...", flush=True)
    fc = {}
    for name, path in (("v2", V2), ("v4", V4), ("v5", V5), ("v6", V6)):
        fc[name] = {cid: run_model(path, p_cases[0] if cid == "p00" else p_cases[1])
                    for cid in ("p00", "p01")}
    fc["pers"] = {cid: persistence(p_cases[0] if cid == "p00" else p_cases[1])
                  for cid in ("p00", "p01")}
    print("%-22s %7s %7s %7s %7s %7s  fit" %
          ("truth (sigma,gamma)", "pers", "v5", "v2", "v6", "v4"))
    print("real anchors:           %7.4f %7.4f %7s %7.4f" %
          (ANCHORS["pers"], ANCHORS["v5"], "?", ANCHORS["v6"]))
    results = []
    for sigma in (0.5, 1.0, 1.25):
        for gamma in (0.0, 0.15, 0.5, 1.0):
            scores = {m: [] for m in fc}
            for seed in (1, 2, 3):
                for j, case in enumerate(p_cases):
                    cid = "p00" if j == 0 else "p01"
                    truth = truth_with_growth(case, sigma, gamma, seed)
                    for m in fc:
                        scores[m].append(grade(truth, fc[m][cid]))
            mean = {m: sum(v) / len(v) for m, v in scores.items()}
            fit = sum((math.log(max(mean[m], 1e-6) / ANCHORS[m])) ** 2
                      for m in ("pers", "v5", "v6"))
            results.append((fit, sigma, gamma, mean))
    results.sort(key=lambda r: r[0])
    for fit, sigma, gamma, mean in results:
        print("sigma=%.2f gamma=%.2f  %7.4f %7.4f %7.4f %7.4f %7.4f  %.4f"
              % (sigma, gamma, mean["pers"], mean["v5"], mean["v2"],
                 mean["v6"], mean["v4"], fit))
    fit, sigma, gamma, mean = results[0]
    print("BEST: sigma=%.2f gamma=%.2f fit=%.4f pers=%.4f v5=%.4f v2=%.4f "
          "v6=%.4f v4=%.4f" % (sigma, gamma, fit, mean["pers"], mean["v5"],
                              mean["v2"], mean["v6"], mean["v4"]))


if __name__ == "__main__":
    main()
