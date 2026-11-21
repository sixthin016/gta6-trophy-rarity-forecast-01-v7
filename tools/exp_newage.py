#!/usr/bin/env python3
"""Refine NEW_AGE_H for v7 (own code, written fresh).

The fan experiment says the only consistent winner is a lower median
(NEW_AGE 0.20 -> 0.15). This refines the grid: 0.08 / 0.10 / 0.12 /
0.15 / 0.18 / 0.20, narrow fan (S_MULT=0.15, S_ALPHA=0.30),
median-preserving. Forecasts cached per case; scored on the old proxy
(13 worlds) and the recalibrated family per sigma (0.5/1.0/1.25/1.5,
seeds 1-3). A value is accepted only if it wins on the recalibrated
mean AND does not regress the old proxy by more than 0.002.
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
from exp_fan import (grade, recalibrated_truth, run_v7)

KIT_IN = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit-v27"
              "/gta6-trophy-rarity-forecast-01-kit/input")


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
    v7.S_MULT = 0.15
    v7.S_ALPHA = 0.30
    print("%-10s %7s %7s %7s %7s %7s %7s" %
          ("NEW_AGE", "proxy", "s=0.5", "s=1.0", "s=1.25", "s=1.5", "mean"))
    for age in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
        v7.NEW_AGE_H = age
        fcs = {cid: run_v7(p_cases[0] if cid == "p00" else p_cases[1])
               for cid in ("p00", "p01")}

        def sc(pairs):
            return sum(grade(t, fcs["p00" if c is p_cases[0] else "p01"])
                       for c, t in pairs) / len(pairs)

        pa = sc(proxy_truths)
        row = [pa] + [sc(recalib[s]) for s in (0.5, 1.0, 1.25, 1.5)]
        mean = sum(row) / len(row)
        print("%-10.2f " % age + " ".join("%7.4f" % v for v in row)
              + " %7.4f" % mean, flush=True)


if __name__ == "__main__":
    main()
