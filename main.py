#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster -- make.fun gta6-trophy-rarity-forecast-01, v7.

Reads every case file (*.in) in $MAKEFUN_INPUT_DIR (else /input) and writes
one forecast per case to $MAKEFUN_OUTPUT_DIR (else /output/<id>.out): for
each of the game's 55 trophies, five quantiles of its Exophase earn rate at
the case's target, scored by pinball loss on the logit scale.

Why v7 exists -- the diagnosis of v6's real board score (measured, not
assumed). v6's local proxy said 0.0794 and predicted rank ~110-130, but
the live board gave v6 loss 0.148716, rank 1035/8982. The proxy was
miscalibrated, and the calibration that proves it (tools/calib_v7.py)
is a joint fit to the real board's own anchors -- persistence 1.51,
v5 1.85, v6 0.148716:
* The old proxy spread the 103 new trophies (4,276 - 943 - 3,230 for
  the 100% trophy) across trophies with per-trophy dispersion far too
  narrow. On truth families whose newcomer rates are lognormally
  dispersed with sigma 1.0-1.25 around the proportional median, v6
  scores 0.12-0.15 -- the real board's number -- while on the old
  narrow worlds it scores 0.072-0.079. The real dispersion is sigma
  ~1.25. The allocation SHAPE (beta) barely matters; the dispersion
  does.
* Two published loss figures in the old notes are impossible and were
  dropped from the calibration: v1/v2's 0.168/0.139 cannot be real --
  v1 and v2 are byte-identical on the practice cases and their fan is
  a sigma=1.0 logit normal, whose pinball floor alone is ~0.197.
* The top of the practice board (loss ~0.000001 for ranks 1-3,
  cutoff 0.0203 at rank 20) is not reachable by any honest model: the
  honest cluster sits at 0.08-0.15. Those entries predict the practice
  answers essentially exactly, which only helps the practice board --
  the prize is decided by the FINAL grade (26 Nov), where the moving
  history below is what runs. v7 improves the honest model; it does
  not chase the peekers.

What v7 changes against v6 (each measured on the recalibrated suites,
tools/exp_fan.py and tools/exp_newage.py; the old proxy is kept as a
guard, not as the target, because it is the miscalibrated one):
1. NEW_AGE_H 0.20 -> 0.15: the newcomers' effective play age. A lower
   median wins consistently on every recalibrated sigma (0.5/1.0/1.25/
   1.5) and the optimum is a shallow basin at 0.12-0.15. The old
   proxy moves only +0.0009.
2. The flat-path fan is median-preserving: v6 drew the overdispersed
   newcomer mean as lam * exp(S_ALPHA*z - 0.5*S_ALPHA^2), which keeps
   the MEAN at lam but shifts the fan's MEDIAN down by
   exp(-0.5*S_ALPHA^2). Negligible at S_ALPHA=0.30, but the correction
   is conceptually wrong for a quantile forecast -- the fan's q50 must
   sit at the dilution median -- so v7 drops it.
3. What was tried and REJECTED by measurement, so it is not in v7:
   widening the fan (S_MULT 0.15->0.35, S_ALPHA 0.30->0.80) loses on
   every suite, old and recalibrated; widening only the above-median
   quantiles (x1.5, x2.0 in logit space) loses everywhere too; adding
   the tracked cohort's own growth into the flat median (gamma 0.15/
   0.5/1.0) fits the real anchors WORSE than gamma=0 -- the tracked
   players effectively earn nothing visible inside the horizon, so
   v6's discarded-growth behaviour is correct and kept.

Model (unchanged from v6 apart from the two changes above):
* Regime: FLAT when the history shows no player-count jump and (nearly)
  no rate movement; else MOVING.
* FLAT path -- one refresh lands at the target (the answer IS the
  refreshed page, as in both practice cases). Median in earners/players
  space: the tracked cohort keeps earning at a damped hazard fitted from
  the trophy's own rate and age; the refresh multiplies players once by
  the published first-refresh multiplier (6.064: 638 -> 3,868); the new
  cohort holds each trophy at the rate a player of ~0.15 h effective age
  reaches under that hazard. Spread: a count fan -- new earners
  K ~ Poisson with a lognormally overdispersed mean (newcomers do not all
  earn alike), refresh multiplier m ~ LogNormal, 3000 draws, fixed seed
  per trophy, so the fan is skewed for rare trophies exactly as counts
  are. The 100% trophy keeps a tight fan at the top.
* MOVING path -- an ensemble scored by the case's own rolling-origin
  backtest: (a) a saturating curve on logit(rate) in log-time (grid over
  decay rates, line included as the slow limit), (b) a logit-linear
  trend, (c) a log-linear trend of log(earners) in log-time (immune to
  the refresh sawtooth that bends rates down), (d) refresh
  extrapolation: the median earners and players added per observed
  refresh times the expected refresh count. All curve fits use the
  recent window only (last 24 h): the launch-night dilution crash must
  not anchor a fit read 108 h out. Candidate weights come from
  backtesting each candidate on earlier cuts of the same history; the
  centre is the weight-averaged logit median. Trophies with (nearly) no
  earners yet borrow their tier's pooled curve. Spread: conformal, from
  the same backtest's standardised errors, with the error variance split
  into a horizon-constant and a horizon-growing part. A trophy whose
  rate line slopes down uses the earners trend, never a falling rate
  line.
* Deterministic: every draw comes from a seed fixed per trophy id; two
  runs on the same input are byte-identical. Standard library only, no
  network.

Priors used here come only from the competition's published text (first
refresh 638 -> 3,868 players; 4,276 trophies earned in the answer; a
refresh every few hours) and from backtests on synthetic worlds built
from those numbers -- never from any per-trophy answer, which is sealed.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# --- priors (published text + backtest tuning, see module docstring) ---
GAP_H = 4.5            # hours between Exophase refreshes (prior median)
MULT_FIRST = 6.064     # published first refresh: 638 -> 3,868 players
NEW_AGE_H = 0.15       # effective play age of newly tracked players (h)
                       # v7: 0.20 -> 0.15, measured best on recalibrated
                       # worlds (shallow optimum, basin 0.12-0.15)
GROWTH = 0.15          # dampening on tracked players' continued earning
S_MULT = 0.15          # lognormal spread of the refresh multiplier (fan)
S_ALPHA = 0.30         # lognormal spread of per-trophy newcomer earning
N_MC = 3000            # Monte-Carlo draws per trophy for the flat fan
SD_FLOOR = 0.05        # moving-path fan never narrower than this (logit)
SD_CAP = 1.20          # ... nor wider than this
CURVE_HALF_LIFE_H = 24.0  # curve family's share halves per this many
                          # horizon hours: share = 1/(1+H/CURVE_HALF_LIFE_H)
K_GRID = (0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.35, 2.0, 3.0)
SEED_BASE = 20261121


# ---------------------------------------------------------------- utilities

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


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(dt):
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def log_time(dt):
    """x = ln(1 + hours since release), the curve axis."""
    return math.log1p(hours_since_release(dt))


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residual sd)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        sd = math.sqrt(sum((y - my) ** 2 for y in ys) / n) if n else 1.0
        return my, 0.0, sd
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        sd = math.sqrt(max(var, 0.0))
    else:
        sd = 0.0
    return a, b, sd


def sat_fit(xs, ys):
    """Fit y = A - B*exp(-k*x) with B >= 0 (non-decreasing in x), plus the
    straight-line limit (k -> 0). Best k by least squares. Returns a
    callable value(x) and the residual sd."""
    n = len(xs)
    if n < 3 or all(y == ys[0] for y in ys):
        c = ys[-1] if ys else 0.0
        return (lambda x, c=c: c), 0.0
    best = None  # (sse, fn)
    for k in K_GRID:
        us = [math.exp(-k * x) for x in xs]
        a, c, _ = ols(us, ys)
        b = -c
        if b < 0:
            sse = sum((y - ys[-1]) ** 2 for y in ys)
            fn = lambda x, v=ys[-1]: v
        else:
            sse = sum((y - (a - b * math.exp(-k * x))) ** 2
                      for x, y in zip(xs, ys))
            fn = (lambda kk, aa, bb: (lambda x: aa - bb * math.exp(-kk * x))
                  )(k, a, b)
        if best is None or sse < best[0]:
            best = (sse, fn)
    a, b, _ = ols(xs, ys)
    if b < 0:
        sse = sum((y - ys[-1]) ** 2 for y in ys)
        fn = lambda x, v=ys[-1]: v
    else:
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        fn = (lambda aa, bb: (lambda x: aa + bb * x))(a, b)
    if best is None or sse < best[0]:
        best = (sse, fn)
    sse, fn = best
    sd = math.sqrt(sse / max(n - 2, 1))
    return fn, sd


def poisson_sample(rng, lam):
    """One Poisson(lam) draw, deterministic under rng."""
    if lam <= 0:
        return 0
    if lam < 10.0:
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            p *= rng.random()
            if p <= limit:
                return k
            k += 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


# ---------------------------------------------------------------- case data

def read_case(path):
    """Parse a case file into per-trophy series.

    Returns (case, series) where series[i] is a list of
    (time, x, rate, earners, players), oldest first, for trophy i.
    """
    case = json.loads(path.read_text(encoding="utf-8"))
    hist = sorted(case.get("history", []), key=lambda r: r["at"])
    n_tr = len(case["trophies"])
    series = [[] for _ in range(n_tr)]
    for read in hist:
        t = parse_time(read["at"])
        x = log_time(t)
        n = read.get("players_tracked") or 0
        rates = read.get("rates") or []
        for i in range(n_tr):
            if i < len(rates) and rates[i] is not None:
                r = float(rates[i])
                series[i].append((t, x, r, r * n / 100.0, float(n)))
    return case, hist, series


def detect_refreshes(hist):
    """Refresh events: (gap_hours, multiplier, read_index) per jump in
    players_tracked between consecutive reads."""
    events = []
    prev = None
    last_refresh_t = parse_time(hist[0]["at"]) if hist else None
    for idx, read in enumerate(hist):
        if prev is not None:
            n0 = prev.get("players_tracked") or 0
            n1 = read.get("players_tracked") or 0
            if n0 > 0 and n1 > n0 * 1.005:
                gap = (parse_time(read["at"]) - last_refresh_t
                       ).total_seconds() / 3600.0
                events.append((max(gap, 0.25), n1 / n0, idx))
                last_refresh_t = parse_time(read["at"])
        prev = read
    return events


def history_is_flat(hist, series):
    """True when nothing moved: no player-count jump and the 75th
    percentile trophy's logit span over the history is under 0.05."""
    if not hist:
        return True
    n0 = hist[0].get("players_tracked") or 0
    for read in hist:
        if (read.get("players_tracked") or 0) > n0 * 1.005:
            return False
    spans = []
    for s in series:
        vals = [logit_pct(r) for (_, _, r, _, _) in s]
        if vals:
            spans.append(max(vals) - min(vals))
    if not spans:
        return True
    spans.sort()
    return spans[int(0.75 * (len(spans) - 1))] < 0.05


# ---------------------------------------------------------------- flat path

def flat_median(rate0, players0, age_h, horizon_h):
    """One refresh lands at the target. Returns
    (earners_at_target, players_at_target, new_earners, multiplier).

    Tracked players keep earning at a damped hazard fitted from the
    trophy's own rate and age; the refresh multiplies players once by
    MULT_FIRST; the new cohort holds the trophy at the rate a player of
    NEW_AGE_H hours reaches under that hazard.
    """
    earners = rate0 * players0 / 100.0
    players = float(players0)
    if rate0 >= 99.9:
        return players, players, 0.0, 1.0
    hazard = -math.log(max(1.0 - rate0 / 100.0, 1e-6)) / max(age_h, 0.5)
    newcomer_rate = 100.0 * (1.0 - math.exp(-hazard * NEW_AGE_H))
    if horizon_h < 1.0:
        # No refresh inside a sub-hour horizon: only tracked growth.
        earners += (players - earners) * (
            1.0 - math.exp(-hazard * horizon_h)) * GROWTH
        return earners, players, 0.0, 1.0
    mult = MULT_FIRST
    new_players = players * mult
    added = new_players - players
    earners += (players - earners) * (
        1.0 - math.exp(-hazard * horizon_h)) * GROWTH
    new_earners = added * newcomer_rate / 100.0
    earners += new_earners
    earners = min(earners, new_players)
    return earners, new_players, new_earners, mult


def count_fan(earners0, players0, new_earners, mult, seed):
    """Quantiles of (E0 + K) / (N0 * m): K ~ Poisson with a lognormally
    overdispersed mean (newcomers earn unevenly), m ~ LogNormal.

    v7: the overdispersion is MEDIAN-preserving -- lam is drawn as
    lam_bar * exp(S_ALPHA * z) with no -0.5*sigma^2 term, so the fan's
    q50 sits exactly at the dilution median. v6's mean-preserving
    correction shifted q50 down by exp(-0.5*S_ALPHA^2)."""
    rng = random.Random(seed)
    lam_bar = max(new_earners, 0.0)
    mu = math.log(max(mult, 1.0))
    samples = []
    for _ in range(N_MC):
        lam = lam_bar * math.exp(S_ALPHA * rng.gauss(0.0, 1.0))
        k = poisson_sample(rng, lam)
        m = math.exp(mu + S_MULT * rng.gauss(0.0, 1.0))
        rate = 100.0 * (earners0 + k) / (players0 * m)
        samples.append(min(max(rate, 0.01), 99.99))
    samples.sort()
    out = []
    for level in LEVELS:
        out.append(round(samples[min(int(level * N_MC), N_MC - 1)], 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


# ---------------------------------------------------------------- moving path

def candidate_centres(pts_rate, pts_earn, target_x, earners0, players_t):
    """Candidate logit medians at the target from the fits available.

    The points passed in must already be windowed to the recent regime
    (last 24 h before the checkpoint): the launch-night regime -- a
    dilution crash followed by recovery -- must not anchor a fit that is
    read 108 h out. Returns a dict: 'sat' (saturating curve on logit
    rate), 'line' (logit-linear trend), 'earn' (log-linear trend of log
    earners, converted to a rate), each possibly None when unusable.
    """
    out = {}
    xs = [x for (x, _) in pts_rate]
    ys = [logit_pct(y) for (_, y) in pts_rate]
    if len(xs) >= 4 and len({round(y, 4) for y in ys}) >= 3:
        fn, _ = sat_fit(xs, ys)
        out["sat"] = fn(target_x)
        a, b, _ = ols(xs, ys)
        if abs(b) > 1e-6:
            out["line"] = a + b * target_x
    xe = [x for (x, _) in pts_earn]
    ye = [math.log(max(e, 0.5)) for (_, e) in pts_earn]
    if len(xe) >= 4 and len({round(y, 4) for y in ye}) >= 3:
        a, b, _ = ols(xe, ye)
        if b > 1e-6:
            e_t = math.exp(a + b * target_x)
            e_t = min(max(e_t, earners0), max(players_t, earners0))
            out["earn"] = logit_pct(100.0 * e_t / players_t)
    return out


def backtest_weights(series, target_dt, horizon_h, gap_h, events, hist):
    """Rolling-origin backtest of the candidate families on this trophy's
    own history. Returns (weights dict, list of (horizon_h, std error)).

    Cuts at 60% and 80% of the reads; each candidate is asked to predict
    the read 6/12/24 h after the cut (when one exists) and scored by
    |logit error| against the SMOOTHED rate (mean logit over reads within
    +-3 h), which removes the refresh-cycle lag a single frozen read
    carries. The 'dilution' candidate (repeat the last rate, shifted by
    the refresh extrapolation) is scored too, so the ensemble can fall
    back to it when the curves mislead.
    """
    n = len(series)
    if n < 12:
        return None, []
    cuts = sorted({max(4, int(n * f)) for f in (0.6, 0.8)})
    errs = {}  # candidate name -> [(h, |logit err|)]
    std_errs = []  # (h, signed std err of the blended forecast)
    for cut in cuts:
        cut_t = series[cut - 1][0]
        for h in (6.0, 12.0, 24.0):
            fut = None
            for row in series[cut:]:
                if (row[0] - cut_t).total_seconds() >= h * 3600.0:
                    fut = row
                    break
            if fut is None:
                continue
            window = [r for r in series
                      if abs((r[0] - fut[0]).total_seconds()) <= 3 * 3600.0]
            if len(window) < 3:
                continue
            truth = sum(logit_pct(r[2]) for r in window) / len(window)
            sub = series[:cut]
            # Window the fits to the last 24 h before the cut: the
            # launch-night regime must not anchor a fit read far out.
            cut24 = log_time(cut_t - timedelta(hours=24.0))
            pts_r = [(x, y) for (_, x, y, _, _) in sub if x >= cut24]
            pts_e = [(x, e) for (_, x, _, e, _) in sub if x >= cut24]
            last_rate = sub[-1][2]
            last_n = sub[-1][4]
            # Dilution candidate at this cut: extrapolate the per-refresh
            # deltas actually observed before the cut (median earners and
            # players added per refresh) times the expected refresh count.
            # Falls back to the flat hazard-age median when no refresh
            # was seen before the cut.
            n_ref = max(1, int(round(h / gap_h))) if h >= 1.0 else 0
            e0 = last_rate * last_n / 100.0
            cut_events = [ev for ev in events if ev[2] < cut]
            if cut_events:
                d_es, d_ns = [], []
                for (_, _, j) in cut_events[-3:]:
                    t_prev = parse_time(hist[j - 1]["at"])
                    t_now = parse_time(hist[j]["at"])
                    e_prev = next((r[3] for r in series if r[0] == t_prev),
                                  None)
                    e_now = next((r[3] for r in series if r[0] == t_now),
                                 None)
                    if e_prev is not None and e_now is not None:
                        d_es.append(e_now - e_prev)
                    d_ns.append((hist[j].get("players_tracked") or 0)
                                - (hist[j - 1].get("players_tracked") or 0))
                if d_es and d_ns:
                    e_dil = e0 + n_ref * median(d_es)
                    n_dil = last_n + n_ref * median(d_ns)
                    n_dil = max(n_dil, last_n)
                    e_dil = min(max(e_dil, e0), n_dil)
                else:
                    e_dil, n_dil, _, _ = flat_median(
                        last_rate, last_n,
                        max(hours_since_release(cut_t), 0.5), h)
            else:
                e_dil, n_dil, _, _ = flat_median(
                    last_rate, last_n,
                    max(hours_since_release(cut_t), 0.5), h)
            cands = candidate_centres(pts_r, pts_e, log_time(fut[0]),
                                      e0, n_dil)
            cands["dil"] = logit_pct(100.0 * min(e_dil, n_dil) / n_dil)
            for name, centre in cands.items():
                errs.setdefault(name, []).append((h, abs(truth - centre)))
            # blended error (equal weights here; the final weights are
            # fitted from these same errors, so this is slightly
            # optimistic -- the fan floor covers the gap).
            vals = [c for c in cands.values()]
            blend = sum(vals) / len(vals)
            # scale of the blend error at this horizon, for the fan
            sd_h = math.sqrt(sum((truth - c) ** 2 for c in vals)
                             / len(vals))
            std_errs.append((h, (truth - blend) / max(sd_h, 1e-6)))
    if not any(errs.values()):
        return None, []
    mean_err = {}
    for name, lst in errs.items():
        mean_err[name] = sum(e for (_, e) in lst) / len(lst) if lst else 9.9
    # Softmax weights on inverse error (temperature 0.15 logit).
    inv = {k: math.exp(-v / 0.15) for k, v in mean_err.items()}
    tot = sum(inv.values())
    weights = {k: v / tot for k, v in inv.items()}
    return weights, std_errs


def moving_fan(std_errs, horizon_h):
    """Conformal fan from backtest standardised errors: fit
    var(h) = c + d*h on the per-horizon robust scales, then take the
    empirical quantiles of the standardised errors times the target
    scale. Falls back to a horizon-scaled normal fan with no backtest."""
    if len(std_errs) < 12:
        sd = min(SD_CAP, max(0.25 * math.sqrt(horizon_h / 24.0), SD_FLOOR))
        return list(Z), sd
    by_h = {}
    for h, e in std_errs:
        by_h.setdefault(h, []).append(e)
    points = []
    for h, es in sorted(by_h.items()):
        es = sorted(es)
        m = len(es)
        q1 = es[int(0.25 * (m - 1))]
        q3 = es[int(0.75 * (m - 1))]
        points.append((h, max((q3 - q1) / 1.349, 1e-6) ** 2))
    xs = [h for h, _ in points]
    ys = [v for _, v in points]
    c, d, _ = ols(xs, ys)
    if d < 0:
        d = 0.0
        c = sum(ys) / len(ys)
    if c < 0:
        c = 0.0
    sd_target = min(SD_CAP, max(math.sqrt(c + d * horizon_h), SD_FLOOR))
    zs = sorted(e for (_, e) in std_errs)
    m = len(zs)
    quants = [zs[min(m - 1, max(0, int(tau * (m - 1))))] for tau in LEVELS]
    return quants, sd_target


# ---------------------------------------------------------------- per-trophy

def forecast_trophy(trophy, s, last_rate, last_players, last_time,
                    horizon_h, target_x, cut_x, gap_h, events, hist, pool):
    """One trophy's five quantiles. `pool` is the tier-pooled series
    (same shape as `s`) for borrowing strength when earners are ~0."""
    tid = trophy["id"]
    if last_rate is None:
        last_rate = 1.0
    if last_rate >= 99.9:
        return {"id": tid, "q": [99.0, 99.5, 99.9, 100.0, 100.0]}
    earners0 = last_rate * last_players / 100.0
    age_h = max(hours_since_release(last_time), 0.5)

    # --- dilution median (both paths start here) ---
    if events:
        d_es, d_ns = [], []
        for (_, _, j) in events[-3:]:
            r_prev = hist[j - 1].get("rates") or []
            r_now = hist[j].get("rates") or []
            n_prev = hist[j - 1].get("players_tracked") or 0
            n_now = hist[j].get("players_tracked") or 0
            idx = tid - 1
            if idx < len(r_prev) and idx < len(r_now) \
                    and r_prev[idx] is not None and r_now[idx] is not None:
                d_es.append(r_now[idx] * n_now / 100.0
                            - r_prev[idx] * n_prev / 100.0)
                d_ns.append(n_now - n_prev)
        if d_es:
            steps = max(1, int(round(horizon_h / gap_h))) \
                if horizon_h >= 1.0 else 0
            earners_t = earners0 + steps * median(d_es)
            players_t = float(last_players) + steps * median(d_ns)
            players_t = max(players_t, last_players)
            earners_t = min(max(earners_t, earners0), players_t)
            mult = players_t / last_players if last_players else 1.0
        else:
            earners_t, players_t, _, mult = flat_median(
                last_rate, last_players, age_h, horizon_h)
    else:
        earners_t, players_t, _, mult = flat_median(
            last_rate, last_players, age_h, horizon_h)
    dilution_rate = 100.0 * earners_t / players_t if players_t > 0 \
        else last_rate

    # --- moving history: backtest-weighted ensemble ---
    # Fits use the recent window only (last 24 h): the launch-night
    # dilution crash must not anchor a fit read 108 h out.
    pts_r = [(x, y) for (_, x, y, _, _) in s if x >= cut_x]
    pts_e = [(x, e) for (_, x, _, e, _) in s if x >= cut_x]
    if len(pts_r) < 4:
        pts_r = [(x, y) for (_, x, y, _, _) in s]
        pts_e = [(x, e) for (_, x, _, e, _) in s]
    cands = candidate_centres(pts_r, pts_e, target_x, earners0, players_t)
    weights, std_errs = backtest_weights(s, last_time, horizon_h, gap_h,
                                         events, hist)
    centres = dict(cands)
    centres["dil"] = logit_pct(dilution_rate)
    # Horizon prudence: the backtest validates candidates at 6-24 h, but
    # the target can be 108 h out, where a curve fit's extrapolation is
    # high-variance. Cap the curve family's total share at
    # 1/(1 + horizon in days) (the dilution share takes the rest), and
    # split that share among the curve candidates by their backtest
    # weights (equal split when there is no backtest). At 108 h the
    # curves get 18%; at 10 h they get 70%.
    curve_keys = [k for k in centres if k != "dil"]
    if curve_keys and "dil" in centres:
        if weights:
            cw = {k: weights.get(k, 0.0) for k in curve_keys}
            tot = sum(cw.values())
            if tot > 0:
                cw = {k: v / tot for k, v in cw.items()}
            else:
                cw = {k: 1.0 / len(curve_keys) for k in curve_keys}
        else:
            cw = {k: 1.0 / len(curve_keys) for k in curve_keys}
        share = 1.0 / (1.0 + horizon_h / CURVE_HALF_LIFE_H)
        centre = ((1.0 - share) * centres["dil"]
                  + share * sum(cw[k] * centres[k] for k in curve_keys))
    else:
        centre = centres["dil"]

    # Trophies with (nearly) no earners yet: borrow the tier pool. The
    # pool's own ensemble centre (same machinery, on the pooled series)
    # pulls the median up from a zero that the horizon will not keep.
    if earners0 < 2.0 and pool is not None:
        pts_pr = [(x, y) for (_, x, y, _, _) in pool if x >= cut_x]
        pts_pe = [(x, e) for (_, x, _, e, _) in pool if x >= cut_x]
        if len(pts_pr) < 4:
            pts_pr = [(x, y) for (_, x, y, _, _) in pool]
            pts_pe = [(x, e) for (_, x, _, e, _) in pool]
        pcands = candidate_centres(pts_pr, pts_pe, target_x,
                                   pool[-1][3] if pool else earners0,
                                   players_t)
        pvals = [v for v in pcands.values()]
        if pvals:
            pool_centre = sum(pvals) / len(pvals)
            w_pool = 0.7 if earners0 < 0.5 else 0.7 * (2.0 - earners0) / 1.5
            centre = (1.0 - w_pool) * centre + w_pool * pool_centre

    quants, sd = moving_fan(std_errs, horizon_h)
    qs = []
    prev = 0.0
    for zq in quants:
        v = round(clamp_pct(inv_logit_pct(centre + zq * sd)), 4)
        v = max(v, prev)
        qs.append(v)
        prev = v
    return {"id": tid, "q": qs}


# ---------------------------------------------------------------- case driver

def forecast_case(case, hist, series):
    if not hist:
        return {"trophies": [
            {"id": tr["id"], "q": [0.05, 0.25, 1.0, 4.0, 20.0]}
            for tr in case["trophies"]]}
    last = hist[-1]
    last_time = parse_time(last["at"])
    last_players = last.get("players_tracked") or 0
    horizon_h = case["horizon_minutes"] / 60.0
    target = parse_time(case["target"])
    target_x = log_time(target)
    at = parse_time(case["at"])
    at_hours = (at - RELEASE).total_seconds() / 3600.0
    cut_x = math.log(1.0 + max(0.0, at_hours - 24.0))
    events = detect_refreshes(hist)
    gap_h = median([g for (g, _, _) in events]) if events else GAP_H
    flat = history_is_flat(hist, series)

    # Tier-pooled series (total earners / total players per read), for
    # borrowing strength on near-zero trophies.
    tiers = {}
    for i, tr in enumerate(case["trophies"]):
        tiers.setdefault(tr["tier"], []).append(i)
    pools = {}
    for tier, idxs in tiers.items():
        pool = []
        for read in hist:
            n = read.get("players_tracked") or 0
            rates = read.get("rates") or []
            tot, m = 0.0, 0
            for i in idxs:
                if i < len(rates) and rates[i] is not None:
                    tot += rates[i] * n / 100.0
                    m += 1
            if m == 0 or n == 0:
                continue
            t = parse_time(read["at"])
            pool.append((t, log_time(t), 100.0 * tot / (m * n),
                         tot, float(n)))
        pools[tier] = pool

    rows = []
    for i, tr in enumerate(case["trophies"]):
        s = series[i]
        last_rate = s[-1][2] if s else None
        # A flat history takes the flat path for every trophy, even if a
        # refresh jump is (wrongly) suspected from a rounding wobble.
        if flat:
            e_t, p_t, new_e, mult = flat_median(
                last_rate if last_rate is not None else 1.0,
                last_players, max(hours_since_release(last_time), 0.5),
                horizon_h)
            if (last_rate is not None) and last_rate >= 99.9:
                rows.append({"id": tr["id"],
                             "q": [99.0, 99.5, 99.9, 100.0, 100.0]})
                continue
            earners0 = (last_rate or 1.0) * last_players / 100.0
            seed = SEED_BASE * 7919 + tr["id"] * 104729
            rows.append({"id": tr["id"],
                         "q": count_fan(earners0, last_players,
                                        max(new_e, 0.0), mult, seed)})
            continue
        rows.append(forecast_trophy(
            tr, s, last_rate, last_players, last_time, horizon_h,
            target_x, cut_x, gap_h, events, hist, pools.get(tr["tier"])))
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        raise SystemExit("no .in files in %s" % INPUT_DIR)
    for path in inputs:
        case, hist, series = read_case(path)
        forecast = forecast_case(case, hist, series)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
