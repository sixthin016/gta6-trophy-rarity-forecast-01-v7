# gta6-trophy-rarity-forecast-01 — v7

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01-v7/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01-v7/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Forecast 55 GTA VI trophies' rarity (share of players) at the horizon,
5 quantiles per trophy, scored by pinball loss on the logit scale.

Competition: [gta6-trophy-rarity-forecast-01](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
on [make.fun](https://make.fun). $250 prize, top 20
(1st $50, 2nd $35, 3rd $25, 4–10 $10, 11–20 $7).
Practice checkpoints 2026-11-19T03:00/03:30Z, 90 min horizon; final
grade at 2026-11-26T00:00Z.

## Run

```bash
python3 main.py <input.json> <output.json>   # one case
python3 main.py                               # reads <case>.in, writes <case>.out
python3 tools/backtest_v7.py all              # full local grade (suites a–e)
```

Deterministic. ~1–3.5 s per case (cap 8 s). Output 2,772 bytes (cap 65,536).

## Model

- **Flat (short history, ~2 checkpoints):** a dilution median per
  trophy — newcomer earners split between the 1,000+ earning
  trophies, scaled by trophy rate and the trophy's own history —
  plus a lognormal fan (S_MULT=0.15 on rate, S_ALPHA=0.30 on
  dilution). Median-preserving fan: q50 sits exactly at the
  dilution median.
- **Moving (long history, ≥6 checkpoints):** an exponential decay
  of the earners' share plus a growing newcomer stream, fitted by
  least squares on the history; the fan comes from the same
  dilution law.

## Why v7 (vs v6, rank 1035, live loss 0.1487)

Recalibrating the truth against the real leaderboard anchors
(persistence 1.51, v5 1.85, v6 0.1487) shows the old proxy's
newcomer dispersion was far too narrow (sigma ~1.25 in reality).
v7 lowers the newcomer-age prior (NEW_AGE_H 0.20 → 0.15) and makes
the fan median-preserving. Local grade (N_MC=3000):

| suite | v7 | v6 |
|---|---|---|
| (a) practice-proxy (13) | 0.0809 | 0.0794 |
| (b) recalibrated sigma 0.5/1.0/1.25 (18) | **0.1175** | 0.1213 |
| (c) final-like (8) | 0.7972 | 0.7960 |
| (d) synthetic (practice+final+CV) | 0.1785 | 0.1778 |

Expected live practice loss ~0.138 (vs v6 0.1487); the top 20
(0.0203) is unreachable for an honest model — the prize is decided
by the final grade, where the moving path is unchanged from v6.

## Files

- `main.py` — the model (v7: NEW_AGE_H=0.15, median-preserving fan)
- `tools/backtest_v7.py` — local grade (suites a–e)
- `tools/calib_v7.py` — recalibration of the truth family to the
  real leaderboard anchors
- `tools/tune_v7.py`, `tools/exp_fan.py`, `tools/exp_newage.py`,
  `tools/calib_growth.py` — tuning experiments (fan width,
  NEW_AGE_H grid, cohort growth — growth rejected by measurement)
- `NOTES-v7.md` — full diagnosis and design notes
- `LICENSE` — MIT

## License

MIT — see `LICENSE`.
