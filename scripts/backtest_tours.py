#!/usr/bin/env python3
"""
Backtest the predictor against held-out tours and report accuracy + spread.

Trains on everything before each tour, predicts every show with its actual
venue_type, and scores hit rate (predicted song appears anywhere in the show)
and set precision (predicted in the right set). No-repeat is stand-scoped, so
multi-stop summer tours are measured fairly (see validator.run_backtest).

Why this exists: the shipped "~36% hit rate" was only ever validated on the
2025 NYE run — one venue, end of the data, the easiest tour to predict. Real
summer tours (multi-venue, 23-26 shows) land lower. Re-run this before opening
a new tour so the public accuracy number reflects the tour you're launching.

Usage:
  python scripts/backtest_tours.py                       # default recent set
  python scripts/backtest_tours.py "2024 Summer Tour"    # specific tour(s)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from phish_engine.data.real_data import load_real_data
from phish_engine.clustering import cluster_songs
from phish_engine.scoring import ScoringWeights
from phish_engine.backtest.validator import run_backtest

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "phish_engine" / "data" / "cache"

DEFAULT_TOURS = [
    "2022 Summer Tour",
    "2023 Summer Tour",
    "2024 Summer Tour",
    "2025 Early Summer Tour",
    "2026 Sphere",  # single-venue stress test
]


def production_weights() -> ScoringWeights:
    cfg = json.loads((ROOT / "optimized_config.json").read_text())
    w, sp = cfg["weights"], cfg["sub_params"]
    return ScoringWeights(
        recency=w["recency"], gap_pressure=w["gap_pressure"],
        slot_affinity=w["slot_affinity"], frequency=w["frequency"],
        venue_affinity=w["venue_affinity"], cluster=w["cluster"],
        recency_decay_rate=sp["recency_decay_rate"], freq_w10=sp["freq_w10"],
        freq_w30=sp["freq_w30"], freq_w90=sp["freq_w90"],
        gap_lognormal_sigma=sp["gap_lognormal_sigma"],
    )


def main() -> None:
    tours = sys.argv[1:] or DEFAULT_TOURS
    weights = production_weights()
    songs, shows, app = load_real_data(CACHE, min_plays=3, start_year=2019)
    swc, *_ = cluster_songs(songs, app)
    cluster_labels = dict(zip(swc.index, swc["cluster_id"]))

    print(f"\n{'tour':24} {'n':>3} {'hit%':>6} {'setP%':>6} {'min%':>6} {'max%':>6} {'sd':>5}")
    summer_hits, summer_prec = [], []
    for t in tours:
        r = run_backtest(songs, shows, app, cluster_labels,
                         validation_tour=t, weights=weights, verbose=False)
        hr = [m["hit_rate"] for m in r["per_show"]]
        print(f"{t:24} {r['n_shows']:>3} {r['avg_hit_rate']*100:6.1f} "
              f"{r['avg_set_precision']*100:6.1f} {min(hr)*100:6.1f} "
              f"{max(hr)*100:6.1f} {np.std(hr)*100:5.1f}")
        if "Summer" in t:
            summer_hits.append(r["avg_hit_rate"])
            summer_prec.append(r["avg_set_precision"])

    if summer_hits:
        print(f"\nSUMMER avg: hit {np.mean(summer_hits)*100:.1f}%  "
              f"setPrecision {np.mean(summer_prec)*100:.1f}%  "
              f"(across {len(summer_hits)} tours)")


if __name__ == "__main__":
    main()
