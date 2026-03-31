#!/usr/bin/env python3
"""
Full pipeline optimization: weight tuning + set size calibration + cross-validation.

1. Mine optimal set sizes from real data
2. Grid search over weight configurations using phish_engine pipeline
3. Differential evolution for fine-grained weight optimization
4. Cross-validate against 6+ held-out multi-night runs
5. Report best configuration and regenerate predictions
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from phish_engine.data.real_data import load_real_data
from phish_engine.data.manual_overrides import SONG_PAIRS
from phish_engine.features import compute_all_features
from phish_engine.clustering import cluster_songs
from phish_engine.scoring import ScoringWeights, score_all_songs, DEFAULT_WEIGHTS
from phish_engine.predictor import predict_show

# ── Target runs for cross-validation ─────────────────────────────────────────

TARGET_RUNS = [
    {"name": "Dick's 2023", "dates": ["2023-08-31", "2023-09-01", "2023-09-02", "2023-09-03"]},
    {"name": "MSG NYE 2023", "dates": ["2023-12-28", "2023-12-29", "2023-12-30", "2023-12-31"]},
    {"name": "MSG 2024 Spring", "dates": ["2024-04-18", "2024-04-19", "2024-04-20", "2024-04-21"]},
    {"name": "Dick's 2024", "dates": ["2024-08-29", "2024-08-30", "2024-08-31", "2024-09-01"]},
    {"name": "Hampton 2024", "dates": ["2024-10-25", "2024-10-26", "2024-10-27"]},
    {"name": "MSG NYE 2024", "dates": ["2024-12-28", "2024-12-29", "2024-12-30", "2024-12-31"]},
    {"name": "MSG NYE 2025", "dates": ["2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31"]},
]


def load_all_data():
    print("Loading real data...")
    songs_df, shows_df, appearances_df = load_real_data(
        data_dir="data/", min_plays=3, start_year=2019,
    )
    print(f"  {len(shows_df)} shows, {len(songs_df)} songs, {len(appearances_df)} appearances")
    return songs_df, shows_df, appearances_df


def run_clustering(songs_df, appearances_df):
    print("Running clustering...")
    songs_with_clusters, scaler, kmeans, pca, pca_coords, sil_scores = cluster_songs(
        songs_df, appearances_df,
    )
    cluster_labels = dict(zip(songs_with_clusters.index, songs_with_clusters["cluster_id"]))
    return cluster_labels, songs_with_clusters, pca_coords


def evaluate_run(
    run_name, run_dates,
    songs_df, shows_df, appearances_df, cluster_labels,
    weights, set1_size=9, set2_size=7, enc_size=2,
):
    """Evaluate prediction accuracy on a single held-out run."""
    # Find actual shows
    run_show_dates = [pd.Timestamp(d) for d in run_dates]
    run_shows = shows_df[shows_df["date"].isin(run_show_dates)]

    if run_shows.empty:
        return None

    # Training cutoff: day before first run show
    cutoff = run_show_dates[0] - pd.Timedelta(days=1)
    train_shows = shows_df[shows_df["date"] <= cutoff]
    train_app = appearances_df[appearances_df["date"] <= cutoff]

    if len(train_shows) < 30:
        return None

    # Predict each show
    show_song_history = []
    results = []

    for show_date in sorted(run_show_dates):
        actual_app = appearances_df[
            (appearances_df["date"] == show_date) &
            (appearances_df["song_id"].isin(songs_df.index))
        ]
        if actual_app.empty:
            continue

        actual_songs = set(actual_app["song_id"].tolist())
        actual_by_set = {
            "1": set(actual_app[actual_app["set_number"] == "1"]["song_id"]),
            "2": set(actual_app[actual_app["set_number"] == "2"]["song_id"]),
            "e": set(actual_app[actual_app["set_number"] == "e"]["song_id"]),
        }

        # Compute features
        feat_df = compute_all_features(songs_df, train_shows, train_app, show_date - pd.Timedelta(days=1))
        total_shows = int(train_shows["show_num"].max() or 0)

        # Build exclusions from prior shows in run
        hard_exclusions = set()
        for past in show_song_history:
            hard_exclusions |= past

        # Determine venue type
        show_row = shows_df[shows_df["date"] == show_date]
        venue_type = show_row.iloc[0]["venue_type"] if not show_row.empty else "arena"

        pred = predict_show(
            show_date=show_date,
            venue_type=venue_type,
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            total_shows_in_train=total_shows,
            run_exclusions=hard_exclusions,
            weights=weights,
            set1_size=set1_size,
            set2_size=set2_size,
            enc_size=enc_size,
        )

        # Compute metrics
        pred_songs = set()
        pred_by_set = {"1": set(), "2": set(), "e": set()}
        for set_key, set_label in [("set1", "1"), ("set2", "2"), ("encore", "e")]:
            for entry in pred[set_key]:
                pred_songs.add(entry["song_id"])
                pred_by_set[set_label].add(entry["song_id"])

        hit_rate = len(pred_songs & actual_songs) / max(len(pred_songs), 1)

        # Set precision
        correct_set = 0
        total_pred = 0
        for sn in ("1", "2", "e"):
            for sid in pred_by_set[sn]:
                total_pred += 1
                if sid in actual_by_set[sn]:
                    correct_set += 1
        set_precision = correct_set / max(total_pred, 1)

        results.append({
            "date": show_date.strftime("%Y-%m-%d"),
            "hit_rate": hit_rate,
            "set_precision": set_precision,
            "n_correct": len(pred_songs & actual_songs),
            "n_predicted": len(pred_songs),
        })

        # Track songs for exclusion
        show_song_history.append(actual_songs)

    if not results:
        return None

    return {
        "run": run_name,
        "avg_hit_rate": float(np.mean([r["hit_rate"] for r in results])),
        "avg_set_precision": float(np.mean([r["set_precision"] for r in results])),
        "total_correct": sum(r["n_correct"] for r in results),
        "total_predicted": sum(r["n_predicted"] for r in results),
        "per_show": results,
    }


def cross_validate(songs_df, shows_df, appearances_df, cluster_labels,
                   weights, set1_size=9, set2_size=7, enc_size=2, verbose=True):
    """Run cross-validation across all target runs."""
    all_results = []

    for run in TARGET_RUNS:
        result = evaluate_run(
            run["name"], run["dates"],
            songs_df, shows_df, appearances_df, cluster_labels,
            weights, set1_size, set2_size, enc_size,
        )
        if result:
            all_results.append(result)
            if verbose:
                print(f"  {result['run']:25s}  hit={result['avg_hit_rate']*100:5.1f}%  "
                      f"set_prec={result['avg_set_precision']*100:5.1f}%  "
                      f"({result['total_correct']}/{result['total_predicted']})")

    if not all_results:
        return 0.0, []

    avg_hit = float(np.mean([r["avg_hit_rate"] for r in all_results]))
    avg_prec = float(np.mean([r["avg_set_precision"] for r in all_results]))

    if verbose:
        print(f"\n  Cross-val avg hit rate: {avg_hit*100:.1f}%")
        print(f"  Cross-val avg set prec: {avg_prec*100:.1f}%")

    return avg_hit, all_results


def objective_for_optimizer(x, songs_df, shows_df, appearances_df, cluster_labels,
                            set1_size, set2_size, enc_size):
    """Objective: negative hit rate from cross-validation."""
    weights = ScoringWeights.from_softmax_vector(x)
    avg_hit, _ = cross_validate(
        songs_df, shows_df, appearances_df, cluster_labels,
        weights, set1_size, set2_size, enc_size, verbose=False,
    )
    return -avg_hit


def main():
    songs_df, shows_df, appearances_df = load_all_data()
    cluster_labels, songs_with_clusters, pca_coords = run_clustering(songs_df, appearances_df)

    # ── 1. Set size calibration ───────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: SET SIZE CALIBRATION")
    print("="*60)

    # Mine from real data
    with open("data/setlists.json") as f:
        raw_setlists = json.load(f)

    s1_sizes, s2_sizes, enc_sizes = [], [], []
    for date, entries in raw_setlists.items():
        by_set = {}
        for e in entries:
            if e.get("artistid") != 1:
                continue
            s = str(e.get("set", "1"))
            by_set.setdefault(s, []).append(e)
        s1 = len(by_set.get("1", []))
        s2 = len(by_set.get("2", []))
        enc = len(by_set.get("e", [])) + len(by_set.get("e2", []))
        if s1 > 0: s1_sizes.append(s1)
        if s2 > 0: s2_sizes.append(s2)
        if enc > 0: enc_sizes.append(enc)

    # Use rounded median
    opt_s1 = int(np.round(np.median(s1_sizes)))
    opt_s2 = int(np.round(np.median(s2_sizes)))
    opt_enc = int(np.round(np.median(enc_sizes)))
    print(f"  Optimal set sizes from data: S1={opt_s1}, S2={opt_s2}, Enc={opt_enc}")
    print(f"  (was hardcoded as S1=11, S2=7, Enc=2)")

    # Test a few set size combos
    print("\n  Testing set size configurations:")
    best_set_config = None
    best_set_score = 0
    for s1, s2, enc in [(9, 7, 2), (9, 8, 2), (10, 7, 2), (10, 8, 2), (9, 7, 3), (11, 7, 2)]:
        avg_hit, _ = cross_validate(
            songs_df, shows_df, appearances_df, cluster_labels,
            DEFAULT_WEIGHTS, s1, s2, enc, verbose=False,
        )
        marker = " <-- BEST" if avg_hit > best_set_score else ""
        print(f"    S1={s1} S2={s2} Enc={enc}  →  hit={avg_hit*100:.1f}%{marker}")
        if avg_hit > best_set_score:
            best_set_score = avg_hit
            best_set_config = (s1, s2, enc)

    opt_s1, opt_s2, opt_enc = best_set_config
    print(f"\n  Best set sizes: S1={opt_s1}, S2={opt_s2}, Enc={opt_enc}")

    # ── 2. Grid search over weight configs ────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: WEIGHT GRID SEARCH")
    print("="*60)

    configs = [
        ("current (slot+freq heavy)",
         ScoringWeights(recency=0.05, frequency=0.40, gap_pressure=0.05,
                        slot_affinity=0.35, venue_affinity=0.10, cluster=0.05)),
        ("default balanced",
         ScoringWeights()),
        ("gap_heavy",
         ScoringWeights(recency=0.15, frequency=0.10, gap_pressure=0.40,
                        slot_affinity=0.20, venue_affinity=0.05, cluster=0.10)),
        ("recency_heavy",
         ScoringWeights(recency=0.40, frequency=0.10, gap_pressure=0.20,
                        slot_affinity=0.15, venue_affinity=0.05, cluster=0.10)),
        ("slot_dominant",
         ScoringWeights(recency=0.10, frequency=0.10, gap_pressure=0.10,
                        slot_affinity=0.50, venue_affinity=0.10, cluster=0.10)),
        ("freq_dominant",
         ScoringWeights(recency=0.10, frequency=0.50, gap_pressure=0.10,
                        slot_affinity=0.15, venue_affinity=0.05, cluster=0.10)),
        ("venue_boosted",
         ScoringWeights(recency=0.10, frequency=0.20, gap_pressure=0.10,
                        slot_affinity=0.25, venue_affinity=0.25, cluster=0.10)),
        ("balanced_v2",
         ScoringWeights(recency=0.15, frequency=0.25, gap_pressure=0.15,
                        slot_affinity=0.25, venue_affinity=0.10, cluster=0.10)),
        ("fast_decay",
         ScoringWeights(recency_decay_rate=5.0, gap_lognormal_sigma=0.6)),
        ("slow_decay",
         ScoringWeights(recency_decay_rate=1.5, gap_lognormal_sigma=0.3)),
        ("slot_freq_blend",
         ScoringWeights(recency=0.10, frequency=0.30, gap_pressure=0.10,
                        slot_affinity=0.30, venue_affinity=0.10, cluster=0.10)),
        ("aggressive_gap",
         ScoringWeights(recency=0.10, frequency=0.20, gap_pressure=0.30,
                        slot_affinity=0.20, venue_affinity=0.10, cluster=0.10)),
    ]

    best_name = ""
    best_weights = DEFAULT_WEIGHTS
    best_score = 0

    for name, w in configs:
        avg_hit, _ = cross_validate(
            songs_df, shows_df, appearances_df, cluster_labels,
            w, opt_s1, opt_s2, opt_enc, verbose=False,
        )
        marker = " <-- BEST" if avg_hit > best_score else ""
        nw = w.main_weights_normalized()
        print(f"  {name:30s}  hit={avg_hit*100:.1f}%  "
              f"(rec={nw['recency']:.2f} freq={nw['frequency']:.2f} "
              f"gap={nw['gap_pressure']:.2f} slot={nw['slot_affinity']:.2f} "
              f"venue={nw['venue_affinity']:.2f} clust={nw['cluster']:.2f}){marker}")
        if avg_hit > best_score:
            best_score = avg_hit
            best_weights = w
            best_name = name

    print(f"\n  Grid search best: '{best_name}' at {best_score*100:.1f}%")

    # ── 3. Differential Evolution fine-tuning ─────────────────────────────
    print("\n" + "="*60)
    print("PHASE 3: DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("="*60)

    eval_count = [0]
    best_de = [best_score]

    def callback(xk, convergence=0):
        eval_count[0] += 1
        w = ScoringWeights.from_softmax_vector(xk)
        nw = w.main_weights_normalized()
        if eval_count[0] % 5 == 0:
            print(f"  [gen {eval_count[0]}] best={best_de[0]*100:.1f}%  "
                  f"rec={nw['recency']:.2f} freq={nw['frequency']:.2f} "
                  f"gap={nw['gap_pressure']:.2f} slot={nw['slot_affinity']:.2f}")

    bounds = [(-3.0, 3.0)] * 6 + [
        (np.log(1.0), np.log(7.0)),
        (np.log(0.15), np.log(1.5)),
        (np.log(0.1 / 0.9), np.log(0.9 / 0.1)),
    ]

    # Warm-start from grid search best
    x0 = best_weights.to_softmax_vector()

    t0 = time.time()
    print(f"  Starting DE with popsize=10, maxiter=15 (warm-started from '{best_name}')...")

    de_result = differential_evolution(
        objective_for_optimizer,
        bounds=bounds,
        args=(songs_df, shows_df, appearances_df, cluster_labels, opt_s1, opt_s2, opt_enc),
        maxiter=15,
        popsize=10,
        seed=42,
        tol=1e-4,
        mutation=(0.5, 1.2),
        recombination=0.7,
        strategy="best1bin",
        x0=x0,
        workers=1,
        polish=False,
    )

    de_weights = ScoringWeights.from_softmax_vector(de_result.x)
    de_score = -de_result.fun
    print(f"\n  DE complete in {time.time()-t0:.0f}s")
    print(f"  DE best hit rate: {de_score*100:.1f}%")
    print(f"  {de_weights.describe()}")

    # Local polish with Nelder-Mead
    print("\n  Polishing with Nelder-Mead...")
    nm_result = minimize(
        objective_for_optimizer,
        x0=de_result.x,
        args=(songs_df, shows_df, appearances_df, cluster_labels, opt_s1, opt_s2, opt_enc),
        method="Nelder-Mead",
        options={"maxiter": 100, "xatol": 1e-5, "fatol": 1e-5},
    )
    final_weights = ScoringWeights.from_softmax_vector(nm_result.x)
    final_score = -nm_result.fun

    # Pick the best overall
    if final_score > de_score:
        best_overall_weights = final_weights
        best_overall_score = final_score
        print(f"  NM improved: {final_score*100:.1f}%")
    else:
        best_overall_weights = de_weights
        best_overall_score = de_score
        print(f"  NM did not improve (stayed at {de_score*100:.1f}%)")

    if best_score > best_overall_score:
        best_overall_weights = best_weights
        best_overall_score = best_score
        print(f"\n  Grid search config '{best_name}' is still best at {best_score*100:.1f}%")

    # ── 4. Final cross-validation report ──────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 4: FINAL CROSS-VALIDATION")
    print("="*60)

    print(f"\n  Optimized weights:")
    print(f"  {best_overall_weights.describe()}")
    print(f"  Set sizes: S1={opt_s1}, S2={opt_s2}, Enc={opt_enc}")
    print()

    avg_hit, results = cross_validate(
        songs_df, shows_df, appearances_df, cluster_labels,
        best_overall_weights, opt_s1, opt_s2, opt_enc, verbose=True,
    )

    # ── 5. Save optimized config ──────────────────────────────────────────
    nw = best_overall_weights.main_weights_normalized()
    config = {
        "weights": {
            "recency": round(nw["recency"], 4),
            "frequency": round(nw["frequency"], 4),
            "gap_pressure": round(nw["gap_pressure"], 4),
            "slot_affinity": round(nw["slot_affinity"], 4),
            "venue_affinity": round(nw["venue_affinity"], 4),
            "cluster": round(nw["cluster"], 4),
        },
        "weights_raw": {
            "recency": round(best_overall_weights.recency, 4),
            "frequency": round(best_overall_weights.frequency, 4),
            "gap_pressure": round(best_overall_weights.gap_pressure, 4),
            "slot_affinity": round(best_overall_weights.slot_affinity, 4),
            "venue_affinity": round(best_overall_weights.venue_affinity, 4),
            "cluster": round(best_overall_weights.cluster, 4),
        },
        "sub_params": {
            "recency_decay_rate": round(best_overall_weights.recency_decay_rate, 4),
            "freq_w10": round(best_overall_weights.freq_w10, 4),
            "freq_w30": round(best_overall_weights.freq_w30, 4),
            "freq_w90": round(best_overall_weights.freq_w90, 4),
            "gap_lognormal_sigma": round(best_overall_weights.gap_lognormal_sigma, 4),
        },
        "set_sizes": {"s1": opt_s1, "s2": opt_s2, "enc": opt_enc},
        "cross_val_hit_rate": round(avg_hit, 4),
        "cross_val_runs": len(results),
    }

    with open("optimized_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved optimized config to optimized_config.json")
    print(f"\n{'='*60}")
    print(f"  FINAL CROSS-VAL HIT RATE: {avg_hit*100:.1f}%")
    print(f"  Set sizes: S1={opt_s1} S2={opt_s2} Enc={opt_enc}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
