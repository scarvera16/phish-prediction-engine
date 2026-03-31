#!/usr/bin/env python3
"""
Fast targeted weight optimization around the known-good region.
Grid search already showed slot+freq heavy is best at 36%.
This does focused exploration around that neighborhood.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phish_engine.data.real_data import load_real_data
from phish_engine.features import compute_all_features
from phish_engine.clustering import cluster_songs
from phish_engine.scoring import ScoringWeights, score_all_songs
from phish_engine.predictor import predict_show

TARGET_RUNS = [
    {"name": "Dick's 2023", "dates": ["2023-08-31", "2023-09-01", "2023-09-02", "2023-09-03"]},
    {"name": "MSG NYE 2023", "dates": ["2023-12-28", "2023-12-29", "2023-12-30", "2023-12-31"]},
    {"name": "MSG 2024 Spring", "dates": ["2024-04-18", "2024-04-19", "2024-04-20", "2024-04-21"]},
    {"name": "Dick's 2024", "dates": ["2024-08-29", "2024-08-30", "2024-08-31", "2024-09-01"]},
    {"name": "Hampton 2024", "dates": ["2024-10-25", "2024-10-26", "2024-10-27"]},
    {"name": "MSG NYE 2024", "dates": ["2024-12-28", "2024-12-29", "2024-12-30", "2024-12-31"]},
    {"name": "MSG NYE 2025", "dates": ["2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31"]},
]


def evaluate(songs_df, shows_df, appearances_df, cluster_labels, weights, s1=9, s2=7, enc=2):
    """Quick cross-val returning average hit rate."""
    all_hits = []

    for run in TARGET_RUNS:
        run_dates = [pd.Timestamp(d) for d in run["dates"]]
        cutoff = run_dates[0] - pd.Timedelta(days=1)
        train_shows = shows_df[shows_df["date"] <= cutoff]
        train_app = appearances_df[appearances_df["date"] <= cutoff]

        if len(train_shows) < 30:
            continue

        show_history = []
        for show_date in sorted(run_dates):
            actual_app = appearances_df[
                (appearances_df["date"] == show_date) &
                (appearances_df["song_id"].isin(songs_df.index))
            ]
            if actual_app.empty:
                continue

            actual_songs = set(actual_app["song_id"].tolist())
            feat_df = compute_all_features(songs_df, train_shows, train_app, show_date - pd.Timedelta(days=1))
            total = int(train_shows["show_num"].max() or 0)

            hard_excl = set()
            for past in show_history:
                hard_excl |= past

            show_row = shows_df[shows_df["date"] == show_date]
            vtype = show_row.iloc[0]["venue_type"] if not show_row.empty else "arena"

            pred = predict_show(
                show_date=show_date, venue_type=vtype,
                songs_df=songs_df, feat_df=feat_df,
                cluster_labels=cluster_labels,
                total_shows_in_train=total,
                run_exclusions=hard_excl,
                weights=weights,
                set1_size=s1, set2_size=s2, enc_size=enc,
            )

            pred_songs = set()
            for sk in ("set1", "set2", "encore"):
                for e in pred[sk]:
                    pred_songs.add(e["song_id"])

            hit = len(pred_songs & actual_songs) / max(len(pred_songs), 1)
            all_hits.append(hit)
            show_history.append(actual_songs)

    return float(np.mean(all_hits)) if all_hits else 0.0


def main():
    print("Loading data...")
    songs_df, shows_df, appearances_df = load_real_data(data_dir="data/", min_plays=3, start_year=2019)

    print("Clustering...")
    songs_wc, _, _, _, _, _ = cluster_songs(songs_df, appearances_df)
    cluster_labels = dict(zip(songs_wc.index, songs_wc["cluster_id"]))

    print(f"\n{'='*70}")
    print("TARGETED WEIGHT OPTIMIZATION")
    print(f"{'='*70}")

    # Focused grid around the known-good region (slot 0.30-0.45, freq 0.30-0.50)
    configs = []
    for rec in [0.02, 0.05, 0.08, 0.12]:
        for freq in [0.30, 0.35, 0.40, 0.45, 0.50]:
            for slot in [0.25, 0.30, 0.35, 0.40, 0.45]:
                for gap in [0.02, 0.05, 0.10]:
                    for venue in [0.05, 0.10, 0.15]:
                        clust = max(0.02, 1.0 - rec - freq - slot - gap - venue)
                        if clust < 0:
                            continue
                        w = ScoringWeights(
                            recency=rec, frequency=freq, gap_pressure=gap,
                            slot_affinity=slot, venue_affinity=venue, cluster=clust,
                        )
                        configs.append(w)

    # Also add sub-parameter variations on the best main weights
    for decay in [1.5, 3.0, 5.0]:
        for sigma in [0.3, 0.45, 0.7]:
            for w10 in [0.2, 0.45, 0.7]:
                w = ScoringWeights(
                    recency=0.05, frequency=0.40, gap_pressure=0.05,
                    slot_affinity=0.35, venue_affinity=0.10, cluster=0.05,
                    recency_decay_rate=decay, gap_lognormal_sigma=sigma,
                    freq_w10=w10, freq_w30=(1-w10)*0.6, freq_w90=(1-w10)*0.4,
                )
                configs.append(w)

    print(f"Testing {len(configs)} weight configurations...")

    best_score = 0
    best_weights = None
    t0 = time.time()

    for i, w in enumerate(configs):
        score = evaluate(songs_df, shows_df, appearances_df, cluster_labels, w)
        if score > best_score:
            best_score = score
            best_weights = w
            nw = w.main_weights_normalized()
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(configs)}] NEW BEST {score*100:.1f}%  "
                  f"rec={nw['recency']:.2f} freq={nw['frequency']:.2f} "
                  f"gap={nw['gap_pressure']:.2f} slot={nw['slot_affinity']:.2f} "
                  f"venue={nw['venue_affinity']:.2f} clust={nw['cluster']:.2f}  "
                  f"decay={w.recency_decay_rate:.1f} sigma={w.gap_lognormal_sigma:.2f} "
                  f"w10={w.freq_w10:.2f}  [{elapsed:.0f}s]")
        elif (i+1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(configs)}] best so far: {best_score*100:.1f}%  [{elapsed:.0f}s]")

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE ({time.time()-t0:.0f}s)")
    print(f"{'='*70}")
    print(f"Best cross-val hit rate: {best_score*100:.1f}%")
    print(f"{best_weights.describe()}")

    # Detailed per-run results
    print(f"\nPer-run breakdown:")
    for run in TARGET_RUNS:
        run_dates = [pd.Timestamp(d) for d in run["dates"]]
        cutoff = run_dates[0] - pd.Timedelta(days=1)
        train_shows = shows_df[shows_df["date"] <= cutoff]
        train_app = appearances_df[appearances_df["date"] <= cutoff]
        if len(train_shows) < 30:
            continue

        show_history = []
        hits = []
        for show_date in sorted(run_dates):
            actual_app = appearances_df[
                (appearances_df["date"] == show_date) &
                (appearances_df["song_id"].isin(songs_df.index))
            ]
            if actual_app.empty:
                continue
            actual_songs = set(actual_app["song_id"].tolist())
            feat_df = compute_all_features(songs_df, train_shows, train_app, show_date - pd.Timedelta(days=1))
            total = int(train_shows["show_num"].max() or 0)
            hard_excl = set()
            for past in show_history:
                hard_excl |= past
            show_row = shows_df[shows_df["date"] == show_date]
            vtype = show_row.iloc[0]["venue_type"] if not show_row.empty else "arena"
            pred = predict_show(
                show_date=show_date, venue_type=vtype,
                songs_df=songs_df, feat_df=feat_df,
                cluster_labels=cluster_labels,
                total_shows_in_train=total,
                run_exclusions=hard_excl,
                weights=best_weights,
                set1_size=9, set2_size=7, enc_size=2,
            )
            pred_songs = set()
            for sk in ("set1", "set2", "encore"):
                for e in pred[sk]:
                    pred_songs.add(e["song_id"])
            correct = pred_songs & actual_songs
            hit = len(correct) / max(len(pred_songs), 1)
            hits.append(hit)
            show_history.append(actual_songs)

        if hits:
            print(f"  {run['name']:25s}  avg hit={np.mean(hits)*100:.1f}%  "
                  f"({sum(int(h*18) for h in hits)}/{len(hits)*18} songs)")

    # Save config
    nw = best_weights.main_weights_normalized()
    config = {
        "weights": {k: round(v, 4) for k, v in nw.items()},
        "sub_params": {
            "recency_decay_rate": round(best_weights.recency_decay_rate, 4),
            "freq_w10": round(best_weights.freq_w10, 4),
            "freq_w30": round(best_weights.freq_w30, 4),
            "freq_w90": round(best_weights.freq_w90, 4),
            "gap_lognormal_sigma": round(best_weights.gap_lognormal_sigma, 4),
        },
        "set_sizes": {"s1": 9, "s2": 7, "enc": 2},
        "cross_val_hit_rate": round(best_score, 4),
    }
    with open("optimized_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved to optimized_config.json")


if __name__ == "__main__":
    main()
