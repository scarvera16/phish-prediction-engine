#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║        PHISH SETLIST PREDICTOR — SPHERE LAS VEGAS 2026          ║
║  K-Means clustering + composite gap/recency/slot/venue scoring   ║
╚══════════════════════════════════════════════════════════════════╝

Pipeline
--------
  1. Load mock show history (2019–2025, ~280 shows, realistic gaps)
  2. Extract dynamic features per song (gap, recency, slot fractions)
  3. K-Means clustering (k=8) into behavioural archetypes
  4. Backtest on NYE 2025 run — validate hit-rate & set precision
  5. Predict 8 Sphere shows (Apr 17–May 2, 2026) with run exclusions

Scoring formula (per song × slot):
  score = 0.25·recency + 0.25·gap_pressure + 0.25·slot_affinity
        + 0.10·frequency + 0.10·venue_affinity + 0.05·cluster_diversity

Usage:
    pip install -r requirements.txt
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from collections import Counter
from tabulate import tabulate

from engine.song_pairs import SONG_PAIRS  # get_songs_df moved; import SONG_PAIRS here
from data.mock_data import generate_show_history
from engine.features import compute_all_features, build_cluster_feature_matrix
from engine.clustering import train_song_clusters, get_cluster_members
from engine.predictor import predict_multi_night_run
from backtest.validator import run_backtest

# ── Target Sphere shows ────────────────────────────────────────────────────
SPHERE_DATES = [
    pd.Timestamp("2026-04-16"),  # Thu - Night 1
    pd.Timestamp("2026-04-17"),  # Fri - Night 2
    pd.Timestamp("2026-04-18"),  # Sat - Night 3
    pd.Timestamp("2026-04-23"),  # Thu - Night 4
    pd.Timestamp("2026-04-24"),  # Fri - Night 5
    pd.Timestamp("2026-04-25"),  # Sat - Night 6
    pd.Timestamp("2026-04-30"),  # Thu - Night 7
    pd.Timestamp("2026-05-01"),  # Fri - Night 8
    pd.Timestamp("2026-05-02"),  # Sat - Night 9
]

VALIDATION_TOUR = "NYE 2025"

WEIGHTS = {
    "recency":           0.25,
    "gap_pressure":      0.25,
    "slot_affinity":     0.25,
    "frequency":         0.10,
    "venue_affinity":    0.10,
    "cluster_diversity": 0.05,
}


def banner(text: str, width: int = 68) -> None:
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def section(text: str) -> None:
    print(f"\n▶  {text}")
    print("─" * 60)


def format_setlist(prediction: dict, songs_df: pd.DataFrame) -> str:
    lines = []
    for set_label, key in [("SET 1", "set1"), ("SET 2", "set2"), ("ENCORE", "encore")]:
        lines.append(f"\n  {set_label}")
        for i, entry in enumerate(prediction[key], 1):
            name  = entry["name"]
            score = entry["score"]
            c     = entry["components"]
            detail = (f"rec={c['recency']:.2f} gap={c['gap_pressure']:.2f} "
                      f"slot={c['slot_affinity']:.2f} venue={c['venue_affinity']:.2f}")
            lines.append(f"    {i:2d}. {name:<35} [score:{score:.3f}  {detail}]")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()

    banner("PHISH SETLIST PREDICTOR  ·  Sphere Las Vegas  ·  Apr–May 2026")

    # ── 1. Song catalog ───────────────────────────────────────────────────────
    section("Song Catalog")
    songs_df = get_songs_df()
    print(f"  Loaded {len(songs_df)} songs across {songs_df['style'].nunique()} styles")
    print(f"  Styles: {', '.join(sorted(songs_df['style'].unique()))}")

    # ── 2. Mock show history ──────────────────────────────────────────────────
    section("Generating Mock Show History  (2019–2025)")
    shows_df, appearances_df = generate_show_history(seed=42)
    print(f"  Shows generated    : {len(shows_df)}")
    print(f"  Total appearances  : {len(appearances_df)}")
    print(f"  Date range         : {shows_df['date'].min().date()} → {shows_df['date'].max().date()}")
    print(f"  Tours              : {shows_df['tour'].nunique()}")
    print(f"  Unique songs played: {appearances_df['song_id'].nunique()}")
    for vtype, cnt in shows_df["venue_type"].value_counts().items():
        print(f"    {vtype:<12}: {cnt} shows")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    section("Feature Engineering")
    train_cutoff = shows_df[shows_df["tour"] != VALIDATION_TOUR]["date"].max()
    feat_df = compute_all_features(songs_df, shows_df, appearances_df, train_cutoff)
    overdue = (feat_df["gap_z_score"] > 1.0).sum()
    print(f"  Training cutoff : {train_cutoff.date()}")
    print(f"  Avg observed gap: {feat_df['avg_gap_actual'].mean():.1f} shows")
    print(f"  Songs overdue   : {overdue} / {len(songs_df)} (gap z-score > 1.0)")
    top_overdue = feat_df.nlargest(5, "gap_z_score").index.tolist()
    print(f"  Most overdue    : {', '.join(songs_df.loc[top_overdue, 'name'].tolist())}")

    # ── 4. K-Means clustering ─────────────────────────────────────────────────
    section("K-Means Song Clustering  (k=8)")
    X_scaled, _ = build_cluster_feature_matrix(songs_df, feat_df)
    kmeans, cluster_labels, cluster_names = train_song_clusters(X_scaled, n_clusters=8, seed=42)

    cluster_table = []
    for cid in sorted(set(cluster_labels.values())):
        members = get_cluster_members(cid, cluster_labels, songs_df, top_n=6)
        cluster_table.append([cid, cluster_names.get(cid, f"Cluster {cid}"), ", ".join(members)])
    print(tabulate(cluster_table, headers=["ID", "Archetype", "Top Members (by play freq)"],
                   tablefmt="simple"))

    # ── 5. Backtesting ────────────────────────────────────────────────────────
    section(f"Backtesting on '{VALIDATION_TOUR}'  (held-out validation)")
    backtest_results = run_backtest(
        songs_df=songs_df,
        shows_df=shows_df,
        appearances_df=appearances_df,
        cluster_labels=cluster_labels,
        validation_tour=VALIDATION_TOUR,
        weights=WEIGHTS,
        verbose=True,
    )

    print()
    print("  Scoring weight configuration:")
    for k, v in WEIGHTS.items():
        bar = "█" * int(v * 50)
        print(f"    {k:<20} {v:.2f}  {bar}")

    # ── 6. Sphere 2026 predictions ────────────────────────────────────────────
    banner("SPHERE LAS VEGAS  ·  9 PREDICTED SHOWS  ·  APRIL–MAY 2026")
    print(f"\n  Venue type : Sphere (immersive A/V)")
    print(f"  Dates      : {SPHERE_DATES[0].date()} → {SPHERE_DATES[-1].date()}")
    print(f"  Note       : Run exclusions accumulate across all 9 nights\n")

    predictions = predict_multi_night_run(
        show_dates=SPHERE_DATES,
        venue_type="sphere",
        songs_df=songs_df,
        shows_df=shows_df,
        appearances_df=appearances_df,
        cluster_labels=cluster_labels,
        weights=WEIGHTS,
    )

    run_songs: set[str] = set()
    for i, pred in enumerate(predictions):
        d = pred["date"]
        day_name = d.strftime("%A, %B %-d, %Y")
        print(f"\n{'─'*68}")
        print(f"  SHOW {i+1} of 9  ·  {day_name}")
        print(f"{'─'*68}")
        print(format_setlist(pred, songs_df))
        for slot_key in ("set1", "set2", "encore"):
            for entry in pred[slot_key]:
                run_songs.add(entry["song_id"])

    # ── 7. Run summary ────────────────────────────────────────────────────────
    banner("9-NIGHT RUN  ·  SUMMARY STATISTICS")

    total_predicted = sum(
        len(p["set1"]) + len(p["set2"]) + len(p["encore"]) for p in predictions
    )
    print(f"\n  Total unique songs across 9 nights : {len(run_songs)}")
    print(f"  Total slot predictions             : {total_predicted}")
    print(f"  Catalog coverage                   : {len(run_songs)}/{len(songs_df)} songs")

    song_counter: Counter = Counter()
    for p in predictions:
        for slot_key in ("set1", "set2", "encore"):
            for entry in p[slot_key]:
                song_counter[entry["song_id"]] += 1

    repeats = [(songs_df.loc[sid, "name"], cnt) for sid, cnt in song_counter.items() if cnt > 1]
    if repeats:
        print(f"\n  Songs appearing more than once (soft exclusion in effect):")
        for name, cnt in sorted(repeats, key=lambda x: -x[1]):
            print(f"    {name}: {cnt} nights")
    else:
        print("\n  Run exclusions working — no songs repeated across 9 nights.")

    # Top-confidence individual picks
    all_entries = []
    for p in predictions:
        d = p["date"]
        for slot_key in ("set1", "set2", "encore"):
            for entry in p[slot_key]:
                all_entries.append((entry["score"], entry["name"],
                                    slot_key.upper(), d.strftime("%b %-d")))
    top_entries = sorted(all_entries, reverse=True)[:12]
    print("\n  Highest-confidence individual predictions:")
    print(tabulate(top_entries, headers=["Score", "Song", "Set", "Date"],
                   tablefmt="simple", floatfmt=".3f"))

    # Sphere affinity leaders from catalog
    top_sphere = songs_df.nlargest(8, "sphere_affinity")[["name", "sphere_affinity", "jam_score"]]
    print("\n  Catalog Sphere-affinity leaders:")
    print(tabulate(top_sphere.values.tolist(),
                   headers=["Song", "Sphere Affinity", "Jam Score"],
                   tablefmt="simple", floatfmt=".2f"))

    # Final validation summary
    print()
    print("  ── Validation baseline (backtest) ───────────────────────────────")
    print(f"     Hit rate avg     : {backtest_results['avg_hit_rate']*100:.1f}%")
    print(f"     Set precision avg: {backtest_results['avg_set_precision']*100:.1f}%")
    print(f"     Correct songs    : {backtest_results['total_correct']} / "
          f"{backtest_results['total_predicted']} predicted")

    print()
    print("  ── Model design notes ───────────────────────────────────────────")
    print("     • Gap pressure + recency = 50% of score (overdue songs ranked highest)")
    print("     • Sphere affinity boosts: Light, Sigma Oasis, 2001, YEM, Carini")
    print("     • Mike's Groove (Mike's → Hydrogen → Weekapaug) injected as a unit")
    print("     • Tweezer Reprise locks into encore closer when Tweezer is in Set 2")
    print("     • Run exclusions grow nightly — hard 65% score penalty for repeats")
    print("     • K-Means cluster diversity term discourages consecutive same-archetype songs")

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
