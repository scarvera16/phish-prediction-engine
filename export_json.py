#!/usr/bin/env python3
"""
Export prediction data as JSON for the React frontend.

Runs the full pipeline and outputs a single JSON file with:
  - All 9 predicted shows with score breakdowns
  - Song catalog with metadata, clusters, and features
  - Cluster definitions and members
  - Run-level stats (exclusions accumulating, repeats, coverage)
  - Scoring weights
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter

from phish_engine.data.real_data import load_real_data, _pn_slug_to_song_id
from phish_engine.data.manual_overrides import SONG_PAIRS
from phish_engine.features import compute_all_features
from phish_engine.clustering import cluster_songs
from phish_engine.predictor import predict_multi_night_run
from phish_engine.backtest.validator import run_backtest
from phish_engine.scoring import ScoringWeights

# Sphere dates from main.py
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

WEIGHTS = ScoringWeights(
    recency=0.05,
    gap_pressure=0.05,
    slot_affinity=0.35,
    frequency=0.40,
    venue_affinity=0.10,
    cluster=0.05,
    recency_decay_rate=3.0,
    freq_w10=0.25,
    freq_w30=0.45,
    freq_w90=0.30,
    gap_lognormal_sigma=0.45,
)


def _name_clusters(songs_with_clusters: pd.DataFrame) -> dict:
    """Assign interpretive names to clusters based on their characteristics."""
    stats = []
    for cid in sorted(songs_with_clusters["cluster_id"].unique()):
        m = songs_with_clusters[songs_with_clusters["cluster_id"] == cid]
        stats.append({
            "cid": int(cid),
            "n": len(m),
            "avg_jam": m["jam_score"].mean(),
            "avg_dur": m["avg_duration_min"].mean(),
            "avg_gap": m["avg_gap"].mean(),
            "avg_energy": m["energy"].mean(),
            "s2_frac": m["s2_frac"].mean(),
            "enc_frac": m["enc_frac"].mean(),
        })

    # Sort by characteristic to assign names
    names = {}
    assigned = set()

    # Highest avg_dur + jam = Jam Vehicles
    by_dur = sorted(stats, key=lambda x: x["avg_dur"], reverse=True)
    for s in by_dur:
        if s["cid"] not in assigned and s["avg_jam"] > 0.2:
            names[s["cid"]] = "Jam Vehicles"
            assigned.add(s["cid"])
            break

    # Highest jam + s2_frac (not already assigned) = Set 2 Explorers
    by_jam = sorted(stats, key=lambda x: x["avg_jam"], reverse=True)
    for s in by_jam:
        if s["cid"] not in assigned and s["s2_frac"] > 0.5:
            names[s["cid"]] = "Set 2 Explorers"
            assigned.add(s["cid"])
            break

    # Highest avg_gap = Bustout Rarities
    by_gap = sorted(stats, key=lambda x: x["avg_gap"], reverse=True)
    for s in by_gap:
        if s["cid"] not in assigned:
            names[s["cid"]] = "Bustout Rarities"
            assigned.add(s["cid"])
            break

    # Highest energy + low gap (frequent) = High-Energy Staples
    by_energy = sorted(stats, key=lambda x: x["avg_energy"] - x["avg_gap"] / 100, reverse=True)
    for s in by_energy:
        if s["cid"] not in assigned and s["avg_energy"] > 0.7:
            names[s["cid"]] = "High-Energy Staples"
            assigned.add(s["cid"])
            break

    # Largest remaining group with moderate energy = Set 1 Staples
    by_size = sorted(stats, key=lambda x: x["n"], reverse=True)
    for s in by_size:
        if s["cid"] not in assigned and s["enc_frac"] < 0.15:
            names[s["cid"]] = "Set 1 Staples"
            assigned.add(s["cid"])
            break

    # Remaining names to assign (no duplicates)
    remaining_names = ["Deep Cuts", "Rotation Players", "Encore Favorites", "Cover Corner"]
    name_idx = 0
    for s in sorted(stats, key=lambda x: x["avg_gap"], reverse=True):
        if s["cid"] not in assigned:
            names[s["cid"]] = remaining_names[name_idx % len(remaining_names)]
            assigned.add(s["cid"])
            name_idx += 1

    return names


def main():
    # ── 1. Load real Phish.net data ────────────────────────────────────────────
    songs_df, shows_df, appearances_df = load_real_data(
        data_dir="data/",
        min_plays=3,
        start_year=2019,
    )
    print(f"  Loaded: {len(shows_df)} shows, {len(songs_df)} songs, {len(appearances_df)} appearances")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    feat_df = compute_all_features(
        songs_df, shows_df, appearances_df,
        as_of_date=pd.Timestamp("2026-04-15"),
    )

    # ── 3. Clustering ─────────────────────────────────────────────────────────
    songs_with_clusters, scaler, kmeans, pca, pca_coords_raw, sil_scores = cluster_songs(
        songs_df, appearances_df,
    )
    cluster_labels = dict(zip(songs_with_clusters.index, songs_with_clusters["cluster_id"]))

    # Assign interpretive names based on cluster characteristics
    cluster_names = _name_clusters(songs_with_clusters)
    # Use PCA coords from cluster_songs (already computed)
    pca_coords = pca_coords_raw.values if hasattr(pca_coords_raw, 'values') else pca_coords_raw

    # ── 4. Predict 9 shows ────────────────────────────────────────────────────
    predictions = predict_multi_night_run(
        show_dates=SPHERE_DATES,
        venue_type="sphere",
        songs_df=songs_df,
        shows_df=shows_df,
        appearances_df=appearances_df,
        cluster_labels=cluster_labels,
        weights=WEIGHTS,
    )

    # ── 5. Build export structure ─────────────────────────────────────────────

    # Load all-time play counts from Phish.net songs.json
    with open("data/songs.json") as _f:
        _raw_songs = json.load(_f)
    alltime_plays = {}
    nicknames = {}
    for _s in _raw_songs:
        _sid = _pn_slug_to_song_id(_s["slug"])
        alltime_plays[_sid] = _s.get("times_played", 0)
        abbr = _s.get("abbr", "").strip()
        if abbr and abbr != _s.get("song", ""):
            nicknames[_sid] = abbr

    # Pre-compute set position stats (opener/closer) from cached setlist data
    position_stats = {}
    try:
        with open("data/setlists.json") as _sf:
            _setlists_raw = json.load(_sf)
        # setlists.json is dict keyed by date → list of entries
        from collections import defaultdict
        shows_by_date = {}
        if isinstance(_setlists_raw, dict):
            shows_by_date = _setlists_raw
        else:
            for entry in _setlists_raw:
                shows_by_date.setdefault(entry["showdate"], []).append(entry)
        # For each song, count opener/closer/encore across all shows
        song_position_counts = defaultdict(lambda: {"appearances": 0, "s1_open": 0, "s1_close": 0, "s2_open": 0, "s2_close": 0, "encore": 0})
        for _date, entries in shows_by_date.items():
            by_set = defaultdict(list)
            for e in entries:
                by_set[e.get("set", "")].append(e)
            for set_key, set_entries in by_set.items():
                sorted_entries = sorted(set_entries, key=lambda x: x.get("position", 0))
                for e in sorted_entries:
                    slug = e.get("slug", "")
                    _sid = _pn_slug_to_song_id(slug)
                    if _sid not in songs_df.index:
                        continue
                    pc = song_position_counts[_sid]
                    pc["appearances"] += 1
                    if set_key in ("e", "e2"):
                        pc["encore"] += 1
                    elif set_key == "1":
                        if sorted_entries[0].get("slug") == slug:
                            pc["s1_open"] += 1
                        if sorted_entries[-1].get("slug") == slug:
                            pc["s1_close"] += 1
                    elif set_key == "2":
                        if sorted_entries[0].get("slug") == slug:
                            pc["s2_open"] += 1
                        if sorted_entries[-1].get("slug") == slug:
                            pc["s2_close"] += 1
        for _sid, pc in song_position_counts.items():
            total = max(pc["appearances"], 1)
            position_stats[_sid] = {
                "s1_opener": round(pc["s1_open"] / total * 100),
                "s1_closer": round(pc["s1_close"] / total * 100),
                "s2_opener": round(pc["s2_open"] / total * 100),
                "s2_closer": round(pc["s2_close"] / total * 100),
                "encore_rate": round(pc["encore"] / total * 100),
            }
        print(f"  Position stats computed for {len(position_stats)} songs")
    except Exception as e:
        print(f"  Warning: Could not compute position stats: {e}")

    # Song catalog with cluster + feature data
    pca_index = list(songs_with_clusters.index)
    catalog = {}
    for sid in songs_df.index:
        s = songs_df.loc[sid]
        f = feat_df.loc[sid]
        pca_idx = pca_index.index(sid) if sid in pca_index else 0
        nick = nicknames.get(sid)
        catalog[sid] = {
            "name":            nick if nick else s["name"],
            "fullName":        s["name"] if nick else None,
            "debut_year":      int(s["debut_year"]),
            "avg_duration":    round(float(s["avg_duration_min"]), 1),
            "jam_score":       round(float(s["jam_score"]), 2),
            "energy":          round(float(s["energy"]), 2),
            "style":           s["style"],
            "sphere_affinity": round(float(s["sphere_affinity"]), 2),
            "cluster_id":      int(cluster_labels.get(sid, -1)),
            "s1_frac":         round(float(s["s1_frac"]), 3),
            "s2_frac":         round(float(s["s2_frac"]), 3),
            "enc_frac":        round(float(s["enc_frac"]), 3),
            "avg_gap":         round(float(s["avg_gap"]), 1),
            "current_gap":     round(float(f["current_gap_shows"]), 0),
            "gap_z_score":     round(float(f["gap_z_score"]), 2),
            "plays":           alltime_plays.get(sid, int(f["total_plays"])),
            "pca_x":           round(float(pca_coords[pca_idx, 0]), 3),
            "pca_y":           round(float(pca_coords[pca_idx, 1]), 3),
            "position_stats":  position_stats.get(sid, {"s1_opener": 0, "s1_closer": 0, "s2_opener": 0, "s2_closer": 0, "encore_rate": 0}),
        }

    # Cluster definitions
    clusters = {}
    for cid in sorted(cluster_names.keys()):
        members = [sid for sid, c in cluster_labels.items() if c == cid]
        clusters[str(cid)] = {
            "name": cluster_names[cid],
            "members": members,
            "count": len(members),
        }

    # Shows with full predictions
    shows_export = []
    run_exclusions_so_far = set()
    for pred in predictions:
        show_data = {
            "show_num":  pred["show_num"],
            "date":      pred["date"].strftime("%Y-%m-%d"),
            "day_name":  pred["date"].strftime("%A"),
            "date_short": pred["date"].strftime("%b %-d"),
            "sets": {},
            "run_exclusions_before": sorted(list(run_exclusions_so_far)),
        }

        show_songs = []
        for set_key in ("set1", "set2", "encore"):
            set_data = []
            for i, entry in enumerate(pred[set_key]):
                song_entry = {
                    "song_id":    entry["song_id"],
                    "name":       entry["name"],
                    "position":   i + 1,
                    "slot":       entry["slot"],
                    "score":      entry["score"],
                    "components": entry["components"],
                    "is_pair":    entry["song_id"] in SONG_PAIRS or entry["song_id"] in SONG_PAIRS.values(),
                    "pair_next":  SONG_PAIRS.get(entry["song_id"]),
                }
                set_data.append(song_entry)
                show_songs.append(entry["song_id"])
            show_data["sets"][set_key] = set_data

        # Compute set-level stats
        for set_key in ("set1", "set2", "encore"):
            entries = show_data["sets"][set_key]
            total_dur = sum(catalog[e["song_id"]]["avg_duration"] for e in entries)
            avg_energy = np.mean([catalog[e["song_id"]]["energy"] for e in entries]) if entries else 0
            avg_score = np.mean([e["score"] for e in entries]) if entries else 0
            show_data["sets"][set_key] = {
                "songs": entries,
                "total_duration": round(total_dur, 1),
                "avg_energy": round(float(avg_energy), 2),
                "avg_score": round(float(avg_score), 3),
                "song_count": len(entries),
            }

        shows_export.append(show_data)
        run_exclusions_so_far.update(show_songs)

    # Run-level heatmap: which songs appear on which nights
    heatmap = {}
    for pred in predictions:
        for set_key in ("set1", "set2", "encore"):
            for entry in pred[set_key]:
                sid = entry["song_id"]
                if sid not in heatmap:
                    heatmap[sid] = {"name": entry["name"], "nights": []}
                heatmap[sid]["nights"].append({
                    "show_num": pred["show_num"],
                    "set": set_key,
                    "score": entry["score"],
                })

    # Bustout candidates: songs with highest gap z-scores not yet predicted
    predicted_songs = set()
    for pred in predictions:
        for set_key in ("set1", "set2", "encore"):
            for entry in pred[set_key]:
                predicted_songs.add(entry["song_id"])

    bustouts = []
    for sid in songs_df.index:
        f = feat_df.loc[sid]
        plays = alltime_plays.get(sid, int(f["total_plays"]))
        gap_z = float(f["gap_z_score"])
        # Weight by log of all-time plays so former staples rank above deep cuts
        # A song with 200 plays and z=5 ranks higher than a song with 4 plays and z=50
        import math
        bustout_score = gap_z * math.log2(max(plays, 2))
        bustouts.append({
            "song_id": sid,
            "name": songs_df.loc[sid, "name"],
            "current_gap": round(float(f["current_gap_shows"]), 0),
            "avg_gap": round(float(songs_df.loc[sid, "avg_gap"]), 1),
            "gap_z_score": round(gap_z, 2),
            "plays": plays,
            "bustout_score": round(bustout_score, 2),
            "predicted": sid in predicted_songs,
        })
    bustouts.sort(key=lambda x: -x["bustout_score"])

    # ── 6. Backtest against NYE 2025 ────────────────────────────────────────────
    backtest = run_backtest(
        songs_df=songs_df,
        shows_df=shows_df,
        appearances_df=appearances_df,
        cluster_labels=cluster_labels,
        validation_tour="2025 NYE Run",
        weights=WEIGHTS,
        verbose=True,
    )

    # ── 7. Assemble final JSON ────────────────────────────────────────────────
    export = {
        "meta": {
            "venue": "Sphere, Las Vegas",
            "dates": f"{SPHERE_DATES[0].strftime('%b %-d')} – {SPHERE_DATES[-1].strftime('%b %-d, %Y')}",
            "total_catalog": len(songs_df),
            "total_predicted_unique": len(predicted_songs),
            "scoring_weights": WEIGHTS.main_weights_normalized(),
            "data_source": "phish.net",
            "training_shows": len(shows_df),
        },
        "catalog": catalog,
        "clusters": clusters,
        "shows": shows_export,
        "heatmap": heatmap,
        "bustouts": bustouts[:15],
        "song_pairs": {k: v for k, v in SONG_PAIRS.items()},
        "backtest": {
            "avg_hit_rate": round(backtest["avg_hit_rate"], 3),
            "avg_set_precision": round(backtest["avg_set_precision"], 3),
            "total_correct": backtest["total_correct"],
            "total_predicted": backtest["total_predicted"],
        },
    }

    with open("prediction_data.json", "w") as f:
        json.dump(export, f, indent=2, default=str)

    print(f"Exported prediction_data.json")
    print(f"  Shows: {len(shows_export)}")
    print(f"  Catalog: {len(catalog)} songs")
    print(f"  Clusters: {len(clusters)}")
    print(f"  Unique songs predicted: {len(predicted_songs)}")
    print(f"  Bustout candidates: {len(bustouts)}")


if __name__ == "__main__":
    main()
