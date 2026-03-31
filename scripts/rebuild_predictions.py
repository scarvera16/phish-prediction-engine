"""
Self-contained prediction rebuild pipeline.

Reads the corrected catalog from prediction-data.ts, constructs DataFrames
for the engine, runs clustering + scoring + multi-night prediction, and
writes back the updated shows/clusters/bustouts to prediction-data.ts.

Usage:
    cd /path/to/phish-setlist-predictor
    python scripts/rebuild_predictions.py
"""

import sys
import os
import re
import json
import numpy as np
import pandas as pd

# Add project root to path so engine/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.scoring import score_all_songs, score_breakdown, DEFAULT_WEIGHTS
from engine.clustering import train_song_clusters, compute_cluster_diversity_bonus
from engine.features import build_cluster_feature_matrix

# ─── CONFIG ──────────────────────────────────────────────────────────────────
VENUE = "Sphere, Las Vegas"
DATES_LABEL = "Apr 16 \u2013 May 2, 2026"
SHOW_DATES = [
    ("2026-04-16", "Thu", "Apr 16"),
    ("2026-04-17", "Fri", "Apr 17"),
    ("2026-04-18", "Sat", "Apr 18"),
    ("2026-04-23", "Thu", "Apr 23"),
    ("2026-04-24", "Fri", "Apr 24"),
    ("2026-04-25", "Sat", "Apr 25"),
    ("2026-04-30", "Thu", "Apr 30"),
    ("2026-05-01", "Fri", "May 1"),
    ("2026-05-02", "Sat", "May 2"),
]

# Sphere-tuned weights (frequency-heavy, gap-light)
WEIGHTS = {
    "recency":          0.05,
    "gap_pressure":     0.05,
    "slot_affinity":    0.35,
    "frequency":        0.40,
    "venue_affinity":   0.10,
    "cluster_diversity":0.05,
}

SET1_SIZE = 11
SET2_SIZE = 7
ENC_SIZE = 2
VENUE_TYPE = "sphere"

# Estimated total shows in training window (2009-2025)
TOTAL_TRAINING_SHOWS = 280

TS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "lib", "prediction-data.ts")


# ─── PARSE CATALOG FROM TS ──────────────────────────────────────────────────
def parse_catalog_from_ts(filepath):
    """Extract catalog entries from prediction-data.ts."""
    with open(filepath, "r") as f:
        content = f.read()

    catalog = {}
    pattern = re.compile(r'"([a-z0-9_]+)":\{([^}]+)\}')
    for m in pattern.finditer(content):
        slug = m.group(1)
        props_str = m.group(2)
        props = {}
        for pm in re.finditer(r'(\w+):("[^"]*"|[^,}]+)', props_str):
            key, val = pm.group(1), pm.group(2)
            if val.startswith('"'):
                props[key] = val.strip('"')
            elif '.' in val:
                props[key] = float(val)
            else:
                try:
                    props[key] = int(val)
                except ValueError:
                    props[key] = val
        catalog[slug] = props
    return catalog


# ─── BUILD DATAFRAMES ────────────────────────────────────────────────────────
def build_songs_df(catalog):
    """Build the songs DataFrame the engine expects from our TS catalog."""
    rows = []
    for slug, s in catalog.items():
        # Map TS fields to engine DataFrame columns
        s1f = s.get("s1f", 0.5)
        s2f = s.get("s2f", 0.3)
        encf = s.get("encf", 0.1)
        jam = s.get("jam", 0.0)
        energy_raw = s.get("energy", 0.5)
        sphere_raw = s.get("sphere", 0.5)

        # Compute slot weights from set fractions
        show_opener_weight = s1f * 1.2 if s1f > 0.5 else s1f * 0.5
        set2_opener_weight = s2f * 1.1 if s2f > 0.4 else s2f * 0.4
        set_closer_weight = max(s2f * 0.8, encf * 1.5)
        s1_weight = s1f
        s2_weight = s2f + jam * 0.3  # jam vehicles get s2 boost
        enc_weight = encf + (0.2 if encf > 0.05 else 0.0)

        rows.append({
            "song_id": slug,
            "name": s.get("fullName", s.get("name", slug)),
            "debut_year": s.get("debut", 2000),
            "avg_duration_min": s.get("dur", 6.0),
            "jam_score": jam * 10,  # engine uses 0-10 scale
            "energy": energy_raw,
            "sphere_affinity": sphere_raw,
            "s1_frac": s1f,
            "s2_frac": s2f,
            "enc_frac": encf,
            "avg_gap": s.get("avgGap", 20.0),
            "style": s.get("style", "rock"),
            "plays": s.get("plays", 1),
            "curGap": s.get("curGap", 10),
            "gapZ": s.get("gapZ", 0.0),
            # Slot weights
            "show_opener_weight": show_opener_weight,
            "set2_opener_weight": set2_opener_weight,
            "set_closer_weight": set_closer_weight,
            "s1_weight": s1_weight,
            "s2_weight": s2_weight,
            "enc_weight": enc_weight,
            # PCA (pass through)
            "pcaX": s.get("pcaX", 0.0),
            "pcaY": s.get("pcaY", 0.0),
        })

    df = pd.DataFrame(rows).set_index("song_id")
    return df


def build_synthetic_feat_df(songs_df):
    """
    Build a synthetic feature DataFrame from catalog data.
    Since we don't have raw show history, we synthesize features
    from the catalog's statistical summaries.
    """
    rows = []
    for song_id in songs_df.index:
        s = songs_df.loc[song_id]
        avg_gap = s["avg_gap"]
        cur_gap = s["curGap"]
        gap_z = s["gapZ"]
        plays = s["plays"]

        # Estimate std_gap from avg_gap (typical ratio ~0.45)
        std_gap = max(avg_gap * 0.45, 1.0)

        rows.append({
            "song_id": song_id,
            "total_plays": plays,
            "last_play_date": pd.NaT,
            "last_play_show_num": max(0, TOTAL_TRAINING_SHOWS - cur_gap),
            "current_gap_shows": cur_gap,
            "avg_gap_actual": avg_gap,
            "std_gap_actual": std_gap,
            "gap_z_score": gap_z,
            "plays_30d": max(1, int(plays / max(avg_gap, 1) * 2)) if cur_gap < 5 else 0,
            "plays_90d": max(1, int(plays / max(avg_gap, 1) * 6)) if cur_gap < 15 else 0,
            "plays_365d": max(1, int(plays / max(avg_gap, 1) * 25)),
            "set1_plays": int(plays * s["s1_frac"]),
            "set2_plays": int(plays * s["s2_frac"]),
            "enc_plays": int(plays * s["enc_frac"]),
            "set1_actual_frac": s["s1_frac"],
            "set2_actual_frac": s["s2_frac"],
            "enc_actual_frac": s["enc_frac"],
            "opener_plays": int(plays * s["s1_frac"] * 0.15),
            "set2_opener_plays": int(plays * s["s2_frac"] * 0.12),
            "set_closer_plays": int(plays * (s["s2_frac"] + s["enc_frac"]) * 0.1),
            "sphere_plays": 0,
            "arena_plays": int(plays * 0.4),
        })

    return pd.DataFrame(rows).set_index("song_id")


# ─── PREDICTION PIPELINE ────────────────────────────────────────────────────
def predict_show_standalone(
    songs_df, feat_df, cluster_labels, run_exclusions, soft_exclusions,
    set1_size=SET1_SIZE, set2_size=SET2_SIZE, enc_size=ENC_SIZE
):
    """Predict a single show setlist (standalone version without data/ import)."""
    from engine.song_pairs import SONG_PAIRS

    w = WEIGHTS
    hard_excl_run = run_exclusions or set()
    soft_excl = soft_exclusions or {}
    chosen_show = []

    def _pick(slot_type, n, hard_excl=None):
        hard = (hard_excl or set()) | set(chosen_show) | hard_excl_run
        scores = score_all_songs(
            slot_type=slot_type, songs_df=songs_df, feat_df=feat_df,
            cluster_labels=cluster_labels, already_chosen=chosen_show,
            excluded=hard, venue_type=VENUE_TYPE,
            total_shows=TOTAL_TRAINING_SHOWS, weights=w,
        )
        for sid, penalty in soft_excl.items():
            if sid in scores.index:
                scores[sid] *= penalty
        scores = scores.sort_values(ascending=False)

        picks = []
        for song_id in scores.index:
            if song_id in chosen_show:
                continue
            bd = score_breakdown(song_id, slot_type, songs_df, feat_df,
                                 cluster_labels, chosen_show, VENUE_TYPE,
                                 TOTAL_TRAINING_SHOWS, w)
            picks.append({
                "song_id": song_id,
                "name": songs_df.loc[song_id, "name"],
                "slot": slot_type,
                "score": round(float(scores[song_id]), 4),
                "components": bd,
            })
            if len(picks) >= n:
                break
        return picks

    def _commit(picks, count=1):
        committed = picks[:count]
        for p in committed:
            chosen_show.append(p["song_id"])
        return committed

    setlist = {"set1": [], "set2": [], "encore": []}

    # Set 1
    openers = _pick("show_opener", 3)
    if openers:
        setlist["set1"] += _commit(openers, 1)

    for _ in range(set1_size - 2):
        picks = _pick("s1_body", 3)
        if picks:
            setlist["set1"] += _commit(picks, 1)

    s1_closers = _pick("s1_closer", 3)
    if s1_closers:
        setlist["set1"] += _commit(s1_closers, 1)

    # Set 2
    s2_openers = _pick("s2_opener", 3)
    if not s2_openers:
        s2_openers = _pick("s2_body", 3)
    if s2_openers:
        s2_opener_pick = _commit(s2_openers, 1)[0]
        setlist["set2"].append(s2_opener_pick)

    # Mike's Groove injection
    groove_available = all(sid not in hard_excl_run and sid not in chosen_show
                          for sid in ["mikes", "hydrogen", "weekapaug"]
                          if sid in songs_df.index)
    if groove_available and "mikes" in songs_df.index:
        s2_scores = score_all_songs("s2_body", songs_df, feat_df, cluster_labels,
                                     chosen_show, hard_excl_run, VENUE_TYPE,
                                     TOTAL_TRAINING_SHOWS, w)
        mikes_score = float(s2_scores.get("mikes", 0.0))
        is_mikes_opener = len(setlist["set2"]) > 0 and setlist["set2"][0]["song_id"] == "mikes"

        if is_mikes_opener or mikes_score > 0.65:
            for seq_id in ["mikes", "hydrogen", "weekapaug"]:
                if seq_id in songs_df.index and seq_id not in chosen_show and seq_id not in hard_excl_run:
                    bd = score_breakdown(seq_id, "s2_body", songs_df, feat_df,
                                         cluster_labels, chosen_show, VENUE_TYPE,
                                         TOTAL_TRAINING_SHOWS, w)
                    setlist["set2"].append({
                        "song_id": seq_id, "name": songs_df.loc[seq_id, "name"],
                        "slot": "s2_sequence", "score": bd["composite"],
                        "components": bd,
                    })
                    chosen_show.append(seq_id)

    while len(setlist["set2"]) < set2_size - 1:
        picks = _pick("s2_body", 3)
        if not picks:
            break
        setlist["set2"] += _commit(picks, 1)

    s2_closers = _pick("s2_closer", 3)
    if s2_closers:
        setlist["set2"] += _commit(s2_closers, 1)

    # Encore
    if ("tweezer" in chosen_show and "tweeprise" in songs_df.index
            and "tweeprise" not in chosen_show and "tweeprise" not in hard_excl_run):
        bd = score_breakdown("tweeprise", "encore", songs_df, feat_df,
                             cluster_labels, chosen_show, VENUE_TYPE,
                             TOTAL_TRAINING_SHOWS, w)
        if enc_size > 1:
            pre_encore = _pick("encore", 3, hard_excl={"tweeprise"})
            if pre_encore:
                setlist["encore"] += _commit(pre_encore, enc_size - 1)
        setlist["encore"].append({
            "song_id": "tweeprise", "name": songs_df.loc["tweeprise", "name"],
            "slot": "encore_closer", "score": round(bd["composite"] * 1.2, 4),
            "components": bd,
        })
        chosen_show.append("tweeprise")
    else:
        enc_picks = _pick("encore", 3)
        if enc_picks:
            setlist["encore"] += _commit(enc_picks, min(enc_size, len(enc_picks)))

    return setlist


def predict_full_run(songs_df, feat_df, cluster_labels):
    """Predict all 9 shows with rolling exclusions."""
    show_song_history = []
    predictions = []

    for i, (date_str, day_name, date_short) in enumerate(SHOW_DATES):
        hard_exclusions = set()
        soft_exclusions = {}

        for lookback in range(1, len(show_song_history) + 1):
            past_songs = show_song_history[-lookback]
            if lookback <= 2:
                hard_exclusions |= past_songs
            elif lookback == 3:
                for sid in past_songs:
                    if sid not in hard_exclusions:
                        soft_exclusions[sid] = min(soft_exclusions.get(sid, 1.0), 0.15)
            elif lookback == 4:
                for sid in past_songs:
                    if sid not in hard_exclusions:
                        soft_exclusions[sid] = min(soft_exclusions.get(sid, 1.0), 0.40)
            else:
                for sid in past_songs:
                    if sid not in hard_exclusions:
                        soft_exclusions[sid] = min(soft_exclusions.get(sid, 1.0), 0.70)

        setlist = predict_show_standalone(
            songs_df, feat_df, cluster_labels,
            hard_exclusions, soft_exclusions,
        )

        predictions.append({
            "show_num": i + 1,
            "date": date_str,
            "day_name": day_name,
            "date_short": date_short,
            "setlist": setlist,
        })

        # Record this show's songs
        this_show_songs = set()
        for slot in ("set1", "set2", "encore"):
            for entry in setlist[slot]:
                this_show_songs.add(entry["song_id"])
        show_song_history.append(this_show_songs)

        print(f"  Show {i+1} ({date_short} {day_name}): "
              f"{len(setlist['set1'])}+{len(setlist['set2'])}+{len(setlist['encore'])} songs")

    return predictions


# ─── TS EXPORT ───────────────────────────────────────────────────────────────
def format_show_for_ts(pred, catalog):
    """Format a single show prediction for TypeScript output."""
    show = pred["setlist"]

    def format_songs(songs):
        parts = []
        for s in songs:
            cat = catalog.get(s["song_id"], {})
            slot_map = {
                "show_opener": "opener", "s1_body": "seq", "s1_closer": "closer",
                "s2_opener": "opener", "s2_body": "seq", "s2_closer": "closer",
                "s2_sequence": "seq", "encore": "seq", "encore_closer": "closer",
            }
            slot = slot_map.get(s["slot"], "seq")
            sc = round(s["score"], 2)
            parts.append(f'{{id:"{s["song_id"]}",slot:"{slot}",sc:{sc}}}')
        return "[" + ",".join(parts) + "]"

    return (f'{{num:{pred["show_num"]},'
            f'date:"{pred["date"]}",'
            f'day:"{pred["day_name"]}",'
            f'set1:{format_songs(show["set1"])},'
            f'set2:{format_songs(show["set2"])},'
            f'encore:{format_songs(show["encore"])},'
            f'excl:0}}')


def format_clusters_for_ts(cluster_names, cluster_counts):
    """Format cluster data for TypeScript."""
    parts = []
    for cid in sorted(cluster_names.keys()):
        name = cluster_names[cid]
        count = cluster_counts.get(cid, 0)
        parts.append(f'    "{cid}": {{name:"{name}",count:{count}}}')
    return "{\n" + ",\n".join(parts) + ",\n  }"


def update_ts_file(ts_path, predictions, catalog, cluster_labels, cluster_names):
    """Update prediction-data.ts with new shows and clusters."""
    with open(ts_path, "r") as f:
        content = f.read()

    # Update cluster labels in catalog entries
    for slug, cid in cluster_labels.items():
        pattern = rf'("{re.escape(slug)}":\{{[^}}]*?)cluster:\d+'
        replacement = rf'\g<1>cluster:{cid}'
        content = re.sub(pattern, replacement, content)

    # Update cluster counts
    cluster_counts = {}
    for cid in cluster_labels.values():
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    clusters_ts = format_clusters_for_ts(cluster_names, cluster_counts)
    # Match the entire clusters block: "clusters: { ... }," including nested {name:...,count:...}
    content = re.sub(
        r'clusters:\s*\{[\s\S]*?\n  \},',
        f'clusters: {clusters_ts},',
        content,
    )

    # Format shows array
    shows_parts = []
    for pred in predictions:
        shows_parts.append("    " + format_show_for_ts(pred, catalog))
    shows_ts = "[\n" + ",\n".join(shows_parts) + ",\n  ]"

    # Replace shows array
    content = re.sub(
        r'shows:\s*\[[\s\S]*?\],\s*songPairs',
        f'shows: {shows_ts},\n  songPairs',
        content,
    )

    # Update meta
    all_songs = set()
    for pred in predictions:
        for slot in ("set1", "set2", "encore"):
            for s in pred["setlist"][slot]:
                all_songs.add(s["song_id"])
    content = re.sub(r'total_predicted_unique:\s*\d+', f'total_predicted_unique: {len(all_songs)}', content)
    content = re.sub(
        r'scoring_weights:\s*\{[^}]+\}',
        f'scoring_weights: {{ recency: {WEIGHTS["recency"]}, gap_pressure: {WEIGHTS["gap_pressure"]}, '
        f'slot_affinity: {WEIGHTS["slot_affinity"]}, frequency: {WEIGHTS["frequency"]}, '
        f'venue_affinity: {WEIGHTS["venue_affinity"]}, cluster_diversity: {WEIGHTS["cluster_diversity"]} }}',
        content,
    )

    with open(ts_path, "w") as f:
        f.write(content)

    return len(all_songs)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHISH SPHERE 2026 — PREDICTION REBUILD")
    print("=" * 60)

    # 1. Parse catalog
    print("\n[1/5] Parsing corrected catalog...")
    catalog = parse_catalog_from_ts(TS_FILE)
    print(f"  {len(catalog)} songs in catalog")

    # 2. Build DataFrames
    print("\n[2/5] Building DataFrames...")
    songs_df = build_songs_df(catalog)
    feat_df = build_synthetic_feat_df(songs_df)
    print(f"  songs_df: {songs_df.shape}")
    print(f"  feat_df: {feat_df.shape}")

    # 3. Run clustering
    print("\n[3/5] Running K-Means clustering...")
    X_scaled, scaler = build_cluster_feature_matrix(songs_df, feat_df)
    kmeans, cluster_labels, cluster_names = train_song_clusters(X_scaled, n_clusters=6, seed=42)
    cluster_counts = {}
    for cid in cluster_labels.values():
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
    print(f"  Clusters: {cluster_names}")
    print(f"  Counts: {cluster_counts}")

    # 4. Predict all 9 shows
    print("\n[4/5] Predicting 9-show run...")
    predictions = predict_full_run(songs_df, feat_df, cluster_labels)

    # Stats
    all_songs = set()
    for pred in predictions:
        for slot in ("set1", "set2", "encore"):
            for s in pred["setlist"][slot]:
                all_songs.add(s["song_id"])
    print(f"\n  Total unique songs across run: {len(all_songs)}")

    # 5. Write back to TS
    print("\n[5/5] Writing to prediction-data.ts...")
    n_unique = update_ts_file(TS_FILE, predictions, catalog, cluster_labels, cluster_names)
    print(f"  Updated with {n_unique} unique songs across 9 shows")

    print("\n" + "=" * 60)
    print("REBUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
