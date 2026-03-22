"""
Feature engineering for the setlist predictor.

Given the show history up to a cutoff date, compute per-song dynamic features
that power both the K-Means clustering and the composite scorer.
"""

import numpy as np
import pandas as pd


def compute_all_features(
    songs_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    appearances_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute dynamic features for every song as of `as_of_date`.

    Returns a DataFrame indexed by song_id with columns:
      total_plays, last_play_date, last_play_show_num,
      current_gap_shows,       # shows since last play (as of cutoff)
      avg_gap_actual,          # mean gap across full history
      std_gap_actual,          # std of gaps
      gap_z_score,             # z-score of current gap vs historical
      plays_30d, plays_90d, plays_365d,
      set1_plays, set2_plays, enc_plays,
      set1_actual_frac, set2_actual_frac, enc_actual_frac,
      opener_plays,            # times opened a show
      set2_opener_plays,
      set_closer_plays,
      sphere_plays,            # plays at Sphere-type venues
      arena_plays,
    """
    # Filter to training window
    hist_shows = shows_df[shows_df["date"] <= as_of_date].copy()
    hist_app = appearances_df[appearances_df["date"] <= as_of_date].copy()

    total_shows = len(hist_shows)
    show_nums = hist_shows.set_index("show_id")["show_num"]
    max_show_num = int(hist_shows["show_num"].max()) if total_shows > 0 else 0

    # Map show_id -> venue_type
    venue_map = hist_shows.set_index("show_id")["venue_type"]

    rows = []
    for song_id in songs_df.index:
        song_app = hist_app[hist_app["song_id"] == song_id].copy()
        song_app = song_app.sort_values("date")

        n_plays = len(song_app)

        if n_plays == 0:
            # Song not played in training window
            rows.append({
                "song_id":              song_id,
                "total_plays":          0,
                "last_play_date":       pd.NaT,
                "last_play_show_num":   0,
                "current_gap_shows":    max_show_num,  # maximum possible gap
                "avg_gap_actual":       float(songs_df.loc[song_id, "avg_gap"]),
                "std_gap_actual":       float(songs_df.loc[song_id, "avg_gap"]) * 0.5,
                "gap_z_score":          3.0,
                "plays_30d":            0,
                "plays_90d":            0,
                "plays_365d":           0,
                "set1_plays":           0,
                "set2_plays":           0,
                "enc_plays":            0,
                "set1_actual_frac":     float(songs_df.loc[song_id, "s1_frac"]),
                "set2_actual_frac":     float(songs_df.loc[song_id, "s2_frac"]),
                "enc_actual_frac":      float(songs_df.loc[song_id, "enc_frac"]),
                "opener_plays":         0,
                "set2_opener_plays":    0,
                "set_closer_plays":     0,
                "sphere_plays":         0,
                "arena_plays":          0,
            })
            continue

        # Gap calculations: list of (show_num) sorted
        played_show_nums = sorted(
            hist_app[hist_app["song_id"] == song_id]["show_id"].map(show_nums).dropna().astype(int).tolist()
        )
        if len(played_show_nums) >= 2:
            gaps = [played_show_nums[i] - played_show_nums[i - 1] for i in range(1, len(played_show_nums))]
            avg_gap_actual = float(np.mean(gaps))
            std_gap_actual = float(np.std(gaps)) if len(gaps) > 1 else avg_gap_actual * 0.4
        else:
            avg_gap_actual = float(songs_df.loc[song_id, "avg_gap"])
            std_gap_actual = avg_gap_actual * 0.4

        last_show_num = played_show_nums[-1]
        current_gap = max_show_num - last_show_num
        denom = max(std_gap_actual, 1.0)
        gap_z = (current_gap - avg_gap_actual) / denom

        last_date = song_app["date"].max()
        cutoff = as_of_date
        plays_30  = int((song_app["date"] >= cutoff - pd.Timedelta(days=30)).sum())
        plays_90  = int((song_app["date"] >= cutoff - pd.Timedelta(days=90)).sum())
        plays_365 = int((song_app["date"] >= cutoff - pd.Timedelta(days=365)).sum())

        s1  = int((song_app["set_number"] == "1").sum())
        s2  = int((song_app["set_number"] == "2").sum())
        enc = int((song_app["set_number"] == "e").sum())
        total_nz = max(n_plays, 1)

        # Opener = position 1 in set 1
        opener_plays     = int(((song_app["set_number"] == "1") & (song_app["position"] == 1)).sum())
        # Set 2 opener = position 1 in set 2
        s2_opener_plays  = int(((song_app["set_number"] == "2") & (song_app["position"] == 1)).sum())
        # Set closer = highest position in set per show (approximate)
        set_closer_plays = int(
            ((song_app["set_number"] == "1") & (song_app["position"] >= 9)).sum() +
            ((song_app["set_number"] == "2") & (song_app["position"] >= 6)).sum()
        )

        # Venue-type plays
        vtype_series = song_app["show_id"].map(venue_map)
        sphere_plays = int((vtype_series == "sphere").sum())
        arena_plays  = int((vtype_series == "arena").sum())

        rows.append({
            "song_id":              song_id,
            "total_plays":          n_plays,
            "last_play_date":       last_date,
            "last_play_show_num":   last_show_num,
            "current_gap_shows":    current_gap,
            "avg_gap_actual":       avg_gap_actual,
            "std_gap_actual":       std_gap_actual,
            "gap_z_score":          gap_z,
            "plays_30d":            plays_30,
            "plays_90d":            plays_90,
            "plays_365d":           plays_365,
            "set1_plays":           s1,
            "set2_plays":           s2,
            "enc_plays":            enc,
            "set1_actual_frac":     s1 / total_nz,
            "set2_actual_frac":     s2 / total_nz,
            "enc_actual_frac":      enc / total_nz,
            "opener_plays":         opener_plays,
            "set2_opener_plays":    s2_opener_plays,
            "set_closer_plays":     set_closer_plays,
            "sphere_plays":         sphere_plays,
            "arena_plays":          arena_plays,
        })

    feat_df = pd.DataFrame(rows).set_index("song_id")
    return feat_df


def build_cluster_feature_matrix(songs_df: pd.DataFrame, feat_df: pd.DataFrame) -> tuple:
    """
    Build the feature matrix used for K-Means clustering.
    Combines static catalog features with dynamic historical features.

    Returns (X_scaled DataFrame, fitted StandardScaler).
    """
    from sklearn.preprocessing import StandardScaler

    X = pd.DataFrame(index=songs_df.index)
    X["jam_score"]     = songs_df["jam_score"]
    X["energy"]        = songs_df["energy"]
    X["sphere_affinity"] = songs_df["sphere_affinity"]
    X["s1_frac"]       = songs_df["s1_frac"]
    X["s2_frac"]       = songs_df["s2_frac"]
    X["enc_frac"]      = songs_df["enc_frac"]
    X["log_avg_gap"]   = np.log1p(songs_df["avg_gap"])
    X["avg_duration"]  = songs_df["avg_duration_min"] / songs_df["avg_duration_min"].max()

    # Dynamic features (blend catalog defaults if song never played)
    X["gap_z"]         = feat_df["gap_z_score"].clip(-3, 3)
    X["s1_act_frac"]   = feat_df["set1_actual_frac"]
    X["s2_act_frac"]   = feat_df["set2_actual_frac"]
    X["enc_act_frac"]  = feat_df["enc_actual_frac"]

    X = X.fillna(0.0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return X_scaled, scaler
