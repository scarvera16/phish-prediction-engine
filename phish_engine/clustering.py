"""
K-Means song clustering with silhouette-based k selection.

Groups songs into behavioural archetypes that inform the composite scorer's
cluster-diversity term: e.g., "High-Energy Rockers", "Jam Vehicles",
"Bustout Rarities", "Crowd Favorites", etc.

Combines V2's adaptive k-selection and enriched features with V1's
cluster diversity bonus and member lookup functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples


DEFAULT_CLUSTER_NAMES = {
    0: "Ambient Explorers",
    1: "High-Energy Rockers",
    2: "Groove Anchors",
    3: "Bustout Rarities",
    4: "Crowd Favorites",
    5: "Jam Vehicles",
}

# Fallback names for V1-style fixed k=8
_CLUSTER_NAMES_8 = {
    0: "Set-2 Jam Vehicles",
    1: "High-Energy Openers",
    2: "Set-1 Showcase Pieces",
    3: "Funk & Groove",
    4: "Mellow / Ballads",
    5: "Prog Epics",
    6: "Encore / Closers",
    7: "Newer / Ambient Jams",
}


# ---------------------------------------------------------------------------
# V2: Enriched feature builder for clustering
# ---------------------------------------------------------------------------

def build_song_features(
    songs_df: pd.DataFrame,
    setlist_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build enriched feature matrix for clustering.

    Features (9 total):
      energy, jam_potential, avg_duration_min, tempo_norm,
      typical_set_enc, play_frequency, avg_gap_norm, gap_cv, debut_year_norm
    """
    play_counts = setlist_df.groupby("song_id").size().rename("obs_play_count")

    date_range_days = (setlist_df["date"].max() - setlist_df["date"].min()).days
    years = max(date_range_days / 365.25, 1.0)

    features = songs_df.copy()
    features = features.join(play_counts, how="left")
    features["obs_play_count"] = features["obs_play_count"].fillna(0)
    features["play_frequency"] = features["obs_play_count"] / years

    # Tempo normalized to [0, 1]
    if "tempo_bpm" in features.columns:
        tmin, tmax = features["tempo_bpm"].min(), features["tempo_bpm"].max()
        features["tempo_norm"] = (features["tempo_bpm"] - tmin) / max(tmax - tmin, 1.0)
    else:
        # Fallback: use energy as proxy for tempo
        features["tempo_norm"] = features["energy"]

    # Gap features
    if "avg_gap_days" in features.columns:
        features["avg_gap_norm"] = features["avg_gap_days"] / 365.0
        features["gap_cv"] = features["std_gap_days"] / features["avg_gap_days"].replace(0, 1.0)
    else:
        features["avg_gap_norm"] = features["avg_gap"] / 50.0
        features["gap_cv"] = 0.5  # default
    features["gap_cv"] = features["gap_cv"].clip(0, 5)

    # Typical set encoding
    if "typical_set" in features.columns:
        features["typical_set_enc"] = (features["typical_set"] - 1) / 2.0
    else:
        # Derive from slot weights
        features["typical_set_enc"] = features["s2_frac"]

    # Debut year normalized
    features["debut_year_norm"] = (features["debut_year"] - features["debut_year"].min()) / \
                                   max(features["debut_year"].max() - features["debut_year"].min(), 1.0)

    # Use jam_score as jam_potential if needed
    if "jam_potential" not in features.columns:
        features["jam_potential"] = features.get("jam_score", 0.5)

    feature_cols = [
        "energy",
        "jam_potential",
        "avg_duration_min",
        "tempo_norm",
        "typical_set_enc",
        "play_frequency",
        "avg_gap_norm",
        "gap_cv",
        "debut_year_norm",
    ]

    return features, feature_cols


# ---------------------------------------------------------------------------
# V2: Silhouette-based k selection
# ---------------------------------------------------------------------------

def select_k_by_silhouette(
    X_scaled: np.ndarray,
    k_range: range = range(3, 10),
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """
    Evaluate silhouette scores for each k in k_range.
    Returns (best_k, {k: silhouette_score}).
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20, max_iter=500)
        km.fit(X_scaled)
        if len(set(km.labels_)) < 2:
            scores[k] = -1.0
            continue
        s = silhouette_score(X_scaled, km.labels_, sample_size=min(500, len(X_scaled)))
        scores[k] = float(s)

    best_k = max(scores, key=scores.get)
    return best_k, scores


# ---------------------------------------------------------------------------
# Main clustering pipeline (V2 with V1 compatibility)
# ---------------------------------------------------------------------------

def cluster_songs(
    songs_df: pd.DataFrame,
    setlist_df: pd.DataFrame,
    n_clusters: int = 6,
    auto_select_k: bool = True,
    k_range: range = range(3, 10),
    random_state: int = 42,
) -> tuple[pd.DataFrame, StandardScaler, KMeans, PCA, np.ndarray, dict[int, float]]:
    """
    Fit k-means on enriched song features.

    If auto_select_k=True, runs silhouette analysis and picks optimal k.
    If auto_select_k=False, uses n_clusters directly.

    Returns:
        songs_with_clusters (pd.DataFrame)  - songs_df + cluster_id column
        scaler (StandardScaler)             - fitted scaler
        kmeans (KMeans)                     - fitted model
        pca (PCA)                           - 2-component PCA
        pca_coords (np.ndarray)             - (n, 2) coords
        silhouette_scores (dict)            - {k: score} for each tested k
    """
    features_df, feature_cols = build_song_features(songs_df, setlist_df)

    X = features_df[feature_cols].values.astype(float)
    # Impute NaNs with column mean
    col_means = np.nanmean(X, axis=0)
    nan_idx = np.isnan(X)
    X[nan_idx] = np.take(col_means, np.where(nan_idx)[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K selection
    if auto_select_k:
        best_k, sil_scores = select_k_by_silhouette(X_scaled, k_range, random_state)
        # Bias toward interpretable k=6 if silhouette diff is small (<0.02)
        target_k = 6
        if abs(sil_scores.get(best_k, 0) - sil_scores.get(target_k, -1)) < 0.02:
            best_k = target_k
        n_clusters = best_k
    else:
        sil_scores = {}

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, max_iter=500)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)

    songs_out = songs_df.copy()
    songs_out["cluster_id"] = cluster_labels

    # Remap cluster IDs to canonical names based on centroid characteristics
    if n_clusters <= len(DEFAULT_CLUSTER_NAMES):
        songs_out = _remap_cluster_ids(songs_out, kmeans.cluster_centers_, feature_cols, n_clusters)

    return songs_out, scaler, kmeans, pca, pca_coords, sil_scores


# ---------------------------------------------------------------------------
# V1 convenience wrapper
# ---------------------------------------------------------------------------

def train_song_clusters(
    X_scaled: pd.DataFrame,
    n_clusters: int = 8,
    seed: int = 42,
) -> tuple:
    """
    Fit K-Means on a pre-scaled feature matrix (V1 interface).

    Returns (kmeans, labels_dict, cluster_names_dict).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20, max_iter=500)
    kmeans.fit(X_scaled.values)

    labels = {song_id: int(label) for song_id, label in zip(X_scaled.index, kmeans.labels_)}

    # Auto-name clusters by dominant feature signature
    cluster_names = _auto_name_clusters(kmeans, X_scaled)

    return kmeans, labels, cluster_names


# ---------------------------------------------------------------------------
# V1: Cluster diversity bonus (used by scoring.score_all_songs)
# ---------------------------------------------------------------------------

def compute_cluster_diversity_bonus(
    song_id: str,
    cluster_labels: dict,
    already_chosen: list,
    songs_df: pd.DataFrame,
) -> float:
    """
    Return a 0-1 bonus rewarding cluster diversity in the emerging setlist.

    If the song's cluster is under-represented relative to already-chosen
    songs, it gets a higher bonus. Encourages variety across the set.
    """
    if not already_chosen:
        return 0.5  # neutral at start

    my_cluster = cluster_labels.get(song_id, -1)
    if my_cluster == -1:
        return 0.5

    chosen_clusters = [cluster_labels.get(s, -1) for s in already_chosen]
    same_count = chosen_clusters.count(my_cluster)
    total_chosen = len(chosen_clusters)

    same_frac = same_count / max(total_chosen, 1)
    diversity_bonus = 1.0 - same_frac
    return float(np.clip(diversity_bonus, 0.0, 1.0))


# ---------------------------------------------------------------------------
# V1: Get cluster members
# ---------------------------------------------------------------------------

def get_cluster_members(
    cluster_id: int,
    labels: dict,
    songs_df: pd.DataFrame,
    top_n: int = 8,
) -> list:
    """Return up to top_n song names belonging to the given cluster."""
    members = [sid for sid, cid in labels.items() if cid == cluster_id]
    members.sort(key=lambda s: songs_df.loc[s, "avg_gap"])
    return [songs_df.loc[s, "name"] for s in members[:top_n]]


# ---------------------------------------------------------------------------
# V2: Cluster balance weights
# ---------------------------------------------------------------------------

def get_cluster_weights(cluster_id: int, recent_cluster_ids: list, n_clusters: int = 6) -> dict:
    """
    Compute cluster balance weights from the last few shows.
    Underrepresented clusters get a higher weight to encourage setlist variety.
    """
    counts = {c: 0 for c in range(n_clusters)}
    for cid in recent_cluster_ids:
        if isinstance(cid, (int, np.integer)) and int(cid) in counts:
            counts[int(cid)] += 1

    total = max(len(recent_cluster_ids), 1)
    weights = {}
    for c in range(n_clusters):
        observed = counts[c] / total
        expected = 1.0 / n_clusters
        ratio = expected / max(observed, expected * 0.1)
        weights[c] = float(min(max(ratio, 0.5), 2.0))

    return weights


# ---------------------------------------------------------------------------
# V2: Per-song silhouette coefficients
# ---------------------------------------------------------------------------

def get_per_song_silhouette(
    songs_df: pd.DataFrame,
    setlist_df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.Series:
    """
    Compute per-song silhouette coefficients for cluster visualization.
    Returns pd.Series indexed by song_id.
    """
    features_df, feature_cols = build_song_features(songs_df, setlist_df)
    X = features_df[feature_cols].values.astype(float)
    col_means = np.nanmean(X, axis=0)
    nan_idx = np.isnan(X)
    X[nan_idx] = np.take(col_means, np.where(nan_idx)[1])
    X_scaled = scaler.transform(X)

    labels = songs_df["cluster_id"].values
    sil_vals = silhouette_samples(X_scaled, labels)
    return pd.Series(sil_vals, index=songs_df.index)


# ---------------------------------------------------------------------------
# Internal: cluster naming heuristics
# ---------------------------------------------------------------------------

def _auto_name_clusters(kmeans: KMeans, X_scaled: pd.DataFrame) -> dict:
    """Assign human-readable names to clusters based on centroid positions."""
    feature_names = X_scaled.columns.tolist()
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)

    names = {}
    for cid in range(kmeans.n_clusters):
        c = centroids.iloc[cid]
        if c.get("enc_frac", 0) > 0.8 or c.get("enc_act_frac", 0) > 0.6:
            names[cid] = "Encore / Closers"
        elif c.get("s2_frac", 0) > 0.7 and c.get("jam_score", 0) > 0.5:
            if c.get("sphere_affinity", 0) > 0.6:
                names[cid] = "Ambient / Sphere Jams"
            elif c.get("energy", 0) > 0.5:
                names[cid] = "Set-2 Jam Vehicles"
            else:
                names[cid] = "Funky / Groove Jams"
        elif c.get("s1_frac", 0) > 0.6 and c.get("energy", 0) > 0.6:
            if c.get("log_avg_gap", 0) < 0:
                names[cid] = "High-Energy Openers"
            else:
                names[cid] = "Set-1 Rockers"
        elif c.get("avg_duration", 0) > 0.5 and c.get("s2_frac", 0) > 0.3:
            names[cid] = "Prog Epics"
        elif c.get("energy", 0) < -0.3:
            names[cid] = "Mellow / Ballads"
        elif c.get("jam_score", 0) > 0.3 and c.get("sphere_affinity", 0) > 0.5:
            names[cid] = "Newer / Ambient Jams"
        else:
            names[cid] = f"Cluster {cid}"

    return names


def _remap_cluster_ids(
    songs_df: pd.DataFrame,
    centroids: np.ndarray,
    feature_cols: list[str],
    n_clusters: int,
) -> pd.DataFrame:
    """Map raw k-means cluster IDs to semantic names via centroid analysis."""
    idx = {f: i for i, f in enumerate(feature_cols)}
    energy_i   = idx.get("energy", 0)
    jam_i      = idx.get("jam_potential", 1)
    gap_i      = idx.get("avg_gap_norm", 6)
    freq_i     = idx.get("play_frequency", 5)
    gap_cv_i   = idx.get("gap_cv", 7)

    cluster_chars = []
    for c in range(n_clusters):
        centroid = centroids[c]
        cluster_chars.append({
            "orig":   c,
            "energy": centroid[energy_i],
            "jam":    centroid[jam_i],
            "gap":    centroid[gap_i],
            "freq":   centroid[freq_i],
            "cv":     centroid[gap_cv_i],
        })

    assigned: dict[int, int] = {}
    used_names: set[int] = set()

    def assign(orig_id: int, name_id: int):
        assigned[orig_id] = name_id
        used_names.add(name_id)

    # 1. Bustout Rarities (3) = highest gap_cv + high avg_gap
    sorted_cv = sorted(cluster_chars, key=lambda x: x["cv"] + x["gap"], reverse=True)
    assign(sorted_cv[0]["orig"], 3)

    # 2. Jam Vehicles (5) = highest jam among unassigned
    sorted_jam = sorted(cluster_chars, key=lambda x: x["jam"], reverse=True)
    for item in sorted_jam:
        if item["orig"] not in assigned:
            assign(item["orig"], 5)
            break

    # 3. High-Energy Rockers (1) = highest energy among unassigned
    sorted_e = sorted(cluster_chars, key=lambda x: x["energy"], reverse=True)
    for item in sorted_e:
        if item["orig"] not in assigned:
            assign(item["orig"], 1)
            break

    # 4. Crowd Favorites (4) = highest play frequency among unassigned
    sorted_f = sorted(cluster_chars, key=lambda x: x["freq"], reverse=True)
    for item in sorted_f:
        if item["orig"] not in assigned:
            assign(item["orig"], 4)
            break

    # 5. Ambient Explorers (0) = lowest energy among unassigned
    sorted_e_asc = sorted(cluster_chars, key=lambda x: x["energy"])
    for item in sorted_e_asc:
        if item["orig"] not in assigned:
            assign(item["orig"], 0)
            break

    # 6. Groove Anchors (2) = last remaining
    for item in cluster_chars:
        if item["orig"] not in assigned:
            assign(item["orig"], 2)

    songs_out = songs_df.copy()
    songs_out["cluster_id"] = songs_out["cluster_id"].map(assigned)
    return songs_out
