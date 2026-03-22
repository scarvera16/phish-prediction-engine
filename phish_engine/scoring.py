"""
Composite scoring with tunable ScoringWeights.

Score = sum(w_i * f_i(song, slot, context))

Components (all normalized to [0, 1]):
  1. recency        - how long since last play (exponential saturation)
  2. gap_pressure   - how overdue relative to historical average (log-normal CDF)
  3. slot_affinity  - catalog weight for this slot type
  4. frequency      - play rate with multi-window blending
  5. venue_affinity - Sphere / arena / outdoor multiplier (Laplace-smoothed)
  6. cluster_diversity - rewards under-represented song archetypes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import lognorm
from typing import Optional


# ---------------------------------------------------------------------------
# Weights container (from V2)
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """
    All tunable hyperparameters for the composite scorer.

    Main weights (6 components) are re-normalized at scoring time via
    main_weights_normalized() so they always form a valid convex combination.
    """
    # --- Main component weights (raw; normalized before use) ---
    recency:        float = 0.25
    frequency:      float = 0.15
    gap_pressure:   float = 0.25
    slot_affinity:  float = 0.20
    venue_affinity: float = 0.05
    cluster:        float = 0.10

    # --- Recency sub-params ---
    recency_decay_rate: float = 3.0   # lambda: score = 1 - exp(-lambda * ratio)

    # --- Frequency sub-params (window mix) ---
    freq_w10: float = 0.45   # weight on last-10-shows rate
    freq_w30: float = 0.35   # weight on last-30-shows rate
    freq_w90: float = 0.20   # weight on last-90-shows rate

    # --- Gap pressure sub-params ---
    gap_lognormal_sigma: float = 0.45  # shape of log-normal (higher = fatter tail)

    def main_weights_normalized(self) -> dict:
        """Return the 6 main weights normalized to sum to 1.0."""
        raw = {
            "recency":        self.recency,
            "frequency":      self.frequency,
            "gap_pressure":   self.gap_pressure,
            "slot_affinity":  self.slot_affinity,
            "venue_affinity": self.venue_affinity,
            "cluster":        self.cluster,
        }
        total = sum(raw.values())
        if total <= 0:
            total = 1.0
        return {k: v / total for k, v in raw.items()}

    @classmethod
    def from_softmax_vector(cls, x: np.ndarray) -> ScoringWeights:
        """
        Reconstruct weights from a raw optimization vector of length 9.

        x[0:6]  -> softmax -> 6 main weights
        x[6]    -> exp+clip -> recency_decay_rate  (1.0-7.0)
        x[7]    -> exp+clip -> gap_lognormal_sigma (0.15-1.5)
        x[8]    -> sigmoid  -> freq_w10 mix
        """
        logits = x[:6]
        exp_l = np.exp(logits - logits.max())
        main = exp_l / exp_l.sum()

        decay = float(np.clip(np.exp(x[6]), 1.0, 7.0))
        sigma = float(np.clip(np.exp(x[7]), 0.15, 1.5))
        w10   = float(1.0 / (1.0 + np.exp(-x[8])))
        rem   = 1.0 - w10
        w30   = rem * 0.6
        w90   = rem * 0.4

        return cls(
            recency=float(main[0]),
            frequency=float(main[1]),
            gap_pressure=float(main[2]),
            slot_affinity=float(main[3]),
            venue_affinity=float(main[4]),
            cluster=float(main[5]),
            recency_decay_rate=decay,
            gap_lognormal_sigma=sigma,
            freq_w10=w10,
            freq_w30=w30,
            freq_w90=w90,
        )

    def to_softmax_vector(self) -> np.ndarray:
        """Invert from_softmax_vector for warm-starting the optimizer."""
        nw = self.main_weights_normalized()
        logits = np.log(np.array([
            nw["recency"], nw["frequency"], nw["gap_pressure"],
            nw["slot_affinity"], nw["venue_affinity"], nw["cluster"]
        ]) + 1e-9)
        x6 = np.log(max(self.recency_decay_rate, 1e-3))
        x7 = np.log(max(self.gap_lognormal_sigma, 1e-3))
        w10_clip = float(np.clip(self.freq_w10, 1e-4, 1 - 1e-4))
        x8 = np.log(w10_clip / (1.0 - w10_clip))
        return np.concatenate([logits, [x6, x7, x8]])

    def describe(self) -> str:
        nw = self.main_weights_normalized()
        lines = [
            "ScoringWeights:",
            f"  recency={nw['recency']:.3f}  decay_rate={self.recency_decay_rate:.2f}",
            f"  frequency={nw['frequency']:.3f}  w10={self.freq_w10:.2f} w30={self.freq_w30:.2f} w90={self.freq_w90:.2f}",
            f"  gap_pressure={nw['gap_pressure']:.3f}  lognormal_sigma={self.gap_lognormal_sigma:.2f}",
            f"  slot_affinity={nw['slot_affinity']:.3f}",
            f"  venue_affinity={nw['venue_affinity']:.3f}",
            f"  cluster={nw['cluster']:.3f}",
        ]
        return "\n".join(lines)


# Default weights (used when none supplied)
DEFAULT_WEIGHTS = ScoringWeights()

# Slot type -> column in songs_df that holds the affinity weight (from V1)
_SLOT_WEIGHT_COL = {
    "show_opener":   "show_opener_weight",
    "s1_body":       "s1_weight",
    "s1_closer":     "set_closer_weight",
    "s2_opener":     "set2_opener_weight",
    "s2_body":       "s2_weight",
    "s2_closer":     "set_closer_weight",
    "encore":        "enc_weight",
}


# ---------------------------------------------------------------------------
# Component score functions (V2 algorithms)
# ---------------------------------------------------------------------------

def compute_recency_score(
    current_gap: float,
    avg_gap: float,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> float:
    """
    Exponential-saturation recency.

    score = 1 - exp(-decay_rate * (gap / avg_gap))

    At gap=0: score=0. At gap=avg_gap: score ~ 0.95 (decay_rate=3.0).
    """
    if avg_gap <= 0:
        avg_gap = 30.0
    if current_gap < 0:
        return 0.0
    ratio = current_gap / max(avg_gap, 1.0)
    score = 1.0 - np.exp(-weights.recency_decay_rate * ratio)
    return float(np.clip(score, 0.0, 1.0))


def compute_gap_pressure(
    days_since_last: float,
    avg_gap: float,
    std_gap: float,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> float:
    """
    Log-normal gap pressure CDF.

    Musical gaps are right-skewed: log-normal captures this better than normal.
    P(gap <= days_since_last | log-normal(mu, sigma)).
    """
    if avg_gap <= 0:
        avg_gap = 30.0
    if days_since_last <= 0:
        return 0.0

    sigma = max(weights.gap_lognormal_sigma, 0.05)
    mu = np.log(max(avg_gap, 1.0)) - (sigma ** 2) / 2.0
    score = lognorm.cdf(days_since_last, s=sigma, scale=np.exp(mu))
    return float(np.clip(score, 0.0, 1.0))


def compute_slot_affinity(
    song_id: str,
    slot_type: str,
    songs_df: pd.DataFrame,
    max_weights: dict,
) -> float:
    """
    Normalised slot affinity: song's weight for this slot / max weight for this slot.
    """
    col = _SLOT_WEIGHT_COL.get(slot_type, "s1_weight")
    song_w = float(songs_df.loc[song_id, col])
    max_w = max_weights.get(col, 1.0)
    return float(np.clip(song_w / max(max_w, 1e-6), 0.0, 1.0))


def compute_frequency_score(total_plays: int, total_shows: int) -> float:
    """
    Baseline play-rate, normalised by the maximum across the catalog.
    Keeps high-rotation songs (Chalk Dust, Sample) competitive.
    """
    if total_shows <= 0:
        return 0.0
    rate = total_plays / total_shows
    return float(np.clip(rate / 0.5, 0.0, 1.0))


def compute_venue_affinity(
    song_id: str,
    venue_type: str,
    songs_df: pd.DataFrame,
) -> float:
    """
    Venue-specific multiplier.
    For the Sphere, use sphere_affinity directly.
    For other venues, mild boost for high-energy songs in outdoor settings.
    """
    if venue_type == "sphere":
        return float(songs_df.loc[song_id, "sphere_affinity"])
    elif venue_type == "outdoor":
        energy = float(songs_df.loc[song_id, "energy"])
        return float(0.5 + 0.5 * energy)
    else:
        return 0.7


# ---------------------------------------------------------------------------
# Batch scoring (from V1 interface, using V2 algorithms)
# ---------------------------------------------------------------------------

def score_all_songs(
    slot_type: str,
    songs_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_labels: dict,
    already_chosen: list,
    excluded: set,
    venue_type: str,
    total_shows: int,
    weights: ScoringWeights | dict | None = None,
) -> pd.Series:
    """
    Compute composite scores for all eligible songs for a given slot.

    Parameters
    ----------
    slot_type      : one of 'show_opener','s1_body','s1_closer',
                     's2_opener','s2_body','s2_closer','encore'
    songs_df       : catalog DataFrame
    feat_df        : dynamic feature DataFrame (from features.py)
    cluster_labels : {song_id: cluster_id}
    already_chosen : songs already selected for this show
    excluded       : hard-excluded songs (already played in run, etc.)
    venue_type     : 'sphere' | 'arena' | 'outdoor' | 'theater'
    total_shows    : number of shows in training window
    weights        : ScoringWeights or legacy dict (defaults to DEFAULT_WEIGHTS)

    Returns
    -------
    pd.Series of composite scores indexed by song_id, sorted descending.
    """
    from .clustering import compute_cluster_diversity_bonus

    # Support both ScoringWeights and legacy dict
    if isinstance(weights, dict):
        w_obj = ScoringWeights(**{k: v for k, v in weights.items()
                                  if k in ScoringWeights.__dataclass_fields__})
    elif weights is None:
        w_obj = DEFAULT_WEIGHTS
    else:
        w_obj = weights

    nw = w_obj.main_weights_normalized()

    # Pre-compute max weights per slot column (for normalisation)
    col = _SLOT_WEIGHT_COL.get(slot_type, "s1_weight")
    max_slot_w = float(songs_df[col].max())

    scores = {}
    for song_id in songs_df.index:
        if song_id in already_chosen or song_id in excluded:
            continue

        f = feat_df.loc[song_id]

        rec   = compute_recency_score(f["current_gap_shows"], f["avg_gap_actual"], w_obj)
        gp    = compute_gap_pressure(f["current_gap_shows"], f["avg_gap_actual"], f["std_gap_actual"], w_obj)
        slot  = compute_slot_affinity(song_id, slot_type, songs_df, {col: max_slot_w})
        freq  = compute_frequency_score(f["total_plays"], total_shows)
        venue = compute_venue_affinity(song_id, venue_type, songs_df)
        clust = compute_cluster_diversity_bonus(song_id, cluster_labels, already_chosen, songs_df)

        composite = (
            nw["recency"]        * rec   +
            nw["gap_pressure"]   * gp    +
            nw["slot_affinity"]  * slot  +
            nw["frequency"]      * freq  +
            nw["venue_affinity"] * venue +
            nw["cluster"]        * clust
        )
        scores[song_id] = composite

    return pd.Series(scores).sort_values(ascending=False)


def score_breakdown(
    song_id: str,
    slot_type: str,
    songs_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_labels: dict,
    already_chosen: list,
    venue_type: str,
    total_shows: int,
    weights: ScoringWeights | dict | None = None,
) -> dict:
    """Return labelled component scores for a single song (for display)."""
    from .clustering import compute_cluster_diversity_bonus

    if isinstance(weights, dict):
        w_obj = ScoringWeights(**{k: v for k, v in weights.items()
                                  if k in ScoringWeights.__dataclass_fields__})
    elif weights is None:
        w_obj = DEFAULT_WEIGHTS
    else:
        w_obj = weights

    nw = w_obj.main_weights_normalized()
    col = _SLOT_WEIGHT_COL.get(slot_type, "s1_weight")
    max_slot_w = float(songs_df[col].max())
    f = feat_df.loc[song_id]

    rec   = compute_recency_score(f["current_gap_shows"], f["avg_gap_actual"], w_obj)
    gp    = compute_gap_pressure(f["current_gap_shows"], f["avg_gap_actual"], f["std_gap_actual"], w_obj)
    slot  = compute_slot_affinity(song_id, slot_type, songs_df, {col: max_slot_w})
    freq  = compute_frequency_score(f["total_plays"], total_shows)
    venue = compute_venue_affinity(song_id, venue_type, songs_df)
    clust = compute_cluster_diversity_bonus(song_id, cluster_labels, already_chosen, songs_df)

    composite = (
        nw["recency"]        * rec   +
        nw["gap_pressure"]   * gp    +
        nw["slot_affinity"]  * slot  +
        nw["frequency"]      * freq  +
        nw["venue_affinity"] * venue +
        nw["cluster"]        * clust
    )
    return {
        "composite":         round(composite, 4),
        "recency":           round(rec, 3),
        "gap_pressure":      round(gp, 3),
        "slot_affinity":     round(slot, 3),
        "frequency":         round(freq, 3),
        "venue_affinity":    round(venue, 3),
        "cluster_diversity": round(clust, 3),
    }


# ---------------------------------------------------------------------------
# Venue index builder (from V2)
# ---------------------------------------------------------------------------

def build_venue_plays_index(setlist_df: pd.DataFrame, shows_df: pd.DataFrame) -> dict:
    """Build {venue: {song_id: count, '__total_shows__': n}} lookup."""
    index = {}
    venue_show_counts = shows_df.groupby("venue")["show_id"].nunique()

    setlist_slim = setlist_df[["show_id", "song_id"]].copy()
    shows_slim = shows_df[["show_id", "venue"]].rename(columns={"venue": "venue_name"})
    merged = setlist_slim.merge(shows_slim, on="show_id", how="left")
    song_venue_counts = merged.groupby(["venue_name", "song_id"]).size()

    for venue, total in venue_show_counts.items():
        index[venue] = {"__total_shows__": int(total)}
    for (venue, sid), count in song_venue_counts.items():
        if venue not in index:
            index[venue] = {"__total_shows__": 1}
        index[venue][sid] = int(count)

    return index
