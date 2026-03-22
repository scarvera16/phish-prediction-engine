"""
Bayesian song probability model for Phish setlist prediction.

Computes P(song | slot_role, show_context) by combining:
  1. Temporal base rate (time-decayed play frequency)
  2. Sphere venue prior (2024 Sphere run as direct template)
  3. Gap pressure (log-normal CDF on show-count gaps)
  4. Song role probabilities (opener/closer/mid/segue distributions)
  5. Segue compatibility matrix (which songs flow into which)
  6. Mandatory pairings (Mike's > Weekapaug, Horse > Silent, etc.)
  7. Era diversity (Classic/CowFunk/Island/Hiatus/Modern)
  8. Bustout / special occasion model
  9. Cover song & never-before-played cover probability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from scipy.stats import lognorm
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Era classification
# ---------------------------------------------------------------------------

ERAS = {
    "classic":       (1983, 1992),
    "cow_funk":      (1993, 1996),
    "island":        (1997, 2004),
    "hiatus_return": (2009, 2014),
    "modern":        (2015, 2030),
}


def classify_era(debut_year: int) -> str:
    for era_name, (start, end) in ERAS.items():
        if start <= debut_year <= end:
            return era_name
    if debut_year < 1983:
        return "classic"
    if 2005 <= debut_year <= 2008:
        return "island"
    return "modern"


# ---------------------------------------------------------------------------
# Transition code semantics
# ---------------------------------------------------------------------------

SEGUE_TRANSITIONS = {2, 3}
SET1_CLOSER_TRANS = 4
SET2_CLOSER_TRANS = 5
SHOW_CLOSER_TRANS = 6

ROLES = [
    "set1_opener", "set1_closer", "set1_mid",
    "set2_opener", "set2_closer", "set2_mid",
    "encore_opener", "show_closer", "encore_mid",
]


# ---------------------------------------------------------------------------
# SongModel: precomputed probability tables from historical data
# ---------------------------------------------------------------------------

@dataclass
class SongModel:
    """All precomputed song probability tables."""

    base_rates: Dict[str, float] = field(default_factory=dict)
    sphere_prior: Dict[str, float] = field(default_factory=dict)
    gap_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    role_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    segue_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    segue_out_rate: Dict[str, float] = field(default_factory=dict)
    segue_in_rate: Dict[str, float] = field(default_factory=dict)
    mandatory_pairs: List[Tuple[str, str, str]] = field(default_factory=list)
    song_eras: Dict[str, str] = field(default_factory=dict)
    is_cover: Dict[str, bool] = field(default_factory=dict)
    covers_per_show: float = 0.0
    new_cover_rate_per_show: float = 0.0
    venue_cover_rates: Dict[str, float] = field(default_factory=dict)
    song_names: Dict[str, str] = field(default_factory=dict)
    avg_duration_sec: Dict[str, float] = field(default_factory=dict)
    jam_potential: Dict[str, float] = field(default_factory=dict)
    energy: Dict[str, float] = field(default_factory=dict)
    run_appearance_rate: Dict[str, float] = field(default_factory=dict)
    all_song_ids: Set[str] = field(default_factory=set)


def build_song_model(
    setlist_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    songs_api: List[dict],
    temporal_decay_lambda: float = 1.5,
    sphere_alpha: float = 0.3,
    mandatory_pair_threshold: float = 0.75,
    min_segue_count: int = 3,
    enrichment_data: Optional[dict] = None,
) -> SongModel:
    """
    Build the complete song probability model from real data.

    Args:
        setlist_df: Enriched setlist entries
        shows_df: Show metadata
        songs_api: Raw song catalog from Phish.net API
        temporal_decay_lambda: Yearly decay rate for base rate
        sphere_alpha: Blend weight for Sphere 2024 prior
        mandatory_pair_threshold: P(B|A) threshold for mandatory pairs
        min_segue_count: Minimum segue observations to include in matrix
        enrichment_data: Optional dict with keys 'avg_duration_min', 'jamchart_rate'
                         indexed by song_id (replaces fetch_phishin_data import)
    """
    model = SongModel()

    df = setlist_df[~setlist_df["isreprise"]].copy()
    if df.empty:
        return model

    model.all_song_ids = set(df["song_id"].unique())

    songs_index = {}
    for s in songs_api:
        sid = s.get("slug", "").replace("-", "_")
        if sid:
            songs_index[sid] = s

    model.base_rates = _compute_base_rates(df, shows_df, temporal_decay_lambda)
    model.sphere_prior = _compute_sphere_prior(df, sphere_alpha)
    model.gap_stats = _compute_gap_stats(df, songs_index)
    model.role_probs = _compute_role_probs(df)
    model.segue_matrix, model.segue_out_rate, model.segue_in_rate = \
        _compute_segue_data(df, min_segue_count)
    model.mandatory_pairs = _detect_mandatory_pairs(df, mandatory_pair_threshold)
    model.song_eras = _compute_eras(df, songs_index)

    # Cover model
    phish_artists = {"phish", "trey anastasio", "mike gordon", "page mcconnell",
                     "jon fishman", "trey anastasio, tom marshall",
                     "anastasio/marshall", "phish, inc."}
    for s in songs_api:
        sid = s.get("slug", "").replace("-", "_")
        artist = s.get("artist", "").lower().strip()
        model.is_cover[sid] = artist not in phish_artists and bool(artist)
    cover_stats = _compute_cover_stats(df, songs_index, model.is_cover)
    model.covers_per_show = cover_stats["covers_per_show"]
    model.new_cover_rate_per_show = cover_stats["new_cover_rate_per_show"]
    model.venue_cover_rates = _compute_venue_cover_rates(df, model.is_cover)

    # Run appearance rate
    if "run_name" in shows_df.columns:
        model.run_appearance_rate = _compute_run_appearance_rate(df, shows_df)

    # Song names
    model.song_names = {sid: row["song_name"]
                        for sid, row in df.drop_duplicates("song_id").set_index("song_id").iterrows()}

    # Duration and jam potential
    dur_means_phishnet = df[df["tracktime_sec"] > 0].groupby("song_id")["tracktime_sec"].mean().to_dict()

    if enrichment_data:
        dur_means = {}
        for sid in model.all_song_ids:
            if sid in enrichment_data and enrichment_data[sid].get("avg_duration_min", 0) > 0:
                dur_means[sid] = enrichment_data[sid]["avg_duration_min"] * 60.0
            elif sid in dur_means_phishnet:
                dur_means[sid] = dur_means_phishnet[sid]
            else:
                dur_means[sid] = 300.0
    else:
        dur_means = dur_means_phishnet
    model.avg_duration_sec = dur_means

    jc_rates_phishnet = df.groupby("song_id")["isjamchart"].mean()
    play_counts = df.groupby("song_id").size()
    for sid in model.all_song_ids:
        if enrichment_data and sid in enrichment_data:
            jc = float(enrichment_data[sid].get("jamchart_rate", 0.0))
        else:
            jc = jc_rates_phishnet.get(sid, 0.0)
        dur = dur_means.get(sid, 300)
        model.jam_potential[sid] = min(10.0, max(1.0, jc * 8.0 + (dur / 60.0) / 3.0))
        freq = play_counts.get(sid, 0) / max(len(shows_df), 1)
        model.energy[sid] = min(10.0, max(1.0, model.jam_potential[sid] * 0.6 + freq * 15.0))

    return model


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def _compute_base_rates(df, shows_df, decay_lambda):
    latest_date = df["date"].max()
    show_dates = shows_df.set_index("show_id")["date"].to_dict()
    show_weights = {}
    for sid, dt in show_dates.items():
        days = (latest_date - dt).days
        show_weights[sid] = np.exp(-decay_lambda * days / 365.0)
    total_weight = sum(show_weights.values()) or 1.0
    song_weighted = defaultdict(float)
    for _, row in df.iterrows():
        w = show_weights.get(row["show_id"], 0.0)
        song_weighted[row["song_id"]] += w
    return {sid: wc / total_weight for sid, wc in song_weighted.items()}


def _compute_sphere_prior(df, alpha):
    sphere_mask = df["venue"].str.contains("Sphere", case=False, na=False)
    sphere_df = df[sphere_mask]
    if sphere_df.empty:
        return {}
    n_shows = sphere_df["show_id"].nunique()
    sphere_counts = sphere_df.groupby("song_id").size()
    return {sid: (count / n_shows) * alpha for sid, count in sphere_counts.items()}


def _compute_gap_stats(df, songs_index):
    stats = {}
    for song_id, grp in df.groupby("song_id"):
        gaps = grp["gap"].dropna().values
        gaps = gaps[gaps > 0].astype(float)
        if len(gaps) >= 3:
            median_gap = float(np.median(gaps))
            log_gaps = np.log(gaps)
            sigma = float(np.std(log_gaps)) if np.std(log_gaps) > 0.05 else 0.5
        elif len(gaps) >= 1:
            median_gap = float(np.median(gaps))
            sigma = 0.5
        else:
            api_song = songs_index.get(song_id, {})
            times = api_song.get("times_played", 10)
            median_gap = max(2800.0 / max(times, 1), 2.0)
            sigma = 0.6
        api_song = songs_index.get(song_id, {})
        current_gap = api_song.get("gap", 0) or 0
        stats[song_id] = {"median": median_gap, "sigma": sigma, "current_gap": float(current_gap)}
    return stats


def _compute_role_probs(df):
    role_counts = defaultdict(Counter)
    total_role_counts = Counter()
    for show_id, show_grp in df.groupby("show_id"):
        for set_num in [1, 2, 3]:
            set_entries = show_grp[show_grp["set_number"] == set_num].sort_values("position")
            if set_entries.empty:
                continue
            entries_list = list(set_entries.itertuples())
            for i, entry in enumerate(entries_list):
                sid = entry.song_id
                is_first = (i == 0)
                is_last = (i == len(entries_list) - 1)
                trans = entry.transition
                if set_num == 1:
                    if is_first:
                        role = "set1_opener"
                    elif is_last or trans == SET1_CLOSER_TRANS:
                        role = "set1_closer"
                    else:
                        role = "set1_mid"
                elif set_num == 2:
                    if is_first:
                        role = "set2_opener"
                    elif is_last or trans == SET2_CLOSER_TRANS:
                        role = "set2_closer"
                    else:
                        role = "set2_mid"
                else:
                    if is_first:
                        role = "encore_opener"
                    elif is_last or trans == SHOW_CLOSER_TRANS:
                        role = "show_closer"
                    else:
                        role = "encore_mid"
                role_counts[sid][role] += 1
                total_role_counts[role] += 1
    role_probs = {}
    for sid, counts in role_counts.items():
        total_appearances = sum(counts.values())
        probs = {}
        for role in ROLES:
            if total_role_counts[role] > 0:
                probs[role] = counts.get(role, 0) / total_appearances
            else:
                probs[role] = 0.0
        probs["_total"] = total_appearances
        role_probs[sid] = probs
    return role_probs


def _compute_segue_data(df, min_count):
    segue_pairs = Counter()
    segue_out_counts = Counter()
    segue_in_counts = Counter()
    total_appearances = Counter()
    for show_id, show_grp in df.groupby("show_id"):
        for set_num in [1, 2, 3]:
            set_entries = show_grp[show_grp["set_number"] == set_num].sort_values("position")
            entries_list = list(set_entries.itertuples())
            for i in range(len(entries_list)):
                sid = entries_list[i].song_id
                trans = entries_list[i].transition
                total_appearances[sid] += 1
                if trans in SEGUE_TRANSITIONS and i + 1 < len(entries_list):
                    next_sid = entries_list[i + 1].song_id
                    segue_pairs[(sid, next_sid)] += 1
                    segue_out_counts[sid] += 1
                    segue_in_counts[next_sid] += 1
    matrix = defaultdict(dict)
    for (from_sid, to_sid), count in segue_pairs.items():
        if count >= min_count:
            from_total = segue_out_counts[from_sid]
            matrix[from_sid][to_sid] = count / from_total
    out_rates = {sid: segue_out_counts.get(sid, 0) / max(total_appearances[sid], 1)
                 for sid in total_appearances}
    in_rates = {sid: segue_in_counts.get(sid, 0) / max(total_appearances[sid], 1)
                for sid in total_appearances}
    return dict(matrix), out_rates, in_rates


def _detect_mandatory_pairs(df, threshold):
    show_songs = defaultdict(set)
    show_song_positions = defaultdict(dict)
    for _, row in df.iterrows():
        show_songs[row["show_id"]].add(row["song_id"])
        show_song_positions[row["show_id"]][row["song_id"]] = row["position"]
    song_shows = defaultdict(set)
    for show_id, songs in show_songs.items():
        for sid in songs:
            song_shows[sid].add(show_id)
    pairs = []
    checked = set()
    frequent_songs = {sid for sid, shows in song_shows.items() if len(shows) >= 5}
    for song_a in frequent_songs:
        shows_a = song_shows[song_a]
        for song_b in frequent_songs:
            if song_a >= song_b:
                continue
            if (song_a, song_b) in checked:
                continue
            checked.add((song_a, song_b))
            shows_b = song_shows[song_b]
            co_shows = shows_a & shows_b
            p_b_given_a = len(co_shows) / len(shows_a)
            p_a_given_b = len(co_shows) / len(shows_b)
            if p_b_given_a > threshold or p_a_given_b > threshold:
                a_before_b = 0
                b_before_a = 0
                for show_id in co_shows:
                    pos_a = show_song_positions[show_id].get(song_a, 0)
                    pos_b = show_song_positions[show_id].get(song_b, 0)
                    if pos_a < pos_b:
                        a_before_b += 1
                    else:
                        b_before_a += 1
                if a_before_b > b_before_a * 2:
                    direction = "a_before_b"
                elif b_before_a > a_before_b * 2:
                    direction = "b_before_a"
                else:
                    direction = "either"
                pairs.append((song_a, song_b, direction))
    return pairs


def _compute_eras(df, songs_index):
    eras = {}
    for song_id in df["song_id"].unique():
        api_song = songs_index.get(song_id, {})
        debut_str = api_song.get("debut", "")
        if debut_str and len(debut_str) >= 4:
            debut_year = int(debut_str[:4])
        else:
            debut_year = 1990
        eras[song_id] = classify_era(debut_year)
    return eras


def _compute_run_appearance_rate(df, shows_df):
    run_sizes = shows_df.groupby("run_name")["show_id"].count()
    multi_runs = run_sizes[run_sizes >= 3].index
    if len(multi_runs) == 0:
        return {}
    run_song_sets = {}
    for run_name in multi_runs:
        run_show_ids = shows_df[shows_df["run_name"] == run_name]["show_id"]
        run_songs = set(df[df["show_id"].isin(run_show_ids)]["song_id"])
        if run_songs:
            run_song_sets[run_name] = run_songs
    n_runs = len(run_song_sets)
    if n_runs == 0:
        return {}
    song_counts = Counter()
    for run_songs in run_song_sets.values():
        for sid in run_songs:
            song_counts[sid] += 1
    return {sid: count / n_runs for sid, count in song_counts.items()}


def _compute_cover_stats(df, songs_index, is_cover):
    n_shows = df["show_id"].nunique()
    if n_shows == 0:
        return {"covers_per_show": 0.0, "new_cover_rate_per_show": 0.0}
    cover_count = sum(1 for _, row in df.iterrows() if is_cover.get(row["song_id"], False))
    avg_covers = cover_count / n_shows
    new_cover_count = 0
    for _, row in df.iterrows():
        if not is_cover.get(row["song_id"], False):
            continue
        api_song = songs_index.get(row["song_id"], {})
        debut = api_song.get("debut", "")
        if debut and str(row["date"].date()) == debut:
            new_cover_count += 1
    return {"covers_per_show": avg_covers, "new_cover_rate_per_show": new_cover_count / n_shows}


def _compute_venue_cover_rates(df, is_cover):
    if "venue" not in df.columns:
        return {}
    show_venues = {}
    for _, row in df.drop_duplicates("show_id").iterrows():
        v = str(row.get("venue", "")).lower()
        if "sphere" in v:
            vtype = "sphere"
        elif "madison square" in v:
            vtype = "msg"
        elif "dick" in v:
            vtype = "dicks"
        else:
            vtype = "other"
        show_venues[row["show_id"]] = vtype
    venue_covers = defaultdict(int)
    venue_shows = defaultdict(set)
    for _, row in df.iterrows():
        vtype = show_venues.get(row["show_id"], "other")
        venue_shows[vtype].add(row["show_id"])
        if is_cover.get(row["song_id"], False):
            venue_covers[vtype] += 1
    rates = {}
    for vtype, show_ids in venue_shows.items():
        n = len(show_ids)
        if n > 0:
            rates[vtype] = venue_covers[vtype] / n
    return rates


# ---------------------------------------------------------------------------
# Scoring functions (used by set_builder)
# ---------------------------------------------------------------------------

def gap_pressure(current_gap: float, median_gap: float, sigma: float) -> float:
    """Log-normal CDF gap pressure on show-count gaps."""
    if median_gap <= 0:
        median_gap = 10.0
    if current_gap <= 0:
        return 0.0
    mu = np.log(max(median_gap, 1.0)) - (sigma ** 2) / 2.0
    score = lognorm.cdf(current_gap, s=sigma, scale=np.exp(mu))
    return float(np.clip(score, 0.0, 1.0))


def role_fit_score(song_id: str, role: str, model: SongModel) -> float:
    """P(role | song) with confidence weighting."""
    probs = model.role_probs.get(song_id, {})
    if not probs:
        return 0.1
    total = probs.get("_total", 1)
    raw_prob = probs.get(role, 0.0)
    confidence = min(total / 30.0, 1.0)
    uniform = 1.0 / len(ROLES)
    blended = confidence * raw_prob + (1 - confidence) * uniform
    if total >= 10:
        dominant_role = max([(r, probs.get(r, 0.0)) for r in ROLES], key=lambda x: x[1])
        if dominant_role[1] > 0.6 and dominant_role[0] != role:
            _role_family = {
                "set1_opener": "set1", "set1_closer": "set1", "set1_mid": "set1",
                "set2_opener": "set2", "set2_closer": "set2", "set2_mid": "set2",
                "encore_opener": "encore", "show_closer": "encore", "encore_mid": "encore",
            }
            dom_family = _role_family.get(dominant_role[0], "?")
            target_family = _role_family.get(role, "?")
            if dom_family != target_family:
                blended *= 0.1
    return blended


def bustout_boost(
    current_gap: float, median_gap: float,
    venue_type: str = "normal", run_length: int = 1,
) -> float:
    """Multiplicative boost for overdue songs, amplified at special venues."""
    if median_gap <= 0:
        median_gap = 10.0
    ratio = current_gap / median_gap
    base = 1.0 / (1.0 + np.exp(-(ratio - 2.0) * 2.0))
    venue_mult = {"sphere": 1.5, "msg_nye": 1.3, "halloween": 2.0, "festival": 1.2}.get(venue_type, 1.0)
    run_mult = 1.0 + 0.05 * min(run_length, 8)
    boost = 1.0 + base * venue_mult * run_mult
    return float(min(boost, 2.0))


def era_diversity_factor(
    song_id: str, current_eras: Counter, model: SongModel, target_min_eras: int = 3,
) -> float:
    """Boost songs from underrepresented eras."""
    era = model.song_eras.get(song_id, "modern")
    total = sum(current_eras.values())
    if total == 0:
        return 1.0
    n_distinct_eras = len([e for e in current_eras if current_eras[e] > 0])
    era_count = current_eras.get(era, 0)
    expected = total / max(len(ERAS), 1)
    if n_distinct_eras < target_min_eras and era_count == 0:
        return 1.5
    elif era_count < expected * 0.5:
        return 1.2
    elif era_count > expected * 2.0:
        return 0.8
    return 1.0


def score_song_for_slot(
    song_id: str,
    role: str,
    model: SongModel,
    played_in_run: Set[str],
    current_eras: Counter,
    venue_type: str = "normal",
    run_length: int = 1,
    preceding_song: Optional[str] = None,
) -> float:
    """
    Composite probability score for placing a song in a specific slot.
    Multiplicative log-linear model. Returns unnormalized score.
    """
    if song_id in played_in_run:
        return 0.0

    base = model.base_rates.get(song_id, 0.001)
    sphere = model.sphere_prior.get(song_id, 0.0)
    if sphere > 0:
        base = base * 0.7 + sphere

    gs = model.gap_stats.get(song_id, {"median": 10, "sigma": 0.5, "current_gap": 5})
    effective_gap = gs["current_gap"]
    gp = gap_pressure(effective_gap, gs["median"], gs["sigma"])

    rf = role_fit_score(song_id, role, model)
    bb = bustout_boost(effective_gap, gs["median"], venue_type, run_length)
    ed = era_diversity_factor(song_id, current_eras, model)

    segue_bonus = 1.0
    if preceding_song is not None:
        if preceding_song in model.segue_matrix:
            segue_prob = model.segue_matrix[preceding_song].get(song_id, 0.0)
            if segue_prob > 0:
                segue_bonus = 1.0 + segue_prob * 5.0
        out_rate = model.segue_out_rate.get(preceding_song, 0.0)
        in_rate = model.segue_in_rate.get(song_id, 0.0)
        if out_rate > 0.3 and in_rate > 0.3:
            segue_bonus = max(segue_bonus, 1.3)

    run_rate = model.run_appearance_rate.get(song_id, 0.0)
    run_boost = 1.0 + run_rate * 0.8

    score = (base ** 0.6) * (0.4 + 0.6 * gp) * (0.3 + 0.7 * rf) * bb * ed * segue_bonus * run_boost
    return float(score)


def print_model_summary(model: SongModel):
    """Print a summary of the built model."""
    print(f"\n{'='*60}")
    print("SONG MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Songs in model:          {len(model.all_song_ids)}")
    print(f"Songs with gap stats:    {len(model.gap_stats)}")
    print(f"Songs with role data:    {len(model.role_probs)}")
    print(f"Segue matrix entries:    {sum(len(v) for v in model.segue_matrix.values())}")
    print(f"Mandatory pairs:         {len(model.mandatory_pairs)}")
    print(f"Covers per show:         {model.covers_per_show:.1f}")
    print(f"New cover rate/show:     {model.new_cover_rate_per_show:.3f}")

    era_counts = Counter(model.song_eras.values())
    print(f"\nEra distribution:")
    for era, count in sorted(era_counts.items(), key=lambda x: -x[1]):
        print(f"  {era:20s} {count:4d} songs")

    top = sorted(model.base_rates.items(), key=lambda x: -x[1])[:15]
    print(f"\nTop 15 songs by temporal base rate:")
    for sid, rate in top:
        name = model.song_names.get(sid, sid)
        gs = model.gap_stats.get(sid, {})
        gap = gs.get("current_gap", "?")
        print(f"  {name:40s} rate={rate:.4f}  gap={gap}")
    print(f"{'='*60}")
