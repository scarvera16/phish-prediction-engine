"""
Setlist predictor.

For each show, builds a complete predicted setlist by greedily selecting
the top-scoring song for each slot, subject to:
  - No repeats within a show
  - Soft penalty for songs played in the same multi-night run
  - Mike's Groove sequence enforced when Mike's Song is predicted
  - Tweezer Reprise appended to encore when Tweezer is in set 2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from .scoring import score_all_songs, score_breakdown, ScoringWeights, DEFAULT_WEIGHTS
from .data.songs import SONG_PAIRS


def predict_show(
    show_date: pd.Timestamp,
    venue_type: str,
    songs_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_labels: dict,
    total_shows_in_train: int,
    run_exclusions: set | None = None,
    soft_exclusions: dict | None = None,
    weights: ScoringWeights | dict | None = None,
    top_k: int = 3,
    set1_size: int = 11,
    set2_size: int = 7,
    enc_size: int = 2,
) -> dict:
    """
    Predict a full show setlist.

    Parameters
    ----------
    show_date               : date of the show (for display only)
    venue_type              : 'sphere' | 'arena' | 'outdoor'
    songs_df, feat_df       : catalog + dynamic features
    cluster_labels          : {song_id: cluster_id}
    total_shows_in_train    : n shows used for training
    run_exclusions          : songs hard-excluded (played in last 1-2 shows)
    soft_exclusions         : {song_id: penalty_multiplier}
    weights                 : ScoringWeights or legacy dict
    top_k                   : return top-k candidates per slot
    set1_size, set2_size, enc_size : target song counts

    Returns
    -------
    dict with keys: 'set1', 'set2', 'encore'
    """
    w = weights or DEFAULT_WEIGHTS
    hard_excl_run = run_exclusions or set()
    soft_excl = soft_exclusions or {}
    chosen_show: list[str] = []

    def _pick(slot_type: str, n: int, hard_excl: set | None = None) -> list[dict]:
        """Score and greedily select n songs for a slot."""
        hard = (hard_excl or set()) | set(chosen_show) | hard_excl_run
        scores = score_all_songs(
            slot_type=slot_type,
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            already_chosen=chosen_show,
            excluded=hard,
            venue_type=venue_type,
            total_shows=total_shows_in_train,
            weights=w,
        )
        # Apply tiered soft penalties
        for sid, penalty in soft_excl.items():
            if sid in scores.index:
                scores[sid] *= penalty

        scores = scores.sort_values(ascending=False)

        picks = []
        for song_id in scores.index:
            if song_id in chosen_show:
                continue
            bd = score_breakdown(song_id, slot_type, songs_df, feat_df, cluster_labels,
                                 chosen_show, venue_type, total_shows_in_train, w)
            picks.append({
                "song_id":    song_id,
                "name":       songs_df.loc[song_id, "name"],
                "slot":       slot_type,
                "score":      round(float(scores[song_id]), 4),
                "components": bd,
            })
            if len(picks) >= n:
                break
        return picks

    def _commit(picks: list[dict], count: int = 1) -> list[dict]:
        """Take the top `count` picks and add them to chosen_show."""
        committed = picks[:count]
        for p in committed:
            chosen_show.append(p["song_id"])
        return committed

    setlist = {"set1": [], "set2": [], "encore": []}

    # -- SET 1 --
    openers = _pick("show_opener", top_k)
    setlist["set1"] += _commit(openers, 1)

    s1_body_n = set1_size - 2
    for _ in range(s1_body_n):
        picks = _pick("s1_body", top_k)
        setlist["set1"] += _commit(picks, 1)

    s1_closers = _pick("s1_closer", top_k)
    setlist["set1"] += _commit(s1_closers, 1)

    # -- SET 2 --
    s2_openers = _pick("s2_opener", top_k)
    if not s2_openers:
        s2_openers = _pick("s2_body", top_k)
    if not s2_openers:
        return setlist
    s2_opener_pick = _commit(s2_openers, 1)[0]
    setlist["set2"].append(s2_opener_pick)

    # Mike's Groove injection
    groove_available = all(sid not in hard_excl_run for sid in ["mikes", "hydrogen", "weekapaug"])
    mikes_score = 0.0
    if groove_available and "mikes" not in chosen_show:
        s2_scores = score_all_songs("s2_body", songs_df, feat_df, cluster_labels,
                                     chosen_show, hard_excl_run, venue_type, total_shows_in_train, w)
        mikes_score = float(s2_scores.get("mikes", 0.0))

    if groove_available and (s2_opener_pick["song_id"] == "mikes" or mikes_score > 0.65):
        for seq_id in ["mikes", "hydrogen", "weekapaug"]:
            if seq_id not in chosen_show and seq_id not in hard_excl_run:
                bd = score_breakdown(seq_id, "s2_body", songs_df, feat_df, cluster_labels,
                                     chosen_show, venue_type, total_shows_in_train, w)
                setlist["set2"].append({
                    "song_id": seq_id,
                    "name":    songs_df.loc[seq_id, "name"],
                    "slot":    "s2_sequence",
                    "score":   bd["composite"],
                    "components": bd,
                })
                chosen_show.append(seq_id)

    while len(setlist["set2"]) < set2_size - 1:
        picks = _pick("s2_body", top_k)
        setlist["set2"] += _commit(picks, 1)

    s2_closers = _pick("s2_closer", top_k)
    setlist["set2"] += _commit(s2_closers, 1)

    # -- ENCORE --
    if "tweezer" in chosen_show and "tweeprise" not in chosen_show and "tweeprise" not in hard_excl_run:
        bd = score_breakdown("tweeprise", "encore", songs_df, feat_df, cluster_labels,
                             chosen_show, venue_type, total_shows_in_train, w)
        if enc_size > 1:
            pre_encore = _pick("encore", top_k, hard_excl={"tweeprise"})
            setlist["encore"] += _commit(pre_encore, enc_size - 1)
        setlist["encore"].append({
            "song_id":    "tweeprise",
            "name":       songs_df.loc["tweeprise", "name"],
            "slot":       "encore_closer",
            "score":      round(bd["composite"] * 1.2, 4),
            "components": bd,
        })
        chosen_show.append("tweeprise")
    else:
        enc_picks = _pick("encore", top_k)
        setlist["encore"] += _commit(enc_picks, min(enc_size, len(enc_picks)))

    return setlist


def predict_multi_night_run(
    show_dates: list,
    venue_type: str,
    songs_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    appearances_df: pd.DataFrame,
    cluster_labels: dict,
    weights: ScoringWeights | dict | None = None,
    set1_size: int = 11,
    set2_size: int = 7,
    enc_size: int = 2,
) -> list[dict]:
    """
    Predict a full multi-night run with rolling exclusion windows.

    Exclusion strategy (mirrors real Phish multi-night behaviour):
      - Songs from previous 1-2 shows: HARD excluded
      - Songs from 3 shows ago: heavy soft penalty (0.15x)
      - Songs from 4 shows ago: moderate penalty (0.40x)
      - Songs from 5+ shows ago: mild penalty (0.70x)
    """
    from .features import compute_all_features

    show_song_history: list[set[str]] = []
    predictions = []

    for i, show_date in enumerate(show_dates):
        cutoff = show_date - pd.Timedelta(days=1)
        feat_df = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
        total_shows = int(shows_df[shows_df["date"] <= cutoff]["show_num"].max() or 0)

        # Hard-exclude ALL songs from previous nights in this run.
        # Real data shows Phish virtually never repeats within a same-venue run.
        hard_exclusions: set[str] = set()
        soft_exclusions: dict[str, float] = {}

        for lookback in range(1, len(show_song_history) + 1):
            hard_exclusions |= show_song_history[-lookback]

        pred = predict_show(
            show_date=show_date,
            venue_type=venue_type,
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            total_shows_in_train=total_shows,
            run_exclusions=hard_exclusions,
            soft_exclusions=soft_exclusions,
            weights=weights,
            set1_size=set1_size,
            set2_size=set2_size,
            enc_size=enc_size,
        )
        pred["date"] = show_date
        pred["show_num"] = i + 1
        predictions.append(pred)

        this_show_songs = set()
        for slot in ("set1", "set2", "encore"):
            for entry in pred[slot]:
                this_show_songs.add(entry["song_id"])
        show_song_history.append(this_show_songs)

    return predictions
