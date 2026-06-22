"""
Backtesting framework.

Validates the predictor against a held-out multi-night run by:
  1. Splitting show history at a cutoff date
  2. Training features on the pre-cutoff window
  3. Predicting each show in the validation window
  4. Comparing predictions to actual setlists

Metrics
-------
  hit_rate          - % of predicted songs that appear anywhere in actual show
  set_precision     - % of songs predicted in the correct set
"""

from collections import Counter

import numpy as np
import pandas as pd
from ..features import compute_all_features
from ..predictor import (
    predict_show, tour_variety_penalties,
    slot_variety_penalties, role_fills, ROLE_SLOTS,
)
from ..stands import detect_stands


def _actual_sets(show_id: str, appearances_df: pd.DataFrame) -> dict:
    """Return {set_number: [song_id, ...]} for a given show."""
    app = appearances_df[appearances_df["show_id"] == show_id]
    result = {}
    for sn in ["1", "2", "e"]:
        subset = app[app["set_number"] == sn].sort_values("position")
        result[sn] = subset["song_id"].tolist()
    return result


def _predicted_sets(prediction: dict) -> dict:
    """Convert prediction dict into {set_number: [song_id, ...]}."""
    return {
        "1": [e["song_id"] for e in prediction["set1"]],
        "2": [e["song_id"] for e in prediction["set2"]],
        "e": [e["song_id"] for e in prediction["encore"]],
    }


def _compute_show_metrics(pred_sets: dict, actual_sets: dict) -> dict:
    all_actual    = set(actual_sets["1"]) | set(actual_sets["2"]) | set(actual_sets["e"])
    all_predicted = set(pred_sets["1"]) | set(pred_sets["2"]) | set(pred_sets["e"])

    hit_rate = len(all_predicted & all_actual) / max(len(all_predicted), 1)

    correct_set = 0
    total_pred  = 0
    for sn_pred, sn_act in [("1", "1"), ("2", "2"), ("e", "e")]:
        for sid in pred_sets[sn_pred]:
            total_pred += 1
            if sid in actual_sets[sn_act]:
                correct_set += 1
    set_precision = correct_set / max(total_pred, 1)

    return {
        "hit_rate":      round(hit_rate,      4),
        "set_precision": round(set_precision, 4),
        "n_predicted":   len(all_predicted),
        "n_actual":      len(all_actual),
        "n_correct":     len(all_predicted & all_actual),
    }


def run_backtest(
    songs_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    appearances_df: pd.DataFrame,
    cluster_labels: dict,
    validation_tour: str,
    weights=None,
    verbose: bool = True,
    soft_variety: bool = True,
    slot_variety: bool = True,
) -> dict:
    """
    Run backtesting on a specific named tour run.

    Parameters
    ----------
    songs_df, shows_df, appearances_df : data
    cluster_labels : {song_id: cluster_id}
    validation_tour : name of the tour to hold out (must match shows_df['tour'])
    weights : ScoringWeights or dict
    verbose : print per-show results

    Returns
    -------
    dict of aggregate metrics + per-show metrics list
    """
    val_shows = shows_df[shows_df["tour"] == validation_tour].sort_values("date")
    if val_shows.empty:
        raise ValueError(f"No shows found for tour: {validation_tour!r}")

    train_cutoff = val_shows["date"].min() - pd.Timedelta(days=1)
    train_shows  = shows_df[shows_df["date"] <= train_cutoff]
    total_train  = len(train_shows)

    if verbose:
        print(f"\n  Validation run : {validation_tour} ({len(val_shows)} shows)")
        print(f"  Training window: up to {train_cutoff.date()} ({total_train} shows)")
        print()

    per_show_metrics = []
    run_exclusions: set[str] = set()
    prev_venue: str | None = None
    stand_ids = detect_stands(val_shows["venue_name"].tolist())
    show_song_history: list[set[str]] = []
    role_history: dict[str, Counter] = {slot: Counter() for slot in ROLE_SLOTS}

    for i, (_, row) in enumerate(val_shows.iterrows()):
        show_id   = row["show_id"]
        show_date = row["date"]
        vtype     = row["venue_type"]

        # No-repeat is scoped to the *stand* (a multi-night stop at one venue),
        # not the whole tour. Phish never repeats within a stand but replays
        # ~58% of songs at later stops, so exclusions reset at each new venue.
        # For a single-venue run (a residency, an MSG NYE stand) this never
        # fires and behaviour matches the old tour-wide accumulation.
        if row["venue_name"] != prev_venue:
            run_exclusions = set()
        prev_venue = row["venue_name"]

        cutoff = show_date - pd.Timedelta(days=1)
        feat_df = compute_all_features(songs_df, train_shows, appearances_df, cutoff)
        n_train = int(train_shows["show_num"].max() or 0)

        soft_excl = (
            tour_variety_penalties(show_song_history, stand_ids, i)
            if soft_variety else None
        )
        slot_pen = slot_variety_penalties(role_history) if slot_variety else None

        pred = predict_show(
            show_date=show_date,
            venue_type=vtype,
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            total_shows_in_train=n_train,
            run_exclusions=run_exclusions,
            soft_exclusions=soft_excl,
            slot_penalties=slot_pen,
            weights=weights,
        )

        pred_sets   = _predicted_sets(pred)
        actual      = _actual_sets(show_id, appearances_df)
        m = _compute_show_metrics(pred_sets, actual)
        m["show_id"]   = show_id
        m["date"]      = show_date.date()
        m["venue"]     = row["venue_name"]

        per_show_metrics.append(m)

        if verbose:
            actual_all = set(actual["1"]) | set(actual["2"]) | set(actual["e"])
            pred_all   = set(pred_sets["1"]) | set(pred_sets["2"]) | set(pred_sets["e"])
            correct    = pred_all & actual_all
            correct_names = [songs_df.loc[s, "name"] for s in correct]
            print(f"  {show_date.date()} @ {row['venue_name'][:35]:<35} "
                  f"hit={m['hit_rate']*100:5.1f}%  set_prec={m['set_precision']*100:5.1f}%  "
                  f"({m['n_correct']}/{m['n_predicted']} songs correct)")
            if correct_names:
                print(f"    Correct picks: {', '.join(correct_names)}")

        actual_songs = set(actual["1"]) | set(actual["2"]) | set(actual["e"])
        run_exclusions |= actual_songs
        show_song_history.append(actual_songs)
        # Track the actual structural roles this show, for the next show's penalty.
        for slot, fills in role_fills(actual["1"], actual["2"], actual["e"]).items():
            role_history[slot].update(fills)

    avg_hit      = float(np.mean([m["hit_rate"]      for m in per_show_metrics]))
    avg_set_prec = float(np.mean([m["set_precision"] for m in per_show_metrics]))
    total_correct = sum(m["n_correct"]   for m in per_show_metrics)
    total_pred    = sum(m["n_predicted"] for m in per_show_metrics)

    summary = {
        "validation_tour":  validation_tour,
        "n_shows":          len(per_show_metrics),
        "avg_hit_rate":     round(avg_hit,      4),
        "avg_set_precision":round(avg_set_prec, 4),
        "total_correct":    total_correct,
        "total_predicted":  total_pred,
        "per_show":         per_show_metrics,
    }

    if verbose:
        print()
        print(f"  -- Backtest Summary --")
        print(f"  Avg Hit Rate     : {avg_hit*100:.1f}%")
        print(f"  Avg Set Precision: {avg_set_prec*100:.1f}%")
        print(f"  Total Correct    : {total_correct} / {total_pred} predicted songs")

    return summary
