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

from collections import Counter

import numpy as np
import pandas as pd
from .scoring import score_all_songs, score_breakdown, ScoringWeights, DEFAULT_WEIGHTS
from .data.songs import SONG_PAIRS

# Soft "variety" penalty for songs played earlier in a tour but in a *different*
# stand. Within a stand, no-repeat is a hard rule; across stands songs recur,
# but not immediately. Measured across 2022-2025 summer tours: a song almost
# never returns within 1-3 shows (~8% of repeats), with returns peaking at +5-7
# shows (median gap 7). So a just-played song is scored down hard and recovers to
# full eligibility by ~VARIETY_RECOVERY shows later.
VARIETY_FLOOR = 0.25       # multiplier for a song played one show ago
VARIETY_RECOVERY = 8       # shows until a song is fully eligible again
RECENT_NO_REPEAT = 3       # hard-exclude anything played in the last N shows,
                           # even across venues — Phish doesn't repeat that fast
VARIETY_MAX_DAYS = 14      # calendar gate on both windows above: "N shows ago"
                           # only means something when the shows are close
                           # together. Past this many days rotation resets (a
                           # mid-tour break: Fenway -> Dick's is 34 days), and
                           # real tours show last-block songs replayed freely
                           # right after a break.


def tour_variety_penalties(
    show_song_history: list[set[str]],
    stand_ids: list[int],
    current_idx: int,
    floor: float = VARIETY_FLOOR,
    recovery: int = VARIETY_RECOVERY,
    show_dates: list | None = None,
) -> dict[str, float]:
    """Soft score multipliers for the show at `current_idx`.

    Songs played earlier in the tour, in an *earlier stand*, are penalized by how
    recently they appeared: strongest for a song played one show ago, ramping
    linearly back to 1.0 (no penalty) by `recovery` shows. Songs in the current
    stand are omitted (they are hard-excluded elsewhere). Empty for the first
    stand of a run, so a single-venue residency is unaffected.
    """
    penalties: dict[str, float] = {}
    for j in range(current_idx):
        if stand_ids[j] == stand_ids[current_idx]:
            continue  # same stand → hard no-repeat handles it
        if (show_dates is not None
                and (show_dates[current_idx] - show_dates[j]).days > VARIETY_MAX_DAYS):
            continue  # across a tour break, recency pressure is gone
        shows_ago = current_idx - j
        mult = min(1.0, floor + (1.0 - floor) * (shows_ago - 1) / max(recovery - 1, 1))
        for sid in show_song_history[j]:
            # Most-recent appearance wins (strongest penalty).
            penalties[sid] = min(penalties.get(sid, 1.0), mult)
    return penalties


# Slot-variety penalty: a song should not keep landing in the *same structural
# spot* across a tour. Measured across 2022-2025 summer tours, a song opens a
# given set at most 3 times in a whole tour (1x ~85%, 2x ~13%, 3x ~2%, 4x never).
# The penalty multiplies a song's score for a role by SLOT_VARIETY_BASE ** (times
# it has already filled that role this tour): free the first time, halved-ish at
# 2, steep at 3, effectively prohibitive at 4. Only the structural roles are
# tracked — interior set position is score-rank, not a meaningful "spot".
SLOT_VARIETY_BASE = 0.4
ROLE_SLOTS = ("show_opener", "s1_closer", "s2_opener", "s2_closer", "encore")


def role_fills(set1_ids: list[str], set2_ids: list[str],
               encore_ids: list[str]) -> dict[str, list[str]]:
    """Which song filled each structural role in a show (opener/closer/encore)."""
    r: dict[str, list[str]] = {}
    if set1_ids:
        r["show_opener"] = [set1_ids[0]]
        r["s1_closer"] = [set1_ids[-1]]
    if set2_ids:
        r["s2_opener"] = [set2_ids[0]]
        r["s2_closer"] = [set2_ids[-1]]
    if encore_ids:
        r["encore"] = list(encore_ids)
    return r


def slot_variety_penalties(
    role_history: dict[str, "Counter"],
    base: float = SLOT_VARIETY_BASE,
) -> dict[str, dict[str, float]]:
    """Per-slot score multipliers from how often each song already filled a role.

    `role_history` maps slot_type -> Counter(song_id -> times filled this tour).
    Returns slot_type -> {song_id: multiplier}. A song can still reappear; it is
    just steered away from the spot it already owns.
    """
    return {
        slot: {sid: base ** cnt for sid, cnt in counter.items() if cnt > 0}
        for slot, counter in role_history.items()
    }


def predict_show(
    show_date: pd.Timestamp,
    venue_type: str,
    songs_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_labels: dict,
    total_shows_in_train: int,
    run_exclusions: set | None = None,
    soft_exclusions: dict | None = None,
    slot_penalties: dict | None = None,
    weights: ScoringWeights | dict | None = None,
    top_k: int = 3,
    set1_size: int = 11,
    set2_size: int = 7,
    enc_size: int = 2,
    set1_budget_min: float | None = None,
    set2_budget_min: float | None = None,
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
    set1_budget_min, set2_budget_min : optional duration budgets (minutes).
        When set, the set is filled until its expected runtime (sum of each
        pick's avg_duration_min) reaches the budget, instead of to a fixed
        song count. Jam-heavy picks then produce shorter sets, matching how
        real sets work: the ~80 minutes is fixed, the song count is not.
        None (default) keeps the fixed-count behaviour.

    Returns
    -------
    dict with keys: 'set1', 'set2', 'encore'
    """
    w = weights or DEFAULT_WEIGHTS
    hard_excl_run = run_exclusions or set()
    soft_excl = soft_exclusions or {}
    slot_pen = slot_penalties or {}
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
        # Apply tiered soft penalties (song-level variety)
        for sid, penalty in soft_excl.items():
            if sid in scores.index:
                scores[sid] *= penalty

        # Apply slot-variety penalty: steer songs away from a spot they already
        # own this tour (keyed by the role we're filling right now).
        for sid, penalty in slot_pen.get(slot_type, {}).items():
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

    def _dur(sid: str) -> float:
        """Expected runtime of a song, minutes (catalog median; 5 if unknown)."""
        try:
            d = float(songs_df.loc[sid, "avg_duration_min"])
            return d if d > 0 else 5.0
        except (KeyError, ValueError):
            return 5.0

    def _set_minutes(key: str) -> float:
        return sum(_dur(p["song_id"]) for p in setlist[key])

    # Leave room for a closer when filling a budgeted set body.
    CLOSER_ALLOWANCE_MIN = 7.0
    MAX_SET_SONGS = 20  # hard cap so a degenerate catalog can't loop forever

    # -- SET 1 --
    openers = _pick("show_opener", top_k)
    setlist["set1"] += _commit(openers, 1)

    if set1_budget_min is not None:
        while (_set_minutes("set1") + CLOSER_ALLOWANCE_MIN < set1_budget_min
               and len(setlist["set1"]) < MAX_SET_SONGS):
            picks = _pick("s1_body", top_k)
            if not picks:
                break
            setlist["set1"] += _commit(picks, 1)
    else:
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

    if set2_budget_min is not None:
        while (_set_minutes("set2") + CLOSER_ALLOWANCE_MIN < set2_budget_min
               and len(setlist["set2"]) < MAX_SET_SONGS):
            picks = _pick("s2_body", top_k)
            if not picks:
                break
            setlist["set2"] += _commit(picks, 1)
    else:
        while len(setlist["set2"]) < set2_size - 1:
            picks = _pick("s2_body", top_k)
            if not picks:
                # Candidate pool exhausted (heavy exclusions) — ship a shorter
                # set rather than spin forever in an unattended roll.
                break
            setlist["set2"] += _commit(picks, 1)

    s2_closers = _pick("s2_closer", top_k)
    setlist["set2"] += _commit(s2_closers, 1)

    # -- ENCORE --
    # Tweezer Reprise auto-follows Tweezer, but not every single time across a
    # tour — gate it by the same slot-variety rule so it doesn't encore 5 nights.
    tweeprise_ok = slot_pen.get("encore", {}).get("tweeprise", 1.0) > 0.1
    if tweeprise_ok and "tweezer" in chosen_show and "tweeprise" not in chosen_show and "tweeprise" not in hard_excl_run:
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
    show_venues: list[str] | None = None,
    venue_types: list[str] | None = None,
    actual_setlists: dict[int, set[str]] | None = None,
) -> list[dict]:
    """
    Predict a full multi-night run with stand-scoped no-repeat.

    `actual_setlists` maps a show index to the set of song_ids actually played
    there (for shows already completed in a rolling re-prediction). When a later
    night of the same stand is predicted, it hard-excludes the *actual* songs of
    the earlier nights rather than the model's own guesses for them, so the
    no-repeat rule holds against reality.

    No-repeat is scoped to the *stand* (a maximal run of consecutive shows at
    one venue): the band never repeats within a stand but replays ~58% of its
    songs at later stops, so exclusions reset at each stand boundary. For a
    single-venue run (a residency) the whole run is one stand and every prior
    night is excluded — identical to the original behaviour.

    Parameters
    ----------
    venue_type   : default venue type, applied to every show unless overridden
    show_venues  : per-show venue *names* for stand detection. Defaults to a
                   single venue, i.e. one stand spanning the whole run.
    venue_types  : per-show venue *types* for scoring. Defaults to `venue_type`.

    For residencies with multi-weekend structure, distributes top songs evenly
    across weekends so each weekend gets its share of crowd favorites. That
    distribution is a single-venue heuristic and is skipped for multi-stand runs.
    """
    from .features import compute_all_features
    from .stands import detect_stands

    n_shows = len(show_dates)

    # ── Stand structure ──
    if show_venues is None:
        show_venues = [venue_type] * n_shows
    stand_ids = detect_stands(show_venues)
    n_stands = len(set(stand_ids))
    show_vtypes = venue_types if venue_types is not None else [venue_type] * n_shows

    # ── Detect weekend structure ──
    # Group shows into weekends (gap of 3+ days = new weekend)
    weekends: list[list[int]] = [[0]]  # list of show indices per weekend
    for i in range(1, n_shows):
        gap_days = (show_dates[i] - show_dates[i - 1]).days
        if gap_days >= 3:
            weekends.append([i])
        else:
            weekends[-1].append(i)

    n_weekends = len(weekends)

    # ── Pre-assign top songs to weekends ──
    # Score all songs once to find the top candidates, then round-robin
    # assign them so each weekend gets an equal share of favorites.
    # Even-distribution across weekends is a single-venue residency heuristic;
    # for a multi-stand tour, songs are *meant* to recur across stops, so skip it.
    weekend_reserves: dict[int, set[str]] = {}  # show_index -> reserved songs
    if n_stands == 1 and n_weekends >= 2:
        cutoff = show_dates[0] - pd.Timedelta(days=1)
        feat_df_init = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
        total_init = int(shows_df[shows_df["date"] <= cutoff]["show_num"].max() or 0)
        initial_scores = score_all_songs(
            "s2_body", songs_df, feat_df_init, cluster_labels,
            [], set(), venue_type, total_init, weights,
        )
        # Also score S1 openers and encore to catch top songs across all slots
        s1_scores = score_all_songs(
            "show_opener", songs_df, feat_df_init, cluster_labels,
            [], set(), venue_type, total_init, weights,
        )
        # Merge top songs from both S1 and S2 rankings
        combined = {}
        for sid in initial_scores.index:
            combined[sid] = max(combined.get(sid, 0), initial_scores[sid])
        for sid in s1_scores.index:
            combined[sid] = max(combined.get(sid, 0), s1_scores[sid])
        sorted_combined = sorted(combined.items(), key=lambda x: -x[1])

        # Reserve top songs: 10 per weekend for 3 weekends = 30 total
        # This ensures each weekend gets marquee S1 openers, S2 jam vehicles, etc.
        songs_per_weekend = 10
        n_top = min(n_weekends * songs_per_weekend, len(sorted_combined))
        top_songs = [sid for sid, _ in sorted_combined[:n_top]]

        # Round-robin assign: #1 → Wk1, #2 → Wk2, #3 → Wk3, #4 → Wk1, etc.
        for rank, sid in enumerate(top_songs):
            weekend_idx = rank % n_weekends
            weekend_reserves.setdefault(weekends[weekend_idx][0], set()).add(sid)

    # ── Predict each show ──
    show_song_history: list[set[str]] = []
    role_history: dict[str, Counter] = {slot: Counter() for slot in ROLE_SLOTS}
    predictions = []

    for i, show_date in enumerate(show_dates):
        cutoff = show_date - pd.Timedelta(days=1)
        feat_df = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
        total_shows = int(shows_df[shows_df["date"] <= cutoff]["show_num"].max() or 0)

        # Hard-exclude songs from previous nights *of the same stand*. New
        # stand → clean slate, so the ~58% of songs that recur across stops
        # are eligible again.
        hard_exclusions: set[str] = set()
        for j in range(i):
            if stand_ids[j] == stand_ids[i]:
                hard_exclusions |= show_song_history[j]

        # Recency no-repeat: Phish essentially never replays a song within a few
        # shows, even at a new venue. Hard-exclude anything played in the last
        # RECENT_NO_REPEAT shows so the model never predicts last night's setlist.
        # Calendar-gated: "last 3 shows" across a multi-week break is not
        # recency, so those shows don't count against the window.
        for j in range(max(0, i - RECENT_NO_REPEAT), i):
            if (show_dates[i] - show_dates[j]).days > VARIETY_MAX_DAYS:
                continue
            hard_exclusions |= show_song_history[j]

        # Soft variety penalty: songs played in earlier stands are scored down
        # by how recently they appeared, so stand-openers don't replay the
        # previous stop's setlist (real tours keep ~41% fresh across stops).
        soft_exclusions: dict[str, float] = tour_variety_penalties(
            show_song_history, stand_ids, i, show_dates=show_dates,
        )
        current_weekend = next(
            wi for wi, shows in enumerate(weekends) if i in shows
        )
        for other_show_idx, reserved in weekend_reserves.items():
            reserve_weekend = next(
                wi for wi, shows in enumerate(weekends) if other_show_idx in shows
            )
            if current_weekend != reserve_weekend:
                hard_exclusions |= reserved

        # Slot-variety: steer songs away from a structural spot they already
        # own earlier in this tour (tour-wide, not stand-scoped).
        slot_pen = slot_variety_penalties(role_history)

        pred = predict_show(
            show_date=show_date,
            venue_type=show_vtypes[i],
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            total_shows_in_train=total_shows,
            run_exclusions=hard_exclusions,
            soft_exclusions=soft_exclusions,
            slot_penalties=slot_pen,
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
        # For a completed show, later same-stand nights must avoid what was
        # *actually* played, not what we guessed. Seed the stand history with the
        # real setlist when we have it.
        if actual_setlists and i in actual_setlists:
            show_song_history.append(set(actual_setlists[i]))
        else:
            show_song_history.append(this_show_songs)

        # Record which song filled each structural role, for next show's penalty.
        ids = {k: [e["song_id"] for e in pred[k]] for k in ("set1", "set2", "encore")}
        for slot, fills in role_fills(ids["set1"], ids["set2"], ids["encore"]).items():
            role_history[slot].update(fills)

    return predictions
