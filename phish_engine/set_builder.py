"""
Sequential set construction with energy arcs, segue chains,
mandatory pairings, and duration budgets.

Builds a single show's setlist by:
  1. Filling anchor positions (openers/closers) from role distributions
  2. Building segue chains in Set 2
  3. Filling middle positions with energy arc fitting
  4. Enforcing mandatory pairings
  5. Tracking duration budget
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .song_model import (
    SongModel,
    score_song_for_slot,
    role_fit_score,
    era_diversity_factor,
    SEGUE_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Set structure calibrated from Sphere 2024
# ---------------------------------------------------------------------------

@dataclass
class SetConfig:
    min_songs: int
    max_songs: int
    duration_target_min: float
    duration_max_min: float


SPHERE_CONFIG = {
    "set1": SetConfig(min_songs=7, max_songs=8, duration_target_min=75, duration_max_min=85),
    "set2": SetConfig(min_songs=7, max_songs=8, duration_target_min=80, duration_max_min=95),
    "encore": SetConfig(min_songs=2, max_songs=2, duration_target_min=12, duration_max_min=20),
}

MSG_CONFIG = {
    "set1": SetConfig(min_songs=7, max_songs=12, duration_target_min=75, duration_max_min=100),
    "set2": SetConfig(min_songs=5, max_songs=12, duration_target_min=80, duration_max_min=110),
    "encore": SetConfig(min_songs=1, max_songs=4, duration_target_min=8, duration_max_min=30),
}

DICKS_CONFIG = {
    "set1": SetConfig(min_songs=7, max_songs=12, duration_target_min=75, duration_max_min=100),
    "set2": SetConfig(min_songs=6, max_songs=11, duration_target_min=80, duration_max_min=100),
    "encore": SetConfig(min_songs=1, max_songs=3, duration_target_min=8, duration_max_min=25),
}

NORMAL_CONFIG = {
    "set1": SetConfig(min_songs=8, max_songs=11, duration_target_min=75, duration_max_min=90),
    "set2": SetConfig(min_songs=7, max_songs=10, duration_target_min=80, duration_max_min=100),
    "encore": SetConfig(min_songs=2, max_songs=3, duration_target_min=12, duration_max_min=20),
}

VENUE_CONFIG_MAP = {
    "sphere": SPHERE_CONFIG,
    "msg": MSG_CONFIG,
    "msg_nye": MSG_CONFIG,
    "dicks": DICKS_CONFIG,
    "festival": DICKS_CONFIG,
}


# ---------------------------------------------------------------------------
# Energy arc targets (1-10 scale)
# ---------------------------------------------------------------------------

SET1_ENERGY = [7, 6, 7, 5, 6, 5, 7, 8, 9]
SET2_ENERGY = [8, 9, 8, 7, 6, 8, 9, 8]
SET2_JAM = [8, 9, 8, 7, 6, 8, 7, 6]
ENCORE_ENERGY = [4, 8, 9]


def _get_energy_target(set_name: str, position: int, set_size: int) -> float:
    """Interpolate energy target for a position within a set."""
    if set_name == "set1":
        arc = SET1_ENERGY
    elif set_name == "set2":
        arc = SET2_ENERGY
    else:
        arc = ENCORE_ENERGY

    if set_size <= 1:
        return arc[0]
    arc_pos = position * (len(arc) - 1) / (set_size - 1)
    idx = int(arc_pos)
    frac = arc_pos - idx
    if idx >= len(arc) - 1:
        return arc[-1]
    return arc[idx] * (1 - frac) + arc[idx + 1] * frac


def _energy_fit(song_energy: float, target: float) -> float:
    """Score how well a song's energy matches the target (0-1)."""
    diff = abs(song_energy - target)
    return max(0.1, 1.0 - diff / 10.0)


# ---------------------------------------------------------------------------
# Set builder
# ---------------------------------------------------------------------------

def build_setlist(
    model: SongModel,
    played_in_run: Set[str],
    venue_type: str = "sphere",
    run_length: int = 8,
    show_number: int = 1,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict[str, SetConfig]] = None,
) -> Dict[str, List[str]]:
    """
    Build a complete setlist for one show.

    Returns {"set1": [song_ids], "set2": [song_ids], "encore": [song_ids]}
    """
    if rng is None:
        rng = np.random.default_rng()
    if config is None:
        config = VENUE_CONFIG_MAP.get(venue_type, NORMAL_CONFIG)

    committed: Set[str] = set(played_in_run)
    current_eras: Counter = Counter()
    setlist: Dict[str, List[str]] = {}

    for set_name in ["set1", "set2", "encore"]:
        cfg = config[set_name]
        set_size = int(rng.integers(cfg.min_songs, cfg.max_songs + 1))

        songs = _build_one_set(
            model=model, set_name=set_name, set_size=set_size,
            committed=committed, current_eras=current_eras,
            venue_type=venue_type, run_length=run_length,
            show_number=show_number, rng=rng, duration_max=cfg.duration_max_min,
        )

        setlist[set_name] = songs
        for sid in songs:
            committed.add(sid)
            era = model.song_eras.get(sid, "modern")
            current_eras[era] += 1

    return setlist


def _build_one_set(
    model, set_name, set_size, committed, current_eras,
    venue_type, run_length, show_number, rng, duration_max,
) -> List[str]:
    """Build one set with anchor-first strategy."""
    if set_name == "set1":
        opener_role, closer_role, mid_role = "set1_opener", "set1_closer", "set1_mid"
    elif set_name == "set2":
        opener_role, closer_role, mid_role = "set2_opener", "set2_closer", "set2_mid"
    else:
        opener_role, closer_role, mid_role = "encore_opener", "show_closer", "encore_mid"

    result: List[Optional[str]] = [None] * set_size
    local_committed = set(committed)
    local_eras = Counter(current_eras)

    # Step 1: Fill anchor positions
    opener = _pick_song_for_role(model, opener_role, local_committed, local_eras,
                                  venue_type, run_length, rng)
    if opener:
        result[0] = opener
        local_committed.add(opener)
        local_eras[model.song_eras.get(opener, "modern")] += 1

    closer = _pick_song_for_role(model, closer_role, local_committed, local_eras,
                                  venue_type, run_length, rng)
    if closer:
        result[-1] = closer
        local_committed.add(closer)
        local_eras[model.song_eras.get(closer, "modern")] += 1

    # Step 2: Handle mandatory pairings
    _enforce_mandatory_pairs(model, result, local_committed, local_eras, set_name)

    # Step 3: Build segue chains for Set 2
    if set_name == "set2":
        _build_segue_chains(model, result, local_committed, local_eras, venue_type,
                           run_length, rng)

    # Step 4: Fill remaining positions
    total_duration = 0.0
    for i in range(len(result)):
        if result[i] is not None:
            dur = model.avg_duration_sec.get(result[i], 300) / 60.0
            total_duration += dur

    for i in range(len(result)):
        if result[i] is not None:
            continue

        preceding = result[i - 1] if i > 0 else None
        energy_target = _get_energy_target(set_name, i, len(result))

        candidates = _score_candidates(
            model=model, role=mid_role, committed=local_committed,
            current_eras=local_eras, venue_type=venue_type,
            run_length=run_length, preceding_song=preceding,
            energy_target=energy_target, set_name=set_name,
            duration_remaining=duration_max - total_duration,
        )

        if not candidates:
            continue

        chosen = _weighted_sample(candidates, rng, top_k=8)
        result[i] = chosen
        local_committed.add(chosen)
        local_eras[model.song_eras.get(chosen, "modern")] += 1
        total_duration += model.avg_duration_sec.get(chosen, 300) / 60.0

    return [s for s in result if s is not None]


def _pick_song_for_role(model, role, committed, current_eras, venue_type, run_length, rng):
    """Pick a song optimized for a specific role (opener/closer)."""
    candidates = []
    for sid in model.all_song_ids:
        if sid in committed:
            continue
        base_rate = model.base_rates.get(sid, 0.0)
        if base_rate < 0.05:
            continue
        base_score = score_song_for_slot(sid, role, model, committed, current_eras, venue_type, run_length)
        rf = role_fit_score(sid, role, model)
        score = base_score * (0.5 + rf * 3.0)
        if score > 0:
            candidates.append((sid, score))
    if not candidates:
        return None
    return _weighted_sample(candidates, rng, top_k=5)


def _enforce_mandatory_pairs(model, result, committed, current_eras, set_name):
    """If one half of a mandatory pair is placed, force-place the other."""
    placed = {s for s in result if s is not None}
    for song_a, song_b, direction in model.mandatory_pairs:
        if song_a in placed and song_b not in committed:
            idx_a = next((i for i, s in enumerate(result) if s == song_a), None)
            if idx_a is None:
                continue
            if direction == "a_before_b":
                target = _find_empty_slot_after(result, idx_a)
            elif direction == "b_before_a":
                target = _find_empty_slot_before(result, idx_a)
            else:
                target = _find_empty_slot_after(result, idx_a) or \
                         _find_empty_slot_before(result, idx_a)
            if target is not None:
                result[target] = song_b
                committed.add(song_b)
                current_eras[model.song_eras.get(song_b, "modern")] += 1
        elif song_b in placed and song_a not in committed:
            idx_b = next((i for i, s in enumerate(result) if s == song_b), None)
            if idx_b is None:
                continue
            if direction == "a_before_b":
                target = _find_empty_slot_before(result, idx_b)
            elif direction == "b_before_a":
                target = _find_empty_slot_after(result, idx_b)
            else:
                target = _find_empty_slot_before(result, idx_b) or \
                         _find_empty_slot_after(result, idx_b)
            if target is not None:
                result[target] = song_a
                committed.add(song_a)
                current_eras[model.song_eras.get(song_a, "modern")] += 1


def _find_empty_slot_after(result, idx):
    for i in range(idx + 1, len(result) - 1):
        if result[i] is None:
            return i
    return None


def _find_empty_slot_before(result, idx):
    for i in range(idx - 1, 0, -1):
        if result[i] is None:
            return i
    return None


def _build_segue_chains(model, result, committed, current_eras, venue_type, run_length, rng, target_chains=3):
    """Build segue chains for Set 2 middle positions."""
    empty_slots = [i for i in range(1, len(result) - 1) if result[i] is None]
    if len(empty_slots) < 2:
        return

    chains_built = 0
    slot_idx = 0
    next_slot_idx = slot_idx

    while chains_built < target_chains and slot_idx < len(empty_slots) - 1:
        start_slot = empty_slots[slot_idx]
        starters = []
        for sid in model.all_song_ids:
            if sid in committed:
                continue
            if model.base_rates.get(sid, 0.0) < 0.03:
                continue
            out_rate = model.segue_out_rate.get(sid, 0.0)
            if out_rate > 0.3:
                base = score_song_for_slot(sid, "set2_mid", model, committed,
                                           current_eras, venue_type, run_length)
                starters.append((sid, base * out_rate))
        if not starters:
            break

        chain_start = _weighted_sample(starters, rng, top_k=5)
        result[start_slot] = chain_start
        committed.add(chain_start)
        current_eras[model.song_eras.get(chain_start, "modern")] += 1

        prev = chain_start
        for next_slot_idx in range(slot_idx + 1, min(slot_idx + 3, len(empty_slots))):
            next_slot = empty_slots[next_slot_idx]
            if result[next_slot] is not None:
                continue
            segue_targets = model.segue_matrix.get(prev, {})
            valid_targets = [(sid, p) for sid, p in segue_targets.items() if sid not in committed]
            if valid_targets:
                next_song = _weighted_sample(valid_targets, rng, top_k=3)
            else:
                fallbacks = []
                for sid in model.all_song_ids:
                    if sid in committed:
                        continue
                    in_rate = model.segue_in_rate.get(sid, 0.0)
                    if in_rate > 0.2:
                        base = score_song_for_slot(sid, "set2_mid", model, committed,
                                                   current_eras, venue_type, run_length)
                        fallbacks.append((sid, base * in_rate))
                if not fallbacks:
                    break
                next_song = _weighted_sample(fallbacks, rng, top_k=5)
            result[next_slot] = next_song
            committed.add(next_song)
            current_eras[model.song_eras.get(next_song, "modern")] += 1
            prev = next_song

        chains_built += 1
        slot_idx = next_slot_idx + 1


def _score_candidates(model, role, committed, current_eras, venue_type, run_length,
                       preceding_song, energy_target, set_name, duration_remaining):
    """Score all candidate songs for a mid-set position."""
    candidates = []
    for sid in model.all_song_ids:
        if sid in committed:
            continue
        dur_min = model.avg_duration_sec.get(sid, 300) / 60.0
        if dur_min > duration_remaining + 5:
            continue
        base = score_song_for_slot(sid, role, model, committed, current_eras,
                                    venue_type, run_length, preceding_song)
        song_energy = model.energy.get(sid, 5.0)
        e_fit = _energy_fit(song_energy, energy_target)
        score = base * e_fit
        if score > 0:
            candidates.append((sid, score))
    candidates.sort(key=lambda x: -x[1])
    return candidates


def _weighted_sample(candidates, rng, top_k=5):
    """Sample from top-k candidates weighted by score (for variety)."""
    top = candidates[:top_k]
    if not top:
        return candidates[0][0] if candidates else ""
    scores = np.array([s for _, s in top])
    scores = scores / max(scores.max(), 1e-9)
    probs = np.exp(scores * 5.0)
    probs = probs / probs.sum()
    idx = rng.choice(len(top), p=probs)
    return top[idx][0]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_setlist(
    setlist: Dict[str, List[str]],
    model: SongModel,
    show_label: str = "",
) -> str:
    """Format a setlist for display."""
    lines = []
    if show_label:
        lines.append(f"\n{'─'*60}")
        lines.append(f"  {show_label}")
        lines.append(f"{'─'*60}")

    for set_name, label in [("set1", "SET 1"), ("set2", "SET 2"), ("encore", "ENCORE")]:
        songs = setlist.get(set_name, [])
        lines.append(f"\n  {label}:")
        for i, sid in enumerate(songs, 1):
            name = model.song_names.get(sid, sid)
            gs = model.gap_stats.get(sid, {})
            gap = gs.get("current_gap", "?")
            era = model.song_eras.get(sid, "?")
            is_cover = "©" if model.is_cover.get(sid, False) else " "
            segue_mark = ""
            if i < len(songs):
                next_sid = songs[i]
                if sid in model.segue_matrix and next_sid in model.segue_matrix[sid]:
                    segue_mark = " >"
            bustout_mark = ""
            median = gs.get("median", 10)
            if gap != "?" and float(gap) > median * 2:
                bustout_mark = " *"
            lines.append(f"    {i:2d}.{is_cover} {name:<38s} gap={gap:<4}{era[:8]:>9s}"
                        f"{segue_mark}{bustout_mark}")

    return "\n".join(lines)
