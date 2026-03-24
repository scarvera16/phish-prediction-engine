"""
Load real Phish.net cached data and transform to engine DataFrame contracts.

Reads shows.json, setlists.json, and songs.json (cached from Phish.net API)
and produces (songs_df, shows_df, appearances_df) compatible with the engine's
feature engineering, scoring, and prediction pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .manual_overrides import SLUG_ALIASES, SPHERE_AFFINITY, STYLE_TAGS
from .venue_map import classify_venue

# ── slug helpers ──────────────────────────────────────────────────────────────

def _pn_slug_to_song_id(slug: str) -> str:
    """Convert a Phish.net slug to an engine song_id.

    Replaces hyphens with underscores, then applies SLUG_ALIASES
    for the 38+ songs with short canonical engine IDs.
    """
    raw = slug.replace("-", "_")
    return SLUG_ALIASES.get(raw, raw)


def _set_value_to_str(set_val: str) -> str:
    """Map Phish.net set field to engine convention ('1', '2', 'e')."""
    if set_val in ("1",):
        return "1"
    if set_val in ("2", "3", "4"):
        return "2"
    if set_val in ("e", "e2", "E", "E2"):
        return "e"
    return "1"  # fallback


def _parse_tracktime(tt: str) -> float | None:
    """Parse 'MM:SS' or 'H:MM:SS' string to minutes. Returns None if empty."""
    if not tt or not isinstance(tt, str):
        return None
    parts = tt.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60.0
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60.0
    except (ValueError, IndexError):
        return None
    return None


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_shows(data_dir: Path, start_year: int) -> pd.DataFrame:
    """Load shows.json → shows_df matching engine contract."""
    with open(data_dir / "shows.json") as f:
        raw = json.load(f)

    rows = []
    for s in raw:
        if s.get("artistid") != 1:
            continue
        year = int(s.get("showyear", 0))
        if year < start_year:
            continue
        rows.append({
            "show_id": str(s["showid"]),
            "date": pd.Timestamp(s["showdate"]),
            "venue_name": s.get("venue", ""),
            "city": s.get("city", ""),
            "state": s.get("state", ""),
            "tour": s.get("tour_name", s.get("tourname", "")),
            "venue_type": classify_venue(s.get("venue", ""), s.get("city", "")),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df["show_num"] = range(1, len(df) + 1)
    return df


def _load_appearances(data_dir: Path, shows_df: pd.DataFrame) -> pd.DataFrame:
    """Load setlists.json → appearances_df matching engine contract."""
    with open(data_dir / "setlists.json") as f:
        raw = json.load(f)

    # Build lookup from showdate → show_id, show_num, date
    show_lookup = {}
    for _, row in shows_df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        show_lookup[date_str] = {
            "show_id": row["show_id"],
            "show_num": row["show_num"],
            "date": row["date"],
        }

    rows = []
    for date_str, entries in raw.items():
        if date_str not in show_lookup:
            continue
        sl = show_lookup[date_str]

        # Group by set to recompute per-set positions
        by_set: dict[str, list] = {}
        for e in entries:
            if e.get("artistid") != 1:
                continue
            set_str = _set_value_to_str(str(e.get("set", "1")))
            by_set.setdefault(set_str, []).append(e)

        for set_str, set_entries in by_set.items():
            # Sort by original position within the set
            set_entries.sort(key=lambda x: x.get("position", 0))
            for pos_idx, e in enumerate(set_entries, start=1):
                dur = _parse_tracktime(e.get("tracktime", ""))
                rows.append({
                    "show_id": sl["show_id"],
                    "song_id": _pn_slug_to_song_id(e.get("slug", "")),
                    "set_number": set_str,
                    "position": pos_idx,
                    "duration_min": dur if dur is not None else 0.0,
                    "date": sl["date"],
                    "show_num": sl["show_num"],
                    "isjamchart": bool(e.get("isjamchart", 0)),
                })

    return pd.DataFrame(rows)


def _build_songs_df(
    data_dir: Path,
    appearances_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    min_plays: int,
) -> pd.DataFrame:
    """Build songs_df from real data, indexed by song_id."""
    # Load Phish.net song catalog for metadata
    with open(data_dir / "songs.json") as f:
        raw_songs = json.load(f)

    pn_catalog: dict[str, dict] = {}
    for s in raw_songs:
        sid = _pn_slug_to_song_id(s.get("slug", ""))
        pn_catalog[sid] = {
            "name": s.get("song", sid),
            "debut_year": _parse_debut_year(s.get("debut", "")),
            "times_played_alltime": s.get("times_played", 0),
        }

    # Count appearances in our training window
    song_counts = appearances_df.groupby("song_id").size()
    eligible = song_counts[song_counts >= min_plays].index.tolist()

    total_shows = len(shows_df)
    max_show_num = shows_df["show_num"].max()

    records = []
    for sid in eligible:
        song_app = appearances_df[appearances_df["song_id"] == sid]
        n_plays = len(song_app)
        meta = pn_catalog.get(sid, {"name": sid, "debut_year": 1990, "times_played_alltime": n_plays})

        # ── Slot weights (derived from actual set/position data) ──
        s1_count = len(song_app[song_app["set_number"] == "1"])
        s2_count = len(song_app[song_app["set_number"] == "2"])
        enc_count = len(song_app[song_app["set_number"] == "e"])

        # Opener / closer detection
        show_opener_count = len(song_app[
            (song_app["set_number"] == "1") & (song_app["position"] == 1)
        ])
        s2_opener_count = len(song_app[
            (song_app["set_number"] == "2") & (song_app["position"] == 1)
        ])
        # Closer: last song in its set per show
        closer_count = 0
        for _, grp in song_app.groupby(["show_id", "set_number"]):
            max_pos = appearances_df[
                (appearances_df["show_id"] == grp.iloc[0]["show_id"]) &
                (appearances_df["set_number"] == grp.iloc[0]["set_number"])
            ]["position"].max()
            if grp["position"].max() == max_pos:
                closer_count += 1

        # ── Avg gap in show-count units ──
        played_show_nums = sorted(song_app["show_num"].unique())
        if len(played_show_nums) >= 2:
            gaps = np.diff(played_show_nums)
            avg_gap = float(np.mean(gaps))
        else:
            avg_gap = float(max_show_num)  # never-repeat fallback

        # ── Jam score from jamchart rate ──
        if "isjamchart" in song_app.columns:
            jam_rate = song_app["isjamchart"].sum() / max(n_plays, 1)
        else:
            jam_rate = 0.0
        jam_score = float(np.clip(jam_rate, 0.0, 1.0))

        # ── Duration: median of last 20 performances with tracktime ──
        # Median is more robust than mean against occasional epic jams.
        # Using recent performances avoids bias from older sparse data.
        recent_app = song_app.sort_values("show_num", ascending=False)
        valid_durations = recent_app[recent_app["duration_min"] > 0]["duration_min"].head(20)
        avg_dur = float(valid_durations.median()) if len(valid_durations) > 0 else 5.0

        # ── Energy heuristic (jam-heavy + loud songs get high energy) ──
        energy = SPHERE_AFFINITY.get(sid, 0.5)  # use sphere as proxy if no override
        if jam_score > 0.3:
            energy = max(energy, 0.7 + jam_score * 0.2)
        energy = float(np.clip(energy, 0.0, 1.0))

        records.append({
            "song_id": sid,
            "name": meta["name"],
            "debut_year": meta["debut_year"],
            "avg_duration_min": round(avg_dur, 1),
            "jam_score": round(jam_score, 3),
            "s1_raw": s1_count,
            "s2_raw": s2_count,
            "enc_raw": enc_count,
            "show_opener_raw": show_opener_count,
            "set2_opener_raw": s2_opener_count,
            "set_closer_raw": closer_count,
            "avg_gap": round(avg_gap, 1),
            "style": STYLE_TAGS.get(sid, "rock"),
            "energy": round(energy, 2),
            "sphere_affinity": SPHERE_AFFINITY.get(sid, 0.5),
        })

    df = pd.DataFrame(records).set_index("song_id")

    # ── Scale raw counts to weights (max ≈ 9.0) ──
    for col_raw, col_weight in [
        ("s1_raw", "s1_weight"),
        ("s2_raw", "s2_weight"),
        ("enc_raw", "enc_weight"),
        ("show_opener_raw", "show_opener_weight"),
        ("set2_opener_raw", "set2_opener_weight"),
        ("set_closer_raw", "set_closer_weight"),
    ]:
        max_val = df[col_raw].max()
        if max_val > 0:
            df[col_weight] = (df[col_raw] / max_val) * 9.0
        else:
            df[col_weight] = 0.0

    # Drop raw columns
    df = df.drop(columns=[
        "s1_raw", "s2_raw", "enc_raw",
        "show_opener_raw", "set2_opener_raw", "set_closer_raw",
    ])

    # ── Compute normalized slot fractions ──
    total_weight = df["s1_weight"] + df["s2_weight"] + df["enc_weight"]
    total_weight = total_weight.replace(0, 1)  # avoid div by zero
    df["s1_frac"] = df["s1_weight"] / total_weight
    df["s2_frac"] = df["s2_weight"] / total_weight
    df["enc_frac"] = df["enc_weight"] / total_weight

    return df


def _parse_debut_year(debut_str: str) -> int:
    """Extract year from 'YYYY-MM-DD' debut date string."""
    if not debut_str:
        return 1990
    try:
        return int(debut_str[:4])
    except (ValueError, IndexError):
        return 1990


# ── public API ────────────────────────────────────────────────────────────────

def load_real_data(
    data_dir: str | Path,
    min_plays: int = 3,
    start_year: int = 2019,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load real Phish.net cached data and transform to engine contracts.

    Args:
        data_dir: Path to directory containing shows.json, setlists.json, songs.json
        min_plays: Minimum appearances in the training window to include a song
        start_year: Earliest year to include shows from

    Returns:
        (songs_df, shows_df, appearances_df) matching engine DataFrame contracts
    """
    data_dir = Path(data_dir)

    shows_df = _load_shows(data_dir, start_year)
    appearances_df = _load_appearances(data_dir, shows_df)
    songs_df = _build_songs_df(data_dir, appearances_df, shows_df, min_plays)

    # Filter appearances to only songs in the catalog
    appearances_df = appearances_df[
        appearances_df["song_id"].isin(songs_df.index)
    ].reset_index(drop=True)

    return songs_df, shows_df, appearances_df
