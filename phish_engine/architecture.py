"""
Architectural prediction layer: conditional Type-II likelihood.

The base engine collapses jam-chart and duration signal into a single
per-song scalar (`jam_score`). This module keeps the signal disaggregated
and asks a different question: given a song in a particular *context*
(set, venue type, performance era), how likely is it to go Type II?

"Type II" here is a proxy, built from two pieces of evidence that the
engine already ingests:

  1. phish.net `isjamchart` flag on the appearance.
  2. A duration outlier: the performance ran long relative to that song's
     own distribution (Phish.in track durations).

We treat a performance as a Type-II *event* if either piece fires. This is
deliberately a proxy, not a label. `isjamchart` flags "notable jam," not
"suspended peak" specifically, and a long version is not always a Type II
departure. So the internal name is `type_ii_likelihood`, and the
suspended-peak language stays in the display layer until the proxy is
validated against hand-labeled jams.

The core problem the model has to solve is sparsity. A cell like
(ghost, set2, sphere) may have two performances in the cache, or zero. Raw
rates on small cells are noise. So every cell is shrunk toward its parent
with a Beta-Binomial empirical-Bayes estimate: the cell rate borrows
strength from the song's global rate, which in turn borrows from the grand
mean across all songs. A cell with no data falls back cleanly to its parent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data.real_data import _pn_slug_to_song_id

# ── tuning constants ──────────────────────────────────────────────────────────

# A performance is a duration outlier if its z-score against the song's own
# duration distribution clears this threshold. ~1.0 keeps the long versions
# without flagging every slightly-above-average play.
DURATION_Z_THRESHOLD = 1.0

# Beta-Binomial shrinkage strength (pseudo-observations pulled from the parent).
# Higher = trust the parent more, the cell less. 8 means a cell needs roughly
# 8 plays before its own rate outweighs the parent prior.
SHRINKAGE_STRENGTH = 8.0

# A song needs at least this many duration samples before its mean/std are
# considered stable enough to z-score against.
MIN_DURATION_SAMPLES = 4

# Performance-era boundary. 3.0 = post-hiatus through the COVID stop,
# 4.0 = the return onward. Keyed off the *show* date, not the song debut.
ERA_4_START_YEAR = 2021


def performance_era(date: pd.Timestamp) -> str:
    """Bucket a show date into a performance era ('3.0' or '4.0')."""
    return "4.0" if date.year >= ERA_4_START_YEAR else "3.0"


def set_slot(set_number: str) -> str:
    """Map the engine set field to a coarse architectural slot."""
    return {"1": "set1", "2": "set2", "e": "encore"}.get(set_number, "set1")


# ── duration alignment ────────────────────────────────────────────────────────

def _load_phishin_by_appearance(data_dir: Path) -> dict[tuple[str, str], float]:
    """Per-appearance durations from Phish.in, keyed by (date_str, song_id).

    appearances_df.duration_min comes from phish.net tracktime, which is only
    ~2% populated. Phish.in covers ~94% of tracks, so we join it back to each
    appearance by date and song. Where a song appears twice in one show we
    take the longer track (the jam is the one we care about).
    """
    path = data_dir / "phishin_tracks.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)

    out: dict[tuple[str, str], float] = {}
    for date_str, show in data.items():
        show_date = show.get("date", date_str)
        for track in show.get("tracks", []):
            sid = _pn_slug_to_song_id(track.get("slug", ""))
            dur_min = track.get("duration", 0) / 60000.0  # ms → minutes
            if dur_min <= 0.5:
                continue
            key = (str(show_date), sid)
            if dur_min > out.get(key, 0.0):
                out[key] = dur_min
    return out


# ── the table ─────────────────────────────────────────────────────────────────

@dataclass
class ConditionalEstimate:
    """One smoothed Type-II likelihood for a context, with provenance."""
    p: float                # shrunken Type-II likelihood
    n: int                  # raw plays in this exact cell
    raw_rate: float | None  # unsmoothed cell rate (None if n == 0)
    fallback_level: str     # which parent the estimate leaned on most


class ArchitectureTable:
    """Conditional Type-II likelihood, queryable by (song, slot, venue, era).

    Build once from the loaded engine frames, then query. Estimates are
    hierarchical: cell → song-global → grand-mean. Sparse and empty cells
    degrade gracefully to the parent instead of returning noise or nothing.
    """

    def __init__(self, evidence: pd.DataFrame):
        self._ev = evidence
        self._grand_mean = float(evidence["type_ii"].mean())
        self._song_rate = (
            evidence.groupby("song_id")["type_ii"].agg(["mean", "size"])
        )
        # Pre-aggregate the full conditioning cell.
        self._cell = (
            evidence.groupby(["song_id", "slot", "venue_type", "era"])["type_ii"]
            .agg(["mean", "size"])
        )

    # — internal: Beta-Binomial shrinkage of a rate toward a prior —
    @staticmethod
    def _shrink(k: float, n: float, prior: float, strength: float) -> float:
        return (k + strength * prior) / (n + strength)

    def _song_prior(self, song_id: str) -> float:
        if song_id in self._song_rate.index:
            row = self._song_rate.loc[song_id]
            return self._shrink(
                row["mean"] * row["size"], row["size"], self._grand_mean,
                SHRINKAGE_STRENGTH,
            )
        return self._grand_mean

    def estimate(
        self,
        song_id: str,
        slot: str | None = None,
        venue_type: str | None = None,
        era: str | None = None,
    ) -> ConditionalEstimate:
        """Smoothed Type-II likelihood for a context.

        Any of slot/venue_type/era may be None to query a marginal. The
        estimate shrinks the matching cell toward the song's global rate.
        """
        prior = self._song_prior(song_id)

        # Collect the matching subset at the requested granularity.
        ev = self._ev[self._ev["song_id"] == song_id]
        level = "song"
        for col, val, name in [
            ("slot", slot, "slot"),
            ("venue_type", venue_type, "venue"),
            ("era", era, "era"),
        ]:
            if val is not None:
                ev = ev[ev[col] == val]
                level = name

        n = int(len(ev))
        if n == 0:
            return ConditionalEstimate(p=prior, n=0, raw_rate=None,
                                       fallback_level="song")
        k = float(ev["type_ii"].sum())
        raw = k / n
        p = self._shrink(k, n, prior, SHRINKAGE_STRENGTH)
        return ConditionalEstimate(p=round(p, 4), n=n, raw_rate=round(raw, 4),
                                   fallback_level=level)

    def song_block(
        self,
        song_id: str,
        min_global: float = 0.25,
        min_cell_n: int = 5,
    ) -> dict | None:
        """Compact per-song architecture block for frontend export, or None.

        Returns None for songs that don't clear the global Type-II floor
        (non-vehicles) or that have no well-sampled venue cell, so the export
        only carries the jam-capable songs the surface is about. Venue rows are
        kept only when they have at least `min_cell_n` plays, except a populated
        Sphere cell is always kept once the residency data lands.

        Shape (tuples keep the serialized payload small):
            {"t2": 0.28, "t2s2": 0.39, "venues": [["arena", 0.70, 26], ...]}
        """
        g = self.estimate(song_id)
        if g.n == 0 or g.p < min_global:
            return None
        s2 = self.estimate(song_id, slot="set2")
        vc = self.venue_contrast(song_id, slot="set2")
        venues = []
        for _, r in vc.iterrows():
            n = int(r["n_plays"])
            if n >= min_cell_n or (r["venue_type"] == "sphere" and n > 0):
                venues.append([
                    r["venue_type"],
                    round(float(r["type_ii_likelihood"]), 2),
                    n,
                ])
        if not venues:
            return None
        block = {"t2": round(g.p, 2), "t2s2": round(s2.p, 2), "venues": venues}

        # Era trend: song-wide (all slots) so the "jammier now?" comparison has
        # enough samples; only a real comparison if BOTH eras are well-sampled.
        # (The cache starts in 2019, so "3.0" here is late-3.0 only; songs that
        # debuted in 4.0 simply have no 3.0 cell and get no era row.)
        ec = self.era_contrast(song_id)
        eras = [
            [r["era"], round(float(r["type_ii_likelihood"]), 2), int(r["n_plays"])]
            for _, r in ec.iterrows() if int(r["n_plays"]) >= min_cell_n
        ]
        if len(eras) == 2:
            block["eras"] = sorted(eras, key=lambda e: e[0])  # 3.0 then 4.0
        return block

    def venue_contrast(self, song_id: str, slot: str = "set2") -> pd.DataFrame:
        """Type-II likelihood for a song/slot across venue types.

        This is the surface the product wants: "Ghost in set 2 runs Type II
        X% at arenas vs Y% outdoors." Returns one row per venue type present
        in the data, sorted by smoothed likelihood.
        """
        venues = sorted(self._ev["venue_type"].unique())
        rows = []
        for vt in venues:
            est = self.estimate(song_id, slot=slot, venue_type=vt)
            rows.append({
                "venue_type": vt,
                "type_ii_likelihood": est.p,
                "n_plays": est.n,
                "raw_rate": est.raw_rate,
            })
        return pd.DataFrame(rows).sort_values(
            "type_ii_likelihood", ascending=False
        ).reset_index(drop=True)

    def era_contrast(self, song_id: str, slot: str | None = None) -> pd.DataFrame:
        """Type-II likelihood across performance eras (song-wide by default).

        Surfaces the "is this song jammier now than it used to be" trend.
        One row per era present in the data, sorted by era label.
        """
        eras = sorted(self._ev["era"].unique())
        rows = []
        for era in eras:
            est = self.estimate(song_id, slot=slot, era=era)
            rows.append({
                "era": era,
                "type_ii_likelihood": est.p,
                "n_plays": est.n,
                "raw_rate": est.raw_rate,
            })
        return pd.DataFrame(rows)


# ── builder ───────────────────────────────────────────────────────────────────

def build_architecture_table(
    songs_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    appearances_df: pd.DataFrame,
    data_dir: str | Path,
) -> ArchitectureTable:
    """Assemble per-appearance Type-II evidence and return a queryable table.

    Joins appearances to show context (venue_type, era), attaches Phish.in
    durations, computes per-song duration z-scores, and marks each appearance
    as a Type-II event if it is jam-charted or a duration outlier.
    """
    data_dir = Path(data_dir)
    ev = appearances_df.copy()

    # — show context: venue_type + performance era —
    show_ctx = shows_df.set_index("show_id")[["venue_type", "date"]]
    ev = ev.join(show_ctx, on="show_id", rsuffix="_show")
    ev["era"] = ev["date_show"].apply(performance_era)
    ev["slot"] = ev["set_number"].apply(set_slot)

    # — duration: prefer Phish.in, fall back to phish.net tracktime —
    pin = _load_phishin_by_appearance(data_dir)
    ev["date_str"] = ev["date"].dt.strftime("%Y-%m-%d")
    ev["dur"] = [
        pin.get((d, s), 0.0)
        for d, s in zip(ev["date_str"], ev["song_id"])
    ]
    fallback = ev["dur"] <= 0.0
    ev.loc[fallback, "dur"] = ev.loc[fallback, "duration_min"]

    # — per-song duration z-score (only where the song has enough samples) —
    valid = ev[ev["dur"] > 0.5]
    stats = valid.groupby("song_id")["dur"].agg(["mean", "std", "size"])
    ev = ev.join(stats, on="song_id", rsuffix="_dstat")
    safe_std = ev["std"].replace(0, np.nan)
    ev["dur_z"] = (ev["dur"] - ev["mean"]) / safe_std
    ev["dur_outlier"] = (
        (ev["dur_z"] >= DURATION_Z_THRESHOLD)
        & (ev["size"] >= MIN_DURATION_SAMPLES)
        & (ev["dur"] > 0.5)
    ).fillna(False)

    # — Type-II event: jam-charted OR a duration outlier —
    ev["isjamchart"] = ev["isjamchart"].astype(bool)
    ev["type_ii"] = (ev["isjamchart"] | ev["dur_outlier"]).astype(int)

    keep = ["song_id", "slot", "venue_type", "era",
            "isjamchart", "dur", "dur_z", "dur_outlier", "type_ii"]
    return ArchitectureTable(ev[keep].reset_index(drop=True))
