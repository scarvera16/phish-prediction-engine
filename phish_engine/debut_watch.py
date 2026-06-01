"""
Debut watch — Trey-driven debut/bustout candidates for an upcoming Phish tour.

Stage 1/2 established (p << 0.0001 vs a year-shuffle null) that Trey's solo/TAB
tour is a genuine *leading indicator for debuts* — but only for fresh material,
and the base setlist model can't surface these at all (never-played songs are
outside its candidate pool). This module produces the ranked candidate list that
feeds a "debut watch" surface, separate from the core scorer.

Prior score blends three honest signals from the preview window:
  - freshness   : new Trey material crosses; 20-yr TAB warhorses essentially never
  - commitment  : how many times Trey played it in the window (featured vs one-off)
  - heat        : how recently he played it before the target date

Pure-Python, no model deps, so it can run from cached JSON each tour.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

PREVIEW_DAYS = 240
FRESH_DAYS = 540          # Trey-debut recency that separates fresh material from warhorses
BUSTOUT_DORMANT_DAYS = 730  # a Phish song unplayed this long counts as a bustout candidate


def _d(s: str) -> date:
    return date.fromisoformat(s[:10])


@dataclass
class DebutCandidate:
    slug: str
    kind: str            # "debut" (new to Phish) | "bustout" (dormant Phish song)
    score: float         # prior in ~[0, 1]
    trey_plays_in_window: int
    trey_first_played: str
    last_phish_play: str | None
    fresh: bool


def debut_candidates(
    trey_setlists: list[dict],
    phish_first_play: dict[str, str],
    phish_last_play: dict[str, str],
    target_date: date,
    preview_days: int = PREVIEW_DAYS,
    fresh_days: int = FRESH_DAYS,
) -> list[DebutCandidate]:
    """Rank Trey-driven debut/bustout candidates for a Phish tour at `target_date`.

    Parameters
    ----------
    trey_setlists      : flat phish.net rows for artistid=2 (need showdate, slug)
    phish_first_play   : slug -> earliest Phish play date (ISO) — defines "new to Phish"
    phish_last_play    : slug -> latest Phish play date (ISO) — defines bustout dormancy
    target_date        : first date of the upcoming Phish tour
    """
    win_lo = target_date - timedelta(days=preview_days)

    plays_in_win: dict[str, list[date]] = defaultdict(list)
    trey_first: dict[str, date] = {}
    for r in trey_setlists:
        slug = r.get("slug")
        if not slug:
            continue
        dt = _d(r["showdate"])
        if dt < target_date:
            trey_first[slug] = min(trey_first.get(slug, dt), dt)
        if win_lo <= dt < target_date:
            plays_in_win[slug].append(dt)

    out: list[DebutCandidate] = []
    for slug, dts in plays_in_win.items():
        first = phish_first_play.get(slug)
        last = phish_last_play.get(slug)
        new_to_phish = not first or _d(first) >= target_date
        if new_to_phish:
            kind = "debut"
        elif last and _d(last) < target_date - timedelta(days=BUSTOUT_DORMANT_DAYS):
            kind = "bustout"
        else:
            continue  # active Phish song — the core model owns this, not debut watch

        trey_age = (target_date - trey_first[slug]).days
        fresh = trey_age <= fresh_days

        # freshness: 1.0 for brand-new Trey material, decaying for old warhorses
        freshness = math.exp(-max(trey_age - fresh_days, 0) / 1825.0)  # 5-yr half-ish decay past the cutoff
        commitment = min(len(dts), 8) / 8.0
        most_recent = max(dts)
        heat = math.exp(-(target_date - most_recent).days / float(preview_days))
        score = round(0.55 * freshness + 0.30 * commitment + 0.15 * heat, 4)

        out.append(DebutCandidate(
            slug=slug, kind=kind, score=score,
            trey_plays_in_window=len(dts),
            trey_first_played=trey_first[slug].isoformat(),
            last_phish_play=last, fresh=fresh,
        ))

    out.sort(key=lambda c: c.score, reverse=True)
    return out
