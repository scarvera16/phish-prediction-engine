#!/usr/bin/env python3
"""
Calibrate the Jam Call hit-detection thresholds (docs/jam-call-spec.md §9,
frontend repo).

The rule under test, per performance:
    jam_event = dur >= ABS_MIN
                OR (dur >= REL_RATIO * career_avg AND dur >= REL_FLOOR)
    never_jammed(song) = jam_score < 0.05 AND career_avg < 6 min
                         AND no modern-era jam-chart entry
    moonshot = jam_event by a never_jammed song

Outputs, per candidate ABS_MIN bar:
  1. distribution of base-tier qualifiers per show (target: median 1-3)
  2. hit rate of a naive "house" strategy (always call the most obvious
     in-rotation jam vehicle) -> informs the base payout
  3. moonshot events (must stay exactly the 3 known historical ones)

Usage:
  python scripts/jam_call_calibrate.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "phish_engine" / "data" / "cache"

REL_RATIO = 2.0
REL_FLOOR_MIN = 8.0
NEVER_JAMMED_AVG_MAX = 6.0
MIN_PLAYS_FOR_REL = 10          # relative rule needs history to trust the avg
ABS_CANDIDATES = [10, 11, 12, 13, 14, 15]

setlists = json.loads((CACHE / "setlists.json").read_text())
phishin = json.loads((CACHE / "phishin_tracks.json").read_text())

# ── career stats from phish.in ────────────────────────────────────────────────
# phish.in gives sandwich/reprise legs their own slugs (tweezer-2, tweezer-3).
# Fold those into the base song, or legs masquerade as tiny "songs" with
# garbage averages. Only strip a -N suffix when the base slug really exists.
_raw_slugs: set[str] = set()
for show in phishin.values():
    for t in show.get("tracks", []):
        if t.get("slug"):
            _raw_slugs.add(t["slug"])

import re as _re
def canon_slug(slug: str) -> str:
    m = _re.fullmatch(r"(.+)-\d+", slug)
    if m and m.group(1) in _raw_slugs:
        return m.group(1)
    return slug

durs: dict[str, list[float]] = defaultdict(list)          # slug -> minutes
for show in phishin.values():
    for t in show.get("tracks", []):
        if t.get("duration") and t.get("slug"):
            durs[canon_slug(t["slug"])].append(t["duration"] / 60000)
career_avg = {s: sum(v) / len(v) for s, v in durs.items()}

# First jam-chart DATE per song: "never jammed" is a statement about the world
# before a given night, or a song's own first jam retroactively disqualifies it.
first_jamchart: dict[str, str] = {}
name_of: dict[str, str] = {}
for d, entries in setlists.items():
    for e in entries:
        if str(e.get("artistid")) != "1" or not e.get("slug"):
            continue
        slug = canon_slug(e["slug"])
        name_of[slug] = e.get("song", e["slug"])
        if e.get("isjamchart") and (slug not in first_jamchart or d < first_jamchart[slug]):
            first_jamchart[slug] = d

def never_jammed(slug: str, as_of: str) -> bool:
    # No jam-chart entry BEFORE this night, short average, real history.
    fj = first_jamchart.get(slug)
    return ((fj is None or fj >= as_of)
            and career_avg.get(slug, 99) < NEVER_JAMMED_AVG_MAX
            and len(durs.get(slug, [])) >= 3)

def jam_event(slug: str, minutes: float, abs_min: float) -> bool:
    if minutes >= abs_min:
        return True
    avg = career_avg.get(slug)
    if avg is None or len(durs[slug]) < MIN_PLAYS_FOR_REL:
        return False
    return minutes >= REL_RATIO * avg and minutes >= REL_FLOOR_MIN

# ── per-show qualifier scan ──────────────────────────────────────────────────
show_dates = sorted(d for d in phishin if phishin[d].get("tracks"))

print(f"{len(show_dates)} shows with duration data\n")
print(f"{'ABS bar':>8} {'med/show':>9} {'p90/show':>9} {'0-qual shows':>13} "
      f"{'house hit%':>11} {'moonshots':>10}")

for abs_min in ABS_CANDIDATES:
    per_show: list[int] = []
    house_hits = 0
    house_calls = 0
    moonshots: list[tuple[str, str, float]] = []
    recent: list[set[str]] = []   # last-3-shows played sets, for the house pick

    for d in show_dates:
        tracks = [(canon_slug(t["slug"]), t["duration"] / 60000)
                  for t in phishin[d]["tracks"] if t.get("duration")]
        quals = [(s, m) for s, m in tracks if jam_event(s, m, abs_min)]
        per_show.append(len(quals))
        for s, m in quals:
            if never_jammed(s, d) and m >= REL_RATIO * career_avg.get(s, 99):
                moonshots.append((d, s, m))

        # Naive house strategy: call the biggest in-rotation jam vehicle not
        # played in the last 3 shows. "In rotation" = played in the last 15
        # shows; "biggest" = highest career average duration (jam vehicles
        # float to the top: Tweezer, Carini, Ghost...).
        rotation: set[str] = set().union(*recent[-15:]) if recent else set()
        excluded: set[str] = set().union(*recent[-3:]) if recent else set()
        pool = [s for s in rotation - excluded if s in career_avg]
        if pool:
            pick = max(pool, key=lambda s: career_avg[s])
            house_calls += 1
            if any(s == pick and jam_event(s, m, abs_min) for s, m in tracks):
                house_hits += 1
        recent.append({s for s, _ in tracks})

    zero = sum(1 for n in per_show if n == 0)
    house = 100 * house_hits / house_calls if house_calls else 0
    print(f"{abs_min:>7}m {st.median(per_show):>9.0f} "
          f"{sorted(per_show)[int(len(per_show)*0.9)]:>9} "
          f"{zero:>6} ({100*zero/len(per_show):>4.1f}%) "
          f"{house:>10.1f} {len(moonshots):>10}")

# detail at the spec's default bar
ABS = 12
print(f"\nMoonshot events at ABS={ABS} (spec expects the 3 known):")
seen = set()
for d in show_dates:
    for t in phishin[d]["tracks"]:
        if not t.get("duration"):
            continue
        s, m = canon_slug(t["slug"]), t["duration"] / 60000
        if (jam_event(s, m, ABS) and never_jammed(s, d)
                and m >= REL_RATIO * career_avg.get(s, 99) and (d, s) not in seen):
            seen.add((d, s))
            print(f"  {d}  {name_of.get(s, s):<30} {m:5.1f}m (avg {career_avg[s]:.1f})")
