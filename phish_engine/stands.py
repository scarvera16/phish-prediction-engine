"""
Stand detection.

A *stand* is a maximal run of consecutive shows at the same venue. It is the
unit Phish actually organizes a tour around, and the unit that governs song
repetition: the band never repeats a song within a stand, but replays ~58% of
its material at later stops. A 3-night Deer Creek is one stand; a one-night stop
on the way there is a stand of size 1.

This matters everywhere downstream:
  - prediction: no-repeat exclusions reset at each stand boundary
  - picks / locking: a stand is a "run" (lock together); a one-nighter locks alone
  - UI: "night 2 of 3 at the Gorge" vs a single show

Gaps don't split a stand as long as no *other* venue intervenes: the Sphere '26
residency was nine nights at one venue across three weekends, and it was a single
no-repeat engagement. So a stand is defined by venue continuity in date order,
not by calendar adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass


def detect_stands(venues: list[str]) -> list[int]:
    """Assign each show (given in date order) to a stand id.

    A new stand begins whenever the venue differs from the previous show.

    >>> detect_stands(["Gorge", "Gorge", "Gorge", "Deer Creek"])
    [0, 0, 0, 1]
    >>> detect_stands(["A", "B", "A"])   # returning to a venue is a new stand
    [0, 1, 2]
    """
    stand_ids: list[int] = []
    sid = -1
    prev = _SENTINEL
    for v in venues:
        if v != prev:
            sid += 1
        stand_ids.append(sid)
        prev = v
    return stand_ids


@dataclass(frozen=True)
class StandPosition:
    stand_id: int
    night: int        # 1-based night within the stand
    size: int         # total nights in the stand
    is_opener: bool   # first night of the stand
    is_closer: bool   # last night of the stand (tends to jam hardest)


def stand_positions(venues: list[str]) -> list[StandPosition]:
    """Per-show stand position metadata, in the same order as `venues`."""
    ids = detect_stands(venues)
    sizes: dict[int, int] = {}
    for sid in ids:
        sizes[sid] = sizes.get(sid, 0) + 1

    out: list[StandPosition] = []
    seen: dict[int, int] = {}
    for sid in ids:
        seen[sid] = seen.get(sid, 0) + 1
        night, size = seen[sid], sizes[sid]
        out.append(StandPosition(
            stand_id=sid, night=night, size=size,
            is_opener=(night == 1), is_closer=(night == size),
        ))
    return out


# Sentinel distinct from any real venue name (including None).
_SENTINEL = object()
