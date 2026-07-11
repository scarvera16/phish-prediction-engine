#!/usr/bin/env python3
"""Repeat-gap tracker for the active tour.

For each song played more than once in the current tour, measure the gap (in
tour shows) between consecutive plays. This is the ground-truth data we need to
validate the engine's RECENT_NO_REPEAT hard-exclusion window: if a real repeat
lands inside that window, the guardrail would have wrongly blocked a correct
prediction (a false positive to watch).

Scopes to a single tour (default: the tour of the most recent played Phish show
in the given year), so it carries forward to summer/fall tours automatically.

Usage:
  scripts/repeat_gap_tracker.py [--year YYYY] [--tour <tourid|name substr>]
                                [--out PATH] [--quiet]

Reads phish_engine/data/cache/setlists.json. Writes a JSON snapshot (default
phish_engine/data/repeat_gaps.json) and prints a summary. Non-fatal by design:
never raises on missing data, so it's safe to drop into the roll pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "phish_engine", "data", "cache", "setlists.json")
DEFAULT_OUT = os.path.join(ROOT, "phish_engine", "data", "repeat_gaps.json")

# Keep the guardrail windows in sync with the engine rather than hardcoding.
try:
    sys.path.insert(0, ROOT)
    from phish_engine.predictor import RECENT_NO_REPEAT, VARIETY_MAX_DAYS
except Exception:
    RECENT_NO_REPEAT = 3
    VARIETY_MAX_DAYS = 14


def _phish_entries(raw: dict) -> list[dict]:
    out = []
    for date, entries in raw.items():
        for e in entries:
            if str(e.get("artistid")) == "1":
                out.append(e)
    return out


def _pick_tour(entries: list[dict], year: int, tour_arg: str | None) -> tuple[str, str]:
    """Return (tourid, tourname) to scope on."""
    in_year = [e for e in entries if str(e.get("showyear")) == str(year)]
    pool = in_year or entries
    if tour_arg:
        for e in pool:
            if str(e.get("tourid")) == tour_arg or tour_arg.lower() in str(e.get("tourname", "")).lower():
                return str(e.get("tourid")), str(e.get("tourname", ""))
    # Default: the tour of the most recent played show in the year.
    latest = max(pool, key=lambda e: e.get("showdate", ""), default=None)
    if latest is None:
        return "", ""
    return str(latest.get("tourid")), str(latest.get("tourname", ""))


def build_report(year: int, tour_arg: str | None) -> dict:
    if not os.path.exists(CACHE):
        return {"error": f"no setlists cache at {CACHE}"}
    with open(CACHE) as f:
        raw = json.load(f)

    entries = _phish_entries(raw)
    tourid, tourname = _pick_tour(entries, year, tour_arg)
    tour = [e for e in entries if str(e.get("tourid")) == tourid]
    if not tour:
        return {"error": f"no played shows for tour {tourid!r} (year {year})"}

    # Ordered list of tour shows, and the set of slugs played at each.
    dates = sorted({e["showdate"] for e in tour})
    idx_of = {d: i for i, d in enumerate(dates)}
    played_at = defaultdict(set)      # slug -> {show_index}
    name_of = {}                      # slug -> display name
    for e in tour:
        slug = e.get("slug", "")
        if not slug:
            continue
        played_at[slug].add(idx_of[e["showdate"]])
        name_of.setdefault(slug, e.get("song", slug))

    def days_between(d1: str, d2: str) -> int:
        from datetime import date
        a = date(*map(int, d1.split("-")))
        b = date(*map(int, d2.split("-")))
        return (b - a).days

    songs = []
    gap_hist = Counter()
    inside_window = []
    for slug, idxs in played_at.items():
        order = sorted(idxs)
        gaps = [order[i + 1] - order[i] for i in range(len(order) - 1)]
        rec = {
            "slug": slug,
            "name": name_of[slug],
            "plays": len(order),
            "show_dates": [dates[i] for i in order],
            "gaps": gaps,
            "min_gap": min(gaps) if gaps else None,
        }
        songs.append(rec)
        for k, g in enumerate(gaps):
            gap_hist[g] += 1
            # Mirror the engine exactly: the hard window is show-index based
            # AND calendar-gated, so a "gap 1" spanning a multi-week tour
            # break is not a guardrail false-positive.
            cal_days = days_between(dates[order[k]], dates[order[k + 1]])
            if g <= RECENT_NO_REPEAT and cal_days <= VARIETY_MAX_DAYS:
                inside_window.append(
                    {"slug": slug, "name": name_of[slug], "gap": g, "days": cal_days})

    repeated = sorted(
        [s for s in songs if s["plays"] >= 2],
        key=lambda s: (s["min_gap"], -s["plays"]),
    )
    return {
        "tour_id": tourid,
        "tour_name": tourname,
        "recent_no_repeat": RECENT_NO_REPEAT,
        "shows_played": len(dates),
        "date_range": [dates[0], dates[-1]],
        "unique_songs": len(played_at),
        "repeated_song_count": len(repeated),
        "gap_histogram": {str(g): gap_hist[g] for g in sorted(gap_hist)},
        "inside_window": sorted(inside_window, key=lambda x: x["gap"]),
        "repeated_songs": repeated,
    }


def print_summary(r: dict) -> None:
    if "error" in r:
        print(f"  repeat-gap tracker: {r['error']}")
        return
    print(f"  Tour: {r['tour_name']} ({r['tour_id']}) — {r['shows_played']} shows played "
          f"[{r['date_range'][0]} .. {r['date_range'][1]}]")
    print(f"  {r['unique_songs']} unique songs, {r['repeated_song_count']} repeated within the tour")
    if r["gap_histogram"]:
        hist = ", ".join(f"{g}:{n}" for g, n in r["gap_histogram"].items())
        print(f"  gap histogram (shows-between-plays : count) -> {hist}")
    win = r["recent_no_repeat"]
    if r["inside_window"]:
        print(f"  ⚠ {len(r['inside_window'])} real repeat(s) inside the RECENT_NO_REPEAT={win} window "
              f"(guardrail would block these):")
        for x in r["inside_window"]:
            print(f"      gap {x['gap']} ({x.get('days', '?')}d)  {x['name']}")
    else:
        print(f"  ✓ no real repeat fell inside the RECENT_NO_REPEAT={win} window")
    top = r["repeated_songs"][:8]
    if top:
        print("  fastest repeaters:")
        for s in top:
            print(f"      min-gap {s['min_gap']}  x{s['plays']}  {s['name']}")


def main() -> int:
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=datetime.datetime.now(datetime.timezone.utc).year,
                    help="tour year (default: current UTC year)")
    ap.add_argument("--tour", default=None, help="tourid or name substring; default = active tour")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        report = build_report(args.year, args.tour)
    except Exception as exc:  # never break the pipeline over the tracker
        report = {"error": f"{type(exc).__name__}: {exc}"}

    out_dir = os.path.dirname(args.out)
    if out_dir:  # a bare filename has no directory to create
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    if not args.quiet:
        print_summary(report)
        print(f"  wrote {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
