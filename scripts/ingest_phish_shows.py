#!/usr/bin/env python3
"""
Ingest new Phish shows into the engine cache (idempotent).

Pulls phish.net setlists for the given year(s), plus Phish.in track
durations, and merges any shows not already present into:
  cache/shows.json, cache/setlists.json, cache/phishin_tracks.json

The cache had been frozen at 2025; this is how it gets refreshed after a
tour. Re-run with the new year(s) once phish.net has the setlists and the
community has charted the jams.

Usage:
  python scripts/ingest_phish_shows.py            # default: current+next year
  python scripts/ingest_phish_shows.py 2026       # a specific year
  python scripts/ingest_phish_shows.py 2026 2027  # several

The phish.net API key is read from PHISHNET_API_KEY or the web app's
.env.local. We never hardcode or print it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "phish_engine" / "data" / "cache"
WEB_ENV = Path("/Users/carveranderson/Documents/Development/phish-setlist-predictor/.env.local")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def load_key() -> str:
    key = os.environ.get("PHISHNET_API_KEY")
    if key:
        return key.strip()
    if WEB_ENV.exists():
        for line in WEB_ENV.read_text().splitlines():
            m = re.match(r'^\s*PHISHNET_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
            if m:
                return m.group(1)
    sys.exit("PHISHNET_API_KEY not found (env or web app .env.local)")


def get(url: str, timeout: int = 90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    if isinstance(d, dict):
        if d.get("error"):
            raise RuntimeError(d.get("error_message", d.get("error")))
        return d.get("data", d)
    return d


def _show_record(entries: list) -> dict:
    """Build a shows.json record from a date's setlist entries."""
    e = entries[0]
    dt = e["showdate"]
    _, m, d = dt.split("-")
    return {
        "showid": e["showid"],
        "showyear": str(e["showyear"]),
        "showmonth": int(m),
        "showday": int(d),
        "showdate": dt,
        "artistid": 1,
        "exclude_from_stats": 0,
        "venueid": e.get("venueid"),
        "venue": e.get("venue", ""),
        "city": e.get("city", ""),
        "state": e.get("state", ""),
        "tour_name": e.get("tourname", ""),
    }


def main() -> None:
    years = sys.argv[1:]
    if not years:
        years = ["2026", "2027"]
    key = load_key()
    base = "https://api.phish.net/v5"

    shows = json.loads((CACHE / "shows.json").read_text())
    setlists = json.loads((CACHE / "setlists.json").read_text())
    phishin = json.loads((CACHE / "phishin_tracks.json").read_text())
    have_dates = {s["showdate"] for s in shows}

    new_dates: list[str] = []
    for y in years:
        rows = [r for r in get(f"{base}/setlists/showyear/{y}.json?apikey={key}")
                if str(r.get("artistid")) == "1"]
        by_date: dict[str, list] = {}
        for r in rows:
            by_date.setdefault(r["showdate"], []).append(r)
        for dt, entries in sorted(by_date.items()):
            if dt in have_dates:
                continue
            shows.append(_show_record(entries))
            setlists[dt] = entries
            have_dates.add(dt)
            new_dates.append(dt)
        print(f"{y}: {len(by_date)} shows on phish.net, "
              f"{sum(1 for d in by_date if d in new_dates)} new")
        time.sleep(0.3)

    if not new_dates:
        print("Cache already current. Nothing to add.")
        return

    # Phish.in durations for the new dates (best-effort; jam-chart signal
    # works without them, durations only sharpen the outlier component).
    pin_ok = 0
    for dt in new_dates:
        try:
            d = get(f"https://phish.in/api/v2/shows/{dt}", timeout=30)
            tracks = [{"slug": t.get("slug", ""), "duration": t.get("duration", 0)}
                      for t in d.get("tracks", [])]
            if tracks:
                phishin[dt] = {"date": dt, "tracks": tracks}
                pin_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  phishin {dt}: {exc}")
        time.sleep(0.3)

    # Match the cache's on-disk formatting (indent=2) so the diff is just the
    # added shows, not a reflow of the whole file.
    (CACHE / "shows.json").write_text(json.dumps(shows, indent=2))
    (CACHE / "setlists.json").write_text(json.dumps(setlists, indent=2))
    (CACHE / "phishin_tracks.json").write_text(json.dumps(phishin, indent=2))

    print(f"\nAdded {len(new_dates)} shows ({new_dates[0]} .. {new_dates[-1]}); "
          f"phishin durations for {pin_ok}.")
    print(f"Cache now: {len(shows)} shows.")


if __name__ == "__main__":
    main()
