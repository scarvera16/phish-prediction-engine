"""
Stage 0 — pull Trey/TAB + multi-year Phish setlists from phish.net into cache.

phish.net v5 covers Trey Anastasio as artistid=2 (TAB + solo + acoustic), so a
single source gives us both sides of the leading-indicator question. Phish is
artistid=1.

Writes to phish_engine/data/cache/:
  - trey_setlists.json   flat list of Trey song entries (1982-present)
  - trey_shows.json      Trey show metadata
  - phish_setlists_by_year.json   {year: [phish song entries]} for the study window

The phish.net API key is read from the web app's .env.local (PHISHNET_API_KEY)
or the PHISHNET_API_KEY env var — we never hardcode or print it.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "phish_engine" / "data" / "cache"
WEB_ENV = Path("/Users/carveranderson/Documents/Development/phish-setlist-predictor/.env.local")

# Phish setlist history to pull for the study (need pre-cutoff summers + debuts).
PHISH_YEARS = list(range(2009, 2027))


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


def get(url: str) -> list:
    # phish.net 403s the default urllib UA; send a browser-like one.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read().decode())
    if isinstance(d, dict):
        if d.get("error"):
            raise RuntimeError(d.get("error_message", d.get("error")))
        return d.get("data", [])
    return d


def main():
    key = load_key()
    base = "https://api.phish.net/v5"
    CACHE.mkdir(parents=True, exist_ok=True)

    print("→ Trey shows (artistid=2)")
    trey_shows = get(f"{base}/shows/artistid/2.json?apikey={key}")
    (CACHE / "trey_shows.json").write_text(json.dumps(trey_shows))
    print(f"  {len(trey_shows)} shows")

    print("→ Trey setlists (artistid=2)")
    trey_sets = get(f"{base}/setlists/artistid/2.json?apikey={key}")
    trey_sets = [r for r in trey_sets if r.get("artistid") == 2]
    (CACHE / "trey_setlists.json").write_text(json.dumps(trey_sets))
    print(f"  {len(trey_sets)} song entries across "
          f"{len({r['showdate'] for r in trey_sets})} shows")

    print("→ Phish setlists by year")
    by_year = {}
    for y in PHISH_YEARS:
        rows = get(f"{base}/setlists/showyear/{y}.json?apikey={key}")
        rows = [r for r in rows if r.get("artistid") == 1]
        by_year[str(y)] = rows
        print(f"  {y}: {len(rows)} entries / {len({r['showdate'] for r in rows})} shows")
        time.sleep(0.3)
    (CACHE / "phish_setlists_by_year.json").write_text(json.dumps(by_year))

    print("done.")


if __name__ == "__main__":
    main()
