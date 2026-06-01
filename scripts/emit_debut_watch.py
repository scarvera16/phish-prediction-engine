"""
Run the debut-watch generator: validate the prior's ranking on history, then
emit the live 2026 candidate list.

Uses phish.net catalog debut / last_played as authoritative full-history Phish
first/last play (our setlist cache only goes back to 2009).
"""
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from phish_engine.debut_watch import debut_candidates, _d

CACHE = Path(__file__).resolve().parent.parent / "phish_engine" / "data" / "cache"
SUMMER_2026_START = date(2026, 7, 3)   # approx; refine to the real first summer date when known


def load():
    trey = json.loads((CACHE / "trey_setlists.json").read_text())
    pby = json.loads((CACHE / "phish_setlists_by_year.json").read_text())
    songs = json.loads((CACHE / "songs.json").read_text())
    first = {s["slug"]: s.get("debut") for s in songs if s.get("slug") and s.get("debut")}
    last = {s["slug"]: s.get("last_played") for s in songs if s.get("slug") and s.get("last_played")}
    return trey, pby, first, last


def summer_debuts(pby, first, year):
    rows = [r for rows in pby.values() for r in rows if r.get("slug")]
    sd = sorted({r["showdate"] for r in rows
                 if r["showdate"][:4] == str(year) and _d(r["showdate"]).month in (6, 7, 8)})
    if len(sd) < 3:
        return None, None
    sdset = {_d(x) for x in sd}
    debuts = {s for s, dd in first.items() if dd and _d(dd) in sdset}
    return _d(sd[0]), debuts


def main():
    trey, pby, first, last = load()

    print("=" * 70)
    print("Prior validation — does the rank put true crossovers near the top?")
    print("=" * 70)
    ks = [5, 10, 20]
    agg = {k: [0, 0] for k in ks}      # k -> [hits, total_debuts]
    for y in range(2009, 2026):
        start, debuts = summer_debuts(pby, first, y)
        if start is None or not debuts:
            continue
        cands = debut_candidates(trey, first, last, start)
        ranked = [c.slug for c in cands]
        for k in ks:
            topk = set(ranked[:k])
            agg[k][0] += len(topk & debuts)
            agg[k][1] += len(debuts)
    for k in ks:
        h, t = agg[k]
        print(f"  recall@{k:<2} = {h}/{t} = {100*h/max(t,1):.1f}%  of all summer debuts caught in Trey top-{k}")

    print("\n" + "=" * 70)
    print(f"LIVE — debut watch for Phish summer 2026 (target {SUMMER_2026_START})")
    print("=" * 70)
    cands = debut_candidates(trey, first, last, SUMMER_2026_START)
    print(f"{'score':>6}  {'kind':<8} {'fresh':<6} {'TABplays':>8}  {'TreyDebut':<11} song")
    for c in cands[:20]:
        print(f"{c.score:>6.3f}  {c.kind:<8} {str(c.fresh):<6} {c.trey_plays_in_window:>8}  "
              f"{c.trey_first_played:<11} {c.slug}")

    payload = [c.__dict__ for c in cands]
    out = CACHE.parent / "debut_watch_2026.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(payload)} candidates -> {out}")


if __name__ == "__main__":
    main()
