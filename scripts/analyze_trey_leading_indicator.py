"""
Stage 1 — does Trey's solo/TAB tour preview the following Phish summer tour?

Descriptive (no model retrain). For each year with both a Trey pre-summer tour
and a Phish summer tour we build a 2x2 table over a realistic song universe:

    exposure = song played in Trey's preview window (the months before summer)
    outcome  = song played in Phish's summer tour

and report relative risk  RR = P(summer | preview) / P(summer | no preview).
RR ~ 1 => no signal. RR > 1 => Trey's tour is a leading indicator.

Sliced by all / Trey-originals / covers, plus a separate DEBUT crossover test
(songs new to Phish that Trey had just been playing) which is where any real
signal should concentrate. Ends with the actionable 2026 candidate list.
"""
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "phish_engine" / "data" / "cache"

PREVIEW_DAYS = 240          # how far back before summer to look at Trey shows
TRAILING_CATALOG_DAYS = 3 * 365  # "rotatable book" = Phish songs played in this trailing window
SUMMER_MONTHS = {6, 7, 8}


def d(s: str) -> date:
    return date.fromisoformat(s[:10])


def load():
    trey = json.loads((CACHE / "trey_setlists.json").read_text())
    phish_by_year = json.loads((CACHE / "phish_setlists_by_year.json").read_text())
    songs = json.loads((CACHE / "songs.json").read_text())
    phish_debut = {s["slug"]: s.get("debut") for s in songs if s.get("slug")}
    return trey, phish_by_year, phish_debut


def clean(rows):
    """Keep real song rows with a slug."""
    return [r for r in rows if r.get("slug")]


def rr_ci(a, b, c, dd):
    """Relative risk + 95% CI from a 2x2: exposed(a hit / b miss), unexposed(c/dd)."""
    n_exp, n_unexp = a + b, c + dd
    if n_exp == 0 or n_unexp == 0 or a == 0 or c == 0:
        p1 = a / n_exp if n_exp else float("nan")
        p0 = c / n_unexp if n_unexp else float("nan")
        rr = (p1 / p0) if (p0 and p0 == p0) else float("nan")
        return p1, p0, rr, (float("nan"), float("nan"))
    p1, p0 = a / n_exp, c / n_unexp
    rr = p1 / p0
    se = math.sqrt((1 / a) - (1 / n_exp) + (1 / c) - (1 / n_unexp))
    lo = math.exp(math.log(rr) - 1.96 * se)
    hi = math.exp(math.log(rr) + 1.96 * se)
    return p1, p0, rr, (lo, hi)


def main():
    trey, phish_by_year, phish_debut = load()
    trey = clean(trey)
    trey_by_date = defaultdict(set)        # showdate -> set(slug)
    trey_orig = {}                         # slug -> is_original (1/0)
    for r in trey:
        trey_by_date[r["showdate"]].add(r["slug"])
        trey_orig.setdefault(r["slug"], r.get("is_original", 0))
    trey_dates = sorted(trey_by_date)

    # Phish plays: slug -> sorted list of showdates; and per-year summer sets
    phish_plays = defaultdict(list)
    phish_rows = []
    for yr, rows in phish_by_year.items():
        for r in clean(rows):
            phish_plays[r["slug"]].append(r["showdate"])
            phish_rows.append(r)
    for s in phish_plays:
        phish_plays[s].sort()

    pooled = defaultdict(lambda: [0, 0, 0, 0])   # slice -> [a,b,c,d]
    per_year = []
    debut_rows = []

    for y in range(2009, 2026):
        # Phish summer tour = Jun-Aug shows that year
        summer_dates = sorted({r["showdate"] for r in phish_rows
                               if r["showdate"][:4] == str(y) and d(r["showdate"]).month in SUMMER_MONTHS})
        if len(summer_dates) < 3:
            continue
        summer_start = d(summer_dates[0])
        summer_set = {r["slug"] for r in phish_rows
                      if r["showdate"] in set(summer_dates)}

        # Trey preview window: shows in [summer_start - PREVIEW_DAYS, summer_start)
        win_lo = summer_start - timedelta(days=PREVIEW_DAYS)
        preview = set()
        n_prev_shows = 0
        for sd in trey_dates:
            if win_lo <= d(sd) < summer_start:
                preview |= trey_by_date[sd]
                n_prev_shows += 1
        if n_prev_shows == 0:
            continue

        # Rotatable book = songs Phish played in trailing window before summer
        cat_lo = summer_start - timedelta(days=TRAILING_CATALOG_DAYS)
        book = {s for s, dates in phish_plays.items()
                if any(cat_lo <= d(x) < summer_start for x in dates)}
        universe = book | preview

        # 2x2 over the universe, plus slices
        cells = defaultdict(lambda: [0, 0, 0, 0])
        for s in universe:
            exposed = s in preview
            hit = s in summer_set
            slices = ["all"]
            if exposed:
                slices.append("orig" if trey_orig.get(s) else "cover")
            for sl in slices:
                idx = (0 if hit else 1) if exposed else (2 if hit else 3)
                cells[sl][idx] += 1
                pooled[sl][idx] += 1

        a, b, c, e = cells["all"]
        p1, p0, rr, _ = rr_ci(a, b, c, e)
        per_year.append((y, n_prev_shows, len(preview), len(summer_set), len(universe), p1, p0, rr))

        # ── Debut crossover: Trey-preview songs NOT yet in Phish's book ──
        for s in preview:
            deb = phish_debut.get(s)
            already = deb and d(deb) < summer_start
            if already:
                continue  # already a Phish song -> not a debut candidate
            debuted = s in summer_set
            debut_rows.append((y, s, "orig" if trey_orig.get(s) else "cover", debuted))

    # ── Report ──
    print("=" * 78)
    print("STAGE 1 — Trey solo/TAB as a leading indicator for Phish summer tours")
    print(f"preview window = {PREVIEW_DAYS}d before summer start; "
          f"universe = Phish trailing-3yr book ∪ Trey-preview songs")
    print("=" * 78)
    print("\nPer-year (rotation signal, 'all' slice):")
    print(f"{'yr':>4} {'TreyShows':>9} {'TreySongs':>9} {'SummerSongs':>11} "
          f"{'P(sum|prev)':>11} {'P(sum|no)':>10} {'RR':>6}")
    for (y, ns, npv, nsum, nuni, p1, p0, rr) in per_year:
        print(f"{y:>4} {ns:>9} {npv:>9} {nsum:>11} {p1:>11.3f} {p0:>10.3f} {rr:>6.2f}")

    print("\nPooled relative risk (all years, 95% CI):")
    for sl in ["all", "orig", "cover"]:
        a, b, c, e = pooled[sl]
        p1, p0, rr, (lo, hi) = rr_ci(a, b, c, e)
        label = {"all": "all songs", "orig": "Trey originals", "cover": "covers"}[sl]
        ci = f"[{lo:.2f}, {hi:.2f}]" if lo == lo else "[n/a]"
        print(f"  {label:<16} exposed {a:>4}/{a+b:<4} hit={p1:.3f}   "
              f"baseline {c:>4}/{c+e:<5} hit={p0:.3f}   RR={rr:.2f} {ci}")

    # Debut crossover summary
    print("\nDEBUT crossover — Trey previewed a song NOT yet in Phish's book:")
    tot = len(debut_rows)
    crossed = [r for r in debut_rows if r[3]]
    print(f"  {tot} preview songs were new-to-Phish; {len(crossed)} were then played "
          f"by Phish that summer ({100*len(crossed)/max(tot,1):.1f}%)")
    by = defaultdict(lambda: [0, 0])
    for (_, _, kind, dbt) in debut_rows:
        by[kind][0] += dbt
        by[kind][1] += 1
    for kind in ("orig", "cover"):
        h, n = by[kind]
        print(f"    {kind:<6} {h}/{n} crossed ({100*h/max(n,1):.1f}%)")
    print("  examples of crossovers (year, song, type):")
    for (y, s, kind, _) in sorted(crossed)[:25]:
        print(f"    {y}  {s:<32} {kind}")

    # ── Actionable: 2026 preview songs that are debut/bustout candidates ──
    print("\n" + "=" * 78)
    print("ACTIONABLE — Trey 2026 preview songs that are debut/bustout candidates")
    print("=" * 78)
    today = date(2026, 6, 1)
    win_lo = today - timedelta(days=PREVIEW_DAYS)
    prev26 = set()
    for sd in trey_dates:
        if win_lo <= d(sd) <= today:
            prev26 |= trey_by_date[sd]
    cands = []
    for s in prev26:
        last = phish_plays.get(s, [])
        last_play = last[-1] if last else None
        # debut candidate: never in Phish book; bustout: not played by Phish in ~2yr
        if not last_play:
            cands.append((s, "DEBUT?", trey_orig.get(s)))
        elif d(last_play) < today - timedelta(days=730):
            cands.append((s, f"bustout (last {last_play})", trey_orig.get(s)))
    print(f"Trey played {len(prev26)} distinct songs in the {PREVIEW_DAYS}d before {today}.")
    print(f"Of those, {len(cands)} are debut/bustout candidates for Phish summer '26:")
    for (s, tag, og) in sorted(cands, key=lambda x: (x[1], x[0])):
        print(f"  {s:<34} {tag:<28} {'original' if og else 'cover'}")


if __name__ == "__main__":
    main()
