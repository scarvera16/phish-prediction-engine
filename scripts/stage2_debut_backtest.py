"""
Stage 2a — does the Trey debut-injector beat chance?  (significance gate)

The base model can NEVER predict a debut: load_real_data(min_plays=3) drops
never-played songs from the candidate pool, so base-model debut recall = 0 by
construction. The proposed feature injects, for a target summer tour, the songs
Trey just played that Phish has never played — as debut candidates.

The question isn't "does Trey play songs that become Phish songs" (some songs
are destined to cross regardless of when). It's whether Trey previews THIS
year's debuts specifically. We test that with a year-shuffle permutation:

    observed   = Σ_Y |TreyPreview_Y ∩ PhishDebuts_Y|
    null        = Σ_Y |TreyPreview_Y ∩ PhishDebuts_σ(Y)|   for random derangements σ

If Trey is a genuine *leading indicator*, observed >> null (low p). We also
test the Stage-1 "fresh material" sharpening (restrict to songs Trey himself
debuted recently) to see if it trades recall for precision.
"""
import json
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "phish_engine" / "data" / "cache"
PREVIEW_DAYS = 240
TREY_FRESH_DAYS = 540          # "fresh material" = Trey debuted it within ~18mo
N_PERM = 20000
SEED = 42


def d(s):
    return date.fromisoformat(s[:10])


def load():
    trey = json.loads((CACHE / "trey_setlists.json").read_text())
    pby = json.loads((CACHE / "phish_setlists_by_year.json").read_text())
    songs = json.loads((CACHE / "songs.json").read_text())
    debut = {s["slug"]: s.get("debut") for s in songs if s.get("slug")}
    return trey, pby, debut


def build(trey, pby, debut, fresh_only):
    trey_play = defaultdict(list)
    for r in trey:
        if r.get("slug"):
            trey_play[r["slug"]].append(r["showdate"])
    for s in trey_play:
        trey_play[s].sort()

    phish_rows = [r for rows in pby.values() for r in rows if r.get("slug")]
    phish_first = {}  # slug -> earliest phish play date in our data
    for r in phish_rows:
        s = r["slug"]
        if s not in phish_first or r["showdate"] < phish_first[s]:
            phish_first[s] = r["showdate"]

    years = {}
    for y in range(2009, 2026):
        sd = sorted({r["showdate"] for r in phish_rows
                     if r["showdate"][:4] == str(y) and d(r["showdate"]).month in (6, 7, 8)})
        if len(sd) < 3:
            continue
        start = d(sd[0])
        sdset = {d(x) for x in sd}   # date objects, to match debut dates

        # Trey preview: new-to-Phish songs played in the window
        preview = set()
        for s, plays in trey_play.items():
            in_win = any(start - timedelta(days=PREVIEW_DAYS) <= d(p) < start for p in plays)
            if not in_win:
                continue
            deb = debut.get(s)
            if deb and d(deb) < start:
                continue  # already in Phish's book -> not a debut candidate
            if fresh_only:
                trey_debut = d(plays[0])
                if (start - trey_debut).days > TREY_FRESH_DAYS:
                    continue  # old TAB warhorse
            preview.add(s)

        # Phish debuts this summer = phish.net debut date inside the summer window
        debuts = {s for s, dd in debut.items() if dd and d(dd) in sdset}
        years[y] = (preview, debuts)
    return years


def evaluate(years, label):
    obs = sum(len(p & dd) for (p, dd) in years.values())
    tot_debuts = sum(len(dd) for (_, dd) in years.values())
    tot_cands = sum(len(p) for (p, _) in years.values())
    recall = obs / max(tot_debuts, 1)
    precision = obs / max(tot_cands, 1)

    # year-shuffle permutation null (derangement of debut-years)
    rng = random.Random(SEED)
    previews = [p for (p, _) in years.values()]
    debutsets = [dd for (_, dd) in years.values()]
    n = len(previews)
    ge = 0
    null_vals = []
    for _ in range(N_PERM):
        perm = list(range(n))
        rng.shuffle(perm)
        # force derangement (no year maps to itself)
        if any(perm[i] == i for i in range(n)):
            for i in range(n):
                if perm[i] == i:
                    j = rng.randrange(n)
                    perm[i], perm[j] = perm[j], perm[i]
        val = sum(len(previews[i] & debutsets[perm[i]]) for i in range(n))
        null_vals.append(val)
        if val >= obs:
            ge += 1
    p = (ge + 1) / (N_PERM + 1)
    null_mean = sum(null_vals) / len(null_vals)

    print(f"\n── {label} ──")
    print(f"  candidates injected : {tot_cands}")
    print(f"  actual Phish debuts : {tot_debuts}   (base-model recall = 0, by construction)")
    print(f"  correct crossovers  : {obs}")
    print(f"  recall  = {recall:.3f}   precision = {precision:.3f}")
    print(f"  permutation null    : mean {null_mean:.2f} crossovers if Trey's preview were")
    print(f"                        matched to a RANDOM year's debuts")
    print(f"  observed {obs} vs null mean {null_mean:.2f}   ->  p = {p:.4f}  "
          f"({'SIGNIFICANT' if p < 0.05 else 'not significant'})")
    return p


def main():
    trey, pby, debut = load()
    print("=" * 74)
    print("STAGE 2a — Trey debut-injector vs year-shuffle null")
    print(f"preview={PREVIEW_DAYS}d  fresh<= {TREY_FRESH_DAYS}d  perms={N_PERM}")
    print("=" * 74)

    years_all = build(trey, pby, debut, fresh_only=False)
    evaluate(years_all, "RULE A: any Trey-preview song new to Phish")

    years_fresh = build(trey, pby, debut, fresh_only=True)
    evaluate(years_fresh, "RULE B: + Trey 'fresh material' filter (debuted in TAB <=18mo)")

    print("\nPer-year recall (Rule A): of each summer's debuts, how many Trey previewed")
    for y, (p, dd) in years_all.items():
        hit = len(p & dd)
        print(f"  {y}: {hit}/{len(dd)} debuts  ({len(p)} candidates injected)")


if __name__ == "__main__":
    main()
