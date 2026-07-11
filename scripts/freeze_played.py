#!/usr/bin/env python3
"""
Freeze already-played shows after a rolling re-prediction.

A rolling export (ROLL_AS_OF=...) re-predicts every show as of the current
point in the tour. But shows that have already locked were bet against the
prediction that was live when they locked, and their scores are settled. So
those must not change. This splices the played/locked shows' blocks from the
previously-live prediction file back into the freshly-rolled one.

Freeze rule: a show is frozen once its date is strictly BEFORE the cutoff.
Picks lock at showtime (7pm venue-local), and the daily rolls run before
that, so on show day the show is still re-predicted (using last night's
actual setlist) and whatever is live at showtime becomes the locked call.
The morning-after roll then freezes it.

Shows are matched by DATE (never show_num, which renumbers if the tour list
changes), and the venue must agree. A played show missing from the live file
aborts the roll: silently substituting a fresh prediction would rewrite a
settled bet.

Only the per-show `shows` blocks are frozen; catalog stats, bustouts, and
super-card aggregates take the fresh tour-so-far values (informational, not
locked bets).

Usage:
  python scripts/freeze_played.py <rolled.json> <live.json> <cutoff YYYY-MM-DD> <out.json>
"""
import json
import re
import sys


def main() -> None:
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    rolled_path, live_path, cutoff, out_path = sys.argv[1:5]

    # ISO comparison below is lexicographic; an unpadded date like 2026-7-10
    # would silently freeze the whole tour.
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
        sys.exit(f"cutoff must be YYYY-MM-DD (zero-padded), got: {cutoff!r}")

    rolled = json.load(open(rolled_path))
    live = json.load(open(live_path))
    live_by_date = {s["date"]: s for s in live["shows"]}

    frozen = []
    merged_shows = []
    for s in rolled["shows"]:
        if s["date"] < cutoff:
            prev = live_by_date.get(s["date"])
            if prev is None:
                sys.exit(
                    f"played show {s['date']} is missing from the live file "
                    f"({live_path}) — refusing to freeze against a fresh "
                    f"prediction. Is the live snapshot from the right tour?"
                )
            if prev.get("venue") != s.get("venue"):
                sys.exit(
                    f"venue mismatch on {s['date']}: live has "
                    f"{prev.get('venue')!r}, rolled has {s.get('venue')!r} — "
                    f"refusing to splice across mismatched tours."
                )
            # Keep the locked call, but under the rolled file's numbering so a
            # renumbered tour list can't misalign show_num with date.
            block = dict(prev)
            block["show_num"] = s["show_num"]
            merged_shows.append(block)
            frozen.append(s["show_num"])
        else:
            merged_shows.append(s)  # fresh rolling prediction
    rolled["shows"] = merged_shows

    json.dump(rolled, open(out_path, "w"), indent=2)
    fresh = [s["show_num"] for s in rolled["shows"] if s["show_num"] not in frozen]
    print(f"froze {len(frozen)} played shows {frozen}; re-predicted {len(fresh)} shows")


if __name__ == "__main__":
    main()
