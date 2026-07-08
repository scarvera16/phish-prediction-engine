#!/usr/bin/env python3
"""
Freeze already-played shows after a rolling re-prediction.

A rolling export (ROLL_AS_OF=...) re-predicts every show as of the current
point in the tour. But shows that have already locked were bet against the
prediction that was live when they locked, and their scores are settled. So
those must not change. This splices the played/locked shows' blocks from the
previously-live prediction file back into the freshly-rolled one.

Freeze rule (agreed): a show is frozen once its date has arrived (<= cutoff),
matching the app's per-show pick locking. Only the per-show `shows` blocks are
frozen; catalog stats, bustouts, and super-card aggregates take the fresh
tour-so-far values (they are informational, not locked bets).

Usage:
  python scripts/freeze_played.py <rolled.json> <live.json> <cutoff YYYY-MM-DD> <out.json>
"""
import json
import sys


def main() -> None:
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    rolled_path, live_path, cutoff, out_path = sys.argv[1:5]

    rolled = json.load(open(rolled_path))
    live = json.load(open(live_path))
    live_by_num = {s["show_num"]: s for s in live["shows"]}

    frozen = []
    merged_shows = []
    for s in rolled["shows"]:
        if s["date"] <= cutoff and s["show_num"] in live_by_num:
            merged_shows.append(live_by_num[s["show_num"]])  # keep the locked call
            frozen.append(s["show_num"])
        else:
            merged_shows.append(s)  # fresh rolling prediction
    rolled["shows"] = merged_shows

    json.dump(rolled, open(out_path, "w"), indent=2)
    fresh = [s["show_num"] for s in rolled["shows"] if s["show_num"] not in frozen]
    print(f"froze {len(frozen)} played shows {frozen}; re-predicted {len(fresh)} shows")


if __name__ == "__main__":
    main()
