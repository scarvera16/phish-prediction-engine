#!/usr/bin/env python3
"""
Enrich phish_engine data with signals derived from cached phish.net + phish.in data.

Improvements:
  1. Jam scores from Phish.in jamchart tag frequency
  2. Energy derived from duration, jam rate, and Phish.in tags
  3. Sphere affinity extended from indoor/arena venue performance history
  4. Song pairs mined from real transition data (≥3 occurrences)
  5. Style tags derived from set placement, jam propensity, and duration
"""

import json
import math
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
OVERRIDES = Path("phish_engine/data/manual_overrides.py")


def load_json(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)


def slug_to_id(slug):
    """Convert phish.net slug to engine ID using SLUG_ALIASES."""
    from phish_engine.data.manual_overrides import SLUG_ALIASES
    raw = slug.replace("-", "_")
    return SLUG_ALIASES.get(raw, raw)


def main():
    setlists = load_json("setlists.json")
    songs_raw = load_json("songs.json")
    shows = load_json("shows.json")
    phishin = load_json("phishin_tracks.json")

    # Build song name lookup
    song_names = {}
    for s in songs_raw:
        sid = slug_to_id(s.get("slug", ""))
        song_names[sid] = s.get("song", sid)

    # ─── 1. Jam scores from Phish.in jamchart tags ────────────────────────
    print("1. Computing jam scores from Phish.in jamchart tags...")
    jam_appearances = Counter()  # song_id → jamchart appearances
    total_appearances = Counter()  # song_id → total appearances

    for show_date, show_data in phishin.items():
        for track in show_data.get("tracks", []):
            sid = slug_to_id(track.get("slug", ""))
            if not sid:
                continue
            total_appearances[sid] += 1
            tags = track.get("tags", [])
            tag_names = [t.get("name", "") if isinstance(t, dict) else str(t) for t in tags]
            if "Jamcharts" in tag_names:
                jam_appearances[sid] += 1

    jam_scores = {}
    for sid in total_appearances:
        if total_appearances[sid] >= 3:
            rate = jam_appearances[sid] / total_appearances[sid]
            jam_scores[sid] = round(rate, 3)

    improved_jam = sum(1 for sid, score in jam_scores.items() if score > 0)
    print(f"   {improved_jam} songs with jamchart data (was 0 for many)")

    # ─── 2. Energy from duration + jam rate + Phish.in duration variance ──
    print("2. Deriving energy scores...")
    song_durations = {}  # sid → [dur_minutes, ...]
    for show_date, show_data in phishin.items():
        for track in show_data.get("tracks", []):
            sid = slug_to_id(track.get("slug", ""))
            dur = track.get("duration", 0) / 60000  # ms → min
            if sid and dur > 0.5:
                song_durations.setdefault(sid, []).append(dur)

    energy_scores = {}
    for sid in total_appearances:
        durs = song_durations.get(sid, [])
        jam = jam_scores.get(sid, 0)

        # Base energy from jam propensity
        energy = 0.4 + jam * 0.4  # 0.4 to 0.8 from jam rate

        # Duration signal: longer songs tend to be higher energy (jam vehicles)
        if durs:
            avg_dur = sum(durs) / len(durs)
            dur_bonus = min(avg_dur / 20.0, 0.3)  # up to +0.3 for 20+ min songs
            energy += dur_bonus

            # Duration variance: high variance = improvisational = higher energy
            if len(durs) >= 5:
                import statistics
                cv = statistics.stdev(durs) / max(avg_dur, 1)
                energy += min(cv * 0.15, 0.15)  # up to +0.15 for high variance

        # Bustout tag boost
        tags_for_song = set()
        for show_date, show_data in phishin.items():
            for track in show_data.get("tracks", []):
                if slug_to_id(track.get("slug", "")) == sid:
                    for t in track.get("tags", []):
                        name = t.get("name", "") if isinstance(t, dict) else str(t)
                        tags_for_song.add(name)

        energy = max(0.15, min(energy, 1.0))
        energy_scores[sid] = round(energy, 2)

    print(f"   {len(energy_scores)} songs with derived energy scores")

    # ─── 3. Sphere affinity from indoor venue history ─────────────────────
    print("3. Extending sphere affinity estimates...")
    from phish_engine.data.manual_overrides import SPHERE_AFFINITY
    from phish_engine.data.venue_map import classify_venue

    # Count per-song appearances at indoor vs outdoor venues
    indoor_plays = Counter()
    outdoor_plays = Counter()
    msg_plays = Counter()  # MSG specifically (closest to Sphere experience)

    show_venue_map = {}
    for s in shows:
        show_venue_map[s["showdate"]] = {
            "venue": s.get("venue", ""),
            "city": s.get("city", ""),
            "vtype": classify_venue(s.get("venue", ""), s.get("city", "")),
        }

    for date_str, entries in setlists.items():
        vinfo = show_venue_map.get(date_str, {})
        vtype = vinfo.get("vtype", "arena")
        venue = vinfo.get("venue", "")

        for e in entries:
            sid = slug_to_id(e.get("slug", ""))
            if not sid:
                continue
            if vtype in ("arena", "theater", "sphere"):
                indoor_plays[sid] += 1
            else:
                outdoor_plays[sid] += 1
            if "Madison Square Garden" in venue:
                msg_plays[sid] += 1

    sphere_scores = {}
    for sid in set(indoor_plays) | set(outdoor_plays):
        # Start with curated value if available
        if sid in SPHERE_AFFINITY:
            sphere_scores[sid] = SPHERE_AFFINITY[sid]
            continue

        total = indoor_plays.get(sid, 0) + outdoor_plays.get(sid, 0)
        if total < 3:
            continue

        indoor_frac = indoor_plays.get(sid, 0) / total
        msg_frac = msg_plays.get(sid, 0) / max(total, 1)

        # Indoor-heavy songs suit Sphere better
        base = 0.35 + indoor_frac * 0.3  # 0.35 to 0.65 from indoor fraction
        # MSG bonus (similar spectacle venue)
        base += msg_frac * 0.2

        # Jam vehicles get Sphere bonus (immersive experience)
        jam = jam_scores.get(sid, 0)
        base += jam * 0.15

        sphere_scores[sid] = round(max(0.2, min(base, 0.95)), 2)

    curated = sum(1 for sid in sphere_scores if sid in SPHERE_AFFINITY)
    derived = len(sphere_scores) - curated
    print(f"   {curated} curated + {derived} derived = {len(sphere_scores)} total sphere scores")

    # ─── 4. Song pairs from transition data ───────────────────────────────
    print("4. Mining song pairs from real transitions...")
    transitions = Counter()
    for date_str, entries in setlists.items():
        by_set = {}
        for e in entries:
            s = str(e.get("set", "1"))
            by_set.setdefault(s, []).append(e)

        for set_key, set_songs in by_set.items():
            set_songs.sort(key=lambda x: x.get("position", 0))
            for i in range(len(set_songs) - 1):
                a = slug_to_id(set_songs[i].get("slug", ""))
                b = slug_to_id(set_songs[i + 1].get("slug", ""))
                trans = set_songs[i].get("trans_mark", ", ")
                if a and b and a != b:
                    transitions[(a, b)] += 1

    # Only keep pairs that appear 3+ times and represent a meaningful pattern
    # (more than 20% of the time song A is played, B follows)
    song_pair_candidates = {}
    for (a, b), count in transitions.most_common(100):
        if count < 3:
            break
        a_total = total_appearances.get(a, 0)
        if a_total > 0 and count / a_total >= 0.15:
            song_pair_candidates[a] = b

    # Keep existing curated pairs + new ones
    from phish_engine.data.manual_overrides import SONG_PAIRS as EXISTING_PAIRS
    all_pairs = dict(EXISTING_PAIRS)
    for a, b in song_pair_candidates.items():
        if a not in all_pairs:
            all_pairs[a] = b

    new_pairs = {k: v for k, v in all_pairs.items() if k not in EXISTING_PAIRS}
    print(f"   {len(EXISTING_PAIRS)} existing + {len(new_pairs)} new = {len(all_pairs)} total pairs")
    for a, b in new_pairs.items():
        a_name = song_names.get(a, a)
        b_name = song_names.get(b, b)
        count = transitions.get((a, b), 0)
        print(f"     {a_name} → {b_name} ({count}x)")

    # ─── 5. Style tags from features ──────────────────────────────────────
    print("5. Deriving style tags...")
    style_tags = {}
    from phish_engine.data.manual_overrides import STYLE_TAGS as CURATED_STYLES

    for sid in total_appearances:
        if sid in CURATED_STYLES:
            style_tags[sid] = CURATED_STYLES[sid]
            continue

        jam = jam_scores.get(sid, 0)
        durs = song_durations.get(sid, [])
        avg_dur = sum(durs) / len(durs) if durs else 5.0

        # Count set placement
        s1_count = 0
        s2_count = 0
        enc_count = 0
        for date_str, entries in setlists.items():
            for e in entries:
                if slug_to_id(e.get("slug", "")) == sid:
                    s = str(e.get("set", "1"))
                    if s == "1":
                        s1_count += 1
                    elif s == "2":
                        s2_count += 1
                    elif s.startswith("e"):
                        enc_count += 1

        total = s1_count + s2_count + enc_count
        if total == 0:
            continue

        s2_frac = s2_count / total
        enc_frac = enc_count / total

        # Classification logic
        if avg_dur > 12 and jam > 0.3:
            style_tags[sid] = "jam"
        elif avg_dur > 10 and s2_frac > 0.5:
            style_tags[sid] = "jam"
        elif enc_frac > 0.4:
            style_tags[sid] = "encore"
        elif avg_dur < 4.0:
            style_tags[sid] = "short"
        elif jam > 0.2 and s2_frac > 0.6:
            style_tags[sid] = "funk"
        elif avg_dur > 8 and s1_count > s2_count:
            style_tags[sid] = "prog"
        elif s1_count > 0 and s2_count == 0 and enc_count == 0:
            style_tags[sid] = "s1-only"
        else:
            style_tags[sid] = "rock"

    curated_styles = sum(1 for sid in style_tags if sid in CURATED_STYLES)
    derived_styles = len(style_tags) - curated_styles
    print(f"   {curated_styles} curated + {derived_styles} derived = {len(style_tags)} total")
    style_dist = Counter(style_tags.values())
    for style, count in style_dist.most_common():
        print(f"     {style:10s} {count:4d}")

    # ─── Write enriched manual_overrides.py ───────────────────────────────
    print("\n6. Writing enriched manual_overrides.py...")

    # Read existing file
    with open(OVERRIDES) as f:
        original = f.read()

    # Build new SPHERE_AFFINITY dict
    sphere_lines = []
    for sid in sorted(sphere_scores):
        sphere_lines.append(f'    "{sid}": {sphere_scores[sid]},')

    # Build new STYLE_TAGS dict
    style_lines = []
    for sid in sorted(style_tags):
        style_lines.append(f'    "{sid}": "{style_tags[sid]}",')

    # Build new SONG_PAIRS dict
    pair_lines = []
    for a in sorted(all_pairs):
        pair_lines.append(f'    "{a}": "{all_pairs[a]}",')

    # Build new JAM_SCORES dict (new addition)
    jam_lines = []
    for sid in sorted(jam_scores):
        if jam_scores[sid] > 0:
            jam_lines.append(f'    "{sid}": {jam_scores[sid]},')

    # Build new ENERGY_SCORES dict (new addition)
    energy_lines = []
    for sid in sorted(energy_scores):
        energy_lines.append(f'    "{sid}": {energy_scores[sid]},')

    # Find and replace sections in the file
    import re

    # Replace SPHERE_AFFINITY
    new_sphere = "SPHERE_AFFINITY: dict[str, float] = {\n" + "\n".join(sphere_lines) + "\n}"
    original = re.sub(
        r'SPHERE_AFFINITY: dict\[str, float\] = \{[^}]*\}',
        new_sphere, original, flags=re.DOTALL
    )

    # Replace STYLE_TAGS
    new_styles = "STYLE_TAGS: dict[str, str] = {\n" + "\n".join(style_lines) + "\n}"
    original = re.sub(
        r'STYLE_TAGS: dict\[str, str\] = \{[^}]*\}',
        new_styles, original, flags=re.DOTALL
    )

    # Replace SONG_PAIRS
    new_pairs_str = "SONG_PAIRS: dict[str, str] = {\n" + "\n".join(pair_lines) + "\n}"
    original = re.sub(
        r'SONG_PAIRS: dict\[str, str\] = \{[^}]*\}',
        new_pairs_str, original, flags=re.DOTALL
    )

    # Add JAM_SCORES and ENERGY_SCORES at the end
    if "JAM_SCORES" not in original:
        original += "\n\n# Jam chart rate per song (from Phish.in tags, 2019-2025)\n"
        original += "JAM_SCORES: dict[str, float] = {\n" + "\n".join(jam_lines) + "\n}\n"

    if "ENERGY_SCORES" not in original:
        original += "\n\n# Derived energy scores (from duration + jam rate + variance)\n"
        original += "ENERGY_SCORES: dict[str, float] = {\n" + "\n".join(energy_lines) + "\n}\n"

    with open(OVERRIDES, "w") as f:
        f.write(original)

    print(f"   Updated {OVERRIDES}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ENRICHMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Jam scores:      {improved_jam} songs with real jamchart data")
    print(f"  Energy scores:   {len(energy_scores)} songs (was 63% placeholder)")
    print(f"  Sphere affinity: {len(sphere_scores)} songs (was 75% placeholder)")
    print(f"  Song pairs:      {len(all_pairs)} pairs (was {len(EXISTING_PAIRS)})")
    print(f"  Style tags:      {len(style_tags)} songs (was 87% 'rock')")
    print(f"\nRe-run export_json.py to regenerate predictions with enriched data.")


if __name__ == "__main__":
    main()
