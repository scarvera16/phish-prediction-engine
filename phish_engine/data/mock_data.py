"""
Mock Phish show history generator.

Produces ~280 shows from January 2019 through December 2025, with realistic:
  - Tour structure (spring / summer / fall / NYE runs)
  - Venue types (arena, theater, outdoor, festival, special)
  - Setlists generated using slot-affinity weights + gap-pressure sampling
  - No song repeats within a multi-night run (soft constraint via penalty)
  - Special "Sphere 2024" shows included as training data
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from .songs import get_songs_df, SONG_PAIRS

# ---------------------------------------------------------------------------
# Venue pool
# ---------------------------------------------------------------------------
VENUES = [
    # (name, city, state, venue_type)
    ("Madison Square Garden",         "New York",        "NY", "arena"),
    ("Fenway Park",                    "Boston",          "MA", "outdoor"),
    ("Dicks Sporting Goods Park",      "Commerce City",   "CO", "outdoor"),
    ("Gorge Amphitheatre",             "George",          "WA", "outdoor"),
    ("Hampton Coliseum",               "Hampton",         "VA", "arena"),
    ("Bill Graham Civic Auditorium",   "San Francisco",   "CA", "arena"),
    ("KFC Yum! Center",                "Louisville",      "KY", "arena"),
    ("Rupp Arena",                     "Lexington",       "KY", "arena"),
    ("Riverbend Music Center",         "Cincinnati",      "OH", "outdoor"),
    ("Merriweather Post Pavilion",     "Columbia",        "MD", "outdoor"),
    ("Northwell Health at Jones Beach","Wantagh",         "NY", "outdoor"),
    ("Saratoga Performing Arts Center","Saratoga Springs","NY", "outdoor"),
    ("Alpine Valley Music Theatre",    "East Troy",       "WI", "outdoor"),
    ("DTE Energy Music Theatre",       "Clarkston",       "MI", "outdoor"),
    ("Xcel Energy Center",             "Saint Paul",      "MN", "arena"),
    ("Climate Pledge Arena",           "Seattle",         "WA", "arena"),
    ("Kia Forum",                      "Inglewood",       "CA", "arena"),
    ("Bridgestone Arena",              "Nashville",       "TN", "arena"),
    ("United Center",                  "Chicago",         "IL", "arena"),
    ("Sphere",                         "Las Vegas",       "NV", "sphere"),
]

VENUE_WEIGHTS = [
    0.10, 0.06, 0.06, 0.06, 0.06,
    0.05, 0.04, 0.04, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.04, 0.04,
    0.04, 0.04, 0.04, 0.04, 0.04,
]

# ---------------------------------------------------------------------------
# Tour calendar skeleton
# ---------------------------------------------------------------------------
_TOUR_CALENDAR = [
    # (label,                  start,                end,               n_shows, multi_night_run_size)
    ("Spring 2019",       date(2019, 4, 19),  date(2019, 5, 5),   13, 3),
    ("Summer 2019",       date(2019, 7, 16),  date(2019, 8, 11),  22, 3),
    ("Fall 2019",         date(2019, 10, 18), date(2019, 11, 3),  12, 2),
    ("NYE 2019",          date(2019, 12, 28), date(2020, 1, 1),    4, 4),
    ("Riviera Maya 2020", date(2020, 1, 22),  date(2020, 1, 26),   4, 4),
    ("Atlantic City 2021",date(2021, 7, 30),  date(2021, 8, 1),    3, 3),
    ("Summer 2021",       date(2021, 8, 6),   date(2021, 9, 5),   24, 3),
    ("Fall 2021",         date(2021, 10, 22), date(2021, 11, 6),  12, 3),
    ("NYE 2021",          date(2021, 12, 28), date(2022, 1, 1),    4, 4),
    ("Spring 2022",       date(2022, 4, 22),  date(2022, 5, 22),  20, 3),
    ("Summer 2022",       date(2022, 7, 15),  date(2022, 8, 28),  30, 3),
    ("Fall 2022",         date(2022, 10, 21), date(2022, 11, 13), 14, 3),
    ("NYE 2022",          date(2022, 12, 28), date(2023, 1, 1),    4, 4),
    ("Spring 2023",       date(2023, 4, 21),  date(2023, 5, 21),  20, 3),
    ("Summer 2023",       date(2023, 7, 14),  date(2023, 8, 20),  28, 3),
    ("Fall 2023",         date(2023, 10, 20), date(2023, 11, 12), 14, 2),
    ("NYE 2023",          date(2023, 12, 28), date(2024, 1, 1),    4, 4),
    ("Sphere 2024",       date(2024, 4, 18),  date(2024, 4, 28),   6, 6),
    ("Summer 2024",       date(2024, 7, 12),  date(2024, 8, 18),  24, 3),
    ("Fall 2024",         date(2024, 10, 18), date(2024, 11, 10), 14, 3),
    ("NYE 2024",          date(2024, 12, 28), date(2025, 1, 1),    5, 5),
    ("Spring 2025",       date(2025, 4, 18),  date(2025, 5, 18),  20, 3),
    ("Summer 2025",       date(2025, 7, 11),  date(2025, 8, 17),  26, 3),
    ("Fall 2025",         date(2025, 10, 17), date(2025, 11, 9),  14, 3),
    ("NYE 2025",          date(2025, 12, 28), date(2026, 1, 1),    4, 4),
]


def _date_range(start: date, end: date, n: int) -> list:
    """Spread n dates across start-end, roughly on weekends."""
    delta = (end - start).days
    step = max(1, delta // n)
    dates = []
    cur = start
    for _ in range(n):
        dates.append(cur)
        cur += timedelta(days=step)
        if cur > end:
            cur = end
    return sorted(set(dates))[:n]


def _sample_songs(
    n: int,
    weight_col: str,
    songs_df: pd.DataFrame,
    used_show: set,
    used_run: set,
    gap_tracker: dict,
    show_num: int,
    rng: np.random.Generator,
    enforce_pair_after: str | None = None,
) -> list:
    """
    Sample n songs for a single slot in a set.
    Weights combine slot affinity x gap pressure, penalised for run repeats.
    """
    candidates = [s for s in songs_df.index if s not in used_show]

    raw_w = []
    for sid in candidates:
        base = float(songs_df.loc[sid, weight_col])
        last = gap_tracker.get(sid, 0)
        cur_gap = show_num - last
        avg_gap = float(songs_df.loc[sid, "avg_gap"])
        gap_factor = min(cur_gap / max(avg_gap, 1.0), 3.0)
        run_pen = 0.05 if sid in used_run else 1.0
        raw_w.append(base * gap_factor * run_pen)

    total = sum(raw_w)
    if total <= 0:
        raw_w = [1.0] * len(candidates)
        total = float(len(candidates))

    probs = np.array(raw_w) / total

    # Handle forced pair (e.g. Hydrogen always follows Mike's)
    if enforce_pair_after and enforce_pair_after in candidates:
        chosen = [enforce_pair_after]
        candidates2 = [c for c in candidates if c != enforce_pair_after]
        if n > 1 and candidates2:
            idx2 = [i for i, c in enumerate(candidates) if c in candidates2]
            p2 = probs[idx2]
            p2 = p2 / p2.sum()
            extra = rng.choice(candidates2, size=min(n - 1, len(candidates2)), replace=False, p=p2).tolist()
            chosen += extra
        return chosen

    n = min(n, len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False, p=probs).tolist()
    return chosen


def generate_show_history(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate mock Phish show history.

    Returns
    -------
    shows_df : DataFrame
        Columns: show_id, date, venue_name, city, state, tour, venue_type, show_num
    appearances_df : DataFrame
        Columns: show_id, song_id, set_number ('1','2','e'), position, duration_min, date, show_num
    """
    rng = np.random.default_rng(seed)
    songs_df = get_songs_df()
    all_ids = list(songs_df.index)

    # gap_tracker: song_id -> show_number of last appearance
    gap_tracker: dict[str, int] = {sid: 0 for sid in all_ids}
    show_num = 0

    shows_rows = []
    appearance_rows = []

    venue_list = VENUES
    sphere_venue = ("Sphere", "Las Vegas", "NV", "sphere")

    for tour_label, t_start, t_end, n_shows, run_size in _TOUR_CALENDAR:
        is_sphere_tour = "Sphere" in tour_label
        show_dates = _date_range(t_start, t_end, n_shows)

        run_exclusions: set[str] = set()
        run_counter = 0

        for show_date in show_dates:
            show_num += 1
            show_id = f"show_{show_num:04d}"
            run_counter += 1

            # Assign venue
            if is_sphere_tour:
                vname, vcity, vstate, vtype = sphere_venue
            else:
                vi = int(rng.choice(len(venue_list) - 1, p=np.array(VENUE_WEIGHTS[:-1]) / sum(VENUE_WEIGHTS[:-1])))
                vname, vcity, vstate, vtype = venue_list[vi]

            shows_rows.append({
                "show_id":    show_id,
                "date":       pd.Timestamp(show_date),
                "venue_name": vname,
                "city":       vcity,
                "state":      vstate,
                "tour":       tour_label,
                "venue_type": vtype,
                "show_num":   show_num,
            })

            used_show: set[str] = set()
            set_appearances = []

            # ---- Set 1 ----
            n_s1 = int(rng.integers(9, 13))

            # Show opener
            opener = _sample_songs(1, "show_opener_weight", songs_df, used_show, run_exclusions, gap_tracker, show_num, rng)[0]
            set_appearances.append((show_id, opener, "1", 1))
            used_show.add(opener)
            gap_tracker[opener] = show_num

            # S1 body
            body_size = n_s1 - 2
            body = _sample_songs(body_size, "s1_weight", songs_df, used_show, run_exclusions, gap_tracker, show_num, rng)
            for i, sid in enumerate(body):
                set_appearances.append((show_id, sid, "1", i + 2))
                used_show.add(sid)
                gap_tracker[sid] = show_num

            # S1 closer
            closer = _sample_songs(1, "set_closer_weight", songs_df, used_show, run_exclusions, gap_tracker, show_num, rng)[0]
            set_appearances.append((show_id, closer, "1", n_s1))
            used_show.add(closer)
            gap_tracker[closer] = show_num

            # ---- Set 2 ----
            n_s2 = int(rng.integers(5, 9))

            # S2 opener
            s2_opener = _sample_songs(1, "set2_opener_weight", songs_df, used_show, run_exclusions, gap_tracker, show_num, rng)[0]
            set_appearances.append((show_id, s2_opener, "2", 1))
            used_show.add(s2_opener)
            gap_tracker[s2_opener] = show_num

            # Inject Mike's Groove ~30% of the time
            pos = 2
            if s2_opener == "mikes" or (rng.random() < 0.30 and "mikes" not in used_show):
                if "mikes" not in used_show:
                    set_appearances.append((show_id, "mikes", "2", pos));    used_show.add("mikes");    gap_tracker["mikes"] = show_num;    pos += 1
                if "hydrogen" not in used_show:
                    set_appearances.append((show_id, "hydrogen", "2", pos)); used_show.add("hydrogen"); gap_tracker["hydrogen"] = show_num; pos += 1
                if "weekapaug" not in used_show:
                    set_appearances.append((show_id, "weekapaug", "2", pos)); used_show.add("weekapaug"); gap_tracker["weekapaug"] = show_num; pos += 1

            # S2 body
            remaining_body = n_s2 - pos
            if remaining_body > 0:
                body2 = _sample_songs(remaining_body, "s2_weight", songs_df, used_show, run_exclusions, gap_tracker, show_num, rng)
                for sid in body2:
                    set_appearances.append((show_id, sid, "2", pos))
                    used_show.add(sid)
                    gap_tracker[sid] = show_num
                    pos += 1

            # S2 closer
            s2_closer = _sample_songs(1, "set_closer_weight", songs_df, used_show, run_exclusions, gap_tracker, show_num, rng)[0]
            set_appearances.append((show_id, s2_closer, "2", pos))
            used_show.add(s2_closer)
            gap_tracker[s2_closer] = show_num

            # ---- Encore ----
            n_enc = int(rng.integers(1, 3))
            # Tweezer Reprise often follows a Tweezer show
            if "tweezer" in used_show and "tweeprise" not in used_show and rng.random() < 0.75:
                enc_songs = ["tweeprise"]
                if n_enc > 1:
                    extra = _sample_songs(n_enc - 1, "enc_weight", songs_df, used_show | {"tweeprise"}, run_exclusions, gap_tracker, show_num, rng)
                    enc_songs = extra + ["tweeprise"]  # tweeprise closes
            else:
                enc_songs = _sample_songs(n_enc, "enc_weight", songs_df, used_show, run_exclusions, gap_tracker, show_num, rng)

            for i, sid in enumerate(enc_songs):
                set_appearances.append((show_id, sid, "e", i + 1))
                used_show.add(sid)
                gap_tracker[sid] = show_num

            appearance_rows.extend(set_appearances)
            run_exclusions.update(used_show)

            # Reset run exclusions after run_size shows
            if run_counter >= run_size:
                run_exclusions = set()
                run_counter = 0

    shows_df = pd.DataFrame(shows_rows)
    appearances_df = pd.DataFrame(appearance_rows, columns=["show_id", "song_id", "set_number", "position"])
    appearances_df["duration_min"] = 0.0  # placeholder; not used in scoring
    appearances_df = appearances_df.merge(shows_df[["show_id", "date", "show_num"]], on="show_id")

    return shows_df, appearances_df
