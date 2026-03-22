"""
Phish song catalog with realistic metadata.

Columns:
  song_id             - unique slug
  name                - display name
  debut_year          - first known performance
  avg_duration_min    - typical performance length (minutes)
  jam_score           - 0-1, how frequently extended into a jam
  s1_weight           - unnormalized affinity for set 1
  s2_weight           - unnormalized affinity for set 2
  enc_weight          - unnormalized affinity for encore
  show_opener_weight  - affinity for opening the whole show
  set2_opener_weight  - affinity for opening set 2
  set_closer_weight   - affinity for closing a set
  avg_gap             - average shows between performances (historical)
  style               - primary genre/style tag
  energy              - 0-1 energy level
  sphere_affinity     - 0-1 suitability for Sphere's immersive tech
"""

import pandas as pd

_SONGS = [
    # --- High-energy show openers ---
    # song_id, name, debut_yr, dur, jam, s1w, s2w, encw, so_w, s2o_w, sc_w, gap, style, energy, sphere
    ("acdcbag",    "AC/DC Bag",               1987,  5.5, 0.20, 8.0, 1.5, 0.3, 7.0, 1.0, 1.0,  4.2, "rock",      0.85, 0.75),
    ("llama",      "Llama",                   1990,  5.0, 0.15, 7.0, 1.5, 0.2, 6.0, 1.0, 0.5,  6.5, "rock",      0.95, 0.80),
    ("chalkdust",  "Chalk Dust Torture",      1991,  7.5, 0.35, 9.0, 3.0, 0.3, 8.0, 2.0, 2.0,  3.8, "rock",      0.90, 0.85),
    ("pyite",      "Punch You in the Eye",    1991,  5.5, 0.15, 7.5, 1.0, 0.2, 6.5, 0.5, 0.5,  7.2, "rock",      0.90, 0.70),
    ("golgi",      "Golgi Apparatus",         1987,  4.5, 0.10, 8.0, 1.0, 0.5, 5.0, 0.5, 1.0,  5.0, "rock",      0.80, 0.65),
    ("wilson",     "Wilson",                  1990,  5.0, 0.20, 8.5, 1.5, 0.3, 4.0, 1.0, 1.5,  4.5, "rock",      0.85, 0.80),
    ("runjim",     "Runaway Jim",             1989,  7.0, 0.30, 7.5, 2.5, 0.2, 6.0, 2.0, 2.0,  5.8, "rock",      0.88, 0.75),
    ("moma",       "The Moma Dance",          1998,  7.0, 0.45, 7.0, 4.0, 0.3, 5.0, 3.0, 2.0,  5.5, "funk",      0.82, 0.85),
    ("possum",     "Possum",                  1987,  8.0, 0.35, 6.0, 3.0, 0.5, 4.0, 2.0, 4.0,  4.0, "rock",      0.90, 0.75),
    ("sample",     "Sample in a Jar",         1992,  5.0, 0.10, 8.0, 2.0, 0.5, 4.0, 1.0, 2.0,  3.5, "rock",      0.80, 0.70),

    # --- Set 1 mid-show songs ---
    ("farmhouse",  "Farmhouse",               1999,  4.5, 0.05, 7.0, 1.5, 0.8, 1.0, 0.5, 1.5,  5.0, "folk",      0.55, 0.60),
    ("heavy",      "Heavy Things",            2000,  4.0, 0.05, 8.0, 1.5, 0.8, 2.0, 0.5, 1.0,  4.0, "pop",       0.65, 0.65),
    ("bouncing",   "Bouncing Around the Room",1990,  3.5, 0.05, 7.5, 1.5, 0.5, 2.0, 0.5, 0.5,  3.8, "pop",       0.55, 0.55),
    ("julius",     "Julius",                  1993,  6.5, 0.20, 7.0, 2.5, 0.5, 3.0, 1.5, 2.0,  6.0, "rock",      0.80, 0.75),
    ("dividesky",  "Divided Sky",             1987, 15.0, 0.50, 7.5, 2.0, 0.3, 2.0, 0.5, 3.0,  7.5, "prog",      0.75, 0.90),
    ("reba",       "Reba",                    1989, 13.0, 0.55, 7.5, 3.0, 0.2, 1.0, 1.0, 4.0,  7.0, "prog",      0.70, 0.85),
    ("guyute",     "Guyute",                  1994, 10.0, 0.25, 8.0, 1.5, 0.2, 1.0, 0.5, 2.0,  9.0, "prog",      0.80, 0.85),
    ("fluffhead",  "Fluffhead",               1989, 11.0, 0.20, 8.0, 1.5, 0.2, 2.0, 0.5, 2.0, 10.0, "prog",      0.85, 0.90),
    ("sparkle",    "Sparkle",                 1991,  4.0, 0.05, 7.5, 1.5, 0.5, 2.0, 0.5, 1.0,  4.5, "pop",       0.70, 0.60),
    ("lawnboy",    "Lawn Boy",                1989,  5.0, 0.05, 7.0, 1.5, 0.8, 0.5, 0.3, 1.5,  7.0, "jazz",      0.50, 0.55),
    ("fee",        "Fee",                     1987,  6.0, 0.10, 7.0, 1.0, 0.5, 0.5, 0.3, 1.0, 10.0, "folk",      0.55, 0.60),
    ("cavern",     "Cavern",                  1990,  5.5, 0.25, 7.5, 2.5, 0.5, 2.0, 1.0, 4.0,  5.0, "rock",      0.85, 0.75),
    ("stash",      "Stash",                   1991, 10.0, 0.55, 6.5, 4.5, 0.3, 1.0, 2.0, 3.0,  6.5, "funk",      0.78, 0.80),
    ("squirmcoil", "The Squirming Coil",      1989,  7.0, 0.20, 6.5, 1.0, 2.5, 0.5, 0.3, 5.0,  8.5, "prog",      0.65, 0.75),
    ("maze",       "Maze",                    1992,  9.0, 0.45, 6.0, 5.0, 0.2, 1.0, 2.0, 2.0,  7.0, "rock",      0.88, 0.85),
    ("czero",      "Character Zero",          1996,  6.0, 0.25, 6.0, 4.5, 1.0, 1.0, 2.0, 5.5,  4.5, "rock",      0.85, 0.80),
    ("waste",      "Waste",                   1996,  4.5, 0.03, 5.0, 1.5, 2.0, 0.2, 0.3, 2.0,  8.0, "folk",      0.40, 0.60),
    ("bug",        "Bug",                     2000,  5.0, 0.05, 5.5, 2.0, 1.5, 0.3, 0.5, 2.5,  8.0, "folk",      0.45, 0.55),
    ("wading",     "Wading in the Velvet Sea",2000,  5.5, 0.03, 5.0, 2.0, 2.0, 0.2, 0.3, 2.5,  9.0, "folk",      0.40, 0.65),
    ("wolfmans",   "Wolfman's Brother",       1993,  7.5, 0.50, 6.0, 5.5, 0.5, 1.5, 3.0, 2.5,  5.5, "funk",      0.78, 0.80),
    ("glide",      "Glide",                   1993,  5.0, 0.10, 5.5, 2.0, 0.5, 0.8, 0.5, 1.5,  9.5, "pop",       0.70, 0.65),
    ("blaze",      "Blaze On",                2015,  6.5, 0.40, 6.0, 5.0, 0.5, 2.5, 2.5, 2.5,  9.5, "rock",      0.80, 0.80),

    # --- Set 2 jam vehicles ---
    ("tweezer",    "Tweezer",                 1990, 18.0, 0.95, 1.5, 9.0, 0.5, 0.2, 7.0, 3.0,  5.8, "rock",      0.85, 0.95),
    ("yem",        "You Enjoy Myself",        1986, 20.0, 0.90, 3.0, 8.5, 0.5, 0.5, 4.0, 4.0,  6.2, "prog",      0.88, 0.95),
    ("antelope",   "Run Like an Antelope",    1986, 11.0, 0.70, 3.5, 7.5, 0.3, 1.0, 3.0, 5.0,  6.5, "rock",      0.95, 0.90),
    ("hood",       "Harry Hood",              1989, 14.0, 0.75, 3.0, 8.0, 0.5, 0.3, 2.5, 6.0,  6.0, "prog",      0.82, 0.90),
    ("bathtub",    "Bathtub Gin",             1990, 15.0, 0.80, 2.5, 8.0, 0.3, 0.3, 3.5, 3.0,  5.5, "rock",      0.85, 0.90),
    ("mikes",      "Mike's Song",             1987, 10.0, 0.80, 2.0, 9.0, 0.2, 0.2, 7.5, 2.5,  7.0, "rock",      0.90, 0.88),
    ("weekapaug",  "Weekapaug Groove",        1987,  8.0, 0.65, 1.5, 8.5, 0.3, 0.2, 3.0, 3.5,  7.0, "rock",      0.90, 0.88),
    ("hydrogen",   "I Am Hydrogen",           1987,  4.0, 0.10, 1.0, 5.0, 0.5, 0.1, 2.0, 1.0,  8.0, "ambient",   0.40, 0.70),
    ("bowie",      "David Bowie",             1987, 15.0, 0.80, 2.0, 8.5, 0.3, 0.3, 4.0, 4.0,  6.5, "rock",      0.88, 0.90),
    ("ghost",      "Ghost",                   1997, 15.0, 0.85, 2.0, 8.5, 0.3, 0.2, 4.5, 3.5,  7.0, "funk",      0.85, 0.92),
    ("carini",     "Carini",                  1997, 14.0, 0.85, 2.0, 8.5, 0.2, 0.2, 4.0, 2.5,  7.5, "rock",      0.88, 0.90),
    ("light",      "Light",                   2009, 14.0, 0.90, 2.0, 9.0, 0.2, 0.2, 5.0, 3.0,  9.0, "ambient",   0.80, 0.98),
    ("sand",       "Sand",                    2000, 13.0, 0.80, 2.0, 8.5, 0.2, 0.2, 4.5, 2.5,  7.5, "funk",      0.82, 0.90),
    ("piper",      "Piper",                   1997, 14.0, 0.85, 2.0, 8.5, 0.2, 0.2, 4.0, 2.5,  7.5, "rock",      0.88, 0.90),
    ("free",       "Free",                    1994, 12.0, 0.70, 2.5, 7.5, 0.3, 0.3, 4.0, 3.0,  7.0, "rock",      0.82, 0.85),
    ("simple",     "Simple",                  1993, 13.0, 0.75, 2.0, 8.0, 0.2, 0.2, 4.5, 2.5,  8.5, "funk",      0.78, 0.85),
    ("dwd",        "Down with Disease",       1994, 15.0, 0.85, 2.0, 8.5, 0.3, 0.3, 5.5, 3.0,  6.5, "rock",      0.88, 0.92),
    ("twooone",    "Also Sprach Zarathustra", 1992, 18.0, 0.85, 1.5, 8.5, 0.5, 0.1, 5.0, 3.5,  8.0, "funk",      0.83, 0.95),
    ("crosseyed",  "Crosseyed and Painless",  1993, 16.0, 0.80, 2.0, 8.0, 0.3, 0.2, 4.0, 3.0,  9.0, "funk",      0.85, 0.90),
    ("drowned",    "Drowned",                 1995, 14.0, 0.80, 1.5, 8.0, 0.3, 0.1, 3.5, 3.0, 11.0, "rock",      0.80, 0.85),
    ("splitopen",  "Split Open and Melt",     1989, 15.0, 0.85, 2.0, 8.5, 0.2, 0.2, 4.0, 3.0,  8.0, "rock",      0.90, 0.88),
    ("slave",      "Slave to the Traffic Light",1985,12.0, 0.60, 1.5, 7.5, 1.0, 0.1, 2.0, 6.5,  6.5, "prog",      0.75, 0.88),
    ("gtbt",       "Good Times Bad Times",    1989,  6.5, 0.30, 1.5, 6.0, 4.0, 0.3, 2.5, 4.0,  6.0, "rock",      0.90, 0.82),
    ("sigma",      "Sigma Oasis",             2019, 12.0, 0.80, 2.0, 8.0, 0.3, 0.2, 4.5, 2.5,  8.0, "ambient",   0.78, 0.95),
    ("steam",      "Steam",                   2015,  8.0, 0.55, 4.0, 6.5, 0.5, 1.5, 3.0, 3.0, 11.0, "funk",      0.80, 0.85),
    ("mercury",    "Mercury",                 2018,  9.0, 0.50, 3.5, 7.0, 0.5, 0.5, 3.5, 3.0, 10.0, "rock",      0.82, 0.85),
    ("nminm",      "No Men In No Man's Land", 2016,  6.5, 0.40, 5.0, 5.5, 0.5, 1.5, 2.5, 2.0, 11.5, "rock",      0.82, 0.80),
    ("drift",      "Drift While You're Sleeping",2019,5.0, 0.15, 4.5, 4.5, 1.0, 0.5, 1.5, 2.5, 12.0, "ambient",   0.60, 0.80),

    # --- Encore songs ---
    ("tweeprise",  "Tweezer Reprise",         1990,  3.5, 0.05, 0.5, 0.5, 8.0, 0.1, 0.1, 7.0,  5.5, "rock",      0.95, 0.85),
    ("lovingcup",  "Loving Cup",              1994,  5.5, 0.10, 0.8, 0.8, 6.0, 0.1, 0.3, 5.0,  7.5, "rock",      0.85, 0.80),
    ("sleepmonkey","Sleeping Monkey",         1991,  3.5, 0.03, 0.3, 0.5, 4.0, 0.1, 0.1, 3.5,  9.0, "folk",      0.65, 0.65),
    ("rockytop",   "Rocky Top",              1983,  2.5, 0.05, 0.3, 0.5, 5.0, 0.1, 0.1, 3.0,  9.0, "bluegrass",  0.80, 0.70),
    ("contact",    "Contact",                 1988,  3.5, 0.03, 0.5, 0.5, 4.0, 0.1, 0.1, 3.5, 10.0, "folk",      0.55, 0.60),
    ("boldaslove", "Bold as Love",            2000,  6.0, 0.15, 0.5, 1.0, 5.0, 0.1, 0.2, 4.5, 15.0, "rock",      0.75, 0.80),
    ("taste",      "Taste",                   1996,  8.0, 0.40, 4.0, 5.5, 1.5, 1.0, 2.0, 3.5,  7.5, "rock",      0.80, 0.82),
]

_COLUMNS = [
    "song_id", "name", "debut_year", "avg_duration_min", "jam_score",
    "s1_weight", "s2_weight", "enc_weight",
    "show_opener_weight", "set2_opener_weight", "set_closer_weight",
    "avg_gap", "style", "energy", "sphere_affinity",
]

# Known sequential pairs: if song A is played, song B often follows
SONG_PAIRS = {
    "mikes":    "hydrogen",
    "hydrogen": "weekapaug",
    "tweezer":  "tweeprise",   # not always, but tweeprise often follows tweezer show
    "sleepmonkey": "rockytop",
}


def get_songs_df() -> pd.DataFrame:
    """Return the Phish song catalog as a DataFrame indexed by song_id."""
    df = pd.DataFrame(_SONGS, columns=_COLUMNS)
    df = df.set_index("song_id")
    # Compute normalised slot fractions (useful for feature engineering)
    total_weight = df["s1_weight"] + df["s2_weight"] + df["enc_weight"]
    df["s1_frac"] = df["s1_weight"] / total_weight
    df["s2_frac"] = df["s2_weight"] / total_weight
    df["enc_frac"] = df["enc_weight"] / total_weight
    return df
