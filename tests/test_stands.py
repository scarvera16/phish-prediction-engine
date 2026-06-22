"""Tests for stand detection (the no-repeat / run unit)."""

from collections import Counter

from phish_engine.stands import detect_stands, stand_positions
from phish_engine.predictor import (
    tour_variety_penalties, VARIETY_FLOOR,
    role_fills, slot_variety_penalties, SLOT_VARIETY_BASE,
)


class TestDetectStands:
    def test_single_venue_is_one_stand(self):
        # The Sphere '26 residency: nine nights, one venue, one stand.
        assert detect_stands(["Sphere"] * 9) == [0] * 9

    def test_multi_night_stand_then_move(self):
        assert detect_stands(["Gorge", "Gorge", "Gorge", "Deer Creek"]) == [0, 0, 0, 1]

    def test_one_nighters(self):
        assert detect_stands(["A", "B", "C"]) == [0, 1, 2]

    def test_returning_to_a_venue_starts_a_new_stand(self):
        assert detect_stands(["A", "B", "A"]) == [0, 1, 2]

    def test_empty(self):
        assert detect_stands([]) == []


class TestStandPositions:
    def test_positions_within_a_stand(self):
        pos = stand_positions(["Gorge", "Gorge", "Gorge", "Deer Creek"])
        assert [p.night for p in pos] == [1, 2, 3, 1]
        assert [p.size for p in pos] == [3, 3, 3, 1]
        assert [p.is_opener for p in pos] == [True, False, False, True]
        # Last night of a stand is the closer (tends to jam hardest).
        assert [p.is_closer for p in pos] == [False, False, True, True]

    def test_one_nighter_is_both_opener_and_closer(self):
        (p,) = stand_positions(["A"])
        assert p.night == 1 and p.size == 1 and p.is_opener and p.is_closer


class TestTourVarietyPenalties:
    def test_first_stand_has_no_penalties(self):
        # A single-venue residency: every show is the same stand, nothing to
        # penalize across stands (the Sphere path is unaffected).
        hist = [{"a", "b"}, {"c", "d"}]
        assert tour_variety_penalties(hist, [0, 0, 0], 2) == {}

    def test_same_stand_songs_not_penalized(self):
        # Songs from the current stand are hard-excluded elsewhere, not softened.
        hist = [{"a"}, {"b"}]
        pen = tour_variety_penalties(hist, [0, 1, 1], 2)
        assert "b" not in pen          # b is in the current stand (id 1)
        assert "a" in pen              # a is from the earlier stand

    def test_more_recent_means_stronger_penalty(self):
        hist = [{"old"}, {"recent"}]   # stands 0 and 1, current is stand 2
        pen = tour_variety_penalties(hist, [0, 1, 2], 2)
        assert pen["recent"] == VARIETY_FLOOR        # played one show ago
        assert pen["old"] > pen["recent"]            # older → weaker penalty

    def test_recovers_to_full_eligibility(self):
        # A song last played `recovery` shows ago carries no penalty.
        hist = [{"x"}] + [{f"f{i}"} for i in range(8)]
        pen = tour_variety_penalties(hist, list(range(10)), 8, floor=0.25, recovery=8)
        assert pen["x"] == 1.0


class TestSlotVariety:
    def test_role_fills_picks_opener_closer_encore(self):
        r = role_fills(["open", "mid", "close1"], ["s2open", "s2close"], ["enc1", "enc2"])
        assert r["show_opener"] == ["open"]
        assert r["s1_closer"] == ["close1"]
        assert r["s2_opener"] == ["s2open"]
        assert r["s2_closer"] == ["s2close"]
        assert r["encore"] == ["enc1", "enc2"]

    def test_role_fills_handles_empty_sets(self):
        r = role_fills([], [], [])
        assert r == {}

    def test_penalty_compounds_with_repeated_role(self):
        hist = {"show_opener": Counter({"chalkdust": 2, "free": 1})}
        pen = slot_variety_penalties(hist)
        # base ** count: opened twice → base^2, once → base^1
        assert pen["show_opener"]["chalkdust"] == SLOT_VARIETY_BASE ** 2
        assert pen["show_opener"]["free"] == SLOT_VARIETY_BASE
        # by the 4th opening the multiplier is tiny — effectively prohibitive
        assert SLOT_VARIETY_BASE ** 3 < 0.1

    def test_unfilled_role_has_no_penalty(self):
        pen = slot_variety_penalties({"encore": Counter()})
        assert pen["encore"] == {}


class TestVenueAffinity:
    @staticmethod
    def _frames():
        import pandas as pd
        rows = ([{"show_id": f"v{i}", "venue_name": "V", "venue_type": "arena", "tour": "T"} for i in range(10)]
                + [{"show_id": f"o{i}", "venue_name": "O", "venue_type": "arena", "tour": "T"} for i in range(10)])
        app = ([{"show_id": f"v{i}", "song_id": "loved"} for i in range(10)]      # all 10 V shows
               + [{"show_id": f"o{i}", "song_id": "loved"} for i in range(2)]     # only 2 O shows
               + [{"show_id": f"v{i}", "song_id": "base"} for i in range(10)]     # everywhere
               + [{"show_id": f"o{i}", "song_id": "base"} for i in range(10)])
        return pd.DataFrame(rows), pd.DataFrame(app)

    def test_over_represented_song_gets_boost(self):
        from phish_engine.scoring import compute_venue_song_affinity
        shows, app = self._frames()
        aff = compute_venue_song_affinity("V", shows, app, min_shows=8)
        assert aff["loved"] > 1.0            # shows up far more at V than the arena baseline
        assert abs(aff["base"] - 1.0) < 0.1  # plays everywhere → roughly neutral

    def test_thin_venue_returns_empty(self):
        from phish_engine.scoring import compute_venue_song_affinity
        shows, app = self._frames()
        assert compute_venue_song_affinity("O", shows, app, min_shows=20) == {}

    def test_nye_runs_excluded(self):
        import pandas as pd
        from phish_engine.scoring import compute_venue_song_affinity
        shows = pd.DataFrame([{"show_id": f"n{i}", "venue_name": "MSG",
                              "venue_type": "arena", "tour": "2024 NYE Run"} for i in range(10)])
        app = pd.DataFrame([{"show_id": f"n{i}", "song_id": "auld"} for i in range(10)])
        # all history is NYE → dropped → not enough qualifying shows → empty
        assert compute_venue_song_affinity("MSG", shows, app, min_shows=4) == {}
