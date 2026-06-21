"""Tests for stand detection (the no-repeat / run unit)."""

from phish_engine.stands import detect_stands, stand_positions


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
