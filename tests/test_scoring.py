"""Tests for the scoring module."""

import numpy as np
import pytest
from phish_engine.scoring import (
    ScoringWeights,
    DEFAULT_WEIGHTS,
    compute_recency_score,
    compute_gap_pressure,
    compute_slot_affinity,
    compute_frequency_score,
    compute_venue_affinity,
    score_all_songs,
    score_breakdown,
)


class TestScoringWeights:
    def test_main_weights_sum_to_one(self):
        w = ScoringWeights()
        nw = w.main_weights_normalized()
        assert abs(sum(nw.values()) - 1.0) < 1e-9

    def test_custom_weights_normalize(self):
        w = ScoringWeights(recency=1.0, frequency=1.0, gap_pressure=1.0,
                           slot_affinity=1.0, venue_affinity=1.0, cluster=1.0)
        nw = w.main_weights_normalized()
        for v in nw.values():
            assert abs(v - 1/6) < 1e-9

    def test_softmax_roundtrip(self):
        w = ScoringWeights(recency=0.3, frequency=0.1, gap_pressure=0.2,
                           slot_affinity=0.15, venue_affinity=0.1, cluster=0.15,
                           recency_decay_rate=2.5, gap_lognormal_sigma=0.6, freq_w10=0.5)
        vec = w.to_softmax_vector()
        w2 = ScoringWeights.from_softmax_vector(vec)
        nw1 = w.main_weights_normalized()
        nw2 = w2.main_weights_normalized()
        for k in nw1:
            assert abs(nw1[k] - nw2[k]) < 0.01

    def test_describe_returns_string(self):
        desc = DEFAULT_WEIGHTS.describe()
        assert "ScoringWeights:" in desc
        assert "recency" in desc


class TestRecencyScore:
    def test_zero_gap_returns_zero(self):
        assert compute_recency_score(0, 30.0) == 0.0

    def test_large_gap_approaches_one(self):
        score = compute_recency_score(100, 10.0)
        assert score > 0.99

    def test_one_avg_gap_high_score(self):
        score = compute_recency_score(30, 30.0)
        assert score > 0.9

    def test_negative_gap_returns_zero(self):
        assert compute_recency_score(-5, 30.0) == 0.0


class TestGapPressure:
    def test_zero_days_returns_zero(self):
        assert compute_gap_pressure(0, 30.0, 10.0) == 0.0

    def test_overdue_returns_high(self):
        score = compute_gap_pressure(100, 10.0, 5.0)
        assert score > 0.9

    def test_on_schedule_returns_moderate(self):
        score = compute_gap_pressure(30, 30.0, 10.0)
        assert 0.3 < score < 0.9


class TestSlotAffinity:
    def test_highest_weight_returns_one(self, songs_df):
        col = "show_opener_weight"
        max_w = float(songs_df[col].max())
        best_song = songs_df[col].idxmax()
        score = compute_slot_affinity(best_song, "show_opener", songs_df, {col: max_w})
        assert abs(score - 1.0) < 1e-6

    def test_low_weight_returns_low(self, songs_df):
        col = "show_opener_weight"
        max_w = float(songs_df[col].max())
        # tweeprise has low show_opener_weight
        score = compute_slot_affinity("tweeprise", "show_opener", songs_df, {col: max_w})
        assert score < 0.1


class TestFrequencyScore:
    def test_zero_shows_returns_zero(self):
        assert compute_frequency_score(10, 0) == 0.0

    def test_high_rate_returns_high(self):
        score = compute_frequency_score(50, 100)  # 50% rate
        assert score > 0.9


class TestVenueAffinity:
    def test_sphere_uses_sphere_affinity(self, songs_df):
        score = compute_venue_affinity("light", "sphere", songs_df)
        assert abs(score - 0.98) < 0.01  # light has sphere_affinity=0.98

    def test_arena_returns_fixed(self, songs_df):
        score = compute_venue_affinity("light", "arena", songs_df)
        assert score == 0.7


class TestScoreAllSongs:
    def test_returns_series(self, songs_df, feat_df, cluster_labels):
        scores = score_all_songs(
            "show_opener", songs_df, feat_df, cluster_labels,
            [], set(), "sphere", 200,
        )
        assert len(scores) > 0
        assert scores.iloc[0] >= scores.iloc[-1]

    def test_excludes_chosen_songs(self, songs_df, feat_df, cluster_labels):
        scores = score_all_songs(
            "show_opener", songs_df, feat_df, cluster_labels,
            ["chalkdust"], set(), "sphere", 200,
        )
        assert "chalkdust" not in scores.index

    def test_excludes_hard_excluded(self, songs_df, feat_df, cluster_labels):
        scores = score_all_songs(
            "show_opener", songs_df, feat_df, cluster_labels,
            [], {"llama"}, "sphere", 200,
        )
        assert "llama" not in scores.index

    def test_accepts_legacy_dict_weights(self, songs_df, feat_df, cluster_labels):
        w = {"recency": 0.25, "gap_pressure": 0.25, "slot_affinity": 0.25,
             "frequency": 0.10, "venue_affinity": 0.10, "cluster": 0.05}
        scores = score_all_songs(
            "show_opener", songs_df, feat_df, cluster_labels,
            [], set(), "sphere", 200, weights=w,
        )
        assert len(scores) > 0


class TestScoreBreakdown:
    def test_returns_all_components(self, songs_df, feat_df, cluster_labels):
        bd = score_breakdown("chalkdust", "show_opener", songs_df, feat_df,
                             cluster_labels, [], "sphere", 200)
        assert "composite" in bd
        assert "recency" in bd
        assert "gap_pressure" in bd
        assert "slot_affinity" in bd
        assert "frequency" in bd
        assert "venue_affinity" in bd
        assert "cluster_diversity" in bd
        assert bd["composite"] > 0
