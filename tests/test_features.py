"""Tests for the features module."""

import pandas as pd
import pytest
from phish_engine.features import compute_all_features, build_cluster_feature_matrix


class TestComputeAllFeatures:
    def test_returns_all_songs(self, songs_df, shows_df, appearances_df):
        cutoff = shows_df["date"].max()
        feat = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
        assert set(feat.index) == set(songs_df.index)

    def test_has_expected_columns(self, feat_df):
        expected = ["total_plays", "current_gap_shows", "avg_gap_actual",
                     "gap_z_score", "plays_30d", "plays_90d", "plays_365d",
                     "set1_plays", "set2_plays", "enc_plays"]
        for col in expected:
            assert col in feat_df.columns

    def test_no_negative_plays(self, feat_df):
        assert (feat_df["total_plays"] >= 0).all()

    def test_fractions_sum_to_one(self, feat_df):
        played = feat_df[feat_df["total_plays"] > 0]
        totals = played["set1_actual_frac"] + played["set2_actual_frac"] + played["enc_actual_frac"]
        assert ((totals - 1.0).abs() < 0.01).all()

    def test_gap_z_score_finite(self, feat_df):
        assert feat_df["gap_z_score"].notna().all()


class TestBuildClusterFeatureMatrix:
    def test_returns_scaled_dataframe(self, songs_df, feat_df):
        X_scaled, scaler = build_cluster_feature_matrix(songs_df, feat_df)
        assert X_scaled.shape[0] == len(songs_df)
        assert X_scaled.shape[1] > 0

    def test_columns_are_standardized(self, songs_df, feat_df):
        X_scaled, _ = build_cluster_feature_matrix(songs_df, feat_df)
        # Mean should be approximately 0 for each column
        for col in X_scaled.columns:
            assert abs(X_scaled[col].mean()) < 0.1
