"""End-to-end integration tests."""

import pandas as pd
import pytest
from phish_engine import (
    get_songs_df,
    generate_show_history,
    compute_all_features,
    build_cluster_feature_matrix,
    train_song_clusters,
    predict_multi_night_run,
    run_backtest,
    ScoringWeights,
)


class TestFullPipeline:
    def test_end_to_end_prediction(self):
        """Full pipeline: catalog -> history -> features -> clusters -> predict."""
        songs_df = get_songs_df()
        shows_df, appearances_df = generate_show_history(seed=42)

        cutoff = shows_df["date"].max() - pd.Timedelta(days=1)
        feat_df = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
        X_scaled, scaler = build_cluster_feature_matrix(songs_df, feat_df)
        kmeans, labels, names = train_song_clusters(X_scaled, n_clusters=8, seed=42)

        # Mock catalog has ~49 songs; with hard run exclusions (20 songs/night),
        # only 2 nights can be fully populated
        dates = [pd.Timestamp(f"2026-04-{d}") for d in [17, 18]]
        preds = predict_multi_night_run(
            show_dates=dates,
            venue_type="sphere",
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=labels,
        )

        assert len(preds) == 2
        for pred in preds:
            total_songs = len(pred["set1"]) + len(pred["set2"]) + len(pred["encore"])
            assert total_songs >= 10

    def test_backtest_produces_results(self):
        """Backtest against a held-out tour."""
        songs_df = get_songs_df()
        shows_df, appearances_df = generate_show_history(seed=42)

        cutoff = shows_df[shows_df["tour"] == "NYE 2025"]["date"].min() - pd.Timedelta(days=1)
        feat_df = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
        X_scaled, _ = build_cluster_feature_matrix(songs_df, feat_df)
        _, labels, _ = train_song_clusters(X_scaled, n_clusters=8, seed=42)

        results = run_backtest(
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=labels,
            validation_tour="NYE 2025",
            verbose=False,
        )

        assert results["n_shows"] > 0
        assert 0 <= results["avg_hit_rate"] <= 1
        assert 0 <= results["avg_set_precision"] <= 1

    def test_custom_weights(self):
        """Pipeline works with custom ScoringWeights."""
        songs_df = get_songs_df()
        shows_df, appearances_df = generate_show_history(seed=42)

        cutoff = shows_df["date"].max() - pd.Timedelta(days=1)
        feat_df = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
        X_scaled, _ = build_cluster_feature_matrix(songs_df, feat_df)
        _, labels, _ = train_song_clusters(X_scaled, n_clusters=8, seed=42)

        weights = ScoringWeights(recency=0.30, gap_pressure=0.20, slot_affinity=0.20,
                                  frequency=0.15, venue_affinity=0.10, cluster=0.05)

        dates = [pd.Timestamp("2026-04-17")]
        preds = predict_multi_night_run(
            show_dates=dates,
            venue_type="sphere",
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=labels,
            weights=weights,
        )
        assert len(preds) == 1
        assert len(preds[0]["set1"]) > 0
