"""Tests for the predictor module."""

import pandas as pd
import pytest
from phish_engine.predictor import predict_show, predict_multi_night_run


class TestPredictShow:
    def test_returns_all_sets(self, songs_df, feat_df, cluster_labels, shows_df):
        total_shows = int(shows_df["show_num"].max())
        pred = predict_show(
            show_date=pd.Timestamp("2026-04-17"),
            venue_type="sphere",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            total_shows_in_train=total_shows,
        )
        assert "set1" in pred
        assert "set2" in pred
        assert "encore" in pred
        assert len(pred["set1"]) > 0
        assert len(pred["set2"]) > 0
        assert len(pred["encore"]) > 0

    def test_no_song_repeats(self, songs_df, feat_df, cluster_labels, shows_df):
        total_shows = int(shows_df["show_num"].max())
        pred = predict_show(
            show_date=pd.Timestamp("2026-04-17"),
            venue_type="sphere",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            total_shows_in_train=total_shows,
        )
        all_songs = []
        for slot in ("set1", "set2", "encore"):
            for entry in pred[slot]:
                all_songs.append(entry["song_id"])
        assert len(all_songs) == len(set(all_songs)), "Duplicate songs found in prediction"

    def test_each_entry_has_components(self, songs_df, feat_df, cluster_labels, shows_df):
        total_shows = int(shows_df["show_num"].max())
        pred = predict_show(
            show_date=pd.Timestamp("2026-04-17"),
            venue_type="sphere",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            total_shows_in_train=total_shows,
        )
        entry = pred["set1"][0]
        assert "song_id" in entry
        assert "name" in entry
        assert "score" in entry
        assert "components" in entry

    def test_respects_hard_exclusions(self, songs_df, feat_df, cluster_labels, shows_df):
        total_shows = int(shows_df["show_num"].max())
        excluded = {"chalkdust", "llama", "tweezer"}
        pred = predict_show(
            show_date=pd.Timestamp("2026-04-17"),
            venue_type="sphere",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=cluster_labels,
            total_shows_in_train=total_shows,
            run_exclusions=excluded,
        )
        all_songs = set()
        for slot in ("set1", "set2", "encore"):
            for entry in pred[slot]:
                all_songs.add(entry["song_id"])
        assert all_songs.isdisjoint(excluded)


class TestPredictMultiNightRun:
    def test_predicts_all_nights(self, songs_df, shows_df, appearances_df, cluster_labels):
        dates = [pd.Timestamp(f"2026-04-{d}") for d in [17, 18, 19, 20]]
        preds = predict_multi_night_run(
            show_dates=dates,
            venue_type="sphere",
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=cluster_labels,
        )
        assert len(preds) == 4
        for pred in preds:
            assert "set1" in pred
            assert "set2" in pred
            assert "encore" in pred

    def test_no_repeats_across_consecutive_nights(self, songs_df, shows_df, appearances_df, cluster_labels):
        dates = [pd.Timestamp(f"2026-04-{d}") for d in [17, 18, 19]]
        preds = predict_multi_night_run(
            show_dates=dates,
            venue_type="sphere",
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=cluster_labels,
        )
        # Songs from night 1 should not appear in night 2
        night1_songs = set()
        for slot in ("set1", "set2", "encore"):
            for entry in preds[0][slot]:
                night1_songs.add(entry["song_id"])
        night2_songs = set()
        for slot in ("set1", "set2", "encore"):
            for entry in preds[1][slot]:
                night2_songs.add(entry["song_id"])
        assert night1_songs.isdisjoint(night2_songs), "Night 1 and Night 2 share songs"
