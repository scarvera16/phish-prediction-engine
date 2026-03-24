"""Tests for the real Phish.net data loader."""
from pathlib import Path

import pandas as pd
import pytest

from phish_engine.data.real_data import load_real_data, _pn_slug_to_song_id
from phish_engine.data.manual_overrides import SLUG_ALIASES
from phish_engine.data.venue_map import classify_venue

DATA_DIR = Path(__file__).parent.parent / "phish_engine" / "data" / "cache"
SKIP_REASON = "Real data cache not available"


def _has_data():
    return (DATA_DIR / "shows.json").exists()


# ── Slug alias tests ──────────────────────────────────────────────────────────

class TestSlugAliases:
    def test_known_alias(self):
        assert _pn_slug_to_song_id("you-enjoy-myself") == "yem"
        assert _pn_slug_to_song_id("divided-sky") == "dividesky"
        assert _pn_slug_to_song_id("also-sprach-zarathustra") == "twooone"

    def test_passthrough(self):
        assert _pn_slug_to_song_id("tweezer") == "tweezer"
        assert _pn_slug_to_song_id("ghost") == "ghost"

    def test_all_aliases_are_unique(self):
        values = list(SLUG_ALIASES.values())
        assert len(values) == len(set(values)), "Duplicate alias targets"


# ── Venue classification tests ────────────────────────────────────────────────

class TestVenueMap:
    def test_sphere(self):
        assert classify_venue("Sphere") == "sphere"

    def test_msg(self):
        assert classify_venue("Madison Square Garden") == "arena"

    def test_dicks(self):
        assert classify_venue("Dick's Sporting Goods Park") == "outdoor"

    def test_keyword_amphitheatre(self):
        assert classify_venue("Some Random Amphitheatre") == "outdoor"

    def test_keyword_arena(self):
        assert classify_venue("Unknown Arena Venue") == "arena"

    def test_default(self):
        assert classify_venue("Totally Unknown Place") == "arena"


# ── Data loader tests (require cache files) ───────────────────────────────────

@pytest.mark.skipif(not _has_data(), reason=SKIP_REASON)
class TestLoadRealData:
    @pytest.fixture(scope="class")
    def data(self):
        return load_real_data(DATA_DIR, min_plays=3, start_year=2019)

    def test_returns_three_dataframes(self, data):
        songs_df, shows_df, appearances_df = data
        assert isinstance(songs_df, pd.DataFrame)
        assert isinstance(shows_df, pd.DataFrame)
        assert isinstance(appearances_df, pd.DataFrame)

    def test_shows_columns(self, data):
        _, shows_df, _ = data
        required = {"show_id", "date", "venue_name", "city", "state", "tour", "venue_type", "show_num"}
        assert required.issubset(set(shows_df.columns))

    def test_shows_sorted_by_date(self, data):
        _, shows_df, _ = data
        assert shows_df["date"].is_monotonic_increasing

    def test_show_num_sequential(self, data):
        _, shows_df, _ = data
        assert shows_df["show_num"].tolist() == list(range(1, len(shows_df) + 1))

    def test_appearances_columns(self, data):
        _, _, appearances_df = data
        required = {"show_id", "song_id", "set_number", "position", "duration_min", "date", "show_num"}
        assert required.issubset(set(appearances_df.columns))

    def test_set_number_values(self, data):
        _, _, appearances_df = data
        assert set(appearances_df["set_number"].unique()).issubset({"1", "2", "e"})

    def test_songs_columns(self, data):
        songs_df, _, _ = data
        required = {
            "name", "debut_year", "avg_duration_min", "jam_score",
            "s1_weight", "s2_weight", "enc_weight",
            "show_opener_weight", "set2_opener_weight", "set_closer_weight",
            "avg_gap", "style", "energy", "sphere_affinity",
            "s1_frac", "s2_frac", "enc_frac",
        }
        assert required.issubset(set(songs_df.columns))

    def test_songs_index_is_song_id(self, data):
        songs_df, _, _ = data
        assert songs_df.index.name == "song_id"

    def test_key_songs_present(self, data):
        songs_df, _, _ = data
        for sid in ["mikes", "hydrogen", "weekapaug", "tweezer", "tweeprise", "yem", "chalkdust", "hood", "dividesky"]:
            assert sid in songs_df.index, f"{sid} missing from songs_df"

    def test_slot_fractions_sum_to_one(self, data):
        songs_df, _, _ = data
        total = songs_df["s1_frac"] + songs_df["s2_frac"] + songs_df["enc_frac"]
        assert ((total - 1.0).abs() < 0.01).all()

    def test_weights_non_negative(self, data):
        songs_df, _, _ = data
        for col in ["s1_weight", "s2_weight", "enc_weight", "show_opener_weight", "set2_opener_weight", "set_closer_weight"]:
            assert (songs_df[col] >= 0).all(), f"{col} has negative values"

    def test_weights_max_about_nine(self, data):
        songs_df, _, _ = data
        for col in ["s1_weight", "s2_weight", "enc_weight"]:
            assert 8.0 <= songs_df[col].max() <= 9.1, f"{col} max is {songs_df[col].max()}"

    def test_reasonable_song_count(self, data):
        songs_df, _, _ = data
        assert 100 < len(songs_df) < 500

    def test_appearances_only_contain_catalog_songs(self, data):
        songs_df, _, appearances_df = data
        unknown = set(appearances_df["song_id"]) - set(songs_df.index)
        assert len(unknown) == 0, f"Appearances contain songs not in catalog: {unknown}"


# ── Integration with features pipeline ────────────────────────────────────────

@pytest.mark.skipif(not _has_data(), reason=SKIP_REASON)
class TestRealDataIntegration:
    def test_features_pipeline(self):
        from phish_engine.features import compute_all_features

        songs_df, shows_df, appearances_df = load_real_data(DATA_DIR, min_plays=3)
        feat_df = compute_all_features(songs_df, shows_df, appearances_df, as_of_date=pd.Timestamp("2026-04-15"))

        assert len(feat_df) == len(songs_df)
        assert "total_plays" in feat_df.columns
        assert (feat_df["total_plays"] >= 0).all()

    def test_full_pipeline(self):
        from phish_engine.features import compute_all_features
        from phish_engine.clustering import cluster_songs
        from phish_engine.predictor import predict_multi_night_run

        songs_df, shows_df, appearances_df = load_real_data(DATA_DIR, min_plays=3)

        songs_with_clusters, *_ = cluster_songs(songs_df, appearances_df)
        cluster_labels = dict(zip(songs_with_clusters.index, songs_with_clusters["cluster_id"]))

        sphere_dates = [pd.Timestamp(f"2026-04-{d}") for d in [16, 18, 20, 22, 24, 26, 28, 30]] + [pd.Timestamp("2026-05-02")]
        predictions = predict_multi_night_run(
            show_dates=sphere_dates,
            venue_type="sphere",
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=cluster_labels,
        )

        assert len(predictions) == 9
        for p in predictions:
            assert "set1" in p
            assert "set2" in p
            assert "encore" in p
            assert len(p["set1"]) > 0
            # No song repeats within a show
            all_songs = [s["song_id"] for s in p["set1"] + p["set2"] + p["encore"]]
            assert len(all_songs) == len(set(all_songs)), "Duplicate song in show"
