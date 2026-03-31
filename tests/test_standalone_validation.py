"""
Standalone validation suite for phish-prediction-engine.

Proves the library is fully functional as an independent package by
exercising every layer of the pipeline end-to-end:

  1. Data layer    – song catalog, mock history generation
  2. Features      – dynamic feature engineering
  3. Clustering    – K-Means training and labeling
  4. Scoring       – composite score computation
  5. Prediction    – single-show and multi-night run prediction
  6. Set builder   – energy-arc-aware set construction (SongModel path)
  7. Backtest      – held-out tour validation with metrics

Each test is independent and uses only the public API.
"""

import numpy as np
import pandas as pd
import pytest

from phish_engine import (
    get_songs_df,
    SONG_PAIRS,
    generate_show_history,
    compute_all_features,
    build_cluster_feature_matrix,
    train_song_clusters,
    score_all_songs,
    score_breakdown,
    ScoringWeights,
    DEFAULT_WEIGHTS,
    predict_show,
    predict_multi_night_run,
    build_setlist,
    SetConfig,
    format_setlist,
    SongModel,
    build_song_model,
    score_song_for_slot,
    run_backtest,
    compute_cluster_diversity_bonus,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def songs_df():
    return get_songs_df()


@pytest.fixture(scope="module")
def mock_history():
    shows_df, appearances_df = generate_show_history(seed=42)
    return shows_df, appearances_df


@pytest.fixture(scope="module")
def features(songs_df, mock_history):
    shows_df, appearances_df = mock_history
    cutoff = shows_df["date"].max()
    feat_df = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
    total_shows = int(shows_df["show_num"].max())
    return feat_df, total_shows


@pytest.fixture(scope="module")
def clusters(songs_df, features):
    feat_df, _ = features
    cluster_matrix, _scaler = build_cluster_feature_matrix(songs_df, feat_df)
    _kmeans, labels, _names = train_song_clusters(cluster_matrix)
    return labels


@pytest.fixture(scope="module")
def song_model(songs_df, mock_history):
    """Build a synthetic SongModel from mock data for set_builder testing."""
    shows_df, appearances_df = mock_history
    model = SongModel()
    model.all_song_ids = set(songs_df.index)
    model.song_names = songs_df["name"].to_dict()
    model.song_eras = {sid: "modern" for sid in songs_df.index}
    model.energy = songs_df["energy"].to_dict()

    # Compute base rates from mock appearances
    total = len(shows_df)
    play_counts = appearances_df["song_id"].value_counts()
    model.base_rates = {sid: count / max(total, 1)
                        for sid, count in play_counts.items()
                        if sid in songs_df.index}

    # Synthetic role probabilities
    for sid in songs_df.index:
        model.role_probs[sid] = {
            "set1_opener": float(songs_df.loc[sid].get("s1_opener_frac", 0.1)),
            "set1_closer": float(songs_df.loc[sid].get("s1_closer_frac", 0.05)),
            "set1_mid":    float(songs_df.loc[sid].get("s1_frac", 0.3)),
            "set2_opener": float(songs_df.loc[sid].get("s2_opener_frac", 0.1)),
            "set2_closer": float(songs_df.loc[sid].get("s2_closer_frac", 0.05)),
            "set2_mid":    float(songs_df.loc[sid].get("s2_frac", 0.3)),
            "encore_opener": float(songs_df.loc[sid].get("enc_frac", 0.1)),
            "show_closer":   float(songs_df.loc[sid].get("enc_frac", 0.1)),
            "encore_mid":    float(songs_df.loc[sid].get("enc_frac", 0.1)),
        }

    # Duration and jam potential from catalog
    model.avg_duration_sec = {sid: float(songs_df.loc[sid].get("avg_duration_min", 5.0)) * 60.0
                              for sid in songs_df.index}
    model.jam_potential = songs_df["jam_score"].to_dict()

    # Gap stats
    model.gap_stats = {sid: {"current_gap": 5, "median": 8, "sigma": 0.5} for sid in songs_df.index}

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLayer:
    def test_song_catalog_loads(self, songs_df):
        assert len(songs_df) >= 60, "Catalog should have 60+ songs"
        required_cols = {"name", "energy", "jam_score", "s1_weight", "s2_weight"}
        assert required_cols.issubset(songs_df.columns)

    def test_song_pairs_defined(self):
        assert len(SONG_PAIRS) > 0, "SONG_PAIRS should be non-empty"
        for pair in SONG_PAIRS:
            assert len(pair) >= 2, "Each pair should have at least 2 elements"

    def test_key_songs_present(self, songs_df):
        must_have = ["tweezer", "mikes", "weekapaug", "fluffhead", "yem", "dividesky"]
        for sid in must_have:
            assert sid in songs_df.index, f"Missing iconic song: {sid}"

    def test_mock_history_realistic(self, mock_history):
        shows_df, appearances_df = mock_history
        assert len(shows_df) >= 200, "Should generate 200+ shows"
        assert len(appearances_df) > 2000, "Should have thousands of appearances"
        venues = shows_df["venue_type"].unique()
        assert len(venues) >= 2, "Multiple venue types expected"

    def test_mock_history_deterministic(self, songs_df):
        s1, a1 = generate_show_history(seed=99)
        s2, a2 = generate_show_history(seed=99)
        assert s1["show_id"].tolist() == s2["show_id"].tolist()

    def test_no_within_show_repeats(self, mock_history):
        _, appearances_df = mock_history
        for show_id, group in appearances_df.groupby("show_id"):
            songs = group["song_id"].tolist()
            assert len(songs) == len(set(songs)), \
                f"Duplicate songs in show {show_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatures:
    def test_features_computed(self, features):
        feat_df, total_shows = features
        assert len(feat_df) > 0
        assert total_shows > 0

    def test_feature_columns_present(self, features):
        feat_df, _ = features
        expected = {"current_gap_shows", "total_plays", "gap_z_score"}
        present = set(feat_df.columns)
        for col in expected:
            assert col in present, f"Missing feature column: {col}"

    def test_gap_values_nonnegative(self, features):
        feat_df, _ = features
        assert (feat_df["current_gap_shows"] >= 0).all(), "Gaps should be non-negative"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

class TestClustering:
    def test_cluster_labels_assigned(self, clusters, songs_df):
        assert len(clusters) > 0
        # Every clustered song should be in the catalog
        for sid in clusters:
            assert sid in songs_df.index

    def test_multiple_clusters(self, clusters):
        unique_labels = set(clusters.values())
        assert len(unique_labels) >= 2, "Should produce multiple clusters"

    def test_diversity_bonus_computable(self, clusters, songs_df):
        chosen = list(clusters.keys())[:5]
        test_song = list(clusters.keys())[5]
        bonus = compute_cluster_diversity_bonus(test_song, clusters, chosen, songs_df)
        assert isinstance(bonus, float)
        assert 0 <= bonus <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCORING
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoring:
    def test_score_all_songs(self, songs_df, features, clusters):
        feat_df, total_shows = features
        scores = score_all_songs(
            slot_type="show_opener",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=clusters,
            already_chosen=[],
            excluded=set(),
            venue_type="arena",
            total_shows=total_shows,
            weights=DEFAULT_WEIGHTS,
        )
        assert len(scores) > 0
        assert (scores >= 0).all(), "Scores should be non-negative"

    def test_score_breakdown(self, songs_df, features, clusters):
        feat_df, total_shows = features
        sid = "tweezer"
        bd = score_breakdown(
            sid, "s2_body", songs_df, feat_df, clusters,
            [], "arena", total_shows, DEFAULT_WEIGHTS,
        )
        assert "composite" in bd
        assert bd["composite"] > 0

    def test_exclusions_respected(self, songs_df, features, clusters):
        feat_df, total_shows = features
        excluded = {"tweezer", "mikes"}
        scores = score_all_songs(
            "s2_body", songs_df, feat_df, clusters,
            [], excluded, "arena", total_shows, DEFAULT_WEIGHTS,
        )
        for sid in excluded:
            assert sid not in scores.index, f"{sid} should be excluded"

    def test_custom_weights(self, songs_df, features, clusters):
        feat_df, total_shows = features
        heavy_recency = ScoringWeights(
            recency=0.80, gap_pressure=0.05, slot_affinity=0.05,
            frequency=0.05, venue_affinity=0.025, cluster=0.025,
        )
        scores = score_all_songs(
            "show_opener", songs_df, feat_df, clusters,
            [], set(), "arena", total_shows, heavy_recency,
        )
        assert len(scores) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SINGLE-SHOW PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrediction:
    def test_predict_single_show(self, songs_df, features, clusters):
        feat_df, total_shows = features
        pred = predict_show(
            show_date=pd.Timestamp("2025-07-01"),
            venue_type="outdoor",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=clusters,
            total_shows_in_train=total_shows,
        )
        assert "set1" in pred and "set2" in pred and "encore" in pred
        assert len(pred["set1"]) > 0
        assert len(pred["set2"]) > 0
        assert len(pred["encore"]) > 0

    def test_no_repeats_within_show(self, songs_df, features, clusters):
        feat_df, total_shows = features
        pred = predict_show(
            show_date=pd.Timestamp("2025-08-15"),
            venue_type="arena",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=clusters,
            total_shows_in_train=total_shows,
        )
        all_songs = []
        for slot in ("set1", "set2", "encore"):
            all_songs.extend(e["song_id"] for e in pred[slot])
        assert len(all_songs) == len(set(all_songs)), "No repeats within a show"

    def test_prediction_entries_have_components(self, songs_df, features, clusters):
        feat_df, total_shows = features
        pred = predict_show(
            show_date=pd.Timestamp("2025-09-01"),
            venue_type="arena",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=clusters,
            total_shows_in_train=total_shows,
        )
        for entry in pred["set1"]:
            assert "song_id" in entry
            assert "name" in entry
            assert "score" in entry
            assert "components" in entry
            assert entry["score"] > 0

    def test_run_exclusions_work(self, songs_df, features, clusters):
        feat_df, total_shows = features
        exclude = {"tweezer", "yem", "dividesky", "mikes"}
        pred = predict_show(
            show_date=pd.Timestamp("2025-09-02"),
            venue_type="arena",
            songs_df=songs_df,
            feat_df=feat_df,
            cluster_labels=clusters,
            total_shows_in_train=total_shows,
            run_exclusions=exclude,
        )
        all_songs = set()
        for slot in ("set1", "set2", "encore"):
            all_songs.update(e["song_id"] for e in pred[slot])
        for sid in exclude:
            assert sid not in all_songs, f"{sid} should be excluded from prediction"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MULTI-NIGHT RUN PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiNightRun:
    def test_multi_night_basic(self, songs_df, mock_history, clusters):
        shows_df, appearances_df = mock_history
        dates = [pd.Timestamp("2025-12-28"), pd.Timestamp("2025-12-29"),
                 pd.Timestamp("2025-12-30"), pd.Timestamp("2025-12-31")]
        preds = predict_multi_night_run(
            show_dates=dates,
            venue_type="arena",
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=clusters,
        )
        assert len(preds) == 4

    def test_no_repeats_across_nights(self, songs_df, mock_history, clusters):
        shows_df, appearances_df = mock_history
        dates = [pd.Timestamp("2025-08-01"), pd.Timestamp("2025-08-02"),
                 pd.Timestamp("2025-08-03")]
        preds = predict_multi_night_run(
            show_dates=dates,
            venue_type="outdoor",
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=clusters,
        )
        all_songs = []
        for pred in preds:
            for slot in ("set1", "set2", "encore"):
                all_songs.extend(e["song_id"] for e in pred[slot])
        assert len(all_songs) == len(set(all_songs)), \
            "Zero repeats across multi-night run"

    def test_each_night_has_full_setlist(self, songs_df, mock_history, clusters):
        shows_df, appearances_df = mock_history
        dates = [pd.Timestamp("2025-07-04"), pd.Timestamp("2025-07-05")]
        preds = predict_multi_night_run(
            show_dates=dates,
            venue_type="outdoor",
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=clusters,
        )
        for i, pred in enumerate(preds):
            assert len(pred["set1"]) >= 7, f"Night {i+1} set1 too short"
            assert len(pred["set2"]) >= 5, f"Night {i+1} set2 too short"
            assert len(pred["encore"]) >= 1, f"Night {i+1} encore missing"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SET BUILDER (SONG MODEL PATH)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSetBuilder:
    def test_song_model_builds(self, song_model):
        assert len(song_model.all_song_ids) > 0
        assert len(song_model.base_rates) > 0
        assert len(song_model.song_names) > 0

    def test_build_setlist_produces_valid_output(self, song_model):
        rng = np.random.default_rng(42)
        setlist = build_setlist(
            model=song_model,
            played_in_run=set(),
            venue_type="arena",
            run_length=3,
            show_number=1,
            rng=rng,
        )
        assert "set1" in setlist and "set2" in setlist and "encore" in setlist
        assert len(setlist["set1"]) >= 7
        assert len(setlist["set2"]) >= 5
        assert len(setlist["encore"]) >= 1

    def test_setlist_no_repeats(self, song_model):
        rng = np.random.default_rng(123)
        setlist = build_setlist(
            model=song_model, played_in_run=set(),
            venue_type="arena", rng=rng,
        )
        all_songs = setlist["set1"] + setlist["set2"] + setlist["encore"]
        assert len(all_songs) == len(set(all_songs))

    def test_format_setlist(self, song_model):
        rng = np.random.default_rng(7)
        setlist = build_setlist(
            model=song_model, played_in_run=set(), rng=rng,
        )
        output = format_setlist(setlist, song_model, "Test Show 2025-07-04")
        assert "SET 1" in output
        assert "SET 2" in output
        assert "ENCORE" in output
        assert len(output) > 100

    def test_multi_show_run_builder(self, song_model):
        """Build a 3-night run using set_builder and verify zero repeats."""
        rng = np.random.default_rng(55)
        played = set()
        all_songs = []
        for night in range(3):
            setlist = build_setlist(
                model=song_model, played_in_run=played,
                venue_type="arena", run_length=3,
                show_number=night + 1, rng=rng,
            )
            for slot in ("set1", "set2", "encore"):
                for sid in setlist[slot]:
                    played.add(sid)
                    all_songs.append(sid)
        assert len(all_songs) == len(set(all_songs)), \
            "No repeats across 3-night set_builder run"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. BACKTEST VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktest:
    def test_backtest_runs(self, songs_df, mock_history, clusters):
        shows_df, appearances_df = mock_history
        tours = shows_df["tour"].dropna().unique()
        if len(tours) == 0:
            pytest.skip("No tours in mock data")
        # Pick a late tour so there's training data
        tour = tours[-1]
        result = run_backtest(
            songs_df=songs_df,
            shows_df=shows_df,
            appearances_df=appearances_df,
            cluster_labels=clusters,
            validation_tour=tour,
            verbose=False,
        )
        assert "avg_hit_rate" in result
        assert "avg_set_precision" in result
        assert result["n_shows"] > 0
        assert 0 <= result["avg_hit_rate"] <= 1
        assert 0 <= result["avg_set_precision"] <= 1

    def test_backtest_produces_per_show_metrics(self, songs_df, mock_history, clusters):
        shows_df, appearances_df = mock_history
        tours = shows_df["tour"].dropna().unique()
        if len(tours) == 0:
            pytest.skip("No tours in mock data")
        tour = tours[-1]
        result = run_backtest(
            songs_df, shows_df, appearances_df, clusters,
            validation_tour=tour, verbose=False,
        )
        assert len(result["per_show"]) == result["n_shows"]
        for m in result["per_show"]:
            assert "hit_rate" in m
            assert "n_correct" in m


# ═══════════════════════════════════════════════════════════════════════════════
# 9. END-TO-END PIPELINE (full flow in one test)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_pipeline_from_scratch(self):
        """
        Demonstrates the complete pipeline a new user would run:
        install → load data → features → clusters → predict → format output.
        """
        # Step 1: Load catalog
        songs = get_songs_df()
        assert len(songs) >= 60

        # Step 2: Generate training data
        shows, appearances = generate_show_history(seed=2025)
        assert len(shows) > 200

        # Step 3: Compute features
        cutoff = shows["date"].max()
        feats = compute_all_features(songs, shows, appearances, cutoff)
        total = int(shows["show_num"].max())

        # Step 4: Train clusters
        cmatrix, _scaler = build_cluster_feature_matrix(songs, feats)
        _km, labels, _names = train_song_clusters(cmatrix)
        assert len(labels) > 0

        # Step 5: Predict a 3-night run
        dates = [pd.Timestamp("2025-12-29"), pd.Timestamp("2025-12-30"),
                 pd.Timestamp("2025-12-31")]
        preds = predict_multi_night_run(
            show_dates=dates, venue_type="arena",
            songs_df=songs, shows_df=shows,
            appearances_df=appearances, cluster_labels=labels,
        )
        assert len(preds) == 3

        # Verify structure and quality
        all_song_ids = []
        for night, pred in enumerate(preds, 1):
            for slot in ("set1", "set2", "encore"):
                for entry in pred[slot]:
                    assert entry["score"] > 0, \
                        f"Night {night} {slot}: {entry['name']} has zero score"
                    all_song_ids.append(entry["song_id"])

        # Zero repeats across 3 nights
        assert len(all_song_ids) == len(set(all_song_ids))

        # Step 6: Also test the SongModel / set_builder path
        model = SongModel()
        model.all_song_ids = set(songs.index)
        model.song_names = songs["name"].to_dict()
        model.song_eras = {sid: "modern" for sid in songs.index}
        model.energy = songs["energy"].to_dict()
        play_counts = appearances["song_id"].value_counts()
        model.base_rates = {sid: c / max(len(shows), 1)
                            for sid, c in play_counts.items() if sid in songs.index}
        for sid in songs.index:
            model.role_probs[sid] = {r: 0.1 for r in [
                "set1_opener", "set1_closer", "set1_mid",
                "set2_opener", "set2_closer", "set2_mid",
                "encore_opener", "show_closer", "encore_mid"]}
        model.avg_duration_sec = {sid: float(songs.loc[sid].get("avg_duration_min", 5)) * 60
                                  for sid in songs.index}
        model.jam_potential = songs["jam_score"].to_dict()
        model.gap_stats = {sid: {"current_gap": 5, "median": 8, "sigma": 0.5} for sid in songs.index}

        rng = np.random.default_rng(2025)
        setlist = build_setlist(model, played_in_run=set(), rng=rng)
        output = format_setlist(setlist, model, "NYE Run N1 — 12/29/2025")
        assert "SET 1" in output
        assert len(setlist["set1"]) >= 7

    def test_prediction_quality_baseline(self, songs_df, mock_history, clusters):
        """
        Verify prediction quality meets minimum thresholds.
        This ensures any user gets the same caliber of predictions.
        Uses the shared fixtures to avoid redundant data generation.
        """
        shows, appearances = mock_history

        # Pick a small NYE tour for fast backtesting
        tours = shows["tour"].dropna().unique()
        nye_tours = [t for t in tours if "NYE" in t]
        # Use a late NYE tour so there's plenty of training data
        tour = nye_tours[-1] if nye_tours else tours[-1]

        result = run_backtest(
            songs_df, shows, appearances, clusters,
            validation_tour=tour, verbose=False,
        )

        # Quality gates: these thresholds represent minimum acceptable
        # prediction quality matching what the front-end app delivered
        assert result["avg_hit_rate"] >= 0.10, \
            f"Hit rate {result['avg_hit_rate']:.1%} below 10% minimum"
        assert result["total_correct"] >= 1, \
            "Should correctly predict at least 1 song"
        assert result["n_shows"] >= 1, \
            "Backtest should cover at least 1 show"
