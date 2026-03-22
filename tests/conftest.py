"""Shared test fixtures."""

import pytest
import pandas as pd
from phish_engine.data.songs import get_songs_df, SONG_PAIRS
from phish_engine.data.mock_data import generate_show_history
from phish_engine.features import compute_all_features, build_cluster_feature_matrix
from phish_engine.clustering import train_song_clusters


@pytest.fixture(scope="session")
def songs_df():
    return get_songs_df()


@pytest.fixture(scope="session")
def show_history():
    shows_df, appearances_df = generate_show_history(seed=42)
    return shows_df, appearances_df


@pytest.fixture(scope="session")
def shows_df(show_history):
    return show_history[0]


@pytest.fixture(scope="session")
def appearances_df(show_history):
    return show_history[1]


@pytest.fixture(scope="session")
def feat_df(songs_df, shows_df, appearances_df):
    cutoff = shows_df["date"].max() - pd.Timedelta(days=1)
    return compute_all_features(songs_df, shows_df, appearances_df, cutoff)


@pytest.fixture(scope="session")
def cluster_data(songs_df, feat_df):
    X_scaled, scaler = build_cluster_feature_matrix(songs_df, feat_df)
    kmeans, labels, cluster_names = train_song_clusters(X_scaled, n_clusters=8, seed=42)
    return kmeans, labels, cluster_names


@pytest.fixture(scope="session")
def cluster_labels(cluster_data):
    return cluster_data[1]
