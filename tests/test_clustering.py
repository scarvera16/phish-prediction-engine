"""Tests for the clustering module."""

import numpy as np
import pytest
from phish_engine.clustering import (
    train_song_clusters,
    compute_cluster_diversity_bonus,
    get_cluster_members,
)


class TestTrainSongClusters:
    def test_returns_correct_types(self, songs_df, feat_df):
        from phish_engine.features import build_cluster_feature_matrix
        X_scaled, _ = build_cluster_feature_matrix(songs_df, feat_df)
        kmeans, labels, names = train_song_clusters(X_scaled, n_clusters=6)
        assert isinstance(labels, dict)
        assert isinstance(names, dict)
        assert len(labels) == len(songs_df)
        assert len(names) == 6

    def test_all_songs_assigned(self, songs_df, feat_df):
        from phish_engine.features import build_cluster_feature_matrix
        X_scaled, _ = build_cluster_feature_matrix(songs_df, feat_df)
        _, labels, _ = train_song_clusters(X_scaled, n_clusters=6)
        assert set(labels.keys()) == set(songs_df.index)

    def test_cluster_ids_in_range(self, songs_df, feat_df):
        from phish_engine.features import build_cluster_feature_matrix
        X_scaled, _ = build_cluster_feature_matrix(songs_df, feat_df)
        _, labels, _ = train_song_clusters(X_scaled, n_clusters=6)
        for cid in labels.values():
            assert 0 <= cid < 6


class TestClusterDiversityBonus:
    def test_neutral_at_start(self, cluster_labels, songs_df):
        score = compute_cluster_diversity_bonus("chalkdust", cluster_labels, [], songs_df)
        assert score == 0.5

    def test_higher_for_different_cluster(self, cluster_labels, songs_df):
        # Pick two songs from different clusters
        clusters_seen = {}
        for sid, cid in cluster_labels.items():
            clusters_seen.setdefault(cid, []).append(sid)

        # Find two clusters with songs
        cluster_ids = list(clusters_seen.keys())
        if len(cluster_ids) >= 2:
            song_a = clusters_seen[cluster_ids[0]][0]
            song_b = clusters_seen[cluster_ids[1]][0]
            # With song_a chosen, song_b from different cluster should score higher
            score_diff = compute_cluster_diversity_bonus(song_b, cluster_labels, [song_a], songs_df)
            score_same = compute_cluster_diversity_bonus(
                clusters_seen[cluster_ids[0]][1] if len(clusters_seen[cluster_ids[0]]) > 1 else song_a,
                cluster_labels, [song_a], songs_df
            )
            assert score_diff >= score_same


class TestGetClusterMembers:
    def test_returns_names(self, cluster_labels, songs_df):
        members = get_cluster_members(0, cluster_labels, songs_df, top_n=5)
        assert isinstance(members, list)
        assert all(isinstance(m, str) for m in members)
        assert len(members) <= 5
