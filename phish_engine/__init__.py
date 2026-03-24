"""
phish_engine - ML-powered setlist prediction engine for Phish shows.

Provides composite scoring, K-Means clustering, multi-night prediction,
energy-arc set building, and backtesting tools.
"""

__version__ = "0.1.0"

from .scoring import (
    ScoringWeights,
    DEFAULT_WEIGHTS,
    score_all_songs,
    score_breakdown,
    compute_recency_score,
    compute_gap_pressure,
    compute_slot_affinity,
    compute_frequency_score,
    compute_venue_affinity,
    build_venue_plays_index,
)
from .features import compute_all_features, build_cluster_feature_matrix
from .clustering import (
    cluster_songs,
    train_song_clusters,
    compute_cluster_diversity_bonus,
    get_cluster_weights,
    get_cluster_members,
)
from .predictor import predict_show, predict_multi_night_run
from .set_builder import build_setlist, SetConfig, format_setlist
from .song_model import SongModel, build_song_model, score_song_for_slot
from .data.songs import get_songs_df, SONG_PAIRS
from .data.mock_data import generate_show_history
from .data.real_data import load_real_data
from .backtest.validator import run_backtest
