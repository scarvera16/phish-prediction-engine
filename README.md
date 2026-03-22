# phish-prediction-engine

ML-powered setlist prediction engine for Phish shows.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from phish_engine import (
    get_songs_df, generate_show_history,
    compute_all_features, build_cluster_feature_matrix,
    train_song_clusters, predict_multi_night_run,
)
import pandas as pd

# Load catalog and generate mock history
songs_df = get_songs_df()
shows_df, appearances_df = generate_show_history(seed=42)

# Compute features and clusters
cutoff = shows_df["date"].max() - pd.Timedelta(days=1)
feat_df = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
X_scaled, scaler = build_cluster_feature_matrix(songs_df, feat_df)
kmeans, labels, names = train_song_clusters(X_scaled, n_clusters=8)

# Predict 4-night run
dates = [pd.Timestamp(f"2026-04-{d}") for d in [17, 18, 19, 20]]
predictions = predict_multi_night_run(
    show_dates=dates,
    venue_type="sphere",
    songs_df=songs_df,
    shows_df=shows_df,
    appearances_df=appearances_df,
    cluster_labels=labels,
)
```

## Components

- **scoring** - 6-component composite scoring with tunable `ScoringWeights`
- **features** - Dynamic feature engineering from show history
- **clustering** - K-Means with silhouette-based k selection
- **predictor** - Multi-night run prediction with rolling exclusions
- **set_builder** - Energy-arc set construction with segue chains
- **song_model** - Bayesian song probability model from real Phish.net data
- **backtest** - Validation framework against held-out tours

## Running Tests

```bash
pytest
```
