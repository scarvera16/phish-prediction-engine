# Phish Prediction Engine

ML-powered setlist prediction library for Phish concerts. Trained on historical phish.net data, installable as a standalone Python package.

## Features

- **6-component composite scoring**: recency, gap pressure (log-normal), slot affinity, frequency (multi-window), venue affinity, cluster diversity
- **Multi-night run prediction**: zero-repeat constraint across nights, weekend-aware song distribution, mandatory pair sequencing (Mike's Groove, etc.)
- **K-Means clustering**: silhouette-based k selection, semantic cluster naming
- **Energy-arc set construction**: alternate prediction path via `SongModel` + `build_setlist()` with segue chain building
- **Backtesting**: held-out tour validation — 36.6% song-level hit rate against real shows
- **Two data modes**: real phish.net cached data, or fully synthetic mock data (no API key needed)

## Installation

```bash
pip install phish-prediction-engine
```

Or in dev mode from source:

```bash
git clone https://github.com/scarvera16/phish-prediction-engine
cd phish-prediction-engine
pip install -e ".[dev]"
```

## Quick Start (no data required)

The library ships with a mock data generator — no API key or cached files needed:

```python
import pandas as pd
from phish_engine import (
    get_songs_df,
    generate_show_history,
    compute_all_features,
    build_cluster_feature_matrix,
    train_song_clusters,
    predict_multi_night_run,
    ScoringWeights,
)

# Generate synthetic training history (~280 shows, 2019-2025)
songs_df = get_songs_df()
shows_df, appearances_df = generate_show_history(seed=42)

# Feature engineering + clustering
cutoff = shows_df["date"].max()
feat_df = compute_all_features(songs_df, shows_df, appearances_df, cutoff)
cluster_matrix, _ = build_cluster_feature_matrix(songs_df, feat_df)
_, cluster_labels, _ = train_song_clusters(cluster_matrix)

# Predict a 4-night NYE run
predictions = predict_multi_night_run(
    show_dates=[
        pd.Timestamp("2025-12-28"),
        pd.Timestamp("2025-12-29"),
        pd.Timestamp("2025-12-30"),
        pd.Timestamp("2025-12-31"),
    ],
    venue_type="arena",
    songs_df=songs_df,
    shows_df=shows_df,
    appearances_df=appearances_df,
    cluster_labels=cluster_labels,
)

for night in predictions:
    print(f"\n--- Night {night['show_num']} ---")
    for entry in night["set1"]:
        print(f"  S1  {entry['name']}  ({entry['score']:.3f})")
    for entry in night["set2"]:
        print(f"  S2  {entry['name']}  ({entry['score']:.3f})")
    for entry in night["encore"]:
        print(f"  E   {entry['name']}  ({entry['score']:.3f})")
```

## Using Real phish.net Data

Cache JSON files from the phish.net API into `data/`:
- `shows.json` — show metadata
- `setlists.json` — full setlist entries
- `songs.json` — song catalog

Then load with:

```python
from phish_engine import load_real_data

songs_df, shows_df, appearances_df = load_real_data("data/", min_plays=3, start_year=2019)
```

The rest of the pipeline (features, clustering, prediction) is identical to the mock path above.

## Export for a Frontend

Generate `prediction_data.json` for use by any frontend or downstream consumer:

```bash
python export_json.py
```

Outputs a single JSON file with predicted setlists, song catalog, cluster assignments, bustout candidates, and backtest results.

## Scripts

| Script | Purpose |
|--------|---------|
| `export_json.py` | Full pipeline → `prediction_data.json` |
| `main.py` | CLI runner with terminal-formatted output |
| `scripts/enrich_data.py` | Enrich data from Phish.in tags + venue history |
| `scripts/fast_optimize.py` | Fast weight sweep via grid search |
| `scripts/optimize_pipeline.py` | Full cross-validated weight optimization |
| `scripts/subparam_optimize.py` | Sub-parameter tuning (recency decay, window weights) |
| `scripts/rebuild_predictions.py` | Rebuild JSON export with current optimized weights |

## Scoring Weights (Optimized)

| Component | Weight | Description |
|-----------|--------|-------------|
| `frequency` | 0.40 | Multi-window play frequency (w10=0.25, w30=0.45, w90=0.30) |
| `slot_affinity` | 0.35 | How well a song fits opener / closer / body / encore slots |
| `venue_affinity` | 0.10 | Indoor / outdoor / Sphere venue preference |
| `recency` | 0.05 | Recency decay (rate=3.0) |
| `gap_pressure` | 0.05 | Log-normal gap pressure (sigma=0.45) |
| `cluster` | 0.05 | Cluster diversity bonus |

Cross-validated across 7 multi-night runs: **36.6% song-level hit rate**.

Customize weights:

```python
from phish_engine import ScoringWeights

weights = ScoringWeights(
    recency=0.25,
    gap_pressure=0.25,
    slot_affinity=0.25,
    frequency=0.15,
    venue_affinity=0.05,
    cluster=0.05,
)
```

## Running Tests

```bash
pytest                          # 101 tests total
pytest tests/test_standalone_validation.py  # 32-test standalone validation suite
```

The standalone validation suite proves the full pipeline is functional end-to-end — data loading, feature engineering, clustering, scoring, single-show and multi-night prediction, set building, and backtesting — using only the public API.

## Public API

```python
from phish_engine import (
    # Data
    get_songs_df, SONG_PAIRS,
    generate_show_history,
    load_real_data,

    # Features
    compute_all_features,
    build_cluster_feature_matrix,

    # Clustering
    train_song_clusters,
    cluster_songs,
    compute_cluster_diversity_bonus,
    get_cluster_members,

    # Scoring
    ScoringWeights, DEFAULT_WEIGHTS,
    score_all_songs, score_breakdown,

    # Prediction
    predict_show,
    predict_multi_night_run,

    # Set builder (energy-arc path)
    build_setlist, SetConfig, format_setlist,
    SongModel, build_song_model,

    # Backtest
    run_backtest,
)
```

## License

MIT
