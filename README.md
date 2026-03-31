# Phish Prediction Engine v2

ML-powered setlist prediction engine for Phish concerts. Uses historical setlist data from phish.net to predict upcoming shows with a multi-component scoring system.

## Features

- **6-component scoring**: recency, frequency (multi-window), gap pressure (log-normal), slot affinity, venue affinity, cluster diversity
- **Multi-night run prediction**: no-repeat constraint, weekend-aware distribution, song pair sequencing
- **K-means clustering**: silhouette-based k selection, PCA visualization, semantic cluster naming
- **Cross-validation**: backtest against 7 held-out multi-night runs (36.6% hit rate)
- **Data enrichment**: Phish.in jamchart tags, venue history, transition mining, style derivation

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from phish_engine.data.real_data import load_real_data
from phish_engine.features import compute_all_features
from phish_engine.clustering import cluster_songs
from phish_engine.predictor import predict_multi_night_run
from phish_engine.scoring import ScoringWeights

# Load data (requires cached phish.net data in data/)
songs_df, shows_df, appearances_df = load_real_data("data/", min_plays=3, start_year=2019)

# Cluster songs
songs_with_clusters, *_ = cluster_songs(songs_df, appearances_df)
cluster_labels = dict(zip(songs_with_clusters.index, songs_with_clusters["cluster_id"]))

# Predict a run
predictions = predict_multi_night_run(
    show_dates=[pd.Timestamp("2026-04-16"), ...],
    venue_type="sphere",
    songs_df=songs_df,
    shows_df=shows_df,
    appearances_df=appearances_df,
    cluster_labels=cluster_labels,
    weights=ScoringWeights(),  # uses optimized defaults
)
```

## Data Setup

Fetch data from phish.net (requires API key):
```bash
PHISHNET_API_KEY=your_key python fetch_show_data.py
```

Or copy cached JSON files into `data/`:
- `shows.json` — show metadata
- `setlists.json` — full setlist entries
- `songs.json` — song catalog

## Export for Frontend

Generate `prediction_data.json` for use by any frontend:

```bash
python export_json.py
```

This outputs a single JSON file with predicted setlists, song catalog, clusters, bustout candidates, and backtest results.

## Scripts

| Script | Purpose |
|--------|---------|
| `export_json.py` | Full pipeline → JSON output |
| `main.py` | CLI runner with terminal output |
| `scripts/enrich_data.py` | Enrich data from Phish.in tags + venue history |
| `scripts/subparam_optimize.py` | Weight optimization via cross-validation |
| `scripts/rebuild_predictions.py` | Rebuild with current weights |

## Scoring Weights (Optimized)

| Weight | Value | Description |
|--------|-------|-------------|
| frequency | 0.40 | Multi-window play frequency (w10=0.25, w30=0.45, w90=0.30) |
| slot_affinity | 0.35 | How well song fits opener/closer/body/encore slots |
| venue_affinity | 0.10 | Indoor/outdoor/sphere venue preference |
| recency | 0.05 | Recency decay (rate=3.0) |
| gap_pressure | 0.05 | Log-normal gap pressure (sigma=0.45) |
| cluster | 0.05 | Cluster diversity bonus |

Cross-validated across 7 multi-night runs: **36.6% song-level hit rate**.

## License

MIT
