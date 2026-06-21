#!/usr/bin/env python3
"""
Patch architecture blocks into an existing prediction_data.json.

The full export pipeline (export_json.py) reads from a populated data/ dir.
This script needs only the engine's cached data and an existing
prediction_data.json, so it can add the conditional Type-II layer to the
frontend payload without re-running prediction or clustering.

Usage:
    python scripts/emit_architecture.py [path/to/prediction_data.json]

Defaults to the sibling frontend repo's prediction_data.json. Adds a compact
"arch" block to each jam-capable catalog song; leaves everything else
untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from phish_engine.data.real_data import load_real_data
from phish_engine.architecture import build_architecture_table

CACHE_DIR = Path(__file__).resolve().parent.parent / "phish_engine" / "data" / "cache"
DEFAULT_TARGET = (
    Path(__file__).resolve().parent.parent.parent
    / "phish-setlist-predictor" / "prediction_data.json"
)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    if not target.exists():
        sys.exit(f"target prediction_data.json not found: {target}")

    songs_df, shows_df, appearances_df = load_real_data(
        CACHE_DIR, min_plays=3, start_year=2019,
    )
    table = build_architecture_table(songs_df, shows_df, appearances_df, CACHE_DIR)

    with open(target) as f:
        data = json.load(f)
    catalog = data.get("catalog", {})

    added = 0
    for sid in catalog:
        if sid not in songs_df.index:
            continue
        block = table.song_block(sid)
        if block is None:
            catalog[sid].pop("arch", None)  # keep idempotent re-runs clean
            continue
        catalog[sid]["arch"] = block
        added += 1

    with open(target, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Patched arch blocks into {added} songs → {target}")
    print(f"  (grand Type-II rate {table._grand_mean:.3f}, "
          f"{len(songs_df)} songs in catalog)")


if __name__ == "__main__":
    main()
