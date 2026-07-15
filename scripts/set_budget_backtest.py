#!/usr/bin/env python3
"""
Backtest fixed-count set building vs duration-budget set building, side by side.

Motivation (2026-07-15): the model predicts every show as 11 + 7 + 2 songs, but
real sets are a ~80-minute time box with a variable song count (modern-era set 2:
4-18 songs, size cv 0.24, duration cv 0.14). A jammed-out set has fewer tracks;
the fixed count can't express that. The budget mode fills a set until its
expected runtime (sum of catalog avg_duration_min) hits the budget instead.

For every show of every held-out tour, both variants are predicted from the
SAME features/exclusions, so the comparison is paired. Metrics:
  precision  = predicted songs that were played / songs predicted
  recall     = predicted songs that were played / songs actually played
  f1         = harmonic mean
  size MAE   = |predicted set size - actual set size| (set1, set2)
  size corr  = corr(predicted set2 size, actual set2 size) across the tour:
               can the model see a short jam-heavy set coming AT ALL?

Usage:
  python scripts/set_budget_backtest.py                      # default tours
  python scripts/set_budget_backtest.py "2024 Summer Tour"
Budgets default to the measured medians (80 / 82); override with
  SET1_BUDGET=78 SET2_BUDGET=80 python scripts/set_budget_backtest.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from phish_engine.data.real_data import load_real_data
from phish_engine.clustering import cluster_songs
from phish_engine.scoring import ScoringWeights
from phish_engine.features import compute_all_features
from phish_engine.predictor import (
    predict_show, tour_variety_penalties, slot_variety_penalties,
    ROLE_SLOTS, role_fills,
)
from phish_engine.stands import detect_stands

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "phish_engine" / "data" / "cache"

DEFAULT_TOURS = [
    "2022 Summer Tour",
    "2023 Summer Tour",
    "2024 Summer Tour",
    "2025 Early Summer Tour",
]

def _budget_env(name: str, default: float) -> float | None:
    raw = os.environ.get(name, str(default)).strip().lower()
    return None if raw in ("", "off", "none") else float(raw)

SET1_BUDGET = _budget_env("SET1_BUDGET", 80)
SET2_BUDGET = _budget_env("SET2_BUDGET", 82)


def production_weights() -> ScoringWeights:
    cfg = json.loads((ROOT / "optimized_config.json").read_text())
    w, sp = cfg["weights"], cfg["sub_params"]
    return ScoringWeights(
        recency=w["recency"], gap_pressure=w["gap_pressure"],
        slot_affinity=w["slot_affinity"], frequency=w["frequency"],
        venue_affinity=w["venue_affinity"], cluster=w["cluster"],
        recency_decay_rate=sp["recency_decay_rate"], freq_w10=sp["freq_w10"],
        freq_w30=sp["freq_w30"], freq_w90=sp["freq_w90"],
        gap_lognormal_sigma=sp["gap_lognormal_sigma"],
    )


def _pred_songs(pred: dict) -> dict[str, list[str]]:
    return {k: [p["song_id"] for p in pred[k]] for k in ("set1", "set2", "encore")}


def _actual_sets(show_id, appearances_df) -> dict[str, list[str]]:
    rows = appearances_df[appearances_df["show_id"] == show_id]
    out = {"set1": [], "set2": [], "encore": []}
    for _, r in rows.iterrows():
        s = str(r["set_number"])
        key = "set2" if s in ("2", "3") else ("encore" if s.startswith("e") else "set1")
        out[key].append(r["song_id"])
    return out


def score(pred: dict[str, list[str]], actual: dict[str, list[str]]) -> dict:
    pred_all = set(pred["set1"]) | set(pred["set2"]) | set(pred["encore"])
    act_all = set(actual["set1"]) | set(actual["set2"]) | set(actual["encore"])
    correct = pred_all & act_all
    prec = len(correct) / len(pred_all) if pred_all else 0.0
    rec = len(correct) / len(act_all) if act_all else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n_pred": len(pred_all), "n_actual": len(act_all), "n_correct": len(correct),
        "precision": prec, "recall": rec, "f1": f1,
        "s1_pred": len(pred["set1"]), "s1_act": len(actual["set1"]),
        "s2_pred": len(pred["set2"]), "s2_act": len(actual["set2"]),
    }


def run_tour(tour, songs_df, shows_df, appearances_df, cluster_labels, weights):
    val = shows_df[shows_df["tour"] == tour].sort_values("date")
    if val.empty:
        print(f"  !! no shows for {tour!r}"); return None
    cutoff0 = val["date"].min() - pd.Timedelta(days=1)
    train = shows_df[shows_df["date"] <= cutoff0]

    stand_ids = detect_stands(val["venue_name"].tolist())
    variants = {"fixed": {}, "budget": {}}
    state = {v: {"run_excl": set(), "prev_venue": None,
                 "hist": [], "roles": {s: Counter() for s in ROLE_SLOTS}}
             for v in variants}
    rows = {v: [] for v in variants}

    for i, (_, row) in enumerate(val.iterrows()):
        cutoff = row["date"] - pd.Timedelta(days=1)
        feat_df = compute_all_features(songs_df, train, appearances_df, cutoff)
        n_train = int(train["show_num"].max() or 0)
        actual = _actual_sets(row["show_id"], appearances_df)
        act_all = set(actual["set1"]) | set(actual["set2"]) | set(actual["encore"])

        for v in variants:
            st = state[v]
            if row["venue_name"] != st["prev_venue"]:
                st["run_excl"] = set()
            st["prev_venue"] = row["venue_name"]
            kw = dict(set1_budget_min=SET1_BUDGET, set2_budget_min=SET2_BUDGET) if v == "budget" else {}
            pred = predict_show(
                show_date=row["date"], venue_type=row["venue_type"],
                songs_df=songs_df, feat_df=feat_df, cluster_labels=cluster_labels,
                total_shows_in_train=n_train,
                run_exclusions=st["run_excl"],
                soft_exclusions=tour_variety_penalties(st["hist"], stand_ids, i),
                slot_penalties=slot_variety_penalties(st["roles"]),
                weights=weights, **kw,
            )
            m = score(_pred_songs(pred), actual)
            m["date"] = str(row["date"].date())
            rows[v].append(m)
            # advance state with ACTUALS (same for both variants by construction)
            st["run_excl"] |= act_all
            st["hist"].append(act_all)
            for slot, fills in role_fills(actual["set1"], actual["set2"], actual["encore"]).items():
                st["roles"][slot].update(fills)

    out = {}
    for v, ms in rows.items():
        s2p = np.array([m["s2_pred"] for m in ms]); s2a = np.array([m["s2_act"] for m in ms])
        s1p = np.array([m["s1_pred"] for m in ms]); s1a = np.array([m["s1_act"] for m in ms])
        out[v] = {
            "precision": float(np.mean([m["precision"] for m in ms])),
            "recall": float(np.mean([m["recall"] for m in ms])),
            "f1": float(np.mean([m["f1"] for m in ms])),
            "correct": int(sum(m["n_correct"] for m in ms)),
            "predicted": int(sum(m["n_pred"] for m in ms)),
            "s1_mae": float(np.mean(np.abs(s1p - s1a))),
            "s2_mae": float(np.mean(np.abs(s2p - s2a))),
            "s2_corr": float(np.corrcoef(s2p, s2a)[0, 1]) if np.std(s2p) > 0 and np.std(s2a) > 0 else float("nan"),
            "s2_sizes_pred": s2p.tolist(), "s2_sizes_act": s2a.tolist(),
        }
    return out


def main() -> None:
    tours = sys.argv[1:] or DEFAULT_TOURS
    weights = production_weights()
    print(f"Budgets: set1={SET1_BUDGET}m set2={SET2_BUDGET}m (fixed-count baseline: 11/7/2)")
    songs_df, shows_df, appearances_df = load_real_data(str(CACHE))
    swc, *_ = cluster_songs(songs_df, appearances_df)
    cluster_labels = dict(zip(swc.index, swc["cluster_id"]))

    agg = {"fixed": [], "budget": []}
    for tour in tours:
        print(f"\n== {tour} ==")
        res = run_tour(tour, songs_df, shows_df, appearances_df, cluster_labels, weights)
        if not res: continue
        for v in ("fixed", "budget"):
            r = res[v]; agg[v].append(r)
            print(f"  {v:<7} prec={r['precision']*100:5.1f}%  rec={r['recall']*100:5.1f}%  "
                  f"f1={r['f1']*100:5.1f}%  ({r['correct']}/{r['predicted']})  "
                  f"s1_MAE={r['s1_mae']:.2f}  s2_MAE={r['s2_mae']:.2f}  s2_corr={r['s2_corr']:+.2f}")
        b, f = res["budget"], res["fixed"]
        print(f"  delta   f1 {(b['f1']-f['f1'])*100:+.1f}pp  s2_MAE {b['s2_mae']-f['s2_mae']:+.2f}")

    print("\n== OVERALL (mean across tours) ==")
    for v in ("fixed", "budget"):
        rs = agg[v]
        if not rs: continue
        print(f"  {v:<7} prec={np.mean([r['precision'] for r in rs])*100:5.1f}%  "
              f"rec={np.mean([r['recall'] for r in rs])*100:5.1f}%  "
              f"f1={np.mean([r['f1'] for r in rs])*100:5.1f}%  "
              f"s1_MAE={np.mean([r['s1_mae'] for r in rs]):.2f}  "
              f"s2_MAE={np.mean([r['s2_mae'] for r in rs]):.2f}  "
              f"s2_corr={np.nanmean([r['s2_corr'] for r in rs]):+.2f}")


if __name__ == "__main__":
    main()
