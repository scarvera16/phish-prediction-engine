#!/usr/bin/env bash
# Rolling re-prediction — run after a show is played and its setlist is on
# phish.net. Regenerates the tour's predictions so upcoming shows reflect what
# has actually been played, while already-played/locked shows stay frozen to
# the call they were scored against.
#
# Manual for now (Phase 2). Phase 3 automates this as a post-show job.
#
# Usage:  scripts/roll.sh [YYYY-MM-DD cutoff, default today] [tour year, default 2026]
# Prereqs: PHISHNET_API_KEY (or the web app's .env.local), the engine .venv.
set -euo pipefail

CUTOFF="${1:-$(date +%F)}"
YEAR="${2:-2026}"
ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND="${FRONTEND_DIR:-$ENGINE/../phish-setlist-predictor}"
PY="$ENGINE/.venv/bin/python"

echo "== 1/4  Ingest actuals for $YEAR (idempotent) =="
"$PY" "$ENGINE/scripts/ingest_phish_shows.py" "$YEAR"

echo "== 2/4  Snapshot the currently-live predictions (the freeze source) =="
cp "$FRONTEND/prediction_data.json" /tmp/roll_live_prev.json

echo "== 3/4  Rolling export, as of the next unplayed show (cutoff $CUTOFF) =="
ROLL_AS_OF="$CUTOFF" "$PY" "$ENGINE/export_json.py"

echo "== 4/4  Freeze played shows (<= $CUTOFF) back to their locked calls =="
"$PY" "$ENGINE/scripts/freeze_played.py" \
  "$ENGINE/prediction_data.json" /tmp/roll_live_prev.json "$CUTOFF" "$ENGINE/prediction_data.json"

echo
echo "Done. To ship (in $FRONTEND):"
echo "  cp $ENGINE/prediction_data.json $FRONTEND/prediction_data.json"
echo "  python scripts/json_to_ts.py           # regenerate prediction-data.ts (271 trained songs)"
echo "  node scripts/backfill-catalog.mjs      # REQUIRED: restores the full ~977-song pick-sheet catalog"
echo "  npm run build                          # verify"
echo "  review the diff, commit, push — Vercel deploys."
