#!/usr/bin/env bash
# Rolling re-prediction, end to end. Run after a show is played and its setlist
# is on phish.net: this regenerates the tour's predictions so upcoming shows
# reflect what was actually played, freezes already-played/locked shows to the
# call they were scored against, and ships it (commit + push -> Vercel deploys).
#
# Idempotent: if nothing new played, it re-predicts to the same result and the
# git commit is a no-op, so it's safe to run on a schedule (see roll.plist).
#
# Usage:  scripts/roll.sh [YYYY-MM-DD cutoff, default today] [tour year, default 2026]
# Prereqs: PHISHNET_API_KEY (or the web app's .env.local), the engine .venv,
#          the frontend repo checked out next to the engine, node installed.
set -euo pipefail

CUTOFF="${1:-$(date -u +%F)}"   # UTC, matching the cloud workflow's clock
YEAR="${2:-$(date -u +%Y)}"
ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND="${FRONTEND_DIR:-$ENGINE/../phish-setlist-predictor}"
PY="$ENGINE/.venv/bin/python"

echo "== 1/6  Ingest actuals for $YEAR (idempotent) =="
"$PY" "$ENGINE/scripts/ingest_phish_shows.py" "$YEAR"

echo "== 2/6  Sync the frontend to main + snapshot the live predictions =="
git -C "$FRONTEND" checkout main
git -C "$FRONTEND" pull --ff-only origin main
# Freeze source = what is actually DEPLOYED (origin/main), never the working
# tree: a half-failed previous roll must not poison the next day's freeze.
LIVE_PREV="$(mktemp /tmp/roll_live_prev.XXXXXX.json)"
git -C "$FRONTEND" show origin/main:prediction_data.json > "$LIVE_PREV"

echo "== 3/6  Rolling export, as of the next unplayed show (cutoff $CUTOFF) =="
ROLL_AS_OF="$CUTOFF" "$PY" "$ENGINE/export_json.py"

echo "== 4/6  Freeze played shows (< $CUTOFF) back to their locked calls =="
"$PY" "$ENGINE/scripts/freeze_played.py" \
  "$ENGINE/prediction_data.json" "$LIVE_PREV" "$CUTOFF" "$ENGINE/prediction_data.json"

echo "== 5/6  Regenerate the frontend data (trained TS + full pick-sheet catalog) =="
cp "$ENGINE/prediction_data.json" "$FRONTEND/prediction_data.json"
( cd "$FRONTEND" \
  && python3 scripts/json_to_ts.py \
  && node scripts/backfill-catalog.mjs )

echo "== Repeat-gap tracker (validates the RECENT_NO_REPEAT window) =="
"$PY" "$ENGINE/scripts/repeat_gap_tracker.py" --year "$YEAR" \
  --out "$FRONTEND/repeat_gaps.json" || echo "  (tracker failed — continuing)"

# Keep the frontend's engine-cache seed fresh (the cloud workflow reads it),
# matching what roll.yml commits back after its own ingest.
cp "$ENGINE/phish_engine/data/cache/shows.json" \
   "$ENGINE/phish_engine/data/cache/setlists.json" \
   "$ENGINE/phish_engine/data/cache/phishin_tracks.json" \
   "$FRONTEND/engine-cache/" 2>/dev/null || true

echo "== 6/6  Ship if anything changed =="
cd "$FRONTEND"
git add src/lib/prediction-data.ts prediction_data.json repeat_gaps.json engine-cache
if git diff --cached --quiet; then
  echo "No prediction change — nothing to deploy."
else
  git commit -q -m "Roll predictions (cutoff $CUTOFF)

Automated rolling re-prediction: upcoming shows reflect what has been
played through $CUTOFF; played/locked shows stay frozen.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  # A human push during the roll shouldn't lose the day's predictions.
  git pull --rebase origin main
  git push origin HEAD:main
  echo "Pushed — Vercel is deploying."
fi
