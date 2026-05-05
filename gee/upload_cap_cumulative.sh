#!/bin/bash
# Upload the pivoted CAP-cumulative CSV to GEE as a tabular
# FeatureCollection via `earthengine upload table` (bypassing geeup
# tabup, which rejects CSVs in practice).
#
# Two-step:
#   1. Stage gee/Data/CAP_Scenario/Cumulative/CAP_Scenario_Cumulative.csv
#      → gs://azhydro/CAP_Scenario_Cumulative.csv  (gsutil cp)
#   2. Submit ingest task → projects/azhydro/assets/az-wu/CAP_Scenario_Cumulative
#      (earthengine upload table; .geo column auto-detected)
#
# Result: ~364 features (52 basins × 7 scenarios), each with
# year-stamped Cum_<year> columns covering 2026–2099.  The
# visualizer reads this asset on every CAP-mode click.
#
# Usage:
#   ./gee/upload_cap_cumulative.sh             # stage + ingest
#   ./gee/upload_cap_cumulative.sh --dry-run   # print commands only
#
# Pre-flight (one-time):
#   gcloud auth login
#   gcloud config set project azhydro
#   earthengine authenticate
#   pip install earthengine-api

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to repo root
REPO_ROOT="$(pwd)"

GCS_BUCKET="azhydro"
SOURCE_CSV="$REPO_ROOT/gee/Data/CAP_Scenario/Cumulative/CAP_Scenario_Cumulative.csv"
ASSET_NAME="CAP_Scenario_Cumulative"
ASSET_ID="projects/azhydro/assets/az-wu/${ASSET_NAME}"
GCS_URI="gs://${GCS_BUCKET}/${ASSET_NAME}.csv"
LOG_FILE="$REPO_ROOT/gee/upload_logs/${ASSET_NAME}.log"
mkdir -p "$(dirname "$LOG_FILE")"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) head -23 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────
for cmd in gsutil earthengine; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' not on PATH." >&2
    exit 1
  fi
done
if [[ ! -f "$SOURCE_CSV" ]]; then
  echo "Error: $SOURCE_CSV not found." >&2
  echo "Generate first:  python gee/pivot_cap_cumulative.py" >&2
  exit 1
fi

echo "=================================================================="
echo "  AZ-Hydro CAP_Scenario_Cumulative upload (GCS + earthengine)"
echo "=================================================================="
echo "  Source CSV : $SOURCE_CSV"
echo "  GCS staging: $GCS_URI"
echo "  Asset ID   : $ASSET_ID"
echo "  Log file   : $LOG_FILE"
[[ $DRY_RUN -eq 1 ]] && echo "  Mode       : DRY-RUN (no commands run)"
echo "=================================================================="
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry-run; would invoke):"
  echo "  gsutil cp $SOURCE_CSV $GCS_URI"
  echo "  earthengine upload table --asset_id=$ASSET_ID $GCS_URI"
  exit 0
fi

start_ts=$(date +%s)
echo "[1/2] Staging in GCS..."
{ gsutil cp "$SOURCE_CSV" "$GCS_URI"; } 2>&1 | tee "$LOG_FILE"

echo "[2/2] Submitting ingest task..."
{ earthengine upload table \
    --asset_id="$ASSET_ID" \
    "$GCS_URI"; } 2>&1 | tee -a "$LOG_FILE"

end_ts=$(date +%s)
echo ""
echo "=================================================================="
echo "  ✓ submitted in $((end_ts - start_ts)) sec"
echo "  Log: $LOG_FILE"
echo "  Asset: $ASSET_ID"
echo ""
echo "  Monitor progress with:"
echo "    earthengine task list | head"
echo "=================================================================="
