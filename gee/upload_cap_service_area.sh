#!/bin/bash
# Upload the CAP service-area CSV to GEE as a single FeatureCollection
# via `earthengine upload table` (bypassing geeup tabup, which
# rejects CSVs in practice — see gee/cap_service_area_to_csv.py
# docstring for the full diagnosis).
#
# Two-step:
#   1. Stage gee/Data/CAP/CAP.csv → gs://azhydro/CAP.csv  (gsutil cp)
#   2. Submit ingest task → projects/azhydro/assets/az-wu/CAP
#      (earthengine upload table; .geo column auto-detected as geometry)
#
# Companion to gee/cap_service_area_to_csv.py.  The visualizer expects
# the asset at projects/azhydro/assets/az-wu/CAP.
#
# Usage:
#   ./gee/upload_cap_service_area.sh             # stage + ingest
#   ./gee/upload_cap_service_area.sh --dry-run   # print commands, run none
#
# Pre-flight (one-time):
#   gcloud auth login
#   gcloud config set project azhydro
#   earthengine authenticate
#   pip install earthengine-api

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to repo root
REPO_ROOT="$(pwd)"

GCS_BUCKET="azhydro"   # same bucket used for AZ_HUC12.geojson — see
                        # azhydro/hydrolibs/dataops.py:803
SOURCE_CSV="$REPO_ROOT/gee/Data/CAP/CAP.csv"
ASSET_NAME="CAP"
ASSET_ID="projects/azhydro/assets/az-wu/${ASSET_NAME}"
GCS_URI="gs://${GCS_BUCKET}/${ASSET_NAME}.csv"
MAX_ERROR_M="1.0"
LOG_FILE="$REPO_ROOT/gee/upload_logs/${ASSET_NAME}_service_area.log"
mkdir -p "$(dirname "$LOG_FILE")"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) head -22 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────
for cmd in gsutil earthengine; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' not on PATH.  Install via:" >&2
    echo "  brew install --cask google-cloud-sdk    # gcloud + gsutil" >&2
    echo "  pip install earthengine-api             # earthengine CLI" >&2
    exit 1
  fi
done
if [[ ! -f "$SOURCE_CSV" ]]; then
  echo "Error: $SOURCE_CSV not found." >&2
  echo "Generate first:  python gee/cap_service_area_to_csv.py" >&2
  exit 1
fi

echo "=================================================================="
echo "  AZ-Hydro CAP service area upload (GCS + earthengine)"
echo "=================================================================="
echo "  Source CSV : $SOURCE_CSV ($(du -k "$SOURCE_CSV" | awk '{print $1}') KB)"
echo "  GCS staging: $GCS_URI"
echo "  Asset ID   : $ASSET_ID"
echo "  Max error  : ${MAX_ERROR_M} m"
echo "  Log file   : $LOG_FILE"
[[ $DRY_RUN -eq 1 ]] && echo "  Mode       : DRY-RUN (no commands run)"
echo "=================================================================="
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry-run; would invoke):"
  echo "  gsutil cp $SOURCE_CSV $GCS_URI"
  echo "  earthengine upload table --asset_id=$ASSET_ID \\"
  echo "    --max_error=$MAX_ERROR_M $GCS_URI"
  exit 0
fi

# ── 1. Stage CSV in GCS ──────────────────────────────────────────────
start_ts=$(date +%s)
echo "[1/2] Staging in GCS..."
{ gsutil cp "$SOURCE_CSV" "$GCS_URI"; } 2>&1 | tee "$LOG_FILE"

# ── 2. Submit ingest task ────────────────────────────────────────────
# `earthengine upload table` auto-detects the `.geo` column as
# the geometry source — no flag needed.
echo "[2/2] Submitting ingest task..."
{ earthengine upload table \
    --asset_id="$ASSET_ID" \
    --max_error="$MAX_ERROR_M" \
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
echo ""
echo "  GCS staging copy is left at $GCS_URI (delete with"
echo "  'gsutil rm $GCS_URI' once the asset is ingested)."
echo "=================================================================="
