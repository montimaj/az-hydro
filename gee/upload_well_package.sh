#!/bin/bash
# Batch-upload the 15 per-category Well_Package CSVs to GEE as
# FeatureCollections via `earthengine upload table` (bypassing geeup
# tabup, which rejects CSVs in practice — see gee/pivot_to_csv.py
# docstring for the full diagnosis).
#
# For each CSV:
#   1. Stage in GCS:  gsutil cp <csv> gs://azhydro/<csv>
#   2. Submit ingest: earthengine upload table --asset_id=... <gcs_uri>
#
# Result on GEE:  one FC per CSV at
#   projects/azhydro/assets/az-wu/Well_Package__<Category>
# (the asset name is the CSV basename without .csv).  Each FC has
# ~170 k well features, each with year-stamped property columns
# (e.g. Total_AF_2024, Total_AF_sigma_2024).  The visualizer can
# chart any well's full history client-side from the property array.
#
# Usage:
#   ./gee/upload_well_package.sh                # upload all 15
#   ./gee/upload_well_package.sh --dry-run      # print commands only
#   ./gee/upload_well_package.sh --filter Total # only categories matching
#
# Pre-flight (one-time):
#   gcloud auth login
#   gcloud config set project azhydro
#   earthengine authenticate
#   pip install earthengine-api
#
# If you haven't generated the CSVs yet:
#   python gee/pivot_to_csv.py

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to repo root
REPO_ROOT="$(pwd)"

# ── Configuration ────────────────────────────────────────────────────
GCS_BUCKET="azhydro"
SOURCE_DIR_REL="gee/Data/Well_Package/csv"
SOURCE_DIR="$REPO_ROOT/$SOURCE_DIR_REL"
DEST_FOLDER="projects/azhydro/assets/az-wu"
LOG_DIR="$REPO_ROOT/gee/upload_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/Well_Package_upload.log"

# Optional CLI flags
DRY_RUN=0
FILTER=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --filter)
      FILTER="$2"
      shift 2
      ;;
    -h|--help)
      head -32 "$0" | grep '^#' | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 1
      ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────
for cmd in gsutil earthengine; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' not on PATH." >&2
    echo "  brew install --cask google-cloud-sdk    # gcloud + gsutil" >&2
    echo "  pip install earthengine-api             # earthengine CLI" >&2
    exit 1
  fi
done

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: source dir not found at $SOURCE_DIR" >&2
  echo "Generate first:  python gee/pivot_to_csv.py" >&2
  exit 1
fi

# Collect every Well_Package__*.csv (portable — works on macOS bash 3.2)
CSV_FILES=()
while IFS= read -r f; do
  CSV_FILES+=("$f")
done < <(find "$SOURCE_DIR" -maxdepth 1 -name 'Well_Package__*.csv' -type f | sort)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "Error: no Well_Package__*.csv files in $SOURCE_DIR/" >&2
  echo "Run: python gee/pivot_to_csv.py" >&2
  exit 1
fi

# Apply optional filter
if [[ -n "$FILTER" ]]; then
  FILTERED=()
  for csv in "${CSV_FILES[@]}"; do
    if [[ "$csv" == *"$FILTER"* ]]; then
      FILTERED+=("$csv")
    fi
  done
  CSV_FILES=("${FILTERED[@]}")
fi

TOTAL=${#CSV_FILES[@]}
TOTAL_MB=$(du -cm "${CSV_FILES[@]}" 2>/dev/null | tail -1 | awk '{print $1}')

# ── Summary banner ────────────────────────────────────────────────────
echo "=================================================================="
echo "  AZ-Hydro Well_Package upload (GCS + earthengine)"
echo "=================================================================="
echo "  GCS bucket : gs://$GCS_BUCKET"
echo "  Source dir : $SOURCE_DIR ($TOTAL CSVs, ${TOTAL_MB} MB)"
echo "  Dest folder: $DEST_FOLDER"
echo "  Log file   : $LOG_FILE"
[[ -n "$FILTER" ]]   && echo "  Filter     : $FILTER"
[[ $DRY_RUN -eq 1 ]] && echo "  Mode       : DRY-RUN (no commands run)"
echo "=================================================================="
echo ""

# ── Upload loop ───────────────────────────────────────────────────────
i=0
: > "$LOG_FILE"  # truncate
for csv in "${CSV_FILES[@]}"; do
  i=$((i + 1))
  base=$(basename "$csv" .csv)              # e.g. Well_Package__Total_GW
  asset_id="$DEST_FOLDER/$base"
  gcs_uri="gs://$GCS_BUCKET/$base.csv"

  echo "[$i/$TOTAL] $base"
  echo "          → $gcs_uri"
  echo "          → $asset_id"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "          (dry-run; would invoke):"
    echo "            gsutil cp $csv $gcs_uri"
    echo "            earthengine upload table --asset_id=$asset_id $gcs_uri"
    echo ""
    continue
  fi

  start_ts=$(date +%s)
  {
    echo "──── [$i/$TOTAL] $base ────"
    echo "[stage] gsutil cp $csv $gcs_uri"
    gsutil cp "$csv" "$gcs_uri"
    echo "[ingest] earthengine upload table --asset_id=$asset_id $gcs_uri"
    earthengine upload table --asset_id="$asset_id" "$gcs_uri"
  } 2>&1 | tee -a "$LOG_FILE"
  end_ts=$(date +%s)
  echo "          ✓ submitted in $((end_ts - start_ts)) sec"
  echo ""
done

# ── Final summary ─────────────────────────────────────────────────────
echo "=================================================================="
echo "  Batch submit complete: $TOTAL upload tasks queued."
echo "  Per-stage log: $LOG_FILE"
echo ""
echo "  Monitor progress:    earthengine task list | head -20"
echo "  GCS staging copies:  gs://$GCS_BUCKET/Well_Package__*.csv"
echo "  Cleanup once done:   gsutil -m rm gs://$GCS_BUCKET/Well_Package__*.csv"
echo "=================================================================="
