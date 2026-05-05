#!/bin/bash
# Upload every metadata.csv dir under gee/Data/ to its target ImageCollection
# in projects/azhydro/assets/az-wu/.
#
# Companion to gee/generate_geeup_metadata.py — that script writes one
# metadata.csv per leaf TIF dir (with asset_collection column pointing
# to the destination IC).  This script reads each CSV's
# asset_collection and runs `geeup upload` for that source dir.
#
# Unit-parent categories (Total_GW_Rasters, Irrigation_Rasters, etc.)
# have 4 metadata.csvs (one per Depth_mm/Depth_ft/Volume_m3/Volume_AF
# sub-dir), all pointing to the SAME asset_collection — geeup will
# create the IC on the first upload and append on the next 3, with
# the `unit` property tagging each image so the visualizer can filter
# on it.
#
# Usage:
#   ./gee/upload_to_gee.sh                # upload everything (with --resume)
#   ./gee/upload_to_gee.sh --dry-run      # print commands without running
#   ./gee/upload_to_gee.sh --filter Total # only upload dirs matching pattern
#
# Pre-flight (one-time):
#   pip install geeup
#   earthengine authenticate
#   geeup auth

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to repo root
REPO_ROOT="$(pwd)"

# ── Configuration ────────────────────────────────────────────────────
USER_EMAIL="monti.majumdar@gmail.com"
WORKERS=10
# geeup needs absolute paths in --source / --metadata; use abs throughout
# and only strip back to relative for log filenames + console display.
DATA_ROOT_REL="gee/Data"
DATA_ROOT="$REPO_ROOT/$DATA_ROOT_REL"
LOG_DIR="$REPO_ROOT/gee/upload_logs"
mkdir -p "$LOG_DIR"

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
      head -25 "$0" | grep '^#' | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 1
      ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────
if ! command -v geeup >/dev/null 2>&1; then
  echo "Error: geeup not installed.  Install with:" >&2
  echo "  pip install geeup" >&2
  exit 1
fi

# ── Collect every metadata.csv (portable — works on macOS bash 3.2) ──
CSV_FILES=()
while IFS= read -r f; do
  CSV_FILES+=("$f")
done < <(find "$DATA_ROOT" -name 'metadata.csv' -type f | sort)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "Error: no metadata.csv files found under $DATA_ROOT/" >&2
  echo "Run: python gee/generate_geeup_metadata.py" >&2
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

# ── Summary banner ────────────────────────────────────────────────────
TOTAL=${#CSV_FILES[@]}
echo "=================================================================="
echo "  AZ-Hydro batch upload to Google Earth Engine"
echo "=================================================================="
echo "  User      : $USER_EMAIL"
echo "  Workers   : $WORKERS"
echo "  Data root : $DATA_ROOT_REL ($DATA_ROOT)"
echo "  Log dir   : $LOG_DIR"
echo "  CSV count : $TOTAL"
[[ -n "$FILTER" ]]   && echo "  Filter    : $FILTER"
[[ $DRY_RUN -eq 1 ]] && echo "  Mode      : DRY-RUN (no uploads)"
echo "=================================================================="
echo ""

# ── Upload loop ───────────────────────────────────────────────────────
i=0
for csv in "${CSV_FILES[@]}"; do
  i=$((i + 1))
  src_dir=$(dirname "$csv")
  # Pull the destination from the second row's asset_collection column (col 6)
  dest=$(awk -F, 'NR==2 {print $NF}' "$csv" | tr -d '\r\n')
  if [[ -z "$dest" ]]; then
    echo "[$i/$TOTAL] SKIP — no asset_collection in $csv" >&2
    continue
  fi
  rel_src=${src_dir#"$DATA_ROOT"/}
  log_basename=$(echo "$rel_src" | tr '/' '_')
  log_file="$LOG_DIR/${log_basename}.log"

  echo "[$i/$TOTAL] $rel_src"
  echo "          → $dest"
  echo "          log: $log_file"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "          (dry-run; would invoke geeup with --workers $WORKERS --resume)"
    echo ""
    continue
  fi

  # The actual upload — log to per-source file + tee to stdout for live progress.
  # `--resume` lets the script be safely re-run; geeup will skip already-uploaded
  # assets and continue from where it left off.
  start_ts=$(date +%s)
  if geeup upload \
      --source "$src_dir" \
      --dest "$dest" \
      --metadata "$csv" \
      --user "$USER_EMAIL" \
      --workers "$WORKERS" \
      --resume 2>&1 | tee "$log_file"; then
    end_ts=$(date +%s)
    echo "          ✓ done in $((end_ts - start_ts)) sec"
  else
    rc=$?
    echo "          ✗ FAILED (exit $rc) — see $log_file" >&2
    echo ""
    # Don't abort the whole batch; log and move on.  Re-run the script with
    # --filter <category-name> to retry just the failed one (--resume picks
    # up where it left off).
    continue
  fi
  echo ""
done

# ── Final summary ─────────────────────────────────────────────────────
echo "=================================================================="
echo "  Batch upload complete."
echo "  Per-source logs: $LOG_DIR/"
echo "  Quick failure check:  grep -L 'done' $LOG_DIR/*.log"
echo "=================================================================="
