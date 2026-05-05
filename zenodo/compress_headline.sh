#!/bin/bash
# Build az-hydro-headline.7z — a focused subset of the AZ-Hydro data
# archive containing only the published Sci Data deliverables
# (predictions with UQ, well package, SW capture, time series,
# validation, CAP scenarios, era-mean figures).  ~8 GB compressed.
#
# For the full bit-identical reproducibility archive, see the
# accompanying az-hydro-data.7z (74 GB) which includes raw inputs,
# Step 2 cross-validation, intermediate predictor stacks, and per-
# component σ rasters.
#
# Same 7-Zip flags as compress_for_zenodo.sh (LZMA2 + solid mode +
# 4-thread / 1 GB dict tuned for 64 GB RAM).

set -euo pipefail
# Script lives in zenodo/; cd to repo root so source paths under Data/
# resolve correctly, and write the .7z back into zenodo/ so all Zenodo
# artefacts stay grouped.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

OUT="$SCRIPT_DIR/az-hydro-headline.7z"
PRED="Data/Outputs/ML_Model_All_Wells_2000m/Full_Prediction_XGBRF"
HEADLINE_README="Data/HEADLINE_README.md"

# Sanity checks
if ! command -v 7z >/dev/null 2>&1; then
  echo "Error: 7-Zip (7z) not installed." >&2; exit 1
fi
[ -d "$PRED" ] || { echo "Error: $PRED not found" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Source set: explicit per-directory list (no tree-wide globbing) so the
# headline contents are deterministic and reviewer-checkable.
# ---------------------------------------------------------------------------
SRC=(
  # Documentation
  "Data/README.md"
  "$HEADLINE_README"

  # ----- Per-year 6-band augmented rasters (prediction + σ + CV +
  #       SNR + lower 95 % CI + upper 95 % CI) — the published per-pixel
  #       per-year product.  Each directory contains 4 unit-convention
  #       sub-dirs: Depth_mm/, Depth_ft/, Volume_m3/, Volume_AF/.  The
  #       in-place augmentation rewrote every *_Rasters/*.tif as a
  #       6-band stack (band 1 = prediction; bands 2–6 = UQ propagation).
  "$PRED/Predicted_Rasters"
  "$PRED/Total_GW_Rasters"
  "$PRED/Total_SW_Rasters"
  "$PRED/Irrigation_Rasters"
  "$PRED/Irrigation_GW_Rasters"
  "$PRED/Irrigation_SW_Rasters"
  "$PRED/Irrigation_CU_Rasters"
  "$PRED/Irrigation_GW_CU_Rasters"
  "$PRED/Irrigation_SW_CU_Rasters"
  "$PRED/Non_Irrigation_Rasters"
  "$PRED/Non_Irrigation_GW_Rasters"
  "$PRED/Non_Irrigation_SW_Rasters"

  # ----- Per-category summary CSVs and time-series figures
  #       (one dir per category — paired with the *_Rasters/ above)
  "$PRED/Total_GW"
  "$PRED/Total_SW"
  "$PRED/Irrigation"
  "$PRED/Irrigation_GW"
  "$PRED/Irrigation_SW"
  "$PRED/Irrigation_CU"
  "$PRED/Irrigation_GW_CU"
  "$PRED/Irrigation_SW_CU"
  "$PRED/Non_Irrigation"
  "$PRED/Non_Irrigation_GW"
  "$PRED/Non_Irrigation_SW"

  # ----- Per-pixel total uncertainty rasters
  "$PRED/Uncertainty/Sigma_Total"

  # ----- Per-well GeoPackage + Parquet (the headline well-level product)
  "$PRED/Well_Package"

  # ----- SW Capture Index outputs (full set: fraction + per-unit rasters)
  "$PRED/SW_Capture"

  # ----- Aggregated summaries and time series
  "$PRED/Annual_Summaries"
  "$PRED/ADWR"
  "$PRED/Basin_Time_Series"
  "$PRED/AMA_INA_Time_Series"
  "$PRED/Subbasin_Time_Series"
  "$PRED/Full_Period_Time_Series.csv"
  "$PRED/Full_Period_Time_Series.png"
  "$PRED/Era_Summary_Bar.png"
  "$PRED/Mean_Annual_Predicted_mm.tif"
  "$PRED/Prediction_Exceedance_Summary.csv"
  "$PRED/Graphical_Abstract_Fig1.png"

  # ----- CAP delivery scenario sweep (CSVs + time-series figures)
  "$PRED/Uncertainty/CAP_Scenario"

  # ----- Era-mean spatial figures + CAP scenario maps (Raster_Maps
  #       includes its CAP_Scenario sub-directory automatically)
  "$PRED/Raster_Maps"

  # ----- Step 4 USGS / NHM / Reitz / PS / CAP-SRP validation outputs
  "$PRED/Withdrawal_Intercomparison"
  "$PRED/CU_Intercomparison"
  "$PRED/Peff_Intercomparison"
  "$PRED/PS_Intercomparison"
  "$PRED/NHM_IE_Basins"
  "$PRED/USGS_Calibration_Bars"
  "$PRED/CAP_SRP_Validation"
)

# Verify every source exists before starting (catch typos early)
echo "=== Source inventory ==="
missing=0
for s in "${SRC[@]}"; do
  if [ ! -e "$s" ]; then
    echo "  MISSING: $s"
    missing=$((missing + 1))
  fi
done
[ "$missing" -gt 0 ] && { echo "Aborting: $missing missing source paths" >&2; exit 1; }

du -sh "${SRC[@]}" 2>/dev/null
echo ""
echo "Total source size:"
du -ck "${SRC[@]}" 2>/dev/null | tail -1 | awk '{print "  "$1/1024/1024" GB"}'
echo ""

# Free disk sanity
free_gb=$(df -g . | awk 'NR==2 {print $4}')
echo "Free disk: ${free_gb} GB on $(pwd)"
echo ""

[ -f "$OUT" ] && { echo "Removing existing $OUT..."; rm -f "$OUT"; }

START=$(date +%s)

# Same flags as compress_for_zenodo.sh — see that file for rationale.
7z a -t7z \
  -m0=LZMA2 \
  -mx=9 \
  -mfb=273 \
  -md=1024m \
  -mmt=4 \
  -ms=on \
  -bb1 -bsp0 -bso1 \
  '-xr!*.DS_Store' \
  '-xr!*.aux.xml' \
  "$OUT" \
  "${SRC[@]}" 2>&1 | awk '{
      "date \"+%H:%M:%S\"" | getline ts
      close("date \"+%H:%M:%S\"")
      printf "[%s] %s\n", ts, $0
      fflush()
  }'

END=$(date +%s)
echo
echo "=== Done in $(( (END-START)/60 )) min ==="
ls -lh "$OUT"
echo
src_kb=$(du -ck "${SRC[@]}" 2>/dev/null | tail -1 | awk '{print $1}')
src_bytes=$((src_kb * 1024))
zip_bytes=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")
ratio=$(awk -v s="$src_bytes" -v z="$zip_bytes" 'BEGIN {printf "%.1f%%", 100*z/s}')
echo "Compression summary:"
echo "  Source   : $(echo "scale=2; $src_bytes/1024/1024/1024" | bc) GB"
echo "  Archive  : $(echo "scale=2; $zip_bytes/1024/1024/1024" | bc) GB"
echo "  Ratio    : $ratio"
