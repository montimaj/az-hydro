#!/bin/bash
# Max-compress Data/Inputs + Data/Outputs into a single .7z archive for
# Zenodo upload using LZMA2 + solid-block mode.
#
# Why .7z instead of .zip?  p7zip 17.05's ZIP writer doesn't implement
# LZMA2 (E_NOTIMPL); the codec only works inside the .7z container in
# this build.  .7z + LZMA2 + solid mode also delivers ~10–15 % better
# compression than ZIP+LZMA2 because of lower per-file metadata
# overhead and cross-file deduplication inside solid blocks.
#
# Tuned for a 64 GB RAM workstation (-md=1024m -mmt=4 → ~48 GB peak,
# 4 cores busy, roughly 2× faster than -md=2048m -mmt=2 at the cost
# of ~5 % worse compression ratio).
#
# Document in the Zenodo description that users need 7-Zip / Keka /
# The Unarchiver / WinRAR to unpack — Apple's stock Archive Utility
# does NOT open .7z files.

set -euo pipefail
# Script lives in zenodo/; cd to repo root so source paths like
# Data/Inputs resolve correctly, and write the .7z back into zenodo/
# so all Zenodo artefacts stay grouped.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

OUT="$SCRIPT_DIR/az-hydro-data.7z"
SRC=("Data/Inputs" "Data/Outputs" "Data/README.md")

# Sanity checks
if ! command -v 7z >/dev/null 2>&1; then
  echo "Error: 7-Zip (7z) not installed.  Install with:" >&2
  echo "  macOS:  brew install p7zip" >&2
  echo "  Linux:  sudo apt-get install p7zip-full" >&2
  exit 1
fi
for d in "${SRC[@]}"; do
  [ -e "$d" ] || { echo "Error: $d not found" >&2; exit 1; }
done

# Free disk space sanity (need ~250 GB headroom)
free_gb=$(df -g . | awk 'NR==2 {print $4}')
echo "Free disk: ${free_gb} GB on $(pwd)"

echo "Source totals:"
du -sh "${SRC[@]}" 2>/dev/null
echo

# Remove stale archive if rerunning
[ -f "$OUT" ] && { echo "Removing existing $OUT..."; rm -f "$OUT"; }

START=$(date +%s)

# 7-Zip flags (tuned for 64 GB RAM):
#   -t7z             output .7z container
#   -m0=LZMA2        first method = LZMA2 (max ratio for the format)
#   -mx=9            ultra compression level
#   -mfb=273         maximum "fast bytes" (best match search)
#   -md=1024m        1 GB dictionary — still wide enough to deduplicate
#                    most of the 1896–2099 per-year stacks; ~5 % worse
#                    ratio than -md=2048m but unlocks 4-way parallelism
#                    inside the 64 GB RAM envelope (12× dict × 4 ≈ 48 GB).
#   -mmt=4           4 parallel LZMA2 blocks → 4 cores busy ⇒ roughly 2×
#                    wall-clock speedup vs -mmt=2 -md=2048m.  Drop back to
#                    -mmt=2 -md=2048m if you want the absolute smallest
#                    archive at the cost of doubled wall-clock.
#   -ms=on           solid mode (default for .7z) — concatenates files into
#                    big blocks before compression so LZMA2 can deduplicate
#                    across files.  This is the main reason .7z beats ZIP
#                    for many-small-files trees like our per-year rasters.
#   -bb1             log each file as it's added (line per file, not in-place)
#   -bsp0            disable in-place progress bar (so logs stay grep-friendly)
#   -bso1            send standard messages to stdout
#   -xr!*.DS_Store   skip macOS dotfiles
#   -xr!*.aux.xml    skip QGIS-generated raster band-statistics sidecars
#   -xr!Inputs copy  skip the local pre-trim backup directory
#
# Output is piped through awk to prepend HH:MM:SS timestamps so the log
# remains useful for ETA estimation hours into the run.  `set -o
# pipefail` (set above) ensures any 7z failure still aborts the script.
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
  '-xr!Inputs copy' \
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
echo "Compression summary:"
src_bytes=$(du -sk "${SRC[@]}" | awk '{s+=$1} END {print s*1024}')
zip_bytes=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")
ratio=$(awk -v s="$src_bytes" -v z="$zip_bytes" 'BEGIN {printf "%.1f%%", 100*z/s}')
echo "  Source   : $(echo "scale=1; $src_bytes/1024/1024/1024" | bc) GB"
echo "  Archive  : $(echo "scale=1; $zip_bytes/1024/1024/1024" | bc) GB"
echo "  Ratio    : $ratio (lower is better)"
echo
echo "Tip: keep the laptop plugged in and run under caffeinate to"
echo "prevent sleep mid-run, e.g.:  caffeinate -d ./zenodo/compress_for_zenodo.sh"
