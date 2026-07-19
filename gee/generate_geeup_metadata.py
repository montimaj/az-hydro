#!/usr/bin/env python3
"""Generate per-directory geeup metadata CSVs for the AZ-Hydro raster
ImageCollections.

For each leaf directory under ``gee/Data/`` that contains one or more
``*.tif`` files (and is not on the skip list), this script writes a
``metadata.csv`` next to the rasters.  The CSV is consumed by

    geeup upload --source <dir> --dest <gee_path> \\
                 --metadata <dir>/metadata.csv -u <user>

and tells geeup to set per-image properties (year + system:time_start)
as it batch-uploads the per-year TIFs into one ImageCollection per leaf
directory.

Skipped directories (handled out-of-band):

    ADWR/         — already uploaded vectors
    Well_Package/ — GeoParquet, separate workflow (see azhydro-visualizer.js)
    Sigma_Total/  — σ is already band 2 of every per-category TIF;
                    no separate ImageCollection needed.

Year is extracted from filenames matching one of these patterns
(checked in order):

    Total_GW_<YEAR>_AF.tif       (most per-category rasters)
    Total_GW_<YEAR>_mm.tif       (depth conventions)
    OOD_Flag_<YEAR>.tif          (OOD raster naming)
    CAP_Scenario_..._<YEAR>_..   (CAP scenario rasters)

Generic fallback: any 4-digit number 1800-2199 in the filename.

Usage::

    python gee/generate_geeup_metadata.py            # generate all
    python gee/generate_geeup_metadata.py --dry-run  # preview only
    python gee/generate_geeup_metadata.py --root gee/Data --asset-prefix \\
        projects/azhydro/assets/az-wu                # custom asset root
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Directories to skip entirely (recursive — any leaf dir under these is skipped)
SKIP_DIRS: set[str] = {
    'ADWR',
    'Well_Package',
    'Sigma_Total',
}

# Recognised unit-convention sub-directory names.  When a parent dir
# contains exactly these four sub-dirs (each holding TIFs), the four
# are consolidated into a single ImageCollection at the parent level,
# with each TIF tagged with a `unit` property so the visualizer can
# filter on it.
UNIT_DIRS: set[str] = {'Depth_mm', 'Depth_ft', 'Volume_m3', 'Volume_AF'}

# Year extraction — checked in order
YEAR_RE = re.compile(r'(?:^|[_/])(\d{4})(?:[_.]|$)')


def extract_year(filename: str) -> int | None:
    """Return the 4-digit year embedded in *filename*, or None."""
    for match in YEAR_RE.finditer(filename):
        year = int(match.group(1))
        if 1800 <= year <= 2199:
            return year
    return None


def is_skipped(dir_path: Path, root: Path) -> bool:
    """Return True if any path component matches SKIP_DIRS."""
    rel = dir_path.relative_to(root)
    return any(part in SKIP_DIRS for part in rel.parts)


def asset_id_for_dir(dir_path: Path, root: Path, asset_prefix: str) -> str:
    """Build the destination GEE asset ID for *dir_path*.

    Path components are flattened with ``__`` separators so each leaf
    gets a unique asset ID.  When called on a unit-parent dir, the
    asset ID is just ``<category>`` (no ``__<unit>`` suffix) — the four
    unit-conventions are consolidated into one ImageCollection with
    a ``unit`` property tagging each image.

    Examples
    --------
    Total_GW_Rasters/  (unit-parent)
        → projects/azhydro/assets/az-wu/Total_GW_Rasters

    Capture/Total_Rasters/  (nested unit-parent)
        → projects/azhydro/assets/az-wu/Capture__Total_Rasters

    Capture/Total_Fraction/  (flat, no units)
        → projects/azhydro/assets/az-wu/Capture__Total_Fraction

    OOD_Rasters/
        → projects/azhydro/assets/az-wu/OOD_Rasters
    """
    rel = dir_path.relative_to(root)
    # Flatten path with __ separator so each leaf becomes a unique asset
    flat = '__'.join(rel.parts)
    return f'{asset_prefix.rstrip("/")}/{flat}'


def write_metadata_csv(
        out_dir: Path,
        rows: list[dict],
        asset_id: str,
) -> tuple[Path, int]:
    """Write a geeup-compatible metadata CSV.

    Columns (always written; unit/scenario blank for non-applicable rows):
        id_no            : filename stem
        system:time_start: ISO date string, year-01-01T00:00:00
        year             : integer year
        unit             : 'Depth_mm' / 'Depth_ft' / 'Volume_m3' /
                           'Volume_AF' / '' (blank for non-unit rasters)
        category         : derived from category dir name
        asset_collection : destination ImageCollection asset ID
    """
    out_csv = out_dir / 'metadata.csv'
    fieldnames = ['id_no', 'system:time_start', 'year', 'unit',
                  'category', 'asset_collection']
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Ensure all fields exist even if blank
            for fld in fieldnames:
                row.setdefault(fld, '')
            writer.writerow(row)
    return out_csv, len(rows)


def is_unit_parent(dir_path: Path) -> bool:
    """True if *dir_path*'s immediate children are exactly the four
    UNIT_DIRS, each containing at least one TIF.

    Used to detect dirs like ``Total_GW_Rasters/`` (which has
    ``Depth_mm/``, ``Depth_ft/``, ``Volume_m3/``, ``Volume_AF/`` as
    children) that should be consolidated into one ImageCollection.
    """
    children = [p for p in dir_path.iterdir() if p.is_dir()]
    if {p.name for p in children} != UNIT_DIRS:
        return False
    # Every unit subdir must contain at least one TIF
    for c in children:
        if not any(p.suffix.lower() == '.tif' for p in c.iterdir()):
            return False
    return True


def collect_unit_parent_rows(
        parent: Path,
        asset_id: str,
) -> tuple[list[dict], int]:
    """Walk a unit-parent dir's four UNIT_DIRS subdirs and return
    consolidated rows + skip count."""
    rows = []
    skipped = 0
    for unit in sorted(UNIT_DIRS):
        sub = parent / unit
        for tif in sorted(sub.iterdir()):
            if tif.suffix.lower() != '.tif':
                continue
            year = extract_year(tif.name)
            if year is None:
                skipped += 1
                continue
            rows.append({
                'id_no': tif.stem,
                'system:time_start': f'{year}-01-01T00:00:00',
                'year': year,
                'unit': unit,
                'category': parent.name,
                'asset_collection': asset_id,
            })
    return rows, skipped


def collect_flat_dir_rows(
        dir_path: Path,
        asset_id: str,
) -> tuple[list[dict], int]:
    """Walk a flat dir of TIFs and return rows + skip count."""
    rows = []
    skipped = 0
    for tif in sorted(dir_path.iterdir()):
        if tif.suffix.lower() != '.tif':
            continue
        year = extract_year(tif.name)
        if year is None:
            skipped += 1
            continue
        rows.append({
            'id_no': tif.stem,
            'system:time_start': f'{year}-01-01T00:00:00',
            'year': year,
            'unit': '',
            'category': dir_path.name,
            'asset_collection': asset_id,
        })
    return rows, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--root', default='gee/Data',
                        help='Root directory to walk (default: gee/Data)')
    parser.add_argument('--asset-prefix', default='projects/azhydro/assets/az-wu',
                        help='GEE asset path prefix for ImageCollections')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what would be written; do not write CSVs')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.stderr.write(f'Error: root directory not found: {root}\n')
        return 1

    print(f'Scanning {root}')
    print(f'Asset prefix: {args.asset_prefix}')
    print(f'Skipping (recursive): {sorted(SKIP_DIRS)}')
    print(f'Mode: {"dry-run" if args.dry_run else "write"}')
    print()

    total_csvs = 0
    total_rasters = 0
    total_skipped_files = 0
    total_cleaned_stale = 0

    # Custom walk: top-down, with unit-parent consolidation.  When we
    # detect a dir whose children are exactly the four UNIT_DIRS, we
    # produce ONE consolidated CSV at the parent and prune the walk
    # under the unit subdirs.  Otherwise we descend normally and emit
    # one CSV per leaf-of-TIFs.
    for dirpath, dirnames, filenames in os.walk(root):
        dpath = Path(dirpath)
        if is_skipped(dpath, root):
            dirnames[:] = []
            continue

        # ── Case 1: unit-parent — write 4 per-unit-subdir CSVs, all
        #           pointing to the SAME consolidated asset_collection.
        # Why: geeup's --source expects TIFs in one dir, but the four
        # unit conventions live in four sibling subdirs.  We write a
        # metadata.csv inside each unit subdir (so geeup can find its
        # TIFs locally) but set asset_collection to the parent-level
        # IC for all four, so geeup uploads all 4 unit subdirs into
        # the same ImageCollection with the `unit` property tagging
        # each image.
        if is_unit_parent(dpath):
            asset_id = asset_id_for_dir(dpath, root, args.asset_prefix)
            rel_display = dpath.relative_to(root)
            print(f'  {rel_display}/  (unit-parent → 4 per-unit CSVs targeting'
                  f' {asset_id.split("/")[-1]})')
            # Clean any stale parent-level metadata.csv from prior consolidation
            stale_parent = dpath / 'metadata.csv'
            if stale_parent.exists() and not args.dry_run:
                stale_parent.unlink()
                total_cleaned_stale += 1
            for unit in sorted(UNIT_DIRS):
                sub = dpath / unit
                rows, skipped = collect_flat_dir_rows(sub, asset_id)
                # Override the unit + category fields (collect_flat_dir_rows
                # was called on the sub-dir and used its name for both).
                # We want unit = the unit subdir name and category =
                # the parent category name (e.g. 'Total_GW_Rasters').
                for r in rows:
                    r['unit'] = unit
                    r['category'] = dpath.name
                if not args.dry_run:
                    csv_path, n_written = write_metadata_csv(sub, rows, asset_id)
                    print(f'      {unit:<10}  {n_written} rows  →  '
                          f'{csv_path.relative_to(root.parent)}')
                else:
                    print(f'      {unit:<10}  {len(rows)} rows would be written')
                total_skipped_files += skipped
                total_csvs += 1
                total_rasters += len(rows) + skipped
            # Prune walk so we don't re-descend into the four unit subdirs
            dirnames[:] = []
            continue

        # ── Case 2: flat dir of TIFs ────────────────────────────────
        tifs = [f for f in filenames if f.lower().endswith('.tif')]
        if not tifs:
            continue
        asset_id = asset_id_for_dir(dpath, root, args.asset_prefix)
        rel_display = dpath.relative_to(root)
        print(f'  {rel_display}/  (flat, {len(tifs)} TIFs)')
        print(f'      → {asset_id}')
        rows, skipped = collect_flat_dir_rows(dpath, asset_id)
        if not args.dry_run:
            csv_path, n_written = write_metadata_csv(dpath, rows, asset_id)
            print(f'      {n_written} rows written ({skipped} no-year skipped)'
                  f'  → {csv_path.relative_to(root.parent)}')
        else:
            print(f'      {len(rows)} rows would be written ({skipped} no-year skipped)')
        total_skipped_files += skipped
        total_csvs += 1
        total_rasters += len(rows) + skipped

    print()
    print(f'Done.  {total_csvs} CSV files {"would be" if args.dry_run else ""} written, '
          f'{total_rasters} TIFs catalogued, {total_skipped_files} files skipped'
          f' (no year in filename).')
    if total_cleaned_stale:
        print(f'Cleaned {total_cleaned_stale} stale leaf-level metadata.csv files '
              f'left over from prior runs.')
    print()
    print('Next steps:')
    print('  1. Authenticate geeup with your Google account (one-time):')
    print('       earthengine authenticate')
    print('       geeup auth')
    print('  2. For each generated metadata.csv, batch-upload its dir to GEE:')
    print('       geeup upload --source <leaf-dir> \\')
    print('                    --dest <asset_collection from CSV> \\')
    print('                    --metadata <leaf-dir>/metadata.csv \\')
    print('                    -u <your-google-account>')
    print('     (Or wrap in a shell loop over each metadata.csv.)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
