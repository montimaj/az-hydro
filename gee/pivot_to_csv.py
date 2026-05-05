#!/usr/bin/env python3
"""Pivot the Well_Package GeoParquet (long-format, one row per
(well, year)) into per-category wide-format CSVs (one row per well,
year-stamped columns) suitable for `earthengine upload table` ingest.

Source: Data/Outputs/.../Full_Prediction_XGBRF/Well_Package/Well_Package_AF.parquet
        (long-format: 34,707,948 rows × 34 cols, 2.1 GB on disk =
        170,137 wells × 204 years (1896–2099) × 1 row each)

Output: gee/Data/Well_Package/csv/Well_Package__<Category>.csv
        (15 CSVs, flat — one per category)

        Each row carries a `.geo` column with a GeoJSON-string Point
        geometry — same convention as gee/pivot_cap_cumulative.py
        and gee/cap_service_area_to_csv.py.  `earthengine upload
        table` auto-detects `.geo` as the geometry column, so all
        412 year-stamped property names survive intact (unlike a
        shapefile pipeline where DBF would truncate them at 10
        characters).

        Uploaded via `earthengine upload table` (NOT geeup tabup,
        which rejected both bare and zipped CSVs).  Files are staged
        in gs://azhydro/ first by the companion upload script.  The
        CSV basename Well_Package__<Cat> becomes the asset name
        under projects/azhydro/assets/az-wu/.

        Per-category sizing (with sigma):
          features    : 170,137 (one row per well)
          properties  : 4 metadata (REGISTRY_I, WATER_USE, .geo,
                        system:index) + 204 prediction + 204 sigma
                        = 412 properties / feature
          raw CSV     : ~250–500 MB (text-encoded floats; `.geo`
                        GeoJSON strings add ~80 B/row)
        With --no-sigma the property count halves to 208 and the
        CSV drops to ~150–250 MB.

Why per-category split:
  GeoParquet has 30 numeric columns × 204 years = 6,120 wide
  properties.  GEE FeatureCollections handle ~hundreds of properties
  per Feature gracefully but degrade past thousands.  Splitting by
  category keeps each FC at ~412 properties (4 metadata + 204
  prediction + 204 sigma), still inside the comfort zone.

Why AF (acre-feet) only:
  The four unit conventions (mm/ft/m³/AF) are exact mathematical
  conversions of each other; uploading all four stores 4× redundant
  data.  AF is the most common unit for AZ water management.  The
  visualizer can convert to other units client-side if needed:
      mm = AF × 1233.48 / pixel_area_m² × 1000
      ft = AF × 1233.48 / pixel_area_m² × 1000 × 0.00328084
      m³ = AF × 1233.48

Memory budget:
  34.7 M rows × ~10 bytes/cell × 34 cols ≈ 12 GB peak in pandas.
  Tested on 64 GB RAM; runs in ~5–10 min per category, ~75–150 min
  total for all 15 categories.

Usage::

    # Run from anywhere — paths are anchored to the script location
    # (gee/), with output landing under gee/Data/Well_Package/csv/.
    python gee/pivot_to_csv.py                     # all 15 categories
    python gee/pivot_to_csv.py --no-sigma          # prediction only (~half size)
    python gee/pivot_to_csv.py --categories Total_GW Total_SW
    python gee/pivot_to_csv.py --dry-run           # preview only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

# Anchor every default path to the script location so the script
# works from any cwd.  Script lives at gee/pivot_to_csv.py;
# repo_root = gee/.. = the az-hydro checkout.
SCRIPT_DIR = Path(__file__).resolve().parent           # gee/
REPO_ROOT  = SCRIPT_DIR.parent                         # az-hydro/

# Source GeoParquet (acre-feet unit convention)
DEFAULT_SRC = (
    REPO_ROOT / 'Data/Outputs/ML_Model_All_Wells_2000m'
                '/Full_Prediction_XGBRF/Well_Package/Well_Package_AF.parquet'
).resolve()

# 15 categories, in the order they appear in the parquet
CATEGORIES: list[str] = [
    'Total',
    'Irrigation',
    'Non_Irrigation',
    'Irrigation_GW',
    'Irrigation_SW',
    'Non_Irrigation_GW',
    'Non_Irrigation_SW',
    'Total_GW',
    'Total_SW',
    'Irrigation_CU',
    'Irrigation_GW_CU',
    'Irrigation_SW_CU',
    'Total_SW_Capture',
    'Irrigation_SW_Capture',
    'Non_Irrigation_SW_Capture',
]


def geometry_to_geojson_str(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Serialize a GeoDataFrame's geometry column to GeoJSON strings.

    GEE's `earthengine upload table` auto-detects the column name
    `.geo` containing GeoJSON-encoded geometry; this is GEE's
    canonical CSV-table convention (works for points, lines, polygons
    alike).  Output is cast to WGS-84 / EPSG:4326 first so coordinates
    land as lon/lat decimal degrees, which is what GEE expects.
    """
    g = gdf
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(epsg=4326)
    return g.geometry.apply(lambda geom: json.dumps(mapping(geom)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--source', type=Path, default=DEFAULT_SRC,
                        help='Path to Well_Package_AF.parquet')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('Data/Well_Package/csv'),
                        help='Output directory (relative to script location, '
                             'i.e. gee/).  Default: '
                             'gee/Data/Well_Package/csv/')
    parser.add_argument('--categories', nargs='+', default=CATEGORIES,
                        choices=CATEGORIES,
                        help='Subset of categories to convert (default: all)')
    parser.add_argument('--no-sigma', action='store_true',
                        help='Drop sigma columns (halve output size)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan only, do not read or write')
    args = parser.parse_args()

    src = args.source
    if not src.is_file():
        print(f'Error: source parquet not found at {src}', file=sys.stderr)
        return 1

    out_dir = (SCRIPT_DIR / args.output_dir).resolve()
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 64)
    print('Well_Package GeoParquet → per-category CSV pivot')
    print('=' * 64)
    print(f'  Source     : {src}')
    print(f'  Output dir : {out_dir}')
    print(f'  Categories : {len(args.categories)} '
          f'({", ".join(args.categories)})')
    print(f'  Sigma cols : {"NO" if args.no_sigma else "YES"}')
    print(f'  Mode       : {"DRY-RUN" if args.dry_run else "WRITE"}')
    print()

    if args.dry_run:
        for cat in args.categories:
            cols = [f'{cat}_AF']
            if not args.no_sigma:
                cols.append(f'{cat}_AF_sigma')
            est_props = 4 + 204 * len(cols)  # 4 metadata + per-year × n_value_cols
            csv_path = out_dir / f'Well_Package__{cat}.csv'
            print(f'  → {csv_path.relative_to(out_dir.parent.parent.parent)}')
            print(f'      sources: {", ".join(cols)}')
            print(f'      ~{est_props} props/feature × 170,137 features')
        return 0

    # ── Load the source parquet ──────────────────────────────────────
    t0 = time.time()
    print('Loading parquet (may take 30-60 sec on 2.3 GB)...')
    gdf = gpd.read_parquet(src)
    print(f'  Loaded {len(gdf):,} rows × {len(gdf.columns)} cols '
          f'in {time.time() - t0:.1f}s')
    print(f'  CRS: {gdf.crs}')
    print()

    # Defensive cleanup
    gdf = gdf.dropna(subset=['REGISTRY_I', 'Year', 'geometry'])
    gdf['Year'] = gdf['Year'].astype(int)
    gdf['REGISTRY_I'] = gdf['REGISTRY_I'].astype(str)

    # Cast all numeric category columns to float32 to halve memory
    for col in gdf.columns:
        if gdf[col].dtype == 'float64':
            gdf[col] = gdf[col].astype('float32')

    n_wells = gdf['REGISTRY_I'].nunique()
    n_years = gdf['Year'].nunique()
    print(f'  {n_wells:,} distinct wells × {n_years} years '
          f'= {n_wells * n_years:,} expected pivot cells per category')
    print()

    # Static well attributes (one row per well — preserve geometry, WATER_USE)
    static = (
        gdf.drop_duplicates(subset='REGISTRY_I', keep='first')
        [['REGISTRY_I', 'WATER_USE', 'geometry']]
        .reset_index(drop=True)
    )
    static_gdf = gpd.GeoDataFrame(static, geometry='geometry', crs=gdf.crs)
    # Pre-compute the .geo column once (same for every category)
    static_geo_col = geometry_to_geojson_str(static_gdf)

    # ── Per-category pivot + CSV write ────────────────────────────────
    for i, cat in enumerate(args.categories, 1):
        cat_t0 = time.time()
        pred_col = f'{cat}_AF'
        sigma_col = f'{cat}_AF_sigma'

        if pred_col not in gdf.columns:
            print(f'[{i}/{len(args.categories)}] {cat}: SKIP — '
                  f'column {pred_col} not in source parquet')
            continue

        value_cols = [pred_col]
        if not args.no_sigma and sigma_col in gdf.columns:
            value_cols.append(sigma_col)

        print(f'[{i}/{len(args.categories)}] Pivoting {cat} '
              f'({len(value_cols)} value cols)...')

        # Pivot: one row per well, columns named "<cat>_AF_<year>" and
        # "<cat>_AF_sigma_<year>".  pivot_table is the simplest call;
        # for performance on this size, a manual groupby+unstack is
        # equivalent and slightly faster but harder to read.
        wide = gdf.pivot_table(
            index='REGISTRY_I',
            columns='Year',
            values=value_cols,
            aggfunc='first',
        )
        # MultiIndex columns: (value_col, year) → flatten to "<col>_<year>"
        wide.columns = [f'{vc}_{yr}' for vc, yr in wide.columns]
        wide = wide.reset_index()

        # Re-attach static well attributes via REGISTRY_I, then add
        # the .geo GeoJSON-string column (drop the shapely geometry).
        static_plain = pd.DataFrame(static_gdf.drop(columns=['geometry']))
        static_plain['.geo'] = static_geo_col.values
        merged = static_plain.merge(wide, on='REGISTRY_I', how='inner')

        out_csv = out_dir / f'Well_Package__{cat}.csv'
        if out_csv.exists():
            out_csv.unlink()
        merged.to_csv(out_csv, index=False)

        size_mb = out_csv.stat().st_size / 1024 / 1024
        elapsed = time.time() - cat_t0
        print(f'      ✓ {len(merged):,} features × '
              f'{len(merged.columns)} cols → '
              f'{out_csv.name} ({size_mb:.1f} MB) in {elapsed:.1f}s')

    print()
    print('=' * 64)
    csvs = sorted(out_dir.glob('Well_Package__*.csv'))
    total_size = sum(p.stat().st_size for p in csvs)
    print(f'  Done in {time.time() - t0:.1f}s. Total upload payload: '
          f'{total_size / 1024 / 1024:.1f} MB across {len(csvs)} CSVs.')
    print(f'  Output dir: {out_dir}')
    print('=' * 64)
    print()
    print('Next step: stage each CSV in gs://azhydro and ingest via')
    print('`earthengine upload table` — run gee/upload_well_package.sh.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
