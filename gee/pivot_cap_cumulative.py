#!/usr/bin/env python3
"""Pivot the CAP-scenario cumulative drawdown CSV into a wide-format
GEE-ingest CSV (with `.geo` GeoJSON-string geometry column) suitable
for `earthengine upload table` ingest.

Source CSV (long format, ~27k rows):
  Data/Outputs/.../CAP_Scenario/CAP_Scenario_Cumulative.csv
    columns: Year, Scenario, Basin, Cumulative_Delta_GW_AF
    52 basins × 7 scenarios × 74 years (2026–2099)

Wide-format output (one feature per [basin, scenario]):
  gee/Data/CAP_Scenario/Cumulative/CAP_Scenario_Cumulative.csv
  Each row has columns:
    Basin             : basin name (matches Groundwater_Basin.BASIN_NAME)
    Scenario          : scenario label (Basic_Coordination_237kAF, ...)
    Cum_2026 .. Cum_2099  : cumulative ΔGW in AF for that year
    .geo              : GeoJSON Point string (basin centroid in WGS-84;
                        `earthengine upload table` auto-detects `.geo`
                        as the geometry column).  The visualizer
                        ignores the geometry and joins on `Basin`.

  Uploaded via `earthengine upload table` (NOT geeup tabup, which
  rejected both bare and zipped CSVs).  Files are staged in
  gs://azhydro/ first by the companion upload script.

After running this script, upload via:
  ./gee/upload_cap_cumulative.sh

The visualizer (gee/azhydro-visualizer.js) reads
  projects/azhydro/assets/az-wu/CAP_Scenario_Cumulative
on every CAP-mode click and charts all 7 scenarios for the basin
under the click point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_CSV = (REPO_ROOT / 'Data/Outputs/ML_Model_All_Wells_2000m'
           '/Full_Prediction_XGBRF/Uncertainty/CAP_Scenario'
           '/CAP_Scenario_Cumulative.csv')

# Basin polygons → centroids for the .geo column (every row needs a
# geometry; the visualizer joins on Basin name and ignores it).
BASIN_SHP_CANDIDATES = [
    REPO_ROOT / 'Data/Inputs/GW_Data/Groundwater_Basin/Groundwater_Basin.shp',
    REPO_ROOT / 'Data/Inputs_Zenodo/GW_Data/Groundwater_Basin.shp',
]

OUT_DIR = REPO_ROOT / 'gee/Data/CAP_Scenario/Cumulative'
OUT_CSV = OUT_DIR / 'CAP_Scenario_Cumulative.csv'


def main() -> int:
    if not SRC_CSV.is_file():
        print(f'Error: source CSV not found: {SRC_CSV}', file=sys.stderr)
        return 1

    basin_shp = next((p for p in BASIN_SHP_CANDIDATES if p.is_file()), None)
    if basin_shp is None:
        print('Error: Groundwater_Basin shapefile not found in any of:',
              file=sys.stderr)
        for p in BASIN_SHP_CANDIDATES:
            print(f'  {p}', file=sys.stderr)
        return 1

    print(f'Reading CSV   : {SRC_CSV}')
    long_df = pd.read_csv(SRC_CSV)
    print(f'  {len(long_df):,} rows × {len(long_df.columns)} cols')

    # Pivot: index=(Basin, Scenario), columns=Year, values=Cumulative_Delta_GW_AF
    wide = long_df.pivot_table(
        index=['Basin', 'Scenario'],
        columns='Year',
        values='Cumulative_Delta_GW_AF',
        aggfunc='first',
    )
    wide.columns = [f'Cum_{int(y)}' for y in wide.columns]
    wide = wide.reset_index()
    print(f'  pivoted to {len(wide):,} rows '
          f'× {len(wide.columns)} cols (1 row per Basin × Scenario)')

    # Attach basin centroid geometry as `.geo` GeoJSON-string column
    # (`earthengine upload table` auto-detects this column name as
    # the feature geometry in CSV inputs — same convention used by
    # gee/pivot_to_csv.py).
    print(f'Reading basins: {basin_shp}')
    basins = gpd.read_file(basin_shp)
    if 'BASIN_NAME' not in basins.columns:
        print(f'Error: BASIN_NAME column missing from {basin_shp}',
              file=sys.stderr)
        print(f'  available: {list(basins.columns)}', file=sys.stderr)
        return 1
    basins = basins[['BASIN_NAME', 'geometry']].copy()
    # Centroid in source CRS, then reproject to WGS-84 for the GeoJSON
    centroids = gpd.GeoDataFrame(
        {'Basin': basins['BASIN_NAME']},
        geometry=basins.geometry.centroid,
        crs=basins.crs,
    ).to_crs(epsg=4326)
    centroids['.geo'] = centroids.geometry.apply(
        lambda g: json.dumps(mapping(g)),
    )
    centroids = centroids.drop(columns=['geometry'])

    # Inner-join — drop any (Basin, Scenario) rows whose Basin isn't
    # in the polygon set (shouldn't happen, but safer than NaN geom).
    merged = wide.merge(centroids, on='Basin', how='inner')
    n_dropped = len(wide) - len(merged)
    if n_dropped:
        unmatched = sorted(set(wide['Basin']) - set(centroids['Basin']))
        print(f'  WARN: dropped {n_dropped} rows for unmatched basins:')
        for b in unmatched[:10]:
            print(f'    - {b}')
    print(f'  joined {len(merged):,} features × {len(merged.columns)} cols '
          f'(incl. .geo column)')

    # Write CSV (the upload script stages it to gs://azhydro and then
    # ingests via `earthengine upload table` — no zip needed).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    merged.to_csv(OUT_CSV, index=False)
    size_mb = OUT_CSV.stat().st_size / 1024 / 1024

    print()
    print(f'Wrote: {OUT_CSV}  ({size_mb:.2f} MB)')
    print()
    print('Next: upload to GEE with')
    print('  ./gee/upload_cap_cumulative.sh')
    return 0


if __name__ == '__main__':
    sys.exit(main())
