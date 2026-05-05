#!/usr/bin/env python3
"""Convert the CAP service-area polygon GeoJSON into a single GEE-
ingest CSV (with `.geo` GeoJSON-string geometry column) for upload
via `earthengine upload table` (after staging in gs://azhydro).

Source (one of):
  Data/Inputs/GW_Data/CAP/CAP_Service_Area.geojson
  Data/Inputs_Zenodo/GW_Data/CAP/CAP_Service_Area.geojson

Output: gee/Data/CAP/CAP.csv
        3 features (MARICOPA / PIMA / PINAL polygons) with columns:
          OBJECTID, NAME, .geo
        The visualizer expects the asset at
          projects/azhydro/assets/az-wu/CAP — that's the asset_id
        the upload script passes to `earthengine upload table`.

Why bypass geeup tabup:
  Empirically, `geeup tabup` failed both ways — bare CSV → "No
  tables to upload"; zipped CSV → same message.  Its docs claim
  CSVs are accepted but in practice it only enumerates zipped
  shapefiles.  `earthengine upload table` natively accepts CSVs
  with a `.geo` GeoJSON-string column (auto-detected), avoiding
  shapefile's 10-character DBF column-name limit.  Files must be
  staged in GCS first; the companion upload script handles that
  via `gsutil cp gs://azhydro/...`.

Usage::

    python gee/cap_service_area_to_csv.py
    ./gee/upload_cap_service_area.sh
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC_CANDIDATES = [
    REPO_ROOT / 'Data/Inputs/GW_Data/CAP/CAP_Service_Area.geojson',
    REPO_ROOT / 'Data/Inputs_Zenodo/GW_Data/CAP/CAP_Service_Area.geojson',
]

OUT_DIR = REPO_ROOT / 'gee/Data/CAP'
OUT_CSV = OUT_DIR / 'CAP.csv'


def main() -> int:
    src = next((p for p in SRC_CANDIDATES if p.is_file()), None)
    if src is None:
        print('Error: CAP_Service_Area.geojson not found in any of:',
              file=sys.stderr)
        for p in SRC_CANDIDATES:
            print(f'  {p}', file=sys.stderr)
        return 1

    print(f'Reading: {src}')
    gdf = gpd.read_file(src)
    print(f'  {len(gdf)} features × {len(gdf.columns)} cols  CRS: {gdf.crs}')

    # Reproject to WGS-84 so the GeoJSON coordinates are lon/lat —
    # GEE's expected default for .geo column geometries.
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    df = pd.DataFrame(gdf.drop(columns=['geometry']))
    df['.geo'] = gdf.geometry.apply(lambda geom: json.dumps(mapping(geom)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    df.to_csv(OUT_CSV, index=False)
    size_kb = OUT_CSV.stat().st_size / 1024

    print(f'Wrote: {OUT_CSV}  ({size_kb:.1f} KB, {len(df)} rows)')
    print()
    print('Next: upload to GEE with')
    print('  ./gee/upload_cap_service_area.sh')
    return 0


if __name__ == '__main__':
    sys.exit(main())
