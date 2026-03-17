"""
Export MACA monthly ET (v3, LULC-varying EToF) as GEE assets for 2026-2099.

Multiplies pre-exported MACA monthly ETo by LULC-varying EToF (composited
from per-class climatologies using the projected LULC map for each year).

This replaces the static EToF used in v2 with a spatially varying EToF that
reflects projected land-use change (agriculture, urban, surface water, other).

Dependencies: Run these first:
  1. export_lulc_stratified_etof.py  (produces 12 images × 4 bands)
  2. export_lulc_ensemble.py         (produces lulc_projection_ensemble)
  3. export_maca_monthly_eto.py      (produces maca_monthly_eto_v2)

Asset: projects/azhydro/assets/maca_monthly_et_v3/ (74 years × 12 months = 888 images)

Usage:
    conda activate azhydro
    python export_maca_monthly_et_v3.py [--start-year 2026] [--end-year 2099] [--no-wait]
"""

import logging

import ee
from config import (
    ASSET_PREFIX,
    MACA_SCALE,
    build_lulc_varying_etof,
    create_ic_asset,
    export_image,
    get_az_geometry,
    get_export_parser,
    init_ee,
    list_existing_assets,
    list_pending_task_descriptions,
    wait_for_tasks,
)

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/maca_monthly_et_v3'
MACA_ETO_ASSET = f'{ASSET_PREFIX}/maca_monthly_eto_v2'
DEFAULT_START = 2026
DEFAULT_END = 2099


def build_and_export(start_year, end_year):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    tasks = []
    for year in range(start_year, end_year + 1):
        maca_eto_year = ee.ImageCollection(MACA_ETO_ASSET) \
            .filterDate(f'{year}-01-01', f'{year + 1}-01-01')

        for m in range(1, 13):
            img_name = f'{year}_{m:02d}'
            img_asset = f'{ASSET_ID}/{img_name}'
            if img_asset in existing:
                continue

            eto = maca_eto_year.filter(
                ee.Filter.calendarRange(m, m, 'month')
            ).first().select('eto')

            etof = build_lulc_varying_etof(year, m)

            img = eto.multiply(etof.select('etof')) \
                .rename('actual_et') \
                .set('system:time_start',
                     ee.Date.fromYMD(year, m, 1).millis()) \
                .set('system:time_end',
                     ee.Date.fromYMD(year, m, 1).advance(1, 'month').millis()) \
                .set('month', m) \
                .set('year', year)

            task = export_image(
                img, img_asset,
                f'maca_et_v3_{img_name}',
                az, MACA_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)
        logger.info(f'  Submitted year {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = get_export_parser('Export MACA monthly ET v3 — LULC-varying EToF (2026-2099)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END

    logger.info(f'Exporting MACA monthly ET v3 for {start}-{end}...')
    tasks = build_and_export(start, end)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    logger.info(f'Asset: {ASSET_ID}')
