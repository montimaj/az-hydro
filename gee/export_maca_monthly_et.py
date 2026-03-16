"""
Export MACA monthly ET as GEE assets for 2026-2099.

Multiplies pre-exported MACA monthly ETo by pre-exported monthly EToF
to produce bias-corrected monthly ET grids.

Dependencies: Run these first:
  1. export_monthly_etof.py
  2. export_maca_monthly_eto.py (produces maca_monthly_eto_v2)

Asset: projects/azhydro/assets/maca_monthly_et_v2/ (74 years × 12 months = 888 images)

Usage:
    conda activate azhydro
    python export_maca_monthly_et.py [--start-year 2026] [--end-year 2099] [--no-wait]
"""

import logging

import ee
from config import (
    ASSET_PREFIX,
    MACA_SCALE,
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


ASSET_ID = f'{ASSET_PREFIX}/maca_monthly_et_v2'
ETOF_ASSET = f'{ASSET_PREFIX}/monthly_etof'
MACA_ETO_ASSET = f'{ASSET_PREFIX}/maca_monthly_eto_v2'
DEFAULT_START = 2026
DEFAULT_END = 2099


def build_and_export(start_year, end_year):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    # Load pre-exported monthly EToF (12 images, one per month)
    etof_ic = ee.ImageCollection(ETOF_ASSET)

    tasks = []
    for year in range(start_year, end_year + 1):
        # Load pre-exported MACA monthly ETo for this year
        maca_eto_year = ee.ImageCollection(MACA_ETO_ASSET) \
            .filterDate(f'{year}-01-01', f'{year + 1}-01-01')

        # Add month property to each MACA ETo image for the join
        maca_eto_year = maca_eto_year.map(lambda img:
            img.set('month', ee.Date(img.get('system:time_start')).get('month'))
        )

        # Join MACA ETo with EToF by month, multiply
        joined = ee.ImageCollection(ee.Join.inner().apply(
            maca_eto_year, etof_ic,
            ee.Filter.equals(leftField='month', rightField='month')
        ))
        monthly_et = joined.map(lambda f:
            ee.Image(f.get('primary')).select('eto')
                .multiply(ee.Image(f.get('secondary')).select('etof'))
                .rename('actual_et')
                .copyProperties(ee.Image(f.get('primary')),
                                ['system:time_start', 'system:time_end', 'month'])
        )

        for m in range(1, 13):
            img_name = f'{year}_{m:02d}'
            img_asset = f'{ASSET_ID}/{img_name}'
            if img_asset in existing:
                continue
            img = monthly_et.filter(
                ee.Filter.calendarRange(m, m, 'month')
            ).first()
            task = export_image(
                img, img_asset,
                f'maca_et_v2_{img_name}',
                az, MACA_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)
        logger.info(f'  Submitted year {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = get_export_parser('Export MACA monthly ET (2026-2099)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END

    logger.info(f'Exporting MACA monthly ET for {start}-{end}...')
    tasks = build_and_export(start, end)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    logger.info(f'Asset: {ASSET_ID}')
