"""
Export annual LULC projection ensemble as GEE assets for 2026-2099.

Takes the pixel-wise mode of four USGS LULC projection scenarios
(B1, B2, A1B, A2) for each year.

No custom asset dependency.

Asset: projects/azhydro/assets/lulc_projection_ensemble/ (74 annual images)

Usage:
    conda activate azhydro
    python export_lulc_ensemble.py [--start-year 2026] [--end-year 2099] [--no-wait]
"""

import ee
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    list_pending_task_descriptions, export_image, wait_for_tasks,
    get_export_parser, ASSET_PREFIX, USGS_LULC_SCALE, USGS_LULC_SCENARIOS
)
import logging

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/lulc_projection_ensemble'
DEFAULT_START = 2026
DEFAULT_END = 2099


def build_annual_lulc(year):
    """
    Build one annual LULC ensemble image for a year by taking the mode
    across B1, B2, A1B, A2 scenarios.
    Returns ee.Image with band 'landcover'.
    """
    lulc_ic = ee.ImageCollection('projects/nwi-usgs/assets/USGS-LULC-CONUS')
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year + 1, 1, 1)

    scenario_imgs = ee.ImageCollection(
        ee.List(USGS_LULC_SCENARIOS).map(lambda scenario:
            lulc_ic.filterDate(start_date, end_date)
                .filterMetadata('scenario', 'equals', scenario)
                .first()
                .rename('landcover')
        )
    )
    return scenario_imgs.reduce(ee.Reducer.mode()) \
        .rename('landcover') \
        .set('system:time_start', start_date.millis()) \
        .set('system:time_end', end_date.millis()) \
        .set('year', year)


def build_and_export(start_year, end_year):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    tasks = []
    for year in range(start_year, end_year + 1):
        img_name = str(year)
        img_asset = f'{ASSET_ID}/{img_name}'
        if img_asset in existing:
            continue
        img = build_annual_lulc(year)
        task = export_image(
            img, img_asset,
            f'lulc_{year}',
            az, USGS_LULC_SCALE,
            as_int=True,
            pending_descriptions=pending
        )
        tasks.append(task)
        if len(tasks) % 10 == 0:
            logger.info(f'  Submitted {len(tasks)} tasks...')

    logger.info(f'  Total: {len(tasks)} tasks submitted')
    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = get_export_parser('Export LULC projection ensemble (2026-2099)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END

    logger.info(f'Exporting LULC projection ensemble for {start}-{end}...')
    tasks = build_and_export(start, end)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    logger.info(f'Asset: {ASSET_ID}')
