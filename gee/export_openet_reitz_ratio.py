"""
Export monthly OpenET/Reitz ET ratio grids (12 images) as a GEE asset.

Computes the ratio of OpenET ET to USGS Reitz Ensemble ET for each month,
averaged across 2000-2018. This is used to bias-correct Reitz ET for 1896-1999.

Asset: projects/azhydro/assets/openet_reitz_et_ratio/ (12 images: month_01 .. month_12)

Usage:
    conda activate azhydro
    python export_openet_reitz_ratio.py [--no-wait]
"""

import logging

import ee
from config import (
    ASSET_PREFIX,
    REITZ_SCALE,
    build_openet_monthly_et_ic,
    create_ic_asset,
    export_image,
    get_az_geometry,
    init_ee,
    list_existing_assets,
    list_pending_task_descriptions,
    wait_for_tasks,
)

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/openet_reitz_et_ratio'


def build_and_export():
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    # OpenET monthly ET 2000-2018 (inclusive; Reitz goes through 9/2018)
    openet_ic = build_openet_monthly_et_ic()
    openet_flt = openet_ic.filterDate('2000-01-01', '2019-01-01') \
        .select('et_ensemble_mad')

    # USGS Reitz Ensemble ET (mm/day → mm/month)
    reitz_ic = ee.ImageCollection('projects/nwi-usgs/assets/USGS-Reitz-Ensemble-ET')
    reitz_monthly = reitz_ic.filterDate('2000-01-01', '2019-01-01').map(lambda img:
        img.rename('reitz_et')
            .multiply(
                ee.Date(img.get('system:time_start')).advance(1, 'month')
                .difference(ee.Date(img.get('system:time_start')), 'day')
            )
            .set('month', ee.Date(img.get('system:time_start')).get('month'))
            .set('year', ee.Date(img.get('system:time_start')).get('year'))
            .copyProperties(img, ['system:time_start', 'system:time_end'])
    )

    # Join on month AND year so each ratio is from the same time step
    join_filter = ee.Filter.And(
        ee.Filter.equals(leftField='month', rightField='month'),
        ee.Filter.equals(leftField='year', rightField='year')
    )
    joined = ee.ImageCollection(ee.Join.inner().apply(
        openet_flt, reitz_monthly, join_filter
    )).map(lambda f:
        ee.Image(f.get('primary')).select('et_ensemble_mad')
            .divide(ee.Image(f.get('secondary')).select('reitz_et'))
            .rename('ratio')
            .set('system:time_start', ee.Image(f.get('primary')).get('system:time_start'))
            .set('month', ee.Image(f.get('primary')).get('month'))
    )

    # Mean ratio per month (12 grids)
    tasks = []
    for m in range(1, 13):
        img_name = f'month_{m:02d}'
        img_asset = f'{ASSET_ID}/{img_name}'
        if img_asset in existing:
            logger.info(f'  Skipping {img_name} (already exists)')
            continue
        monthly_mean = joined.filter(ee.Filter.calendarRange(m, m, 'month')).mean() \
            .rename('ratio') \
            .set('month', m) \
            .set('system:time_start', ee.Date.fromYMD(2000, m, 1).millis())
        task = export_image(
            monthly_mean, img_asset,
            f'openet_reitz_ratio_{img_name}',
            az, REITZ_SCALE,
            pending_descriptions=pending
        )
        tasks.append(task)
        logger.info(f'  Submitted {img_name}')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    import argparse
    parser = argparse.ArgumentParser(description='Export OpenET/Reitz ET ratio')
    parser.add_argument('--no-wait', action='store_true')
    args = parser.parse_args()

    logger.info('Exporting OpenET/Reitz ET ratio (12 monthly grids)...')
    tasks = build_and_export()
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    logger.info(f'Asset: {ASSET_ID}')
