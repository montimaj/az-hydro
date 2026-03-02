"""
Export monthly EToF (crop coefficient = ET/ETo) grids (12 images) as a GEE asset.

Computes the ratio of OpenET ET to gridMET ETo for each month,
averaged across 2000-2025. This is used to derive MACA future ET
from MACA ETo for 2026-2099.

Asset: projects/azhydro/assets/monthly_etof/ (12 images: month_01 .. month_12)

Usage:
    conda activate azhydro
    python export_monthly_etof.py [--no-wait]
"""

import ee
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    export_image, wait_for_tasks, ASSET_PREFIX, GRIDMET_SCALE,
    build_openet_monthly_et_ic
)


ASSET_ID = f'{ASSET_PREFIX}/monthly_etof'


def build_and_export():
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)

    # OpenET monthly ET 2000-2025
    openet_ic = build_openet_monthly_et_ic()
    openet_flt = openet_ic.filterDate('2000-01-01', '2026-01-01') \
        .select('et_ensemble_mad')

    # gridMET monthly ETo 2000-2025
    gridmet_eto = ee.ImageCollection(
        'projects/openet/assets/reference_et/conus/gridmet/monthly/v1'
    ).filterDate('2000-01-01', '2026-01-01').select('eto').map(lambda img:
        img.set('month', ee.Date(img.get('system:time_start')).get('month'))
            .set('year', ee.Date(img.get('system:time_start')).get('year'))
    )

    # Join on month AND year so each EToF is from the same time step
    join_filter = ee.Filter.And(
        ee.Filter.equals(leftField='month', rightField='month'),
        ee.Filter.equals(leftField='year', rightField='year')
    )
    joined = ee.ImageCollection(ee.Join.inner().apply(
        openet_flt, gridmet_eto, join_filter
    )).map(lambda f:
        ee.Image(f.get('primary')).select('et_ensemble_mad')
            .divide(ee.Image(f.get('secondary')).select('eto'))
            .rename('etof')
            .set('system:time_start', ee.Image(f.get('primary')).get('system:time_start'))
            .set('month', ee.Image(f.get('primary')).get('month'))
    )

    # Mean EToF per month (12 grids)
    tasks = []
    for m in range(1, 13):
        img_name = f'month_{m:02d}'
        img_asset = f'{ASSET_ID}/{img_name}'
        if img_asset in existing:
            print(f'  Skipping {img_name} (already exists)')
            continue
        monthly_mean = joined.filter(ee.Filter.calendarRange(m, m, 'month')).mean() \
            .rename('etof') \
            .set('month', m) \
            .set('system:time_start', ee.Date.fromYMD(2000, m, 1).millis())
        task = export_image(
            monthly_mean, img_asset,
            f'monthly_etof_{img_name}',
            az, GRIDMET_SCALE
        )
        tasks.append(task)
        print(f'  Submitted {img_name}')

    return tasks


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export monthly EToF')
    parser.add_argument('--no-wait', action='store_true')
    args = parser.parse_args()

    print('Exporting monthly EToF (12 grids)...')
    tasks = build_and_export()
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    print(f'Asset: {ASSET_ID}')
