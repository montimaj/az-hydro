"""
Export monthly Hargreaves/gridMET ETo ratio grids (12 images) as a GEE asset.

Computes the ratio of gridMET ETo to PRISM Hargreaves ETo for each month,
averaged across 1979-2025. This is used to bias-correct Hargreaves ETo
for 1896-1978 when daily data is unavailable.

Asset: projects/azhydro/assets/hargreaves_gridmet_eto_ratio/ (12 images: month_01 .. month_12)

Usage:
    conda activate azhydro
    python export_hargreaves_gridmet_ratio.py [--no-wait]
"""

import ee
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    export_image, wait_for_tasks, ASSET_PREFIX, PRISM_SCALE,
    calc_prism_monthly_eto
)


ASSET_ID = f'{ASSET_PREFIX}/hargreaves_gridmet_eto_ratio'


def build_and_export():
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)

    # gridMET monthly ETo 1979-2025
    gridmet_eto = ee.ImageCollection(
        'projects/openet/assets/reference_et/conus/gridmet/monthly/v1'
    ).filterDate('1979-01-01', '2026-01-01').select('eto')

    # PRISM monthly data 1979-2025 → Hargreaves ETo
    prism_monthly = ee.ImageCollection('OREGONSTATE/PRISM/ANm') \
        .filterDate('1979-01-01', '2026-01-01')
    prism_eto = prism_monthly.map(calc_prism_monthly_eto)

    # Add 'month' and 'year' properties to both for paired join
    gridmet_eto = gridmet_eto.map(lambda img:
        img.set('month', ee.Date(img.get('system:time_start')).get('month'))
            .set('year', ee.Date(img.get('system:time_start')).get('year'))
    )
    prism_eto = prism_eto.map(lambda img:
        img.set('month', ee.Date(img.get('system:time_start')).get('month'))
            .set('year', ee.Date(img.get('system:time_start')).get('year'))
    )

    # Join on month AND year so each ratio is from the same time step
    join_filter = ee.Filter.And(
        ee.Filter.equals(leftField='month', rightField='month'),
        ee.Filter.equals(leftField='year', rightField='year')
    )
    joined = ee.ImageCollection(ee.Join.inner().apply(
        gridmet_eto, prism_eto, join_filter
    )).map(lambda f:
        ee.Image(f.get('primary')).select('eto')
            .divide(ee.Image(f.get('secondary')).select('prism_eto'))
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
            print(f'  Skipping {img_name} (already exists)')
            continue
        monthly_mean = joined.filter(ee.Filter.calendarRange(m, m, 'month')).mean() \
            .rename('ratio') \
            .set('month', m) \
            .set('system:time_start', ee.Date.fromYMD(2000, m, 1).millis())
        task = export_image(
            monthly_mean, img_asset,
            f'hargreaves_gridmet_ratio_{img_name}',
            az, PRISM_SCALE
        )
        tasks.append(task)
        print(f'  Submitted {img_name}')

    return tasks


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export Hargreaves/gridMET ETo ratio')
    parser.add_argument('--no-wait', action='store_true')
    args = parser.parse_args()

    print('Exporting Hargreaves/gridMET ETo ratio (12 monthly grids)...')
    tasks = build_and_export()
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    print(f'Asset: {ASSET_ID}')
