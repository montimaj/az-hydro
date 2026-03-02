"""
Export bias-corrected PRISM Hargreaves ETo as monthly GEE assets for 1896-1978.

For each year, computes Hargreaves ETo from PRISM monthly tmax/tmin, then
applies the pre-exported Hargreaves/gridMET ratio to bias-correct it.

Dependency: Run export_gridmet_hargreaves_ratio.py first.

Asset: projects/azhydro/assets/prism_hargreaves_eto/ (83 years × 12 months = 996 images)

Usage:
    conda activate azhydro
    python export_prism_hargreaves_eto.py [--start-year 1896] [--end-year 1978] [--no-wait]
"""

import ee
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    list_pending_task_descriptions, export_image, wait_for_tasks,
    get_export_parser, ASSET_PREFIX, PRISM_SCALE, calc_prism_monthly_eto
)


ASSET_ID = f'{ASSET_PREFIX}/prism_hargreaves_eto'
RATIO_ASSET = f'{ASSET_PREFIX}/gridmet_hargreaves_eto_ratio'
DEFAULT_START = 1896
DEFAULT_END = 1978


def build_and_export(start_year, end_year):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    # Load pre-exported ratio grids
    ratio_ic = ee.ImageCollection(RATIO_ASSET)

    tasks = []
    for year in range(start_year, end_year + 1):
        # PRISM monthly data for this year
        prism_monthly = ee.ImageCollection('OREGONSTATE/PRISM/ANm') \
            .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
        prism_eto = prism_monthly.map(calc_prism_monthly_eto)
        prism_eto = prism_eto.map(lambda img:
            img.set('month', ee.Date(img.get('system:time_start')).get('month'))
        )

        # Join with ratio and bias-correct
        joined = ee.ImageCollection(ee.Join.inner().apply(
            prism_eto, ratio_ic,
            ee.Filter.equals(leftField='month', rightField='month')
        ))
        corrected = joined.map(lambda f:
            ee.Image(f.get('primary')).select('prism_eto')
                .multiply(ee.Image(f.get('secondary')).select('ratio'))
                .rename('eto')
                .copyProperties(ee.Image(f.get('primary')),
                                ['system:time_start', 'system:time_end'])
                .set('month', ee.Image(f.get('primary')).get('month'))
                .set('year', year)
        )

        # Export each month
        for m in range(1, 13):
            img_name = f'{year}_{m:02d}'
            img_asset = f'{ASSET_ID}/{img_name}'
            if img_asset in existing:
                continue
            monthly_img = corrected.filter(
                ee.Filter.calendarRange(m, m, 'month')
            ).first()
            task = export_image(
                monthly_img, img_asset,
                f'prism_eto_{img_name}',
                az, PRISM_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)

        print(f'  Submitted year {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    parser = get_export_parser('Export PRISM Hargreaves ETo (1896-1978)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END

    print(f'Exporting PRISM Hargreaves ETo for {start}-{end}...')
    tasks = build_and_export(start, end)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    print(f'Asset: {ASSET_ID}')
