"""
Export bias-corrected USGS Reitz Ensemble ET as monthly GEE assets for 1896-1999.

For each year, applies the pre-exported OpenET/Reitz ratio to Reitz ET
to produce bias-corrected monthly ET grids.

Dependency: Run export_openet_reitz_ratio.py first.

Asset: projects/azhydro/assets/usgs_adjusted_et/ (104 years × 12 months = 1248 images)

Usage:
    conda activate azhydro
    python export_usgs_adjusted_et.py [--start-year 1896] [--end-year 1999] [--no-wait]
"""

import ee
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    list_pending_task_descriptions, export_image, wait_for_tasks,
    get_export_parser, ASSET_PREFIX, REITZ_SCALE
)


ASSET_ID = f'{ASSET_PREFIX}/usgs_adjusted_et'
RATIO_ASSET = f'{ASSET_PREFIX}/openet_reitz_et_ratio'
DEFAULT_START = 1896
DEFAULT_END = 1999


def build_and_export(start_year, end_year):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    # Load pre-exported ratio grids
    ratio_ic = ee.ImageCollection(RATIO_ASSET)
    reitz_ic = ee.ImageCollection('projects/nwi-usgs/assets/USGS-Reitz-Ensemble-ET')

    tasks = []
    for year in range(start_year, end_year + 1):
        # Filter to this year first, then convert mm/day → mm/month
        reitz_year = reitz_ic.filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
            .map(lambda img:
                img.rename('reitz_et')
                    .multiply(
                        ee.Date(img.get('system:time_start')).advance(1, 'month')
                        .difference(ee.Date(img.get('system:time_start')), 'day')
                    )
                    .set('month', ee.Date(img.get('system:time_start')).get('month'))
                    .copyProperties(img, ['system:time_start', 'system:time_end'])
            )

        # Join with ratio and bias-correct
        joined = ee.ImageCollection(ee.Join.inner().apply(
            reitz_year, ratio_ic,
            ee.Filter.equals(leftField='month', rightField='month')
        ))
        adjusted = joined.map(lambda f:
            ee.Image(f.get('primary')).select('reitz_et')
                .multiply(ee.Image(f.get('secondary')).select('ratio'))
                .rename('actual_et')
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
            monthly_img = adjusted.filter(
                ee.Filter.calendarRange(m, m, 'month')
            ).first()
            task = export_image(
                monthly_img, img_asset,
                f'usgs_et_{img_name}',
                az, REITZ_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)

        print(f'  Submitted year {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    parser = get_export_parser('Export USGS adjusted ET (1896-1999)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END

    print(f'Exporting USGS adjusted ET for {start}-{end}...')
    tasks = build_and_export(start, end)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    print(f'Asset: {ASSET_ID}')
