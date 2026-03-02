"""
Export MACA bias-corrected monthly ETo as GEE assets for 2026-2099.

Builds a daily MACA ensemble (mean across all GCMs × scenarios), computes daily
ETo via openet.refetgee.Daily.maca(), sums to monthly, and applies the gridMET
bias-correction ratios.

No custom asset dependency — uses existing gridMET ratios at
projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/

Asset: projects/azhydro/assets/maca_monthly_eto/ (74 years × 12 months = 888 images)

Usage:
    conda activate azhydro
    python export_maca_monthly_eto.py [--start-year 2026] [--end-year 2099] [--no-wait]
"""

import ee
from openet.refetgee import Daily
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    list_pending_task_descriptions, export_image, wait_for_tasks,
    get_export_parser, build_daily_maca_ensemble,
    ASSET_PREFIX, MACA_SCALE, MONTH_NAMES
)


ASSET_ID = f'{ASSET_PREFIX}/maca_monthly_eto'
DEFAULT_START = 2026
DEFAULT_END = 2099


def build_monthly_maca_eto(year):
    """
    Build bias-corrected monthly MACA ETo for a single year.
    Returns ee.ImageCollection with 12 images (band 'eto', mm/month).
    """
    # Step 1: build daily MACA ensemble
    daily_ic = build_daily_maca_ensemble(year)

    # Step 2: compute daily ETo via openet.refetgee
    lat_img = ee.Image.pixelLonLat().select('latitude')
    elev = ee.Image('NASA/NASADEM_HGT/001').select('elevation')

    def calc_daily_eto(daily_img):
        eto = Daily.maca(input_img=daily_img, lat=lat_img, elev=elev).eto
        return eto.rename('eto') \
            .set('system:time_start', daily_img.get('system:time_start'))

    daily_eto = daily_ic.map(calc_daily_eto)

    # Step 3: sum to monthly
    year_ee = ee.Number(year)
    months = ee.List.sequence(1, 12)
    monthly_eto = ee.ImageCollection(months.map(lambda m:
        daily_eto.filter(ee.Filter.calendarRange(m, m, 'month')).sum()
            .set('system:time_start', ee.Date.fromYMD(year_ee, m, 1).millis())
            .set('system:time_end',
                 ee.Date.fromYMD(year_ee, m, 1).advance(1, 'month').millis())
            .set('month', m)
    ))

    # Step 4: apply gridMET bias-correction ratios
    gridmet_ratio_base = \
        'projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/'
    ratio_ic = ee.ImageCollection([
        ee.Image(f'{gridmet_ratio_base}{name}').set('month', i + 1)
        for i, name in enumerate(MONTH_NAMES)
    ])

    joined = ee.ImageCollection(ee.Join.inner().apply(
        monthly_eto, ratio_ic,
        ee.Filter.equals(leftField='month', rightField='month')
    ))
    corrected = joined.map(lambda f:
        ee.Image(f.get('primary')).select('eto')
            .multiply(ee.Image(f.get('secondary')))
            .rename('eto')
            .copyProperties(ee.Image(f.get('primary')),
                            ['system:time_start', 'system:time_end', 'month'])
    )
    return corrected


def build_and_export(start_year, end_year):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    tasks = []
    for year in range(start_year, end_year + 1):
        monthly_eto = build_monthly_maca_eto(year)
        for m in range(1, 13):
            img_name = f'{year}_{m:02d}'
            img_asset = f'{ASSET_ID}/{img_name}'
            if img_asset in existing:
                continue
            img = monthly_eto.filter(
                ee.Filter.calendarRange(m, m, 'month')
            ).first()
            task = export_image(
                img, img_asset,
                f'maca_eto_{img_name}',
                az, MACA_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)
        print(f'  Submitted year {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    parser = get_export_parser('Export MACA monthly ETo (2026-2099)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END

    print(f'Exporting MACA monthly ETo for {start}-{end}...')
    tasks = build_and_export(start, end)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    print(f'Asset: {ASSET_ID}')
