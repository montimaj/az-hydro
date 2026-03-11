"""
Export MACA bias-corrected monthly ETo as GEE assets for 2026-2099.

Computes monthly ETo independently for each GCM × scenario combination:
  1. For each model/scenario, build daily MACA data.
  2. Compute daily ETo via openet.refetgee.Daily.maca().
  3. Sum to monthly.
  4. Apply gridMET bias-correction ratios.
Then takes the ensemble mean across all model/scenario monthly ETo images.

This preserves non-linear relationships in the Penman-Monteith equation,
avoiding the low bias that results from averaging climate inputs first.

No custom asset dependency — uses existing gridMET ratios at
projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/

Asset: projects/azhydro/assets/maca_monthly_eto_v2/ (74 years × 12 months = 888 images)

Usage:
    conda activate azhydro
    python export_maca_monthly_eto.py [--start-year 2026] [--end-year 2099] [--no-wait]
"""

import ee
from concurrent.futures import ThreadPoolExecutor, as_completed
from openet.refetgee import Daily
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    list_pending_task_descriptions, export_image, wait_for_tasks,
    get_export_parser, build_daily_maca_single,
    ASSET_PREFIX, MACA_SCALE, MACA_MODELS, MACA_SCENARIOS, MONTH_NAMES
)
import logging

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/maca_monthly_eto_v2'
DEFAULT_START = 2026
DEFAULT_END = 2099


def _build_monthly_eto_single(year, model, scenario):
    """
    Build bias-corrected monthly ETo for a single GCM and scenario.
    Returns ee.ImageCollection with 12 images (band 'eto', mm/month).
    """
    daily_ic = build_daily_maca_single(year, model, scenario)

    lat_img = ee.Image.pixelLonLat().select('latitude')
    elev = ee.Image('NASA/NASADEM_HGT/001').select('elevation')

    def calc_daily_eto(daily_img):
        eto = Daily.maca(input_img=daily_img, lat=lat_img, elev=elev).eto
        return eto.rename('eto') \
            .set('system:time_start', daily_img.get('system:time_start'))

    daily_eto = daily_ic.map(calc_daily_eto)

    year_ee = ee.Number(year)
    months = ee.List.sequence(1, 12)
    monthly_eto = ee.ImageCollection(months.map(lambda m:
        daily_eto.filter(ee.Filter.calendarRange(m, m, 'month')).sum()
            .set('system:time_start', ee.Date.fromYMD(year_ee, m, 1).millis())
            .set('system:time_end',
                 ee.Date.fromYMD(year_ee, m, 1).advance(1, 'month').millis())
            .set('month', m)
    ))

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


def build_monthly_maca_eto(year):
    """
    Build ensemble-mean bias-corrected monthly MACA ETo for a single year.

    Computes monthly ETo for each GCM × scenario independently, then
    averages across all members to produce the ensemble mean.
    Returns ee.ImageCollection with 12 images (band 'eto', mm/month).
    """
    # Collect monthly ETo from every model × scenario
    all_monthly = []
    for model in MACA_MODELS:
        for scenario in MACA_SCENARIOS:
            all_monthly.append(_build_monthly_eto_single(year, model, scenario))

    # Flatten all model/scenario monthly images into a single collection
    all_images = ee.List([])
    for ic in all_monthly:
        all_images = all_images.cat(ic.toList(12))
    merged = ee.ImageCollection(all_images)

    # For each month, average across all model/scenario members
    year_ee = ee.Number(year)
    months = ee.List.sequence(1, 12)
    n_members = len(all_monthly)

    def _ensemble_month(m):
        return merged.filter(ee.Filter.eq('month', m)).mean().rename('eto') \
            .set('system:time_start', ee.Date.fromYMD(year_ee, m, 1).millis()) \
            .set('system:time_end',
                 ee.Date.fromYMD(year_ee, m, 1).advance(1, 'month').millis()) \
            .set('month', m) \
            .set('n_members', n_members)

    return ee.ImageCollection(months.map(_ensemble_month))


def build_and_export(start_year, end_year, max_workers=4):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    def _submit_year(year):
        """Build graph and submit export tasks for one year."""
        year_tasks = []
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
                f'maca_eto_v2_{img_name}',
                az, MACA_SCALE,
                pending_descriptions=pending
            )
            year_tasks.append(task)
        return year, year_tasks

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_submit_year, year): year
            for year in range(start_year, end_year + 1)
        }
        for future in as_completed(futures):
            year, year_tasks = future.result()
            tasks.extend(year_tasks)
            logger.info(f'  Submitted year {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = get_export_parser('Export MACA monthly ETo (2026-2099)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END

    logger.info(f'Exporting MACA monthly ETo for {start}-{end}...')
    tasks = build_and_export(start, end)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    logger.info(f'Asset: {ASSET_ID}')
