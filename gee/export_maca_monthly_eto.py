"""
Export MACA bias-corrected monthly ETo as GEE assets for 2026-2099.

Uses a flat-pipeline approach to avoid GEE out-of-memory errors:
  1. Load ALL MACA daily images for the year (all 20 GCMs × 2 scenarios).
  2. Map daily ETo (openet.refetgee.Daily.maca) over the entire collection.
  3. For each month, sum all daily ETo and divide by n_members (40)
     to get the ensemble-mean monthly ETo.
  4. Apply gridMET bias-correction ratios.

This preserves non-linear relationships in the Penman-Monteith equation
(ETo is computed per-image, not from averaged inputs) while keeping the
computation graph simple enough for GEE to evaluate without OOM.

No custom asset dependency — uses existing gridMET ratios at
projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/

Asset: projects/azhydro/assets/maca_monthly_eto_v2/ (74 years × 12 months = 888 images)

Usage:
    conda activate azhydro
    python export_maca_monthly_eto.py [--start-year 2026] [--end-year 2099] [--no-wait]
"""

import ee
from openet.refetgee import Daily
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    list_pending_task_descriptions, export_image, wait_for_tasks,
    get_export_parser,
    ASSET_PREFIX, MACA_SCALE, MACA_BANDS, MACA_MODELS, MACA_SCENARIOS,
    MONTH_NAMES
)
import logging

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/maca_monthly_eto_v2'
DEFAULT_START = 2026
DEFAULT_END = 2099
N_MEMBERS = len(MACA_MODELS) * len(MACA_SCENARIOS)  # 40


def build_monthly_maca_eto(year):
    """
    Build ensemble-mean bias-corrected monthly MACA ETo for a single year.

    Uses a flat pipeline to keep the computation graph small:
      - Maps daily ETo over ALL MACA images (all models/scenarios) at once.
      - Sums per month and divides by n_members (40) for ensemble mean.
      - Applies gridMET bias-correction ratios.

    Returns ee.ImageCollection with 12 images (band 'eto', mm/month).
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year + 1, 1, 1)

    # Load ALL MACA images for the year (20 models × 2 scenarios × 365 days)
    maca_ic = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
        .filterDate(start_date, end_date).select(MACA_BANDS)

    lat_img = ee.Image.pixelLonLat().select('latitude')
    elev = ee.Image('NASA/NASADEM_HGT/001').select('elevation')

    # Compute daily ETo for every individual image (one map over entire collection)
    def _daily_eto(img):
        eto = Daily.maca(input_img=img, lat=lat_img, elev=elev).eto
        return eto.rename('eto') \
            .set('system:time_start', img.get('system:time_start'))

    daily_eto = maca_ic.map(_daily_eto)

    # For each month: sum all daily ETo, divide by n_members → ensemble mean
    year_ee = ee.Number(year)
    months = ee.List.sequence(1, 12)

    monthly_eto = ee.ImageCollection(months.map(lambda m:
        daily_eto.filter(ee.Filter.calendarRange(m, m, 'month'))
            .sum().divide(N_MEMBERS)
            .rename('eto')
            .set('system:time_start', ee.Date.fromYMD(year_ee, m, 1).millis())
            .set('system:time_end',
                 ee.Date.fromYMD(year_ee, m, 1).advance(1, 'month').millis())
            .set('month', m)
    ))

    # Bias correction with gridMET ratios
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
                f'maca_eto_v2_{img_name}',
                az, MACA_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)
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
