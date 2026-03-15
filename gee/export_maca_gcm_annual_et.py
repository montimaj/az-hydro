"""
Export per-GCM annual ET as GEE assets for 2026-2099.

For each of the 5 representative GCMs, computes bias-corrected monthly ETo
internally (same flat pipeline as the ensemble version), multiplies by monthly
EToF, and sums to annual ET before export.

Annual export is sufficient because the downstream XGBoost model operates on
annual predictor variables.

Dependencies: Run these first:
  1. export_monthly_etof.py              (produces monthly_etof)
  2. export_maca_gcm_annual_eto.py       (produces maca_gcm_annual_eto)

Note: per-GCM monthly ETo is recomputed internally rather than loaded from the
per-GCM annual ETo asset, because the monthly intermediate is needed for the
EToF multiplication.  The per-GCM annual ETo asset exists separately for the
Peff clamp safeguard in dataops.py.

Asset: projects/azhydro/assets/maca_gcm_annual_et/
       ({model}_{year}, 5 GCMs × 74 years = 370 images)

Usage:
    conda activate azhydro
    python export_maca_gcm_annual_et.py [--start-year 2026] [--end-year 2099] [--no-wait]
    python export_maca_gcm_annual_et.py --gcm CCSM4          # single GCM
    python export_maca_gcm_annual_et.py --gcm all             # all 5 (default)
"""

import ee
from openet.refetgee import Daily
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    list_pending_task_descriptions, export_image, wait_for_tasks,
    get_export_parser,
    ASSET_PREFIX, MACA_SCALE, MACA_BANDS, MACA_SCENARIOS,
    MACA_REPRESENTATIVE_GCMS, MONTH_NAMES
)
import logging

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/maca_gcm_annual_et'
ETOF_ASSET = f'{ASSET_PREFIX}/monthly_etof'
DEFAULT_START = 2026
DEFAULT_END = 2099
N_SCENARIOS = len(MACA_SCENARIOS)  # 2


def build_annual_maca_gcm_et(year, model):
    """
    Build annual ET for a single GCM (both scenarios averaged) for one year.

    Internally computes bias-corrected monthly ETo (same flat pipeline as
    the ensemble version), multiplies by monthly EToF, and sums to annual.

    Returns ee.Image with band 'actual_et' (mm/year).
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year + 1, 1, 1)

    # --- Compute bias-corrected monthly per-GCM ETo (same as gcm_annual_eto) ---
    maca_ic = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('model', model)) \
        .select(MACA_BANDS)

    lat_img = ee.Image.pixelLonLat().select('latitude')
    elev = ee.Image('NASA/NASADEM_HGT/001').select('elevation')

    def _daily_eto(img):
        eto = Daily.maca(input_img=img, lat=lat_img, elev=elev).eto
        return eto.rename('eto') \
            .set('system:time_start', img.get('system:time_start'))

    daily_eto = maca_ic.map(_daily_eto)

    months = ee.List.sequence(1, 12)
    monthly_eto = ee.ImageCollection(months.map(lambda m:
        daily_eto.filter(ee.Filter.calendarRange(m, m, 'month'))
            .sum().divide(N_SCENARIOS)
            .rename('eto')
            .set('month', m)
    ))

    # Apply gridMET bias-correction ratios
    gridmet_ratio_base = \
        'projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/'
    ratio_ic = ee.ImageCollection([
        ee.Image(f'{gridmet_ratio_base}{name}').set('month', i + 1)
        for i, name in enumerate(MONTH_NAMES)
    ])

    joined_eto = ee.ImageCollection(ee.Join.inner().apply(
        monthly_eto, ratio_ic,
        ee.Filter.equals(leftField='month', rightField='month')
    ))
    corrected_eto = joined_eto.map(lambda f:
        ee.Image(f.get('primary')).select('eto')
            .multiply(ee.Image(f.get('secondary')))
            .rename('eto')
            .set('month', ee.Image(f.get('primary')).get('month'))
    )

    # --- Multiply monthly ETo × EToF → monthly ET, then sum to annual ---
    etof_ic = ee.ImageCollection(ETOF_ASSET)

    joined_et = ee.ImageCollection(ee.Join.inner().apply(
        corrected_eto, etof_ic,
        ee.Filter.equals(leftField='month', rightField='month')
    ))
    monthly_et = joined_et.map(lambda f:
        ee.Image(f.get('primary')).select('eto')
            .multiply(ee.Image(f.get('secondary')).select('etof'))
            .rename('actual_et')
    )

    return monthly_et.sum().rename('actual_et') \
        .set('system:time_start', start_date.millis()) \
        .set('system:time_end', end_date.millis()) \
        .set('model', model) \
        .set('year', year)


def build_and_export(start_year, end_year, gcms):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    tasks = []
    for model in gcms:
        for year in range(start_year, end_year + 1):
            img_name = f'{model}_{year}'
            img_asset = f'{ASSET_ID}/{img_name}'
            if img_asset in existing:
                continue
            img = build_annual_maca_gcm_et(year, model)
            task = export_image(
                img, img_asset,
                f'gcm_et_{model}_{year}',
                az, MACA_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)
            logger.info(f'  Submitted {model} {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = get_export_parser('Export per-GCM annual MACA ET (2026-2099)')
    parser.add_argument('--gcm', type=str, default='all',
                        choices=MACA_REPRESENTATIVE_GCMS + ['all'],
                        help='Which GCM to export (default: all)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END
    gcms = MACA_REPRESENTATIVE_GCMS if args.gcm == 'all' else [args.gcm]

    logger.info(f'Exporting per-GCM annual MACA ET for {start}-{end}, '
                f'GCMs: {gcms}...')
    tasks = build_and_export(start, end, gcms)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
