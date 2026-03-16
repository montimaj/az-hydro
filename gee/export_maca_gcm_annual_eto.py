"""
Export per-GCM annual ETo as GEE assets for 2026-2099.

For each of the 5 representative GCMs, averages across both RCP scenarios
(rcp45, rcp85) to produce a 2-member ensemble-mean ETo that captures
that GCM's climate signal.  Monthly ETo is computed internally using the same
flat-pipeline approach and gridMET bias-correction ratios as the full 40-member
ensemble (export_maca_monthly_eto.py), then summed to annual before export.

Annual export is sufficient because the downstream XGBoost model operates on
annual predictor variables.  Monthly resolution is only needed for intermediate
computation (gridMET bias correction is per-month) and is handled server-side.

These per-GCM assets enable climate-model uncertainty quantification (σ_MACA)
by running each GCM's ETo/ET/Peff chain through the downstream pipeline
and computing inter-GCM spread.  Annual ETo is also used at tile-download
time for the Peff clamp safeguard in dataops.py.

Dependency: gridMET bias-correction ratios (pre-existing OpenET asset)

Asset: projects/azhydro/assets/maca_gcm_annual_eto/
       ({model}_{year}, 5 GCMs × 74 years = 370 images)

Usage:
    conda activate azhydro
    python export_maca_gcm_annual_eto.py [--start-year 2026] [--end-year 2099] [--no-wait]
    python export_maca_gcm_annual_eto.py --gcm CCSM4          # single GCM
    python export_maca_gcm_annual_eto.py --gcm all             # all 5 (default)
"""

import logging

import ee
from config import (
    ASSET_PREFIX,
    MACA_BANDS,
    MACA_REPRESENTATIVE_GCMS,
    MACA_SCALE,
    MACA_SCENARIOS,
    MONTH_NAMES,
    create_ic_asset,
    export_image,
    get_az_geometry,
    get_export_parser,
    init_ee,
    list_existing_assets,
    list_pending_task_descriptions,
    wait_for_tasks,
)
from openet.refetgee import Daily

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/maca_gcm_annual_eto'
DEFAULT_START = 2026
DEFAULT_END = 2099
N_SCENARIOS = len(MACA_SCENARIOS)  # 2


def build_annual_maca_gcm_eto(year, model):
    """
    Build bias-corrected annual MACA ETo for a single GCM (both scenarios
    averaged) for one year.

    Computes monthly ETo using the same flat-pipeline approach as the 40-member
    ensemble version, but filtered to the specified model only (2 scenarios ×
    365 days ≈ 730 images/year).  Monthly values are summed to annual.

    Returns ee.Image with band 'eto' (mm/year).
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year + 1, 1, 1)

    # Load MACA daily images for this model only (both scenarios)
    maca_ic = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('model', model)) \
        .select(MACA_BANDS)

    lat_img = ee.Image.pixelLonLat().select('latitude')
    elev = ee.Image('NASA/NASADEM_HGT/001').select('elevation')

    # Compute daily ETo for every image
    def _daily_eto(img):
        eto = Daily.maca(input_img=img, lat=lat_img, elev=elev).eto
        return eto.rename('eto') \
            .set('system:time_start', img.get('system:time_start'))

    daily_eto = maca_ic.map(_daily_eto)

    # For each month: sum daily ETo, divide by n_scenarios (2)
    year_ee = ee.Number(year)
    months = ee.List.sequence(1, 12)

    monthly_eto = ee.ImageCollection(months.map(lambda m:
        daily_eto.filter(ee.Filter.calendarRange(m, m, 'month'))
            .sum().divide(N_SCENARIOS)
            .rename('eto')
            .set('month', m)
    ))

    # Bias correction with gridMET ratios (same ratios as ensemble version)
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
    )

    # Sum monthly → annual
    return corrected.sum().rename('eto') \
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
            img = build_annual_maca_gcm_eto(year, model)
            task = export_image(
                img, img_asset,
                f'gcm_eto_{model}_{year}',
                az, MACA_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)
            logger.info(f'  Submitted {model} {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = get_export_parser('Export per-GCM annual MACA ETo (2026-2099)')
    parser.add_argument('--gcm', type=str, default='all',
                        choices=MACA_REPRESENTATIVE_GCMS + ['all'],
                        help='Which GCM to export (default: all)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END
    gcms = MACA_REPRESENTATIVE_GCMS if args.gcm == 'all' else [args.gcm]

    logger.info(f'Exporting per-GCM annual MACA ETo for {start}-{end}, '
                f'GCMs: {gcms}...')
    tasks = build_and_export(start, end, gcms)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
