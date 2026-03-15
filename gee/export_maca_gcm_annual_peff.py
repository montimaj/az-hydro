"""
Export per-GCM annual USDA SCS effective precipitation as GEE assets for 2026-2099.

For each of the 5 representative GCMs, computes monthly Peff internally using:
  - ETo: bias-corrected monthly per-GCM MACA ETo (same flat pipeline as ensemble)
  - Precip: per-GCM MACA daily precipitation (both scenarios averaged, flat pipeline)

Monthly Peff is computed using the USDA SCS formula (nonlinear, requires monthly
resolution) and then summed to annual before export.  The formula, soil data,
and parameters (mad_factor=1, rz_depth_m=2m) are identical to the full ensemble
version (export_monthly_peff.py).

Only 2026-2099 is exported — historical Peff (1896-2025) uses PRISM/gridMET
observations equally for all GCMs.

No asset dependencies beyond the pre-existing gridMET bias-correction ratios
(monthly ETo is computed internally).

Asset: projects/azhydro/assets/maca_gcm_annual_peff/
       ({model}_{year}, 5 GCMs × 74 years = 370 images)

Usage:
    conda activate azhydro
    python export_maca_gcm_annual_peff.py [--start-year 2026] [--end-year 2099] [--no-wait]
    python export_maca_gcm_annual_peff.py --gcm CCSM4          # single GCM
    python export_maca_gcm_annual_peff.py --gcm all             # all 5 (default)
"""

import ee
from openet.refetgee import Daily
from config import (
    init_ee, get_az_geometry, create_ic_asset, list_existing_assets,
    list_pending_task_descriptions, export_image, wait_for_tasks,
    get_export_parser,
    ASSET_PREFIX, PRISM_SCALE, MACA_BANDS, MACA_SCENARIOS,
    MACA_REPRESENTATIVE_GCMS, MONTH_NAMES
)
import logging

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/maca_gcm_annual_peff'
DEFAULT_START = 2026
DEFAULT_END = 2099
N_SCENARIOS = len(MACA_SCENARIOS)  # 2

# USDA SCS parameters (consistent with export_monthly_peff.py and dataops.py)
MAD_FACTOR = 1
RZ_DEPTH_M = [2] * 12
RZ_INCHES = [rz * 39.37 for rz in RZ_DEPTH_M]


def build_annual_gcm_peff(year, model):
    """
    Compute annual USDA SCS effective precipitation for a single year and GCM.

    Computes bias-corrected monthly ETo and monthly precip internally, applies
    the nonlinear USDA SCS formula per month, and sums to annual.

    Returns ee.Image with band 'peff' (mm/year).
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year + 1, 1, 1)

    # --- Compute bias-corrected monthly per-GCM ETo ---
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

    # Bias correction with gridMET ratios
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

    # --- Compute per-GCM monthly precipitation ---
    maca_pr = maca_ic.select(['pr'])
    year_ee = ee.Number(year)
    monthly_precip = ee.ImageCollection(months.map(lambda m:
        maca_pr.filter(ee.Filter.calendarRange(m, m, 'month'))
            .sum().divide(N_SCENARIOS)
            .rename('pr')
            .set('month', m)
            .set('system:time_start', ee.Date.fromYMD(year_ee, m, 1).millis())
    ))

    # --- Join ETo and precip, apply USDA SCS formula ---
    awc = ee.Image('projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite')
    rz_inches = ee.List(RZ_INCHES)
    ep_scale = ee.ImageCollection('OREGONSTATE/PRISM/ANm').first() \
        .projection().nominalScale()

    joined = ee.ImageCollection(
        ee.Join.inner().apply(
            corrected_eto, monthly_precip,
            ee.Filter.equals(leftField='month', rightField='month')
        )
    ).map(lambda feature:
        ee.Image.cat(
            ee.Image(feature.get('primary')),
            ee.Image(feature.get('secondary'))
        ).copyProperties(
            ee.Image(feature.get('primary')), ['month']
        ).set('system:time_start',
              ee.Image(feature.get('secondary')).get('system:time_start'))
    )

    def calculate_ep(img):
        # Convert mm to inches
        pr = img.select('pr').divide(25.4).rename('pr')
        eto = img.select('eto').divide(25.4).rename('eto')

        # Month index (1-based) for root zone depth lookup
        month = ee.Number(img.get('month'))

        # d = MAD * AWC * root zone depth (eq. 2-85)
        d = awc.multiply(MAD_FACTOR) \
            .multiply(ee.Number(rz_inches.get(month.subtract(1))))

        # Soil storage factor (eq. 2-85)
        sf = d.multiply(0.295164) \
            .add(0.531747) \
            .subtract(d.pow(2).multiply(0.057697)) \
            .add(d.pow(3).multiply(0.003804)) \
            .rename('sf')

        # SCS effective precipitation (eq. 2-84)
        ep = sf.multiply(
            pr.pow(0.82416).multiply(0.70917).subtract(0.11556)
        ).multiply(
            ee.Image.constant(10).pow(eto.multiply(0.02426))
        ).rename('peff')

        # Clamp: ep <= pr, ep <= eto, ep >= 0; convert to mm
        ep_cleaned = ep.where(ep.gte(pr), pr) \
            .where(ep.gt(eto), eto) \
            .clamp(0, 10000) \
            .multiply(25.4)

        # Final safeguard: clamp against original mm precip
        pr_mm = img.select('pr')
        ep_cleaned = ep_cleaned.where(ep_cleaned.gt(pr_mm), pr_mm) \
            .clamp(0, 10000)

        return ee.Image(ep_cleaned) \
            .setDefaultProjection(crs='EPSG:4326', scale=ep_scale)

    monthly_peff = joined.map(calculate_ep)

    # Sum monthly → annual
    return monthly_peff.sum().rename('peff') \
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
            img = build_annual_gcm_peff(year, model)
            task = export_image(
                img, img_asset,
                f'gcm_peff_{model}_{year}',
                az, PRISM_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)
            logger.info(f'  Submitted {model} {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = get_export_parser(
        'Export per-GCM annual USDA SCS effective precipitation (2026-2099)')
    parser.add_argument('--gcm', type=str, default='all',
                        choices=MACA_REPRESENTATIVE_GCMS + ['all'],
                        help='Which GCM to export (default: all)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END
    gcms = MACA_REPRESENTATIVE_GCMS if args.gcm == 'all' else [args.gcm]

    logger.info(f'Exporting per-GCM annual MACA peff for {start}-{end}, '
                f'GCMs: {gcms}...')
    tasks = build_and_export(start, end, gcms)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
