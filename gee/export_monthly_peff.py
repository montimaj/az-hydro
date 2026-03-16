"""
Export monthly USDA SCS effective precipitation as GEE assets for 1896-2099.

Computes monthly effective precipitation using the USDA-SCS method (eq. 2-84/2-85)
with the following inputs per era:
  - 1896-1978: PRISM Hargreaves ETo (pre-exported asset) + PRISM precipitation
  - 1979-2025: gridMET ETo + PRISM precipitation
  - 2026-2099: MACA ETo (pre-exported asset, per-model/scenario ensemble) + MACA precipitation (flat pipeline, sum/40)

Parameters: mad_factor=1, rz_depth_m=2m for all months (consistent with UCRB comparisons).

Dependencies (Level 2 assets):
  - prism_hargreaves_eto (for 1896-1978)
  - maca_monthly_eto_v2 (for 2026-2099)

Asset: projects/azhydro/assets/monthly_peff_v2/ (204 years × 12 months = 2448 images)

Usage:
    conda activate azhydro
    python export_monthly_peff.py [--start-year 1896] [--end-year 2099] [--no-wait]
"""

import logging

import ee
from config import (
    ASSET_PREFIX,
    MACA_MODELS,
    MACA_SCENARIOS,
    PRISM_SCALE,
    create_ic_asset,
    export_image,
    get_az_geometry,
    get_export_parser,
    init_ee,
    list_existing_assets,
    list_pending_task_descriptions,
    wait_for_tasks,
)

logger = logging.getLogger(__name__)


ASSET_ID = f'{ASSET_PREFIX}/monthly_peff_v2'
PRISM_ETO_ASSET = f'{ASSET_PREFIX}/prism_hargreaves_eto'
MACA_ETO_ASSET = f'{ASSET_PREFIX}/maca_monthly_eto_v2'
DEFAULT_START = 1896
DEFAULT_END = 2099
N_MEMBERS = len(MACA_MODELS) * len(MACA_SCENARIOS)  # 40

# USDA SCS parameters (consistent with dataops.py call site)
MAD_FACTOR = 1
RZ_DEPTH_M = [2] * 12  # 2 m root zone depth for all months
RZ_INCHES = [rz * 39.37 for rz in RZ_DEPTH_M]  # meters to inches


def get_monthly_eto(year):
    """
    Get monthly ETo ImageCollection for a given year from the appropriate source.
    Returns ee.ImageCollection with band 'eto' (mm) and property 'month'.
    """
    start_date = f'{year}-01-01'
    end_date = f'{year + 1}-01-01'

    if year <= 1978:
        # Pre-exported bias-corrected PRISM Hargreaves ETo
        monthly_eto = ee.ImageCollection(PRISM_ETO_ASSET) \
            .filterDate(start_date, end_date)
    elif year <= 2025:
        # gridMET native monthly ETo
        monthly_eto = ee.ImageCollection(
            'projects/openet/assets/reference_et/conus/gridmet/monthly/v1'
        ).filterDate(start_date, end_date).select('eto')
    else:
        # Pre-exported bias-corrected MACA monthly ETo
        monthly_eto = ee.ImageCollection(MACA_ETO_ASSET) \
            .filterDate(start_date, end_date)

    # Ensure consistent band name and month property
    monthly_eto = monthly_eto.map(lambda img:
        img.rename('eto')
            .set('month', ee.Date(img.get('system:time_start')).get('month'))
            .copyProperties(img, ['system:time_start', 'system:time_end'])
    )
    return monthly_eto


def get_monthly_precip(year):
    """
    Get monthly precipitation ImageCollection for a given year.
    Returns ee.ImageCollection with band 'pr' (mm) and property 'month'.
    """
    start_date = f'{year}-01-01'
    end_date = f'{year + 1}-01-01'

    if year > 2025:
        # Flat pipeline: sum all MACA daily precip per month, divide by n_members
        maca_ic = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
            .filterDate(start_date, end_date).select(['pr'])
        months = ee.List.sequence(1, 12)
        year_ee = ee.Number(year)
        pr_monthly = ee.ImageCollection(months.map(lambda m:
            maca_ic.filter(ee.Filter.calendarRange(m, m, 'month'))
                .sum().divide(N_MEMBERS)
                .rename('pr')
                .set('month', m)
                .set('system:time_start', ee.Date.fromYMD(year_ee, m, 1).millis())
                .set('system:time_end',
                     ee.Date.fromYMD(year_ee, m, 1).advance(1, 'month').millis())
        ))
    else:
        # PRISM monthly precipitation for 1896-2025
        pr_monthly = ee.ImageCollection('OREGONSTATE/PRISM/ANm') \
            .filterDate(start_date, end_date) \
            .select(['ppt'], ['pr']) \
            .map(lambda img:
                img.set('month', ee.Date(img.get('system:time_start')).get('month'))
            )
    return pr_monthly


def build_monthly_peff(year):
    """
    Compute 12 monthly USDA SCS effective precipitation images for a single year.
    Returns ee.ImageCollection with band 'peff' (inches, converted to mm at export).
    """
    monthly_eto = get_monthly_eto(year)
    monthly_precip = get_monthly_precip(year)

    # AWC from OpenET soil dataset (in inches)
    awc = ee.Image('projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite')
    rz_inches = ee.List(RZ_INCHES)

    # Scale for setDefaultProjection
    ep_scale = ee.ImageCollection('OREGONSTATE/PRISM/ANm').first().projection().nominalScale()

    # Join ETo and precipitation on month
    joined = ee.ImageCollection(
        ee.Join.inner().apply(
            monthly_eto, monthly_precip,
            ee.Filter.equals(leftField='month', rightField='month')
        )
    ).map(lambda feature:
        ee.Image.cat(
            ee.Image(feature.get('primary')),
            ee.Image(feature.get('secondary'))
        ).copyProperties(
            ee.Image(feature.get('primary')),
            ['system:time_start', 'system:time_end', 'month']
        )
    )

    def calculate_ep(img):
        # Convert mm to inches
        pr = img.select('pr').divide(25.4).rename('pr')
        eto = img.select('eto').divide(25.4).rename('eto')

        # Month index (1-based) for root zone depth lookup
        month = ee.Number.parse(ee.Date(img.get('system:time_start')).format('MM'))

        # d = MAD * AWC * root zone depth (eq. 2-85)
        d = awc.multiply(MAD_FACTOR).multiply(ee.Number(rz_inches.get(month.subtract(1))))

        # Soil storage factor (eq. 2-85)
        # sf = 0.531747 + 0.295164*d - 0.057697*d^2 + 0.003804*d^3
        sf = d.multiply(0.295164) \
            .add(0.531747) \
            .subtract(d.pow(2).multiply(0.057697)) \
            .add(d.pow(3).multiply(0.003804)) \
            .rename('sf')

        # SCS effective precipitation (eq. 2-84)
        # ep = sf * (0.70917 * pr^0.82416 - 0.11556) * 10^(0.02426 * eto)
        ep = sf.multiply(
            pr.pow(0.82416).multiply(0.70917).subtract(0.11556)
        ).multiply(
            ee.Image.constant(10).pow(eto.multiply(0.02426))
        ).rename('peff')

        # Clamp: ep <= pr, ep <= eto, ep >= 0; convert to mm; final clamp against mm precip
        ep_cleaned = ep.where(ep.gte(pr), pr) \
            .where(ep.gt(eto), eto) \
            .clamp(0, 10000) \
            .multiply(25.4)  # inches → mm

        # Final safeguard: clamp against original mm precip to handle reprojection artifacts
        pr_mm = img.select('pr')
        ep_cleaned = ep_cleaned.where(ep_cleaned.gt(pr_mm), pr_mm) \
            .clamp(0, 10000)

        return ee.Image(ep_cleaned) \
            .setDefaultProjection(crs='EPSG:4326', scale=ep_scale) \
            .copyProperties(img, ['system:time_start', 'system:time_end', 'month']) \
            .set('year', year)

    return joined.map(calculate_ep)


def build_and_export(start_year, end_year):
    init_ee()
    az = get_az_geometry()
    create_ic_asset(ASSET_ID)
    existing = list_existing_assets(ASSET_ID)
    pending = list_pending_task_descriptions()

    tasks = []
    for year in range(start_year, end_year + 1):
        monthly_peff = build_monthly_peff(year)

        for m in range(1, 13):
            img_name = f'{year}_{m:02d}'
            img_asset = f'{ASSET_ID}/{img_name}'
            if img_asset in existing:
                continue
            img = monthly_peff.filter(
                ee.Filter.calendarRange(m, m, 'month')
            ).first()
            task = export_image(
                img, img_asset,
                f'peff_v2_{img_name}',
                az, PRISM_SCALE,
                pending_descriptions=pending
            )
            tasks.append(task)

        logger.info(f'  Submitted year {year} ({len(tasks)} total tasks)')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = get_export_parser('Export monthly USDA SCS effective precipitation (1896-2099)')
    args = parser.parse_args()
    start = args.start_year or DEFAULT_START
    end = args.end_year or DEFAULT_END

    logger.info(f'Exporting monthly USDA SCS peff for {start}-{end}...')
    tasks = build_and_export(start, end)
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    logger.info(f'Asset: {ASSET_ID}')
