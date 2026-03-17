"""
Shared configuration and utility functions for GEE asset export scripts.

Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

Usage:
    All export scripts import from this module. Run with the azhydro conda env:
    $ conda activate azhydro
    $ python export_<collection>.py [--start-year YYYY] [--end-year YYYY] [--no-wait]
"""

import argparse
import logging
import time
from functools import wraps
from http.client import RemoteDisconnected

import ee

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (ee.EEException, RemoteDisconnected, ConnectionError, TimeoutError, OSError)
MAX_RETRIES = 10


def _retry(func):
    """Decorator: retry on EE / network errors with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except _RETRYABLE_ERRORS as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = min(2 ** attempt, 120)
                logger.warning(f'  [{func.__name__}] {e} (attempt {attempt}/{MAX_RETRIES}). '
                      f'Retrying in {wait}s...')
                time.sleep(wait)
    return wrapper

# ─── Constants ────────────────────────────────────────────────────────────────

GEE_PROJECT = 'azhydro'
ASSET_PREFIX = f'projects/{GEE_PROJECT}/assets'

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

MACA_SCENARIOS = ['rcp45', 'rcp85']
MACA_MODELS = [
    'bcc-csm1-1', 'bcc-csm1-1-m', 'BNU-ESM', 'CanESM2', 'CCSM4',
    'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M',
    'HadGEM2-CC365', 'HadGEM2-ES365', 'inmcm4', 'IPSL-CM5A-MR',
    'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM',
    'MIROC-ESM-CHEM', 'MRI-CGCM3', 'NorESM1-M'
]
MACA_BANDS = ['tasmax', 'tasmin', 'pr', 'rsds', 'uas', 'vas', 'huss']

# Representative GCMs spanning the Southwest US climate spread for uncertainty
# quantification.  Selected from the 20-model MACA ensemble to capture hot-dry,
# hot-wet, cool-dry, cool-wet, and central tendency corners of projected
# temperature × precipitation space (cf. Rupp et al., 2013).
MACA_REPRESENTATIVE_GCMS = [
    'CCSM4',           # central / median
    'CNRM-CM5',        # cool-wet
    'HadGEM2-ES365',   # hot-dry
    'MIROC-ESM-CHEM',  # hot-wet
    'inmcm4',          # cool-dry  (lowest climate sensitivity in CMIP5)
]

USGS_LULC_SCENARIOS = ['B1', 'B2', 'A1B', 'A2']

# Native scales (meters) for each source
PRISM_SCALE = 4638.3
GRIDMET_SCALE = 4638.3
MACA_SCALE = 4638.3
OPENET_SCALE = 30
REITZ_SCALE = 800
USGS_LULC_SCALE = 250


# ─── Initialization ──────────────────────────────────────────────────────────

@_retry
def init_ee():
    """Initialize Earth Engine with high-volume endpoint."""
    ee.Initialize(
        project=GEE_PROJECT,
        opt_url='https://earthengine-highvolume.googleapis.com'
    )


def get_az_geometry():
    """Get Arizona state boundary as ee.Geometry from TIGER/Census."""
    return ee.FeatureCollection('TIGER/2018/States') \
        .filter(ee.Filter.eq('STATEFP', '04')).geometry()


# ─── Asset management ────────────────────────────────────────────────────────

@_retry
def create_ic_asset(asset_id):
    """Create an ImageCollection asset if it doesn't exist."""
    try:
        ee.data.getAsset(asset_id)
    except ee.EEException:
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, asset_id)
        logger.info(f'Created ImageCollection: {asset_id}')


def asset_exists(asset_id):
    """Check if a GEE asset exists."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


@_retry
def list_existing_assets(collection_id):
    """List all asset IDs in an ImageCollection. Returns a set for O(1) lookups."""
    existing = set()
    try:
        params = {'parent': collection_id, 'pageSize': 1000}
        while True:
            result = ee.data.listAssets(params)
            for asset in result.get('assets', []):
                existing.add(asset['name'])
            token = result.get('nextPageToken')
            if not token:
                break
            params['pageToken'] = token
    except ee.EEException:
        pass
    return existing


@_retry
def list_pending_task_descriptions():
    """Return a set of descriptions for all PENDING/RUNNING/READY tasks."""
    pending = set()
    for op in ee.data.listOperations():
        meta = op.get('metadata', {})
        if meta.get('state') in ('PENDING', 'RUNNING', 'READY'):
            pending.add(meta.get('description', ''))
    return pending


def _wait_for_queue_capacity(max_queue=2900, poll_interval=60):
    """Block until the task queue has room (below max_queue active/ready tasks)."""
    while True:
        try:
            ops = ee.data.listOperations()
        except _RETRYABLE_ERRORS as e:
            logger.warning(f'  [_wait_for_queue_capacity] {e}. Retrying in {poll_interval}s...')
            time.sleep(poll_interval)
            continue
        busy = sum(1 for op in ops
                   if op.get('metadata', {}).get('state') in
                   ('PENDING', 'RUNNING', 'READY'))
        if busy < max_queue:
            return
        logger.info(f'  Queue full ({busy} tasks). Waiting {poll_interval}s...')
        time.sleep(poll_interval)


@_retry
def export_image(image, asset_id, description, region, scale, crs='EPSG:4326',
                 as_int=False, pending_descriptions=None, existing_assets=None):
    """Export an ee.Image to a GEE asset, clipped to region.
    Automatically waits if the task queue is near the 3000-task limit.
    Skips if the asset already exists or a task with the same description is queued.

    When *existing_assets* is provided (a set of asset IDs), the caller's
    pre-fetched set is used instead of making a per-image ``asset_exists``
    API call, avoiding redundant requests during large batch exports."""
    if existing_assets is not None:
        if asset_id in existing_assets:
            logger.info(f'  Skipping {description} (asset already exists)')
            return None
    elif asset_exists(asset_id):
        logger.info(f'  Skipping {description} (asset already exists)')
        return None
    desc = description[:100]
    if pending_descriptions is not None and desc in pending_descriptions:
        logger.info(f'  Skipping {desc} (task already queued)')
        return None
    _wait_for_queue_capacity()
    out = image.clip(region).toInt() if as_int else image.clip(region).toFloat()
    task = ee.batch.Export.image.toAsset(
        image=out,
        description=desc,
        assetId=asset_id,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13
    )
    task.start()
    return task


def wait_for_tasks(tasks, check_interval=30):
    """Wait for all export tasks to complete and report status."""
    tasks = [t for t in tasks if t is not None]
    if not tasks:
        logger.info('No tasks to wait for.')
        return
    logger.info(f'Waiting for {len(tasks)} tasks...')
    while True:
        active = [t for t in tasks if t.active()]
        if not active:
            break
        logger.info(f'  {len(active)} tasks still active...')
        time.sleep(check_interval)
    failed = 0
    for t in tasks:
        status = t.status()
        if status['state'] == 'FAILED':
            logger.error(f'  FAILED: {status["description"]} - {status.get("error_message", "")}')
            failed += 1
    logger.info(f'Done. {len(tasks) - failed}/{len(tasks)} succeeded.')


def get_export_parser(description):
    """Create an argparse parser with common arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--start-year', type=int, default=None,
                        help='Start year (inclusive)')
    parser.add_argument('--end-year', type=int, default=None,
                        help='End year (inclusive)')
    parser.add_argument('--no-wait', action='store_true',
                        help='Submit tasks and exit without waiting')
    return parser


# ─── Shared GEE computation functions ────────────────────────────────────────

def calc_prism_monthly_eto(monthly_img):
    """
    Calculate monthly ETo from PRISM using Hargreaves method (openet.refetgee).
    Returns ee.Image with band 'prism_eto' (mm/month).
    """
    from openet.refetgee import Daily

    img_date = ee.Date(monthly_img.get('system:time_start'))
    month = img_date.get('month')
    year = img_date.get('year')
    ndays = ee.Number(
        ee.Date.fromYMD(year, month, 1).advance(1, 'month')
        .difference(ee.Date.fromYMD(year, month, 1), 'day')
    )
    doy = ee.Date(monthly_img.get('system:time_start')) \
        .getRelative('day', 'year').add(ndays.divide(2).floor())
    tmax = monthly_img.select('tmax')
    tmin = monthly_img.select('tmin')
    lat_img = ee.Image.pixelLonLat().select('latitude')
    daily_obj = Daily(
        tmax=tmax, tmin=tmin,
        ea=ee.Image.constant(1), rs=ee.Image.constant(1),
        uz=ee.Image.constant(1), zw=ee.Number(2),
        elev=ee.Image.constant(0), lat=lat_img, doy=doy
    )
    return daily_obj.pet_hargreaves.multiply(ndays) \
        .rename('prism_eto') \
        .set('system:time_start', monthly_img.get('system:time_start')) \
        .set('system:time_end', monthly_img.get('system:time_end'))


def build_openet_monthly_et_ic():
    """
    Build monthly OpenET ET ImageCollection for 2000-2025.
    Returns ee.ImageCollection with band 'et_ensemble_mad' and properties
    'system:time_start', 'system:time_end', 'month', 'year'.
    """
    openet_v2 = ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0')
    openet_v2_1 = ee.ImageCollection('projects/openet/assets/ensemble/conus/gridmet/monthly/v2_1')
    months = ee.List.sequence(1, 12)

    def _build_monthly(src_ic, year_list):
        return ee.ImageCollection(year_list.map(lambda y:
            months.map(lambda m:
                src_ic.filter(ee.Filter.calendarRange(y, y, 'year'))
                    .filter(ee.Filter.calendarRange(m, m, 'month'))
                    .select('et_ensemble_mad').sum()
                    .rename('et_ensemble_mad')
                    .set('system:time_start', ee.Date.fromYMD(y, m, 1).millis())
                    .set('system:time_end', ee.Date.fromYMD(y, m, 1).advance(1, 'month').millis())
                    .set('month', m).set('year', y)
            )
        ).flatten())

    ic_v2 = _build_monthly(openet_v2, ee.List.sequence(2000, 2024))
    ic_v2_1 = _build_monthly(openet_v2_1, ee.List.sequence(2025, 2025))
    return ic_v2.merge(ic_v2_1)


def build_daily_maca_ensemble(year):
    """
    Build daily MACA ensemble for a single year by averaging all models/scenarios.
    Returns ee.ImageCollection with one image per day.
    """
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year + 1, 1, 1)
    nDays = endDate.difference(startDate, 'day')
    maca_ic = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
        .filterDate(startDate, endDate).select(MACA_BANDS)
    dayOffsets = ee.List.sequence(0, nDays.subtract(1))
    return ee.ImageCollection(dayOffsets.map(lambda offset:
        maca_ic.filterDate(
            startDate.advance(offset, 'day'),
            startDate.advance(ee.Number(offset).add(1), 'day')
        ).mean()
        .set('system:time_start', startDate.advance(offset, 'day').millis())
        .set('system:time_end', startDate.advance(ee.Number(offset).add(1), 'day').millis())
    ))


def build_daily_maca_single(year, model, scenario):
    """
    Build daily MACA ImageCollection for a single model and scenario.
    Returns ee.ImageCollection with one image per day.
    """
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year + 1, 1, 1)
    nDays = endDate.difference(startDate, 'day')
    maca_ic = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
        .filterDate(startDate, endDate) \
        .filter(ee.Filter.eq('model', model)) \
        .filter(ee.Filter.eq('scenario', scenario)) \
        .select(MACA_BANDS)
    dayOffsets = ee.List.sequence(0, nDays.subtract(1))
    return ee.ImageCollection(dayOffsets.map(lambda offset:
        maca_ic.filterDate(
            startDate.advance(offset, 'day'),
            startDate.advance(ee.Number(offset).add(1), 'day')
        ).first()
        .set('system:time_start', startDate.advance(offset, 'day').millis())
        .set('system:time_end', startDate.advance(ee.Number(offset).add(1), 'day').millis())
    ))


# ─── LULC-stratified EToF helpers ────────────────────────────────────────────

LULC_ETOF_ASSET = f'{ASSET_PREFIX}/lulc_stratified_etof'


def build_lulc_varying_etof(year, month):
    """
    Composite per-LULC-class EToF climatologies into a single EToF grid
    using the projected LULC map for *year*.

    The raw USGS LULC ensemble is reclassified (13→1 AGRI, 2/6→2 URBAN,
    1→3 SW), matching ``_get_lulc_image`` in ``dataops.py``.  The 'other'
    (natural/barren) EToF is used as the base layer, then pixels classified
    as AGRI, URBAN, or SW are overwritten with their class-specific EToF.

    Returns ee.Image with band 'etof'.
    """
    lulc_ic = ee.ImageCollection(f'{ASSET_PREFIX}/lulc_projection_ensemble')
    lulc_raw = lulc_ic.filterDate(
        ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(ee.Number(year).add(1), 1, 1)
    ).first()
    # Reclassify raw USGS classes → 1=AGRI, 2=URBAN, 3=SW (same as _get_lulc_image)
    lulc = lulc_raw.remap([13, 2, 6, 1], [1, 2, 2, 3])

    etof_img = ee.Image(f'{LULC_ETOF_ASSET}/month_{month:02d}')

    # Start with 'other' as the base (unmask so .where() can write into
    # pixels where etof_other has no data — e.g. always-agri NLCD pixels)
    return etof_img.select('etof_other').unmask(0) \
        .where(lulc.eq(1), etof_img.select('etof_agri')) \
        .where(lulc.eq(2), etof_img.select('etof_urban')) \
        .where(lulc.eq(3), etof_img.select('etof_sw')) \
        .rename('etof') \
        .set('month', month)
