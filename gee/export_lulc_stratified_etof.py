"""
Export LULC-stratified monthly EToF (crop coefficient = ET/ETo) as GEE assets.

Computes per-LULC-class climatological EToF by masking OpenET and gridMET ETo
to each LULC class (Agriculture, Urban, Surface Water, Other) before computing
ET/ETo per month averaged across 2000-2024.  Each of the 12 monthly images has
4 bands (etof_agri, etof_urban, etof_sw, etof_other) that can be composited at
runtime using a projected LULC map, yielding year-varying EToF that captures
land-use-driven ET variability.

Dependencies: None (uses public OpenET, gridMET ETo, and NLCD collections).

Asset: projects/azhydro/assets/lulc_stratified_etof/ (12 images × 4 bands)

Usage:
    conda activate azhydro
    python export_lulc_stratified_etof.py [--no-wait]
"""

import logging

import ee
from config import (
    ASSET_PREFIX,
    GRIDMET_SCALE,
    build_openet_monthly_et_ic,
    create_ic_asset,
    export_image,
    get_az_geometry,
    init_ee,
    list_existing_assets,
    list_pending_task_descriptions,
    wait_for_tasks,
)

logger = logging.getLogger(__name__)


# NLCD class → pipeline LULC class mapping:
#   82 (Cultivated Crops) → AGRI
#   21-24 (Developed) → URBAN
#   11 (Open Water) → SW
#   Everything else → OTHER
LULC_CLASSES = {
    'agri': 'Agriculture',
    'urban': 'Urban',
    'sw': 'Surface Water',
    'other': 'Other/Natural',
}

ETOF_START = 2000
ETOF_END = 2024


def _build_nlcd_mask(nlcd_img, lulc_key):
    """Build a binary mask for a LULC class from an NLCD image."""
    if lulc_key == 'agri':
        return nlcd_img.eq(82)
    elif lulc_key == 'urban':
        return nlcd_img.gte(21).And(nlcd_img.lte(24))
    elif lulc_key == 'sw':
        return nlcd_img.eq(11)
    else:  # other
        is_agri = nlcd_img.eq(82)
        is_urban = nlcd_img.gte(21).And(nlcd_img.lte(24))
        is_sw = nlcd_img.eq(11)
        return is_agri.Or(is_urban).Or(is_sw).Not()


def build_and_export():
    init_ee()
    az = get_az_geometry()
    pending = list_pending_task_descriptions()

    # OpenET monthly ET 2000-2024
    openet_ic = build_openet_monthly_et_ic()
    openet_flt = openet_ic.filterDate(f'{ETOF_START}-01-01', f'{ETOF_END + 1}-01-01') \
        .select('et_ensemble_mad')

    # gridMET monthly ETo 2000-2024
    gridmet_eto = ee.ImageCollection(
        'projects/openet/assets/reference_et/conus/gridmet/monthly/v1'
    ).filterDate(f'{ETOF_START}-01-01', f'{ETOF_END + 1}-01-01').select('eto').map(lambda img:
        img.set('month', ee.Date(img.get('system:time_start')).get('month'))
            .set('year', ee.Date(img.get('system:time_start')).get('year'))
    )

    # NLCD annual landcover
    nlcd_ic = ee.ImageCollection(
        'projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER'
    )

    # Join OpenET and gridMET by year+month to get paired ET/ETo
    join_filter = ee.Filter.And(
        ee.Filter.equals(leftField='month', rightField='month'),
        ee.Filter.equals(leftField='year', rightField='year')
    )
    joined = ee.ImageCollection(ee.Join.inner().apply(
        openet_flt, gridmet_eto, join_filter
    )).map(lambda f:
        ee.Image(f.get('primary')).select('et_ensemble_mad')
            .addBands(ee.Image(f.get('secondary')).select('eto'))
            .set('system:time_start', ee.Image(f.get('primary')).get('system:time_start'))
            .set('month', ee.Image(f.get('primary')).get('month'))
            .set('year', ee.Image(f.get('primary')).get('year'))
    )

    # Build per-class masked EToF collections
    class_etof_ics = {}
    for lulc_key in LULC_CLASSES:
        logger.info(f'Building {LULC_CLASSES[lulc_key]} EToF...')

        def _mask_and_ratio(img, _lulc_key=lulc_key):
            year = ee.Number(img.get('year'))
            nlcd_year = year.max(2001).min(2024)
            nlcd_img = nlcd_ic.filterDate(
                ee.Date.fromYMD(nlcd_year, 1, 1),
                ee.Date.fromYMD(nlcd_year.add(1), 1, 1)
            ).first()
            mask = _build_nlcd_mask(nlcd_img, _lulc_key)
            # Mask OpenET ET by NLCD class, then divide by ETo at export scale
            et_masked = img.select('et_ensemble_mad').updateMask(mask)
            return et_masked.divide(img.select('eto')).rename(f'etof_{_lulc_key}') \
                .set('month', img.get('month')) \
                .set('year', img.get('year'))

        class_etof_ics[lulc_key] = joined.map(_mask_and_ratio)

    # Export 12 multi-band images (one per month, 4 bands each)
    asset_id = f'{ASSET_PREFIX}/lulc_stratified_etof'
    create_ic_asset(asset_id)
    existing = list_existing_assets(asset_id)

    tasks = []
    for m in range(1, 13):
        img_name = f'month_{m:02d}'
        img_asset = f'{asset_id}/{img_name}'
        if img_asset in existing:
            logger.info(f'  Skipping {img_name} (already exists)')
            continue

        bands = []
        for lulc_key in LULC_CLASSES:
            band = class_etof_ics[lulc_key].filter(
                ee.Filter.eq('month', m)
            ).mean().rename(f'etof_{lulc_key}')
            bands.append(band)

        img = bands[0].addBands(bands[1]).addBands(bands[2]).addBands(bands[3]) \
            .set('month', m) \
            .set('system:time_start', ee.Date.fromYMD(2000, m, 1).millis())
        task = export_image(
            img, img_asset,
            f'lulc_etof_{img_name}',
            az, GRIDMET_SCALE,
            pending_descriptions=pending,
        )
        tasks.append(task)
        logger.info(f'  Submitted {img_name}')

    return tasks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    import argparse
    parser = argparse.ArgumentParser(description='Export LULC-stratified monthly EToF (12 images × 4 bands)')
    parser.add_argument('--no-wait', action='store_true')
    args = parser.parse_args()

    logger.info('Exporting LULC-stratified monthly EToF (12 images × 4 bands)...')
    tasks = build_and_export()
    if tasks and not args.no_wait:
        wait_for_tasks(tasks)
    logger.info(f'Asset: {ASSET_PREFIX}/lulc_stratified_etof')
