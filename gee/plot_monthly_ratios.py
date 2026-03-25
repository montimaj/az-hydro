"""
Visualize monthly ratio grids (OpenET gridMET/Hargreaves ETo, EToF,
OpenET/Reitz ET, OpenET gridMET/PRISM ETo 1981–2021) overlaid with AZ
groundwater basin boundaries.

Produces per ratio:
  1. A 4x3 spatial map (Jan-Dec) with basin overlays.
  2. A basin-averaged heatmap (basins x months).

Dependencies: earthengine-api, rasterio, geopandas, matplotlib, numpy.
Requires: Run the three ratio export scripts first so the GEE assets exist.

Usage:
    conda activate azhydro
    python plot_monthly_ratios.py [--output-dir DIR]
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import argparse
import io
import logging
import os
from collections import defaultdict

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import matplotlib.colors as mcolors
from rasterstats import zonal_stats

from config import (
    ASSET_PREFIX, GRIDMET_SCALE, MONTH_NAMES, PRISM_SCALE, REITZ_SCALE,
    init_ee,
)

logger = logging.getLogger(__name__)

# ─── Ratio definitions ─────────────────────────────────────────────────────

RATIOS = {
    'gridmet_hargreaves_eto_ratio': {
        'asset': f'{ASSET_PREFIX}/gridmet_hargreaves_eto_ratio',
        'band': 'ratio',
        'title': 'OpenET GridMET ETo / PRISM Hargreaves ETo (1979–2025)',
        'label': 'Ratio',
        'scale': PRISM_SCALE,
        'norm_group': 'eto_ratio',
    },
    'monthly_etof': {
        'asset': f'{ASSET_PREFIX}/monthly_etof',
        'band': 'etof',
        'title': 'EToF (OpenET ET / OpenET GridMET ETo, 2000-2024)',
        'label': 'EToF',
        'scale': GRIDMET_SCALE,
    },
    'openet_reitz_et_ratio': {
        'asset': f'{ASSET_PREFIX}/openet_reitz_et_ratio',
        'band': 'ratio',
        'title': 'OpenET ET / USGS Ensemble ET (2000-2018)',
        'label': 'Ratio',
        'scale': REITZ_SCALE,
    },
    'prism_gridmet_eto_ratio': {
        'asset': f'{ASSET_PREFIX}/prism_gridmet_ratio/monthly',
        'band': 'eto',
        'title': 'OpenET GridMET ETo / PRISM ETo (1981–2021)',
        'label': 'Ratio',
        'scale': PRISM_SCALE,
        'img_names': [n.lower() for n in MONTH_NAMES],
        'norm_group': 'eto_ratio',
    },
}

BASIN_SHP = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'Data', 'Inputs', 'GW_Data', 'Groundwater_Basin', 'Groundwater_Basin.shp',
)

# Discrete diverging colormap centered on 1 (blue < 1, white = 1, red > 1)
_RATIO_COLORS = [
    '#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',  # blues
    '#f7f7f7',                                                # white center
    '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f',  # reds
]
_RATIO_CMAP = mcolors.LinearSegmentedColormap.from_list('ratio_div', _RATIO_COLORS, N=256)


def _make_ratio_norm(vmin, vmax):
    """Create a BoundaryNorm with discrete steps centered on 1."""
    max_dev = max(1 - vmin, vmax - 1)
    lo = max(1 - max_dev, 0.1)  # clamp: ratios are strictly positive
    hi = 1 + max_dev
    # Build two halves so 1 is always a boundary midpoint
    n_below = 6
    n_above = 6
    lower = np.linspace(lo, 1, n_below + 1)        # lo ... 1
    upper = np.linspace(1, hi, n_above + 1)[1:]     # (1 ... hi], skip duplicate 1
    boundaries = np.concatenate([lower, upper])
    return mcolors.BoundaryNorm(boundaries, 256)


def _download_ee_image(image, region, scale, band, cache_dir=None,
                       cache_name=None):
    """Download a single-band ee.Image as a numpy array with its transform.

    If *cache_dir* and *cache_name* are provided, the GeoTIFF is saved to
    disk on the first call and read from the cache on subsequent runs.

    Returns (array, transform, crs_wkt).
    """
    # Check cache first
    if cache_dir and cache_name:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'{cache_name}.tif')
        if os.path.isfile(cache_path):
            logger.info(f'  Using cached {cache_name}')
            with rasterio.open(cache_path) as src:
                return src.read(1), src.transform, src.crs.to_wkt()
    else:
        cache_path = None

    import urllib.request
    url = image.select(band).getDownloadURL({
        'region': region,
        'scale': scale,
        'format': 'GEO_TIFF',
    })
    resp = urllib.request.urlopen(url)
    data = resp.read()

    # Save to cache
    if cache_path:
        with open(cache_path, 'wb') as f:
            f.write(data)
        logger.info(f'  Cached {cache_name}')

    with rasterio.open(io.BytesIO(data)) as src:
        arr = src.read(1)
        transform = src.transform
        crs_wkt = src.crs.to_wkt()
    return arr, transform, crs_wkt


def _download_ratio(ratio_key, info, az_geom, output_dir):
    """Download all 12 monthly images for one ratio. Returns (arrays, transforms, crs_wkt)."""
    cache_dir = os.path.join(output_dir, '.cache', ratio_key)
    arrays = []
    transforms = []
    crs_wkt = None
    img_names = info.get('img_names', [f'month_{m:02d}' for m in range(1, 13)])
    for m in range(12):
        name = img_names[m]
        img_id = f'{info["asset"]}/{name}'
        img = ee.Image(img_id)
        arr, tfm, cw = _download_ee_image(
            img, az_geom, info['scale'], info['band'],
            cache_dir=cache_dir, cache_name=name)
        arrays.append(arr)
        transforms.append(tfm)
        if crs_wkt is None:
            crs_wkt = cw
    return arrays, transforms, crs_wkt


def _plot_ratio(ratio_key, info, basins_gdf, arrays, transforms, crs_wkt,
                output_dir, norm=None):
    """Create a 4x3 (rows x cols) figure for one ratio collection."""
    fig, axes = plt.subplots(4, 3, figsize=(14, 18))

    if norm is None:
        all_valid = np.concatenate([a[(np.isfinite(a)) & (a > 0)].ravel() for a in arrays])
        vmin = min(np.percentile(all_valid, 1), 1 - 1e-6)
        vmax = max(np.percentile(all_valid, 99), 1 + 1e-6)
        norm = _make_ratio_norm(vmin, vmax)

    basins_plot = basins_gdf.to_crs(crs_wkt)

    for m in range(12):
        row, col = divmod(m, 3)
        ax = axes[row, col]
        arr = arrays[m]
        tfm = transforms[m]

        arr_masked = np.where((np.isfinite(arr)) & (arr > 0), arr, np.nan)

        extent = [
            tfm.c, tfm.c + tfm.a * arr.shape[1],
            tfm.f + tfm.e * arr.shape[0], tfm.f,
        ]
        im = ax.imshow(
            arr_masked, extent=extent, origin='upper',
            cmap=_RATIO_CMAP, norm=norm,
        )
        basins_plot.boundary.plot(ax=ax, color='black', linewidth=0.5)
        ax.set_title(MONTH_NAMES[m], fontweight='bold', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(bottom=0.06, top=0.94, hspace=0.20, wspace=0.10)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.set_label(info['title'], fontweight='bold')

    base = os.path.join(output_dir, ratio_key)
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved {base}.png')


def _compute_basin_means(arrays, transforms, basins_plot):
    """Compute zonal means per basin per month. Returns list of 12 lists."""
    basin_means = []
    for m in range(12):
        arr = arrays[m].astype('float64')
        arr[(~np.isfinite(arr)) | (arr <= 0)] = np.nan
        stats = zonal_stats(
            basins_plot, arr, affine=transforms[m], stats=['mean'], nodata=np.nan,
        )
        basin_means.append([s['mean'] for s in stats])
    return basin_means


def _plot_basin_avg(ratio_key, info, basins_plot, basin_means, output_dir,
                    norm=None):
    """Create a 4x3 choropleth figure showing basin-averaged ratios per month."""
    if norm is None:
        all_means = [v for month_vals in basin_means for v in month_vals if v is not None]
        vmin = min(np.nanpercentile(all_means, 1), 1 - 1e-6)
        vmax = max(np.nanpercentile(all_means, 99), 1 + 1e-6)
        norm = _make_ratio_norm(vmin, vmax)

    fig, axes = plt.subplots(4, 3, figsize=(14, 18))

    for m in range(12):
        row, col = divmod(m, 3)
        ax = axes[row, col]
        gdf = basins_plot.copy()
        gdf['mean_ratio'] = basin_means[m]
        gdf.plot(
            column='mean_ratio', ax=ax, cmap=_RATIO_CMAP,
            norm=norm, edgecolor='black', linewidth=0.5,
            legend=False, missing_kwds={'color': 'lightgray'},
        )
        ax.set_title(MONTH_NAMES[m], fontweight='bold', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(bottom=0.06, top=0.94, hspace=0.20, wspace=0.10)
    sm = plt.cm.ScalarMappable(cmap=_RATIO_CMAP, norm=norm)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.set_label(f'{info["title"]} (Basin Averages)', fontweight='bold')

    base = os.path.join(output_dir, f'{ratio_key}_basin_avg')
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved {base}.png')


def _compute_az_means(arrays):
    """Compute AZ-wide spatial mean per month. Returns list of 12 floats."""
    means = []
    for m in range(12):
        arr = arrays[m].astype('float64')
        valid = arr[(np.isfinite(arr)) & (arr > 0)]
        means.append(np.mean(valid) if valid.size > 0 else np.nan)
    return means


def _compute_ama_ina_means(arrays, transforms, ama_ina_gdf):
    """Compute area-weighted AMA/INA mean per month. Returns list of 12 floats."""
    means = []
    for m in range(12):
        arr = arrays[m].astype('float64')
        arr[(~np.isfinite(arr)) | (arr <= 0)] = np.nan
        stats = zonal_stats(
            ama_ina_gdf, arr, affine=transforms[m],
            stats=['mean', 'count'], nodata=np.nan,
        )
        total_pixels = sum(s['count'] for s in stats if s['count'])
        if total_pixels > 0:
            weighted = sum(s['mean'] * s['count'] for s in stats
                           if s['mean'] is not None and s['count'])
            means.append(weighted / total_pixels)
        else:
            means.append(np.nan)
    return means


def _plot_monthly_mean_ts(ratio_key, info, means, output_dir, ylim=None):
    """Plot AZ-wide monthly mean time series for one ratio."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 13)

    ax.plot(x, means, 'o-', color='#2166ac', linewidth=2, markersize=6)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='1:1')
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Mean Ratio', fontweight='bold')
    ax.set_title(f'AZ-Wide Monthly Mean: {info["title"]}', fontweight='bold')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{ratio_key}_monthly_mean_ts.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved {out_path}')


AMA_INA_NAMES = [
    'SANTA CRUZ AMA', 'PRESCOTT AMA', 'TUCSON AMA', 'PINAL AMA',
    'PHOENIX AMA', 'DOUGLAS AMA', 'JOSEPH CITY INA',
    'HARQUAHALA INA', 'HUALAPAI VALLEY INA', 'WILLCOX AMA',
]


def _plot_monthly_mean_ts_ama_ina(ratio_key, info, means, output_dir,
                                  ylim=None):
    """Plot AMA/INA-only monthly mean time series for one ratio."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 13)

    ax.plot(x, means, 'o-', color='#2166ac', linewidth=2, markersize=6)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='1:1')
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Mean Ratio', fontweight='bold')
    ax.set_title(f'AMA/INA Monthly Mean: {info["title"]}', fontweight='bold')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{ratio_key}_monthly_mean_ts_ama_ina.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize monthly ratio grids with GW basin overlays')
    parser.add_argument(
        '--output-dir', default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'Data', 'Outputs', 'GEE_Ratios'),
        help='Output directory (default: Data/Outputs/GEE_Ratios/)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    init_ee()
    az_geom = ee.FeatureCollection('TIGER/2018/States') \
        .filter(ee.Filter.eq('STATEFP', '04')).geometry()

    logger.info(f'Loading basins from {BASIN_SHP}')
    basins_gdf = gpd.read_file(BASIN_SHP)

    # Phase 1: download all ratio data and compute means
    downloaded = {}
    for key, info in RATIOS.items():
        logger.info(f'Downloading {key}...')
        arrays, transforms, crs_wkt = _download_ratio(
            key, info, az_geom, args.output_dir)
        basins_plot = basins_gdf.to_crs(crs_wkt)
        basin_means = _compute_basin_means(arrays, transforms, basins_plot)
        az_means = _compute_az_means(arrays)
        ama_ina_gdf = basins_plot[basins_plot['BASIN_NAME'].isin(AMA_INA_NAMES)]
        ama_ina_means = _compute_ama_ina_means(arrays, transforms, ama_ina_gdf)
        downloaded[key] = {
            'arrays': arrays, 'transforms': transforms, 'crs_wkt': crs_wkt,
            'basins_plot': basins_plot, 'basin_means': basin_means,
            'az_means': az_means, 'ama_ina_means': ama_ina_means,
        }

    # Phase 2: compute shared norms/ylims for grouped ratios
    groups = defaultdict(list)
    for key, info in RATIOS.items():
        grp = info.get('norm_group')
        if grp:
            groups[grp].append(key)

    group_spatial_norms = {}
    group_basin_norms = {}
    group_az_ylims = {}
    group_ama_ina_ylims = {}
    for grp, keys in groups.items():
        # Shared spatial norm
        all_valid = np.concatenate([
            a[(np.isfinite(a)) & (a > 0)].ravel()
            for k in keys for a in downloaded[k]['arrays']
        ])
        vmin = min(np.percentile(all_valid, 1), 1 - 1e-6)
        vmax = max(np.percentile(all_valid, 99), 1 + 1e-6)
        group_spatial_norms[grp] = _make_ratio_norm(vmin, vmax)

        # Shared basin-avg norm
        all_means = [
            v for k in keys
            for month_vals in downloaded[k]['basin_means']
            for v in month_vals if v is not None
        ]
        vmin_b = min(np.nanpercentile(all_means, 1), 1 - 1e-6)
        vmax_b = max(np.nanpercentile(all_means, 99), 1 + 1e-6)
        group_basin_norms[grp] = _make_ratio_norm(vmin_b, vmax_b)

        # Shared AZ-wide time series y-limits
        all_az = [v for k in keys for v in downloaded[k]['az_means']
                  if not np.isnan(v)]
        pad = 0.05 * (max(all_az) - min(all_az)) if all_az else 0.1
        group_az_ylims[grp] = (min(all_az) - pad, max(all_az) + pad)

        # Shared AMA/INA time series y-limits
        all_ai = [v for k in keys for v in downloaded[k]['ama_ina_means']
                  if not np.isnan(v)]
        pad_ai = 0.05 * (max(all_ai) - min(all_ai)) if all_ai else 0.1
        group_ama_ina_ylims[grp] = (min(all_ai) - pad_ai, max(all_ai) + pad_ai)

    # Phase 3: plot
    for key, info in RATIOS.items():
        d = downloaded[key]
        grp = info.get('norm_group')
        sp_norm = group_spatial_norms.get(grp) if grp else None
        ba_norm = group_basin_norms.get(grp) if grp else None
        az_ylim = group_az_ylims.get(grp) if grp else None
        ai_ylim = group_ama_ina_ylims.get(grp) if grp else None

        logger.info(f'Plotting {key}...')
        _plot_ratio(key, info, basins_gdf, d['arrays'], d['transforms'],
                    d['crs_wkt'], args.output_dir, norm=sp_norm)
        logger.info(f'Plotting basin averages for {key}...')
        _plot_basin_avg(key, info, d['basins_plot'], d['basin_means'],
                        args.output_dir, norm=ba_norm)
        logger.info(f'Plotting AZ-wide time series for {key}...')
        _plot_monthly_mean_ts(key, info, d['az_means'], args.output_dir,
                              ylim=az_ylim)
        logger.info(f'Plotting AMA/INA time series for {key}...')
        _plot_monthly_mean_ts_ama_ina(key, info, d['ama_ina_means'],
                                      args.output_dir, ylim=ai_ylim)

    logger.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    main()
