"""
Master script to run all GEE asset exports in dependency order.

Dependency graph:
  Level 1 (no dependencies):
    - export_gridmet_hargreaves_ratio.py
    - export_openet_reitz_ratio.py
    - export_monthly_etof.py
    - export_lulc_ensemble.py
  Level 2 (depends on Level 1 ratios):
    - export_prism_hargreaves_eto.py  (needs gridmet_hargreaves ratio)
    - export_usgs_adjusted_et.py      (needs openet_reitz ratio)
    - export_maca_monthly_eto.py      (no custom dep, uses gridMET ratios)
  Level 3 (depends on Level 2):
    - export_maca_monthly_et.py       (needs monthly_etof + maca_monthly_eto_v2)
    - export_monthly_peff.py          (needs prism_hargreaves_eto + maca_monthly_eto_v2)
  Level 4 — per-GCM uncertainty (no inter-level deps; uses gridMET ratios + etof):
    - export_maca_gcm_annual_eto.py   (5 GCMs × 74 years = 370 images)
    - export_maca_gcm_annual_et.py    (5 GCMs × 74 years = 370 images)
    - export_maca_gcm_annual_peff.py  (5 GCMs × 74 years = 370 images)

Usage:
    conda activate azhydro
    python run_all_exports.py [--level {1,2,3,4,all}]
"""

import argparse
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


LEVELS = {
    1: [
        ('export_gridmet_hargreaves_ratio.py', 'gridMET/Hargreaves ETo ratio (12 images)'),
        ('export_openet_reitz_ratio.py', 'OpenET/Reitz ET ratio (12 images)'),
        ('export_monthly_etof.py', 'Monthly EToF (12 images)'),
        ('export_lulc_ensemble.py', 'LULC projection ensemble (74 images)'),
    ],
    2: [
        ('export_prism_hargreaves_eto.py', 'PRISM Hargreaves ETo 1896-1978 (996 images)'),
        ('export_usgs_adjusted_et.py', 'USGS adjusted ET 1896-1999 (1248 images)'),
        ('export_maca_monthly_eto.py', 'MACA monthly ETo 2026-2099 (888 images)'),
    ],
    3: [
        ('export_maca_monthly_et.py', 'MACA monthly ET 2026-2099 (888 images)'),
        ('export_monthly_peff.py', 'Monthly USDA SCS peff 1896-2099 (2448 images)'),
    ],
    4: [
        ('export_maca_gcm_annual_eto.py', 'Per-GCM annual ETo 2026-2099 (370 images)'),
        ('export_maca_gcm_annual_et.py', 'Per-GCM annual ET 2026-2099 (370 images)'),
        ('export_maca_gcm_annual_peff.py', 'Per-GCM annual peff 2026-2099 (370 images)'),
    ],
}


def run_script(script_name, extra_args=None):
    """Run an export script as a subprocess."""
    cmd = [sys.executable, script_name]
    if extra_args:
        cmd.extend(extra_args)
    logger.info(f'Running: {script_name}')
    logger.info('=' * 60)
    result = subprocess.run(cmd, cwd=sys.path[0] or '.')
    if result.returncode != 0:
        logger.warning(f'{script_name} exited with code {result.returncode}')
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run all GEE export scripts in dependency order'
    )
    parser.add_argument(
        '--level', type=str, default='all',
        choices=['1', '2', '3', '4', 'all'],
        help='Which dependency level to run (default: all)'
    )
    parser.add_argument(
        '--no-wait', action='store_true',
        help='Pass --no-wait to each export script'
    )
    parser.add_argument(
        '--keep-going', action='store_true',
        help='Continue running subsequent scripts even if one fails'
    )
    args = parser.parse_args()

    if args.level == 'all':
        levels_to_run = [1, 2, 3, 4]
    else:
        levels_to_run = [int(args.level)]

    extra = ['--no-wait'] if args.no_wait else []

    failed_scripts = []
    for level in levels_to_run:
        logger.info(f'# Level {level}')
        logger.info('#' * 60)
        for script, desc in LEVELS[level]:
            logger.info(f'  -> {desc}')
            rc = run_script(script, extra)
            if rc != 0:
                failed_scripts.append(script)
                if not args.keep_going:
                    logger.error(f'{script} failed (exit code {rc}). '
                                 f'Stopping. Use --keep-going to continue.')
                    sys.exit(rc)

    if failed_scripts:
        logger.warning(f'Completed with failures: {failed_scripts}')
    else:
        logger.info('All export levels completed successfully.')
    logger.info('=' * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
