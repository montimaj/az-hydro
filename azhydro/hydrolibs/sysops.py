"""
Provides methods for different system operations.
Source: https://code.usgs.gov/map/wu/aiwum-2.0-hydromap_ml-mirror/-/blob/main/aiwum2/maplibs/sysops.py
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import logging
import os
import shutil
from glob import glob

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Keywords identifying non-consumptive well uses in the ADWR Well Registry.
# Used to filter wells before pump-capacity and well-package calculations.
# Matched via str.contains to handle comma-separated combo labels.
NON_CONSUMPTIVE_USES = [
    'MONITORING', 'NO WATER USE', 'TEST', 'UNKNOWN', 'RESERVED',
    'NO USE CODE ON NOI', 'REMEDIATION', 'DEWATERING', 'DRAINAGE',
    'OTHER - MINERAL EXPLORE',
]


def derive_well_start_year(gdf, default_year: int = 1896) -> np.ndarray:
    """Derive installation year for each well in *gdf*.

    Priority: ``INSTALLED`` date → ``APPLICATIO`` date → *default_year*.
    The ADWR sentinel value 1899 (meaning "unknown") is treated as missing.

    Args:
        gdf: GeoDataFrame with optional ``INSTALLED`` / ``APPLICATIO`` columns.
        default_year: Fallback year for wells with no date information.

    Returns:
        1-D int32 array of start years, same length as *gdf*.
    """
    _SENTINEL_YEAR = 1899
    n = len(gdf)
    start_year = np.full(n, default_year, dtype=np.int32)
    for date_col in ('INSTALLED', 'APPLICATIO'):
        if date_col not in gdf.columns:
            continue
        dates = pd.to_datetime(gdf[date_col], errors='coerce')
        years = dates.dt.year
        fill_mask = ((start_year == default_year) & dates.notnull()
                     & (years != _SENTINEL_YEAR))
        start_year[fill_mask] = years[fill_mask].values
    n_fallback = (start_year == default_year).sum()
    logger.info(f'Well start years: {n - n_fallback} from registry dates, '
                f'{n_fallback} using conservative default ({default_year})')
    return start_year


def az_nodata() -> float:
    """Return fixed no data value for rasters.

    Returns
        float: Pre-defined/hard-coded default no data value for the USGS MAP Project.
    """
    return -32767.0


def makedirs(directory_list: tuple[str, ...] | str) -> None:
    """Create directory for storing files.

    Args:
        directory_list (tuple (str, ...) or str): Tuple of directories to create.

    Returns:
        None
    """
    if isinstance(directory_list, str):
        directory_list = [directory_list]
    for directory_name in directory_list:
        if directory_name is not None:
            os.makedirs(directory_name, exist_ok=True)


def make_proper_dir_name(dir_str: str) -> str | None:
    """Append os.sep to dir if not present.

    Args:
        dir_str (str): Directory path.

    Returns:
        str or None: Corrected directory path or None if dir_str is None.
    """
    if dir_str is None:
        return None
    sep = [os.sep, '/']
    if dir_str[-1] not in sep:
        return dir_str + os.sep
    return dir_str


def boolean_string(s: str) -> bool:
    """Return True/False based on a boolean string.

    Args:
        s: String object.

    Returns:
        bool: True (is s is 'True') or False (if s is 'False').

    Raises:
        ValueError if s does not belong to {'False', 'True'}.
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def copy_files(
        input_dir: str,
        target_dir: str,
        pattern: str = '*.tif',
        prefix: str = '',
        verbose: bool = True
) -> None:
    """Copy files from input directories to target directory.

    Args:
        input_dir (str): Input directory.
        target_dir (str): Target directory.
        pattern (str): File pattern.
        prefix (str): Prefix string for output file.
        verbose (str): Set True to get info on copy.

    Returns:
        None
    """
    file_list = glob(os.path.join(input_dir, pattern))
    for f in file_list:
        file_name = f[f.rfind(os.sep) + 1:]
        outfile = os.path.join(target_dir, f'{prefix}{file_name}')
        if verbose:
            logger.info(f'Copying {f} to {outfile} ...')
        shutil.copyfile(f, outfile)


def copy_file(
        input_file: str,
        output_file: str,
        suffix: str = '',
        ext: str = '',
        verbose: bool = True
) -> None:
    """
    Copy a single file.

    Args:
        input_file (str): Input file name.
        output_file (str): Output file name (should not contain extension) if either suffix or ext is set.
        suffix (str): Suffix string to append to output_file. Should be empty if output_file contains file extension.
        ext (str): Extension of output file. Should be empty if output_file contains file extension.
        verbose (bool): Set True to get info on copy.

    Returns:
        None.
    """

    if suffix or ext:
        output_file += suffix + ext
    if verbose:
        logger.info(f'Copying {input_file} to {output_file} ...')
    shutil.copyfile(input_file, output_file)


def round_to_n_nonzero(x: float, n: int = 2) -> float:
    """
    Round a number to n significant non-zero digits.
    
    Args:
        x (float): Number to round.
        n (int): Number of significant digits.
        
    Returns:
        float: Rounded number.
    """
    import math
    if x != x:  # NaN check
        return x
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
