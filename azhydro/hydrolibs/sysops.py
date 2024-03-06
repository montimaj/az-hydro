"""
Provides methods for different system operations.
Source: https://code.usgs.gov/map/wu/aiwum-2.0-hydromap_ml-mirror/-/blob/main/aiwum2/maplibs/sysops.py
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import os
import shutil
import sys
from glob import glob
from os.path import dirname, abspath
sys.path.insert(0, dirname(abspath(__file__)))


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
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)


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
    file_list = glob(input_dir + pattern)
    for f in file_list:
        file_name = f[f.rfind(os.sep) + 1:]
        outfile = f'{target_dir}{prefix}{file_name}'
        if verbose:
            print('Copying', f, 'to', outfile, '...')
        shutil.copyfile(f, outfile)
