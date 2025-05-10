import sys
from os.path import dirname, abspath
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()
sys.path.insert(0, dirname(abspath(__file__)))