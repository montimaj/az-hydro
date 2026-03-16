import logging as _logging
from osgeo import gdal

_gdal_logger = _logging.getLogger('osgeo.gdal')

def _gdal_error_handler(err_class, err_num, err_msg):
    if err_class == gdal.CE_Warning:
        _gdal_logger.warning('GDAL warning %d: %s', err_num, err_msg)
    elif err_class >= gdal.CE_Failure:
        _gdal_logger.error('GDAL error %d: %s', err_num, err_msg)

gdal.PushErrorHandler(_gdal_error_handler)
gdal.UseExceptions()