"""
Main driver file for running the project.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

from hydrolibs.dataops import download_gee_data

if __name__ == '__main__':
    az_gw_basin = '../Data/Inputs/GW_Data/Groundwater_Basin.geojson'
    az_state = '../Data/Inputs/GW_Data/AZ.geojson'
    input_dir = '../Data/Inputs/'
    gcloud_project = 'azhydro'
    gcloud_bucket = 'azhydro'
    start_year = 1991
    end_year = 2023
    skip_download = False
    tile_size = 10000
    num_workers = 32
    download_gee_data(
        az_state,
        gcloud_project,
        gcloud_bucket,
        input_dir,
        start_year,
        end_year,
        skip_download,
        tile_size,
        num_workers
    )