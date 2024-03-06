# AZ-Hydro

Maintainers: [Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/) [sayantan.majumdar@dri.edu], [Ryan G. Smith](https://www.engr.colostate.edu/ce/ryan-g-smith/) [ryan.g.smith@colostate.edu]

<img src="../Readme_Figures/DRITaglineLogoTransparentBackground.png" height="45"/> &nbsp; <img src="../Readme_Figures/CSU-Signature-C-357.png" height="55"/> 


Note: This software has been successfully tested on [Alienware M17R1 2020](https://www.dell.com/en-us/gaming/alienware) (Windows 10 Home) and the [Apple MacBook Pro 2023](https://www.apple.com/macbook-pro/) (macOS Sonoma 14.3.1).

## Citations
Majumdar, S., Smith, R., Conway, B. D., & Lakshmi, V. (2022). Advancing remote sensing and machine learning‐driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals. Hydrological Processes, 36(11), e14757. https://doi.org/10.1002/hyp.14757


## Running the project

### 1. Download and install Anaconda/Miniconda
Either [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the Python 3 packages. 
It is recommended to install the latest version of Anaconda or miniconda (Python >= 3.10). If Anaconda or miniconda is already installed, skip this step. 

**For Windows users:** Once installed, open the Anaconda terminal (called Ananconda Prompt), and run ```conda init powershell``` to add ```conda``` to Windows PowerShell path.

**For Linux/Mac users:** Make sure ```conda``` is added to path. Typically, conda is automatically added to path after installation. It may be necessary to restart the current shell session to add conda to path.

The conda package manager can be updated by running the following command: ```conda update conda```

Anaconda is a Python distribution and environment manager. Miniconda is a free minimal installer for conda. These will help in installing the correct packages and Python version to run the codes.

### 2. Clone or download the repository

Download the repository from the compressed file link at the top right of the repository webpage, or clone the repository using Git.
Unzip all zipped files.  Several of the input datasets in this repository are zipped for efficient storage and must be unzipped before they can be used to run this project.

#### Repository disk space requirements
//TODO

### 3. Creating the conda environment and installing packages
Open Linux/Mac terminal or Windows PowerShell and run the following:
```
conda create -y -n azhydro python=3.11
conda activate azhydro
conda install -y -c conda-forge rioxarray geopandas lightgbm py-xgboost earthengine-api xee rasterstats seaborn openpyxl
conda install -y -c conda-forge dask-ml dask-jobqueue swifter
```

### 4. Google Earth Engine Authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) some of the predictor datasets from the GEE
data repository. After completing step 3, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). The Google Cloud CLI tools
may be required for this GEE authentication step. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk). You also have to create a gcloud project to use the GEE API. 

### 5. Running AZHydro
//TODO