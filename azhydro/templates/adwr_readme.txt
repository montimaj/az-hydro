================================================================
AZ-Hydro per-basin and per-subbasin water-use time series
Companion delivery for ADWR
================================================================

Pipeline: ML (XGBRF) + density-ratio partition + 6-component
         quadrature uncertainty quantification.  Trained on
         per-well ADWR meter records 1984-2024; predictions
         span 1896-2099 at 2 km grid resolution.

Source:  Zenodo deposit DOI 10.5281/zenodo.19057935 (version 1.2)
         (full citations listed below).


----------------------------------------------------------------
Citation
----------------------------------------------------------------

If you use this data delivery, please cite the Zenodo data
archive plus the relevant companion paper(s):

   Data archive (this Zenodo deposit):

   Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., Wogenstahl, C.,
   Tadych, D.E., & Meeker, L. (2026).  AZ-Hydro -- Historical and Projected
   Arizona Annual Water Use: Software, Input Data, Models,
   Raster and Well Package Predictions, and Validation at 2 km
   Resolution (1896-2099).  Zenodo.
   https://doi.org/10.5281/zenodo.19057935


   Sci Data data paper (in preparation):

   Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., Wogenstahl, C.,
   Tadych, D.E., & Meeker, L. (2026).  Historical and projected
   freshwater withdrawals and irrigation consumptive use for
   Arizona, 1896-2099.
   In prep. for Nature Scientific Data.


   AGU Earth's Future analysis paper (in preparation):

   Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., Wogenstahl, C.,
   Tadych, D.E., & Meeker, L. (2026).  Where Arizona's Water Goes: Declining
   Agricultural Dominance and Rising Urban Demand Drive a
   Two-Century Shift in Withdrawal Patterns (1896-2099).  In
   prep. for AGU Earth's Future.


----------------------------------------------------------------
What is in this folder
----------------------------------------------------------------

Sixteen CSV files: one per category per spatial level (basin or
subbasin).  Each file is a long-format time series covering 1896-
2099 with one row per (Year, Basin/Subbasin) combination.

Categories (8 total = 7 partition outputs + Irrigation CU):

   1. Total_Predicted     -- total annual water use (= GW + SW)
   2. Total_GW            -- total annual groundwater pumping
   3. Total_SW            -- total annual surface-water deliveries
   4. Irrigation_GW       -- irrigation groundwater pumping
   5. Irrigation_SW       -- irrigation surface-water deliveries
   6. Non_Irrigation_GW   -- non-irrigation groundwater pumping
   7. Non_Irrigation_SW   -- non-irrigation surface-water deliveries
   8. Irrigation_CU       -- irrigation consumptive use
                            (= Irrigation x Irrigation Efficiency)

Internal partition identities (hold at every basin and every year,
within rounding):

   Total_Predicted   = Total_GW + Total_SW
   Total_Predicted   = Irrigation + Non_Irrigation
   Total_GW          = Irrigation_GW + Non_Irrigation_GW
   Total_SW          = Irrigation_SW + Non_Irrigation_SW
   Irrigation_CU     = Irrigation x Irrigation Efficiency (USGS NHM
                       basin-level efficiency, applied per pixel)

Spatial levels:

   * Basin    -- 52 Arizona groundwater basins (statewide coverage
                 including AMAs, INAs, and unmanaged basins)
   * Subbasin -- 82 ADWR groundwater sub-basin polygons (further
                 subdivision of the 52 basins, mostly within the
                 AMAs/INAs)


----------------------------------------------------------------
File listing (16 total)
----------------------------------------------------------------

Basin level (~52 basins x 1896-2099 = up to ~10,600 rows each):

   Basin_Total_Predicted.csv
   Basin_Total_GW.csv
   Basin_Total_SW.csv
   Basin_Irrigation_GW.csv
   Basin_Irrigation_SW.csv
   Basin_Non_Irrigation_GW.csv
   Basin_Non_Irrigation_SW.csv
   Basin_Irrigation_CU.csv

Subbasin level (~82 subbasins x 1896-2099 = up to ~16,700 rows each):

   Subbasin_Total_Predicted.csv
   Subbasin_Total_GW.csv
   Subbasin_Total_SW.csv
   Subbasin_Irrigation_GW.csv
   Subbasin_Irrigation_SW.csv
   Subbasin_Non_Irrigation_GW.csv
   Subbasin_Non_Irrigation_SW.csv
   Subbasin_Irrigation_CU.csv

Note: Row counts vary by category.  The *_SW files list every
(basin, year) combination, with explicit zero rows where there is no
surface water.  The Total_Predicted, *_GW and Irrigation_CU files omit
(basin, year) combinations with no predicted withdrawal (for example,
AGUA FRIA has no rows before 1922).  A missing row means zero; it is
not missing data.


----------------------------------------------------------------
Column schema
----------------------------------------------------------------

Basin files (7 columns):

   Year            -- integer, 1896-2099
   Basin           -- ADWR basin name (e.g. "PHOENIX AMA",
                      "TUCSON AMA", "AGUA FRIA")
   Mean_Depth_mm   -- mean withdrawal depth over the basin's
                      active pixels, i.e. 2 km pixels with at least
                      5 mm of predicted withdrawal that year (mm)
   Mean_Depth_ft   -- same as Mean_Depth_mm but in feet
   Volume_m3       -- total annual volume in cubic meters
   Volume_AF       -- total annual volume in acre-feet
   Era             -- era label (Hindcast 1896-1983,
                      Historical 1984-2025, Projection 2026-2099)

Subbasin files (8 columns):

   Year, Subbasin, Mean_Depth_mm, Mean_Depth_ft, Volume_m3,
   Volume_AF, Era, Parent_Basin

   Where Parent_Basin links the subbasin to its parent ADWR basin
   (one of the 52 basin names used in the Basin files).  Subbasins
   outside the AMA/INA system list "Other" as their Parent_Basin.


----------------------------------------------------------------
Era definitions
----------------------------------------------------------------

   Hindcast   1896-1983   Before the ADWR meter record; model
                          retrodiction checked against USGS OFR
                          94-476 (1915-1990) and USGS Circular
                          statewide totals.
   Historical 1984-2025   ADWR meter era.  1984-2024 is the metered
                          training window; 2025 uses the latest
                          observed climate and land-use inputs.
   Projection 2026-2099   Forward predictions driven by the MACA v2
                          CMIP5 ensemble mean (20 GCMs x RCP 4.5 and
                          8.5), the USGS FORE-SCE land-use scenario
                          mode, USBR streamflow projections, and the
                          CAP shortage set by the 2027-2028 Operating
                          Guidelines, held through 2099 as an
                          assumption.


----------------------------------------------------------------
Loading the files
----------------------------------------------------------------

Python:

   import pandas as pd
   df = pd.read_csv('Basin_Total_GW.csv')
   df.query('Basin == "PHOENIX AMA"').plot(x='Year', y='Volume_AF')

R:

   df <- read.csv('Basin_Total_GW.csv')
   subset(df, Basin == "PHOENIX AMA")

Excel:

   File -> Open -> Basin_Total_GW.csv -> headers in row 1.
   Use AutoFilter on the Basin or Era columns.


----------------------------------------------------------------
Sanity checks
----------------------------------------------------------------

Within each file the following hold per row (matched on Year +
Basin, treating a missing row as zero), to rounding:

   Basin_Irrigation_GW.csv + Basin_Non_Irrigation_GW.csv
       == Basin_Total_GW.csv

   Basin_Total_GW.csv + Basin_Total_SW.csv
       == Basin_Total_Predicted.csv

Summing basins does NOT reproduce the statewide total exactly.  Each
basin total includes every 2 km pixel that touches the basin
polygon, so pixels on a shared boundary count in both basins, and
the sum over the 52 basins exceeds the statewide total by about
2-3 % (e.g. 2020 Total_GW: 3.28 MAF summed over basins vs 3.16 MAF
statewide).  For statewide totals use
Outputs/.../Annual_Summaries/Total_<category>.csv.


----------------------------------------------------------------
Source rasters and uncertainty
----------------------------------------------------------------

These CSVs are basin-aggregated time series.  The full per-pixel
prediction rasters with uncertainty bands (sigma, CV, SNR, lower
95% CI, upper 95% CI) and per-well Well Package (GeoParquet, four
unit conventions: mm / ft / m^3 / acre-feet) with uncertainty
disaggregation are in the Zenodo deposit at the DOI cited above.

Files of interest in the Zenodo deposit:

   az-hydro-headline.7z                ~8.7 GB
       Predicted_Rasters/, Total_GW_Rasters/, Total_SW_Rasters/,
       Irrigation_GW_Rasters/, Irrigation_SW_Rasters/, ... and
       Well_Package/ -- the per-pixel and per-well published
       products with full uncertainty.
       Recommended starting point.

   az-hydro-data.7z                    ~80 GB
       Full reproducibility archive (raw inputs + Step 2
       cross-validation + intermediate predictor stacks +
       per-component sigma rasters).


----------------------------------------------------------------
Methodology summary
----------------------------------------------------------------

Step 1.  XGBRF model trained on per-well ADWR meter records (1984-
         2024) using 22 predictors (climate, ET, effective
         precipitation, land-cover fractions and densities,
         irrigated and groundwater-irrigation fractions, soils,
         water-table depth, well, surface-water-rights and canal
         densities, watershed streamflow, coordinates) at 2 km
         grid resolution.

Step 2.  Density-ratio partition decomposes the total prediction
         into Irrigation/Non-Irrigation x GW/SW.  The few free
         partition parameters were tuned to statewide records:
         USGS OFR 94-476 (1915-1990), USGS Circulars (1950-2015),
         ADWR Annual Report totals and the 2019 ADWR shares.  The
         ML totals are never adjusted.

Step 3.  Six-component quadrature uncertainty:
         sigma_MACA   (5 GCMs; 2026-2099)
         sigma_Model  (10 XGBRF random seeds, t-corrected)
         sigma_Irr    (irrigated-area input; 1896-2025)
         sigma_LULC   (4 USGS FORE-SCE scenarios; 2026-2099)
         sigma_GW     (5 recent ADWR Well Registry snapshots,
                      t-corrected)
         sigma_USBR   (5 CMIP3 Upper Colorado streamflow members,
                      t-corrected; 1950-2099)
         Consumptive-use uncertainty (sigma_CU) is propagated
         from these through the irrigation efficiency.

Step 4.  Validation against USGS NHM (Haynes 2023, Martin 2025),
         USGS public-supply reanalysis (Luukkonen 2023), Reitz ET
         (Reitz 2023), CAP delivery records, SRP delivery records,
         and ADWR Annual Reports.


----------------------------------------------------------------
Validation summary
----------------------------------------------------------------

Statewide agency reconciliation (the partition was tuned to these
records, so they show calibration adequacy, not an independent test):

   2017 ADWR total           model 6.81 MAF vs ADWR 7.00 MAF (-2.7 %)
   2015 USGS GW withdrawal   model 2.96 MAF vs USGS 3.09 MAF (-4.1 %)
   2019 ADWR GW share        model 45.9 % vs ADWR 41 %
   2019 ADWR irrigation
   share                     model 74.2 % vs ADWR 72 %

Independent checks against USGS NHM, USGS Reitz, the USGS public-
supply reanalysis, and CAP and SRP delivery records are described
in the Sci Data paper.


----------------------------------------------------------------
License
----------------------------------------------------------------

CC-BY-4.0 (Creative Commons Attribution 4.0 International).
   https://creativecommons.org/licenses/by/4.0/

Source code (BSD 3-Clause "Revised") is available at
https://github.com/montimaj/az-hydro.


----------------------------------------------------------------
Contact
----------------------------------------------------------------

Sayantan Majumdar -- sayantan.majumdar@dri.edu
Desert Research Institute, Reno, NV, USA

For citations, see the "Citation" section above.
