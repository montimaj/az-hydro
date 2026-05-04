================================================================
AZ-Hydro per-basin and per-subbasin water-use time series
Companion delivery for ADWR
================================================================

Pipeline: ML (XGBRF) + density-ratio partition + 6-component
         quadrature uncertainty quantification.  Trained on
         per-well ADWR meter records 1984-2024; predictions
         span 1896-2099 at 2 km grid resolution.

Source:  Zenodo deposit DOI 10.5281/zenodo.19057936
         (full citations listed below).


----------------------------------------------------------------
Citation
----------------------------------------------------------------

If you use this data delivery, please cite the Zenodo data
archive plus the relevant companion paper(s):

   Data archive (this Zenodo deposit):

   Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., &
   Wogenstahl, C. (2026).  AZ-Hydro -- Historical and Projected
   Arizona Annual Water Use: Software, Input Data, Models,
   Raster and Well Package Predictions, and Validation at 2 km
   Resolution (1896-2099).  Zenodo.
   https://doi.org/10.5281/zenodo.19057936


   Sci Data data paper (in preparation):

   Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., &
   Wogenstahl, C. (2026).  Historical and projected groundwater/
   surface-water withdrawals, irrigation consumptive use, and
   pumping-induced surface water capture for Arizona, 1896-2099.
   In prep. for Nature Scientific Data.


   AGU Earth's Future analysis paper (in preparation):

   Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., &
   Wogenstahl, C. (2026).  Where Arizona's Water Goes: Declining
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

Internal partition identities (true at every basin and every year):

   Total_Predicted   = Total_GW + Total_SW
   Total_Predicted   = Irrigation + Non_Irrigation
   Total_GW          = Irrigation_GW + Non_Irrigation_GW
   Total_SW          = Irrigation_SW + Non_Irrigation_SW
   Irrigation_CU     = Irrigation x basin-mean Irrigation Efficiency

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

Note: Row counts vary by category.  Basins/subbasins outside an
irrigation footprint contribute no rows to Irrigation_* / CU files,
and basins outside the surface-water-rights envelope contribute no
rows to *_SW files in pre-rights years.  This is partition behavior,
not data loss: the omitted (basin, year, category) combinations are
structural zeros.


----------------------------------------------------------------
Column schema
----------------------------------------------------------------

Basin files (7 columns):

   Year            -- integer, 1896-2099
   Basin           -- ADWR basin name (e.g. "PHOENIX AMA",
                      "TUCSON AMA", "AGUA FRIA")
   Mean_Depth_mm   -- area-weighted mean withdrawal depth over
                      all pixels in the basin (mm)
   Mean_Depth_ft   -- same as Mean_Depth_mm but in feet
   Volume_m3       -- total annual volume in cubic meters
   Volume_AF       -- total annual volume in acre-feet
   Era             -- era label (Hindcast 1896-1979,
                      Historical 1980-2025, Projection 2026-2099)

Subbasin files (8 columns):

   Year, Subbasin, Mean_Depth_mm, Mean_Depth_ft, Volume_m3,
   Volume_AF, Era, Parent_Basin

   Where Parent_Basin links the subbasin to its parent ADWR basin
   (one of the 52 basin names used in the Basin files).  Subbasins
   outside the AMA/INA system list "Other" as their Parent_Basin.


----------------------------------------------------------------
Era definitions
----------------------------------------------------------------

   Hindcast   1896-1979   Pre-meter-era model retrodiction
                          calibrated against USGS Circulars
                          1950-1980 statewide totals.
   Historical 1980-2025   Period covered by (or directly adjacent
                          to) ADWR meter records.  1984-2024 is
                          the metered training window; 1980-1983
                          and 2025 lean on USGS Circulars and
                          ADWR Annual Reports for tuning anchors
                          and on the latest observed climate /
                          LULC inputs.
   Projection 2026-2099   Forward predictions driven by MACA-
                          downscaled CMIP5 climate (5 GCMs) and
                          USGS FORE-SCE LULC scenarios.


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

The following identities should hold to within partition-rounding
noise (sub-1% mismatch in some basins/years; aggregated totals
agree to >99.9%):

   Sum over basins of Basin_Total_GW.csv[Volume_AF, year=Y]
       == statewide annual Total_GW for year Y (in
       Outputs/.../Annual_Summaries/Total_GW.csv).

   Basin_Irrigation_GW.csv + Basin_Non_Irrigation_GW.csv
       == Basin_Total_GW.csv (per-row sum, matched on
       Year + Basin).

   Basin_Total_GW.csv + Basin_Total_SW.csv
       == Basin_Total_Predicted.csv (per-row sum).


----------------------------------------------------------------
Source rasters and uncertainty
----------------------------------------------------------------

These CSVs are basin-aggregated time series.  The full per-pixel
prediction rasters with uncertainty bands (sigma, CV, SNR, lower
95% CI, upper 95% CI) and per-well Well Package (GeoParquet, four
unit conventions: mm / ft / m^3 / acre-feet) with uncertainty
disaggregation are in the Zenodo deposit at the DOI cited above.

Files of interest in the Zenodo deposit:

   az-hydro-headline.7z                ~8.8 GB
       Predicted_Rasters/, Total_GW_Rasters/, Total_SW_Rasters/,
       Irrigation_GW_Rasters/, Irrigation_SW_Rasters/, ... and
       Well_Package/ -- the per-pixel and per-well published
       products with full uncertainty.
       Recommended starting point.

   az-hydro-data.7z                    ~74 GB
       Full reproducibility archive (raw inputs + Step 2
       cross-validation + intermediate predictor stacks +
       per-component sigma rasters).


----------------------------------------------------------------
Methodology summary
----------------------------------------------------------------

Step 1.  XGBRF model trained on per-well ADWR meter records (1984-
         2024) using 16 predictor bands (climate, ET, Peff,
         irrigation fraction, well density, canal density, water-
         rights density, etc.) at 2 km grid resolution.

Step 2.  Density-ratio partition decomposes the total prediction
         into Irrigation/Non-Irrigation x GW/SW using era-mapped
         factors anchored to USGS Circulars 1950-2015 and ADWR
         Annual Reports 2016-2024.

Step 3.  Six-component quadrature uncertainty:
         sigma_MACA   (5 GCMs)
         sigma_Model  (10 XGBRF random seeds, t-corrected)
         sigma_LULC   (4 USGS FORE-SCE scenarios)
         sigma_USBR   (5 CMIP3 Upper Colorado streamflow members,
                      t-corrected)
         sigma_GW     (5 recent ADWR Well Registry snapshots,
                      t-corrected)
         sigma_CU     (analytic propagation through Irrigation
                      Efficiency)

Step 4.  Validation against USGS NHM (Haynes 2023, Martin 2025),
         USGS public-supply reanalysis (Luukkonen 2023), Reitz ET
         (Reitz 2023), CAP delivery records, SRP delivery records,
         and ADWR Annual Reports.


----------------------------------------------------------------
Validation summary
----------------------------------------------------------------

Statewide-anchor matches (independent or partially-calibrated):

   2016 ADWR Total           model 6.72 MAF vs ADWR ~7.0 MAF
                             (within -0.28 MAF)
   2017 ADWR Total           model 6.81 MAF vs ADWR ~7.0 MAF
                             (within -0.19 MAF, model 95% CI
                             brackets the anchor)
   2017 ADWR GW share        model 44.9% vs ADWR 41% (within 4 pp)
   2015 USGS GW pumping      model 2.96 MAF vs USGS 3.09 MAF
                             (within -0.13 MAF)
   2019-20 ADWR irrigation
   share                     model 73.8% (2-year mean) vs ADWR 74%
                             (essentially exact)


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
