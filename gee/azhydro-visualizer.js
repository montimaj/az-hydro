/**
 * AZ-Hydro Visualizer — interactive GEE app for exploring the 1896–2099
 * AZ water-use prediction stack at pixel, basin, and sub-basin scales.
 *
 * Paste into https://code.earthengine.google.com (or publish as a
 * Google App via Apps > Publish).
 *
 * Source: https://github.com/montimaj/az-hydro
 * Citation: Majumdar et al. (2026), Zenodo DOI 10.5281/zenodo.19057936
 *
 * ─────────────────────────────────────────────────────────────────────
 * What this app shows
 * ─────────────────────────────────────────────────────────────────────
 *   • A year slider (1896–2099) drives a single per-category raster
 *     layer, with optional overlays for basin / sub-basin / well
 *     boundaries.
 *   • A category dropdown picks the variable (Total Predicted, Total
 *     GW, Total SW, Irrigation/Non-Irrigation × GW/SW, Irrigation CU,
 *     Capture (Total / Irrigation / Non-Irrigation, augmented),
 *     Capture Fraction (Total / Irrigation / Non-Irrigation,
 *     dimensionless 0–1 share), OOD Probability, CAP scenarios).
 *   • A unit-convention dropdown picks Depth_mm / Depth_ft /
 *     Volume_m3 / Volume_AF where available.
 *   • A band selector picks which of the 6 augmented bands to view
 *     (prediction, σ, CV, SNR, lower 95 % CI, upper 95 % CI) for the
 *     per-category rasters.  OOD has a single band.
 *   • A "Compare" toggle in the sidebar splits the view into a
 *     side-by-side ui.SplitPanel (linked zoom + pan).  The MAIN
 *     category dropdown drives the LEFT map; a "Right-map category"
 *     dropdown (revealed on toggle) drives the RIGHT.  Year, unit,
 *     band, and CAP scenario/window apply to both.  Drag the wipe
 *     divider to compare.  Clicks on either map populate the time-
 *     series charts using THAT map's category.
 *   • CAP scenarios use a different UX — pick from 7 shortfall
 *     scenarios (Basic Coordination, DCP Tier 0–3, Tier 2a/b,
 *     Extreme Shortage) × 2 windows (2027–2060 mid-century, 2027–2099
 *     full horizon).  The year slider / unit / band controls are
 *     hidden because each CAP raster is a cumulative integral of
 *     ΔGW (AF) over the chosen window — no per-year time series.
 *   • Click anywhere on the map to populate four panels.  Every chart
 *     follows the unit dropdown, and every chart shows prediction +
 *     95 % CI envelope (when the category is augmented):
 *       (1) pixel time series at the click point (raw raster sample,
 *           2 km native scale; bands 1, 5, 6 = pred / lower / upper);
 *       (2) basin time series (mean over the basin containing the
 *           click — pred + mean lower/upper CI);
 *       (3) sub-basin time series (same, sub-basin scale);
 *       (4) nearest well's published Well_Package time series with
 *           prediction ± 95 % CI (capacity-disaggregated per-well
 *           value — differs from the pixel value when a 2 km cell
 *           contains multiple wells).  Pulled from
 *           projects/azhydro/assets/az-wu/Well_Package__<Cat>; AF
 *           values are converted client-side to the selected unit.
 *     The band dropdown is a MAP-only control (it picks which of the
 *     6 augmented bands to render); time-series charts always show
 *     prediction with a CI envelope.
 *     For CAP scenarios this becomes a 3-line readout (pixel + basin +
 *     sub-basin cumulative ΔGW) — no per-well row, since CAP has no
 *     Well_Package counterpart and the raster value at the well's
 *     pixel is already shown.
 *
 * ─────────────────────────────────────────────────────────────────────
 * Asset paths used by this app
 * ─────────────────────────────────────────────────────────────────────
 *   Vectors (already uploaded — see gee/Data/ADWR/):
 *     projects/azhydro/assets/az-wu/Groundwater_Basin
 *     projects/azhydro/assets/az-wu/ADWR_Groundwater_Subbasin
 *     projects/azhydro/assets/az-wu/Well_Registry_2024
 *
 *   ImageCollections (one per category × unit-convention; populate by
 *   running `python gee/generate_geeup_metadata.py` then `geeup
 *   upload` on each leaf dir under gee/Data/):
 *     projects/azhydro/assets/az-wu/Predicted_Rasters__Depth_mm
 *     projects/azhydro/assets/az-wu/Total_GW_Rasters__Depth_mm
 *     ... (etc., naming convention: <Dir>__<Unit>)
 *
 * ─────────────────────────────────────────────────────────────────────
 */

// ═════════════════════════════════════════════════════════════════════
//  Configuration
// ═════════════════════════════════════════════════════════════════════

var ASSET_ROOT = 'projects/azhydro/assets/az-wu';

// Category name -> display label and asset-prefix mapping.
// Entry order controls the dropdown order.
var CATEGORIES = [
  {key: 'Predicted_Rasters',         label: 'Total Predicted (GW+SW)',   augmented: true},
  {key: 'Total_GW_Rasters',          label: 'Total GW',                  augmented: true},
  {key: 'Total_SW_Rasters',          label: 'Total SW',                  augmented: true},
  {key: 'Irrigation_Rasters',        label: 'Irrigation (Total)',        augmented: true},
  {key: 'Irrigation_GW_Rasters',     label: 'Irrigation GW',             augmented: true},
  {key: 'Irrigation_SW_Rasters',     label: 'Irrigation SW',             augmented: true},
  {key: 'Non_Irrigation_Rasters',    label: 'Non-Irrigation (Total)',    augmented: true},
  {key: 'Non_Irrigation_GW_Rasters', label: 'Non-Irrigation GW',         augmented: true},
  {key: 'Non_Irrigation_SW_Rasters', label: 'Non-Irrigation SW',         augmented: true},
  {key: 'Irrigation_CU_Rasters',     label: 'Irrigation CU',             augmented: true},
  {key: 'Irrigation_GW_CU_Rasters',  label: 'Irrigation GW CU',          augmented: true},
  {key: 'Irrigation_SW_CU_Rasters',  label: 'Irrigation SW CU',          augmented: true},

  // Capture — augmented per-pixel pumping-induced SW capture
  // volumes, in mm/ft/m³/AF.  Same 6-band stack as the other
  // augmented categories.
  {key: 'Capture__Total_Rasters',
    label: 'Capture (Total)',                                          augmented: true},
  {key: 'Capture__Irrigation_Rasters',
    label: 'Capture — Irrigation',                                     augmented: true},
  {key: 'Capture__Non_Irrigation_Rasters',
    label: 'Capture — Non-Irrigation',                                 augmented: true},

  // Capture Fraction — dimensionless 0–1 share of pumping that
  // captures surface water (single band, no unit conventions).
  {key: 'Capture__Total_Fraction',
    label: 'Capture Fraction (Total)',                                 augmented: false,
    units: false, fraction: true},
  {key: 'Capture__Irrigation_Fraction',
    label: 'Capture Fraction — Irrigation',                            augmented: false,
    units: false, fraction: true},
  {key: 'Capture__Non_Irrigation_Fraction',
    label: 'Capture Fraction — Non-Irrigation',                        augmented: false,
    units: false, fraction: true},

  {key: 'OOD_Rasters',               label: 'OOD Probability',           augmented: false,
    units: false,  // single-band, no Depth_mm/Volume_AF subdirs
    subPath: ''},
  {key: 'CAP_Scenario__Pixel_Rasters',
    label: 'CAP Scenario (cumulative ΔGW, AF)',
    augmented: false,
    units: false,    // always AF
    cap: true},      // year slider doesn't apply — use scenario + window
];

// CAP scenario shortfalls (matches Raster_Maps/CAP_Scenario/Pixel_Rasters/* basenames).
// For each (scenario, window) the asset's system:index is
//   'CAP_Scenario_Pixel_<scenario>_cum_AF_<window>'
var CAP_SCENARIOS = [
  {key: 'Basic_Coordination_237kAF', label: 'Basic Coordination (237 kAF)'},
  {key: 'DCP_Tier0_192kAF_cut',      label: 'DCP Tier 0 (192 kAF cut)'},
  {key: 'DCP_Tier1_512kAF_cut',      label: 'DCP Tier 1 (512 kAF cut)'},
  {key: 'DCP_Tier2a_592kAF_cut',     label: 'DCP Tier 2a (592 kAF cut)'},
  {key: 'DCP_Tier2b_640kAF_cut',     label: 'DCP Tier 2b (640 kAF cut)'},
  {key: 'DCP_Tier3_720kAF_cut',      label: 'DCP Tier 3 (720 kAF cut)'},
  {key: 'Extreme_Shortage_0kAF',     label: 'Extreme Shortage (0 kAF)'},
];
var CAP_WINDOWS = [
  {key: '2027_2060', label: '2027–2060 (mid-century)'},
  {key: '2027_2099', label: '2027–2099 (full horizon)'},
];
var DEFAULT_CAP_SCENARIO = 'DCP_Tier1_512kAF_cut';
var DEFAULT_CAP_WINDOW = '2027_2099';

// ── Well_Package FeatureCollections (per-category) ──────────────────
// Each FC has ~300k well features, each with year-stamped properties:
//   <Cat>_AF_<year>          (prediction)
//   <Cat>_AF_sigma_<year>    (1σ)
// Category names follow the parquet — different from raster keys
// (no "_Rasters" suffix; "Predicted" → "Total").  OOD and CAP have
// no well-level analog.
function wellPackageCategoryFor(catKey) {
  var map = {
    'Predicted_Rasters':         'Total',
    'Total_GW_Rasters':          'Total_GW',
    'Total_SW_Rasters':          'Total_SW',
    'Irrigation_Rasters':        'Irrigation',
    'Irrigation_GW_Rasters':     'Irrigation_GW',
    'Irrigation_SW_Rasters':     'Irrigation_SW',
    'Non_Irrigation_Rasters':    'Non_Irrigation',
    'Non_Irrigation_GW_Rasters': 'Non_Irrigation_GW',
    'Non_Irrigation_SW_Rasters': 'Non_Irrigation_SW',
    'Irrigation_CU_Rasters':     'Irrigation_CU',
    'Irrigation_GW_CU_Rasters':  'Irrigation_GW_CU',
    'Irrigation_SW_CU_Rasters':  'Irrigation_SW_CU',
    // Capture (augmented rasters only — fractions have no
    // well-level analog in the parquet)
    'Capture__Total_Rasters':         'Total_Capture',
    'Capture__Irrigation_Rasters':    'Irrigation_Capture',
    'Capture__Non_Irrigation_Rasters': 'Non_Irrigation_Capture',
  };
  return map[catKey] || null;  // null for OOD / CAP / Capture Fraction
}

// Search radius for the nearest-well lookup (m).  20 km is generous on
// purpose: well markers visually cover tens of km of ground at typical
// zoom, so users routinely click on what *looks* like a well but is
// actually 5–15 km away on the ground.  Within the buffer we rank
// candidates by true distance to the click and surface the chosen
// well's distance in the chart title, so the user can judge whether the
// match is meaningful for their click.
var WELL_SEARCH_RADIUS_M = 20000;

// AF → unit conversion (well-level only — pixel rasters live in their
// own unit-specific ICs).  Pixel area = 2000² = 4,000,000 m².
//   1 AF = 1233.48 m³ → 0.30837 mm depth = 0.001012 ft depth
function unitConversionFromAF(unit) {
  switch (unit) {
    case 'Volume_AF': return {factor: 1.0,         label: 'AF / yr'};
    case 'Volume_m3': return {factor: 1233.48,     label: 'm³ / yr'};
    case 'Depth_mm':  return {factor: 0.30837,     label: 'mm / yr'};
    case 'Depth_ft':  return {factor: 0.0010117,   label: 'ft / yr'};
    default:          return {factor: 1.0,         label: 'AF / yr'};
  }
}

// Augmented per-category band order (matches the rasterops.py augment step)
var AUGMENTED_BANDS = [
  {key: 'b1', name: 'Prediction'},
  {key: 'b2', name: 'sigma (1σ)'},
  {key: 'b3', name: 'CV'},
  {key: 'b4', name: 'SNR'},
  {key: 'b5', name: 'Lower 95% CI'},
  {key: 'b6', name: 'Upper 95% CI'},
];

var UNITS = ['Depth_mm', 'Depth_ft', 'Volume_m3', 'Volume_AF'];

var YEAR_MIN = 1896;
var YEAR_MAX = 2099;
var DEFAULT_YEAR = 2024;
var DEFAULT_CATEGORY = 'Predicted_Rasters';
var DEFAULT_UNIT = 'Volume_AF';
var DEFAULT_BAND_KEY = 'b1';  // Prediction

// Color palettes — Spectral_r style for predictions, RdYlBu for σ/CV
var PALETTE_PREDICTION = [
  '00007F', '0000FF', '00FFFF', '7FFF7F', 'FFFF00', 'FF7F00', 'FF0000', '7F0000',
];
var PALETTE_SIGMA = ['ffffd9', 'edf8b1', 'c7e9b4', '7fcdbb', '41b6c4',
                     '1d91c0', '225ea8', '253494', '081d58'];
var PALETTE_CAP_DELTA = ['white', 'yellow', 'orange', 'red', 'darkred'];

// Default visualization stretch (overridden per-band via dropdown helpers below)
var VIS_DEFAULTS = {
  prediction_mm: {min: 0, max: 500, palette: PALETTE_PREDICTION},
  prediction_AF: {min: 0, max: 5000, palette: PALETTE_PREDICTION},
  sigma_mm:      {min: 0, max: 200, palette: PALETTE_SIGMA},
  sigma_AF:      {min: 0, max: 2000, palette: PALETTE_SIGMA},
  cv:            {min: 0, max: 1.5, palette: PALETTE_SIGMA},
  snr:           {min: 0, max: 5, palette: PALETTE_SIGMA},
  ood:           {min: 0, max: 1, palette: ['blue', 'cyan', 'green',
                                             'yellow', 'orange', 'red']},
  // Capture Fraction — dimensionless 0–1 share, same scale +
  // palette family as OOD probability.
  fraction:      {min: 0, max: 1, palette: ['ffffe0', 'fed976', 'feb24c',
                                             'fd8d3c', 'fc4e2a', 'b10026']},
  // CAP cumulative ΔGW (AF / pixel summed over the scenario window)
  cap_delta_AF:  {min: 0, max: 50000, palette: PALETTE_CAP_DELTA},
};

// ═════════════════════════════════════════════════════════════════════
//  Vector layers (load once)
// ═════════════════════════════════════════════════════════════════════

var basins = ee.FeatureCollection(ASSET_ROOT + '/Groundwater_Basin');
var subbasins = ee.FeatureCollection(ASSET_ROOT + '/ADWR_Groundwater_Subbasin');

// ── Wells: keep only consumptive-use wells ──────────────────────────
// The ADWR Well Registry has ~238 k wells, of which ~58 k (24 %) are
// purely non-consumptive — MONITORING, NO WATER USE, TEST, DEWATERING,
// REMEDIATION, OTHER - MINERAL EXPLORE, and combinations of these.
// They never pump for use, so they have no meaningful water-use time
// series and shouldn't be the "nearest well" surfaced on a map click
// or appear in the ADWR-Wells overlay.
//
// A well is considered consumptive if its WATER_USE string contains
// at least ONE consumptive keyword.  Mixed-use wells (e.g.
// "DOMESTIC, MONITORING") stay in: they pump for domestic use AND
// were also used for monitoring.  Pure "MONITORING, NO WATER USE"
// wells drop out because none of the keywords match.  Ambiguous codes
// (RESERVED, UNKNOWN, NO USE CODE ON NOI) are included on the
// conservative side, since they aren't explicitly non-consumptive.
var CONSUMPTIVE_USE_KEYWORDS = [
  'IRRIGATION', 'DOMESTIC', 'STOCK', 'MUNICIPAL', 'INDUSTRIAL',
  'COMMERCIAL', 'MINING', 'UTILITY', 'RECOVERY', 'RECREATION',
  'SUBDIVISION', 'DRAINAGE', 'OTHER - PRODUCTION',
  'RESERVED', 'UNKNOWN', 'NO USE CODE',
];
var consumptiveWellFilter = ee.Filter.or.apply(
  null,
  CONSUMPTIVE_USE_KEYWORDS.map(function(kw) {
    return ee.Filter.stringContains('WATER_USE', kw);
  })
);
var wells = ee.FeatureCollection(ASSET_ROOT + '/Well_Registry_2024')
              .filter(consumptiveWellFilter);

var capServiceArea = ee.FeatureCollection(ASSET_ROOT + '/CAP');
var az = ee.FeatureCollection('TIGER/2018/States').filter(ee.Filter.eq('STATEFP', '04'));

// ═════════════════════════════════════════════════════════════════════
//  Map instances — left + right for split-pane comparison
// ═════════════════════════════════════════════════════════════════════
//
//  Single mode (default): only `leftMap` is shown in ui.root.  All
//  controls drive it.
//
//  Split mode (Compare toggle on): leftMap + rightMap shown side by
//  side inside a `ui.SplitPanel` with a draggable wipe divider.  The
//  ui.Map.Linker keeps zoom + pan synchronized between the two maps.
//  The MAIN category dropdown drives the LEFT map; a separate "Right
//  map category" dropdown drives the right.  Year slider, unit, band,
//  and overlays are shared (both maps update on slider drag).
//
//  Click handler is bound to both — clicking either side populates
//  the time-series charts using THAT map's active category.
//
//  IMPORTANT: GEE's ui widget tree refuses to re-parent a `ui.Map`
//  once it has been attached to a `ui.SplitPanel` — the second
//  attach raises "Unable to set parent component".  And attaching
//  it directly to ui.root in single mode renders the map at the
//  half-pane width left by the SplitPanel.  The reliable workaround
//  is to RECREATE both maps on every Compare-toggle, copying the
//  current center/zoom across so visual state is preserved.
var leftMap, rightMap, mapLinker;
var splitActive = false;
function getActiveMaps() { return splitActive ? [leftMap, rightMap] : [leftMap]; }

/** Build fresh leftMap + rightMap instances and re-bind their
 * click handlers.  Called once at startup, then again on every
 * Compare-toggle so each layout gets clean (un-parented) widgets. */
function createMaps(centerLngLat, zoom) {
  leftMap  = ui.Map();
  rightMap = ui.Map();
  leftMap.setOptions('HYBRID');
  rightMap.setOptions('HYBRID');
  leftMap.style().set({cursor: 'crosshair', stretch: 'both'});
  rightMap.style().set({cursor: 'crosshair', stretch: 'both'});
  leftMap.onClick(function(coords) {
    handleClick(coords, catDropdown.getValue(),      'L');
  });
  rightMap.onClick(function(coords) {
    handleClick(coords, rightCatDropdown.getValue(), 'R');
  });
  mapLinker = ui.Map.Linker([leftMap, rightMap]);
  if (centerLngLat && zoom != null) {
    leftMap.setCenter(centerLngLat[0], centerLngLat[1], zoom);
    rightMap.setCenter(centerLngLat[0], centerLngLat[1], zoom);
  } else {
    leftMap.centerObject(az, 7);
    rightMap.centerObject(az, 7);
  }
}
createMaps();

// Split basins by AMA / INA / other.  EXACT name match — substring
// matching mis-classifies PINAL AMA as INA (substring "INA") and
// WESTERN MEXICAN DRAINAGE as INA (substring "INA" in "DRAINAGE").
// Names below are the canonical BASIN_NAME values from
// Data/Inputs/GW_Data/Groundwater_Basin/Groundwater_Basin.shp.
var AMA_BASIN_NAMES = [
  'DOUGLAS AMA', 'PHOENIX AMA', 'PINAL AMA', 'PRESCOTT AMA',
  'RANEGRAS PLAIN', 'SANTA CRUZ AMA', 'TUCSON AMA', 'WILLCOX AMA',
];
var INA_BASIN_NAMES = [
  'HARQUAHALA INA', 'HUALAPAI VALLEY INA', 'JOSEPH CITY INA',
];
var amaBasins   = basins.filter(ee.Filter.inList('BASIN_NAME', AMA_BASIN_NAMES));
var inaBasins   = basins.filter(ee.Filter.inList('BASIN_NAME', INA_BASIN_NAMES));
var otherBasins = basins.filter(
  ee.Filter.inList('BASIN_NAME',
                   AMA_BASIN_NAMES.concat(INA_BASIN_NAMES)).not()
);

// Style helpers for vector overlays.  Saturated colors so the lines
// stay readable over HYBRID (satellite + terrain) basemap imagery —
// dark grays disappear against the brown/green terrain, AND the
// stroke colours must NOT collide with PALETTE_PREDICTION /
// PALETTE_SIGMA / PALETTE_CAP_DELTA or the boundary lines vanish
// against the matching part of the raster.
//
// Color story:
//   • Other basins (41) – pale yellow (default basin "context" stroke)
//   • AMA basins (8)    – orange (regulated active-management areas)
//   • INA basins (3)    – purple (irrigation non-expansion areas;
//                         purple is absent from every raster palette)
//   • CAP service areas – Maricopa = red, Pima = blue, Pinal = green
//   • Sub-basins        – white (lighter, secondary)
//   • Wells             – hot pink point markers
var basinStyle      = {color: 'FFD600', fillColor: '00000000', width: 1.6};
var amaStyle        = {color: 'FF8C00', fillColor: '00000000', width: 2.2};
var inaStyle        = {color: '9C27B0', fillColor: '00000000', width: 2.2};
var subbasinStyle   = {color: 'FFFFFF', fillColor: '00000000', width: 0.9};
var wellStyle       = {color: 'FF1493', pointSize: 3, pointShape: 'circle'};
var capAreaColors = {
  MARICOPA: 'FF3030',  // bright red
  PIMA:     '1E90FF',  // dodger blue
  PINAL:    '32CD32',  // lime green
};

// ═════════════════════════════════════════════════════════════════════
//  Helpers — build asset paths + load IC + image-for-year
// ═════════════════════════════════════════════════════════════════════

/** Build the GEE asset ID for a category.  All four unit conventions
 * live in the SAME ImageCollection — the generator (`gee/generate_geeup_metadata.py`)
 * tags each image with a `unit` property so we filter on (year, unit)
 * to pick the right one. */
function assetIdFor(catKey) {
  return ASSET_ROOT + '/' + catKey;
}

/** Look up category metadata by key. */
function categoryMeta(catKey) {
  for (var i = 0; i < CATEGORIES.length; i++) {
    if (CATEGORIES[i].key === catKey) return CATEGORIES[i];
  }
  return null;
}

/** Get the Image for a specific year (and unit, when applicable) from
 * an ImageCollection. */
function imageForYear(ic, year, unit, hasUnits) {
  var f = ic.filter(ee.Filter.eq('year', year));
  if (hasUnits && unit) {
    f = f.filter(ee.Filter.eq('unit', unit));
  }
  return ee.Image(f.first()).clip(az);
}

/** Choose a default vis stretch given (category, unit, band). */
function defaultVis(catKey, unit, bandKey) {
  var meta = categoryMeta(catKey);
  if (meta && meta.cap) return VIS_DEFAULTS.cap_delta_AF;
  if (meta && meta.fraction) return VIS_DEFAULTS.fraction;
  if (catKey === 'OOD_Rasters') return VIS_DEFAULTS.ood;
  if (bandKey === 'b1') {
    return unit && unit.indexOf('AF') >= 0 ? VIS_DEFAULTS.prediction_AF
         : unit && unit.indexOf('m3') >= 0 ? VIS_DEFAULTS.prediction_AF  // m³ scale similar
         : VIS_DEFAULTS.prediction_mm;
  }
  if (bandKey === 'b2') {
    return unit && unit.indexOf('AF') >= 0 ? VIS_DEFAULTS.sigma_AF : VIS_DEFAULTS.sigma_mm;
  }
  if (bandKey === 'b3') return VIS_DEFAULTS.cv;
  if (bandKey === 'b4') return VIS_DEFAULTS.snr;
  // CI bands — same stretch as prediction
  return defaultVis(catKey, unit, 'b1');
}

/** Build the system:index for a CAP scenario raster.  Matches the
 * id_no column in gee/Data/CAP_Scenario/Pixel_Rasters/metadata.csv. */
function capImageId(scenarioKey, windowKey) {
  return 'CAP_Scenario_Pixel_' + scenarioKey + '_cum_AF_' + windowKey;
}

// ═════════════════════════════════════════════════════════════════════
//  UI panel
// ═════════════════════════════════════════════════════════════════════

var sidePanel = ui.Panel({
  style: {width: '380px', padding: '8px'},
});

sidePanel.add(ui.Label('AZ-Hydro Visualizer',
  {fontWeight: 'bold', fontSize: '20px', margin: '0 0 4px 0'}));
sidePanel.add(ui.Label('Arizona water-use predictions, 1896–2099, 2 km grid',
  {fontSize: '12px', color: '#555', margin: '0 0 12px 0'}));

// ── Category selector ────────────────────────────────────────────────
sidePanel.add(ui.Label('Category', {fontWeight: 'bold', margin: '8px 0 2px 0'}));
var catDropdown = ui.Select({
  items: CATEGORIES.map(function(c) { return {label: c.label, value: c.key}; }),
  value: DEFAULT_CATEGORY,
  onChange: function() {
    syncControlVisibility();
    populateStretchDefaults();
    refreshLayer();
  },
});
sidePanel.add(catDropdown);

// ── Unit-convention selector (hidden for CAP / OOD) ──────────────────
var unitLabel = ui.Label('Unit', {fontWeight: 'bold', margin: '8px 0 2px 0'});
sidePanel.add(unitLabel);
var unitDropdown = ui.Select({
  items: UNITS,
  value: DEFAULT_UNIT,
  onChange: function() { populateStretchDefaults(); refreshLayer(); },
});
sidePanel.add(unitDropdown);

// ── Band selector (hidden for non-augmented categories) ──────────────
var bandLabel = ui.Label('Band', {fontWeight: 'bold', margin: '8px 0 2px 0'});
sidePanel.add(bandLabel);
var bandDropdown = ui.Select({
  items: AUGMENTED_BANDS.map(function(b) { return {label: b.name, value: b.key}; }),
  value: DEFAULT_BAND_KEY,
  onChange: function() { populateStretchDefaults(); refreshLayer(); },
});
sidePanel.add(bandDropdown);

// ── Color stretch (manual override of the auto-selected min / max) ──
sidePanel.add(ui.Label('Color stretch (min / max)',
                       {fontWeight: 'bold', margin: '8px 0 2px 0'}));
var stretchMinBox = ui.Textbox({
  placeholder: 'min',
  style: {width: '100px', margin: '0 4px 0 0'},
  onChange: refreshLayer,
});
var stretchMaxBox = ui.Textbox({
  placeholder: 'max',
  style: {width: '100px'},
  onChange: refreshLayer,
});
sidePanel.add(ui.Panel(
  [stretchMinBox, stretchMaxBox],
  ui.Panel.Layout.Flow('horizontal')
));
sidePanel.add(ui.Label(
  'Auto-fills with the 2nd / 98th percentile of the current image ' +
  '(masking zeros) on category / unit / band / CAP change.  Edit ' +
  'either box to override; year-slider drags keep the active ' +
  'stretch so changes between years stay comparable.',
  {fontSize: '10px', color: '#777', margin: '0 0 4px 0'}
));

/** Refill the stretch textboxes with a DATA-DRIVEN default — the
 * 2nd / 98th percentile of the current image's pixel distribution
 * across AZ.  Called when the category / unit / band / CAP scenario
 * dropdowns change so the colour ramp always tracks the actual
 * value range and outliers don't blow out the legend.
 *
 * Sequence per call:
 *   1. Clear the boxes synchronously (so the immediate refreshLayer
 *      doesn't keep the previous category's percentile values).
 *   2. Build the source image for the current settings.
 *   3. reduceRegion(percentile([2, 98])) over AZ at 4 km scale
 *      (coarse-but-fast; AZ ~75 k pixels at 4 km).
 *   4. evaluate() — when the result arrives, populate the boxes and
 *      trigger another refreshLayer with the data-driven stretch.
 *
 * setValue(..., false) suppresses the textbox onChange so we don't
 * double-trigger refresh during the populate step. */
function populateStretchDefaults() {
  stretchMinBox.setValue('', false);
  stretchMaxBox.setValue('', false);
  computePercentileStretch(function(p2, p98) {
    var dv = defaultVis(catDropdown.getValue(),
                        unitDropdown.getValue(),
                        bandDropdown.getValue());
    var minVal = (p2  !== null && p2  !== undefined) ? roundForDisplay(p2)  : dv.min;
    var maxVal = (p98 !== null && p98 !== undefined) ? roundForDisplay(p98) : dv.max;
    stretchMinBox.setValue(String(minVal), false);
    stretchMaxBox.setValue(String(maxVal), false);
    refreshLayer();
  });
}

/** Async helper: compute the 2nd / 98th percentile of the current
 * layer's pixel values across AZ.  Calls callback(p2, p98) on
 * success, or callback(null, null) on failure / no data.  Year is
 * pinned to DEFAULT_YEAR so the stretch stays consistent as the
 * user drags the year slider — recomputed only on category /
 * unit / band / CAP-scenario change. */
function computePercentileStretch(callback) {
  var catKey  = catDropdown.getValue();
  var unit    = unitDropdown.getValue();
  var bandKey = bandDropdown.getValue();
  var meta    = categoryMeta(catKey);

  var ic = ee.ImageCollection(assetIdFor(catKey));
  var img;

  if (meta && meta.cap) {
    var sc  = capScenarioDropdown.getValue();
    var win = capWindowDropdown.getValue();
    img = ee.Image(ic.filter(ee.Filter.eq('system:index',
                                          capImageId(sc, win))).first())
              .clip(az).select(0);
  } else {
    var hasUnits = !(meta && meta.units === false);
    img = imageForYear(ic, DEFAULT_YEAR, unit, hasUnits);
    img = (meta && meta.augmented) ? img.select(bandIndex(bandKey))
                                   : img.select(0);
  }

  // Mask zeros so the percentiles reflect the non-zero data range
  // (matches the masked render in renderLayerToMap).
  img = img.updateMask(img.neq(0)).rename('v');

  var stats = img.reduceRegion({
    reducer:    ee.Reducer.percentile([2, 98]),
    geometry:   az.geometry(),
    scale:      4000,
    maxPixels:  1e9,
    bestEffort: true,
    tileScale:  4,
  });

  stats.evaluate(function(d, err) {
    if (err || !d) { callback(null, null); return; }
    callback(d.v_p2, d.v_p98);
  });
}

/** Round a number to ~3 significant figures for display in the
 * stretch textboxes / legend (e.g., 487.31 → 487, 0.0732 → 0.0732). */
function roundForDisplay(v) {
  if (v === null || v === undefined) return v;
  var n = Number(v);
  if (n === 0) return 0;
  return Number(n.toPrecision(3));
}

// ── Year slider (hidden for CAP — single cumulative window per asset) ──
var yearLabel = ui.Label('Year', {fontWeight: 'bold', margin: '8px 0 2px 0'});
sidePanel.add(yearLabel);
var yearSlider = ui.Slider({
  min: YEAR_MIN, max: YEAR_MAX, value: DEFAULT_YEAR, step: 1,
  style: {stretch: 'horizontal'},
  onChange: refreshLayer,
});
sidePanel.add(yearSlider);

// ── CAP-only: scenario + window selectors (shown only for CAP cat) ──
var capScenarioLabel = ui.Label('CAP scenario',
                                {fontWeight: 'bold', margin: '8px 0 2px 0'});
sidePanel.add(capScenarioLabel);
var capScenarioDropdown = ui.Select({
  items: CAP_SCENARIOS.map(function(s) { return {label: s.label, value: s.key}; }),
  value: DEFAULT_CAP_SCENARIO,
  onChange: function() { populateStretchDefaults(); refreshLayer(); },
});
sidePanel.add(capScenarioDropdown);

var capWindowLabel = ui.Label('CAP window',
                              {fontWeight: 'bold', margin: '8px 0 2px 0'});
sidePanel.add(capWindowLabel);
var capWindowDropdown = ui.Select({
  items: CAP_WINDOWS.map(function(w) { return {label: w.label, value: w.key}; }),
  value: DEFAULT_CAP_WINDOW,
  onChange: function() { populateStretchDefaults(); refreshLayer(); },
});
sidePanel.add(capWindowDropdown);

// ── Compare (split-pane) controls ────────────────────────────────────
sidePanel.add(ui.Label('Compare', {fontWeight: 'bold',
                                    margin: '12px 0 2px 0'}));
var compareToggle = ui.Checkbox({
  label: 'Side-by-side comparison',
  value: false,
  onChange: function(v) {
    splitActive = v;
    setRootLayout();
    syncControlVisibility();
    refreshLayer();
  },
});
sidePanel.add(compareToggle);

var rightCatLabel = ui.Label('Right-map category',
                              {fontSize: '11px', color: '#555',
                               margin: '4px 0 2px 0'});
sidePanel.add(rightCatLabel);
var rightCatDropdown = ui.Select({
  items: CATEGORIES.map(function(c) { return {label: c.label, value: c.key}; }),
  value: 'Total_SW_Rasters',  // deliberately different from default left
  onChange: function() { syncControlVisibility(); refreshLayer(); },
});
sidePanel.add(rightCatDropdown);

// Right-side CAP scenario + window — only meaningful when split
// mode is active AND the right map shows a CAP category.  This
// lets the user wipe between two different CAP shortfall scenarios
// or windows on the same map.
var rightCapScenarioLabel = ui.Label('Right-map CAP scenario',
                                      {fontSize: '11px', color: '#555',
                                       margin: '4px 0 2px 0'});
sidePanel.add(rightCapScenarioLabel);
var rightCapScenarioDropdown = ui.Select({
  items: CAP_SCENARIOS.map(function(s) { return {label: s.label, value: s.key}; }),
  // Default to a different scenario than the left so the side-by-side
  // is meaningful out of the box.
  value: 'Extreme_Shortage_0kAF',
  onChange: function() { populateStretchDefaults(); refreshLayer(); },
});
sidePanel.add(rightCapScenarioDropdown);

var rightCapWindowLabel = ui.Label('Right-map CAP window',
                                    {fontSize: '11px', color: '#555',
                                     margin: '4px 0 2px 0'});
sidePanel.add(rightCapWindowLabel);
var rightCapWindowDropdown = ui.Select({
  items: CAP_WINDOWS.map(function(w) { return {label: w.label, value: w.key}; }),
  value: DEFAULT_CAP_WINDOW,
  onChange: function() { populateStretchDefaults(); refreshLayer(); },
});
sidePanel.add(rightCapWindowDropdown);

sidePanel.add(ui.Label(
  'Year, unit, band apply to both maps.  Each map has its own CAP ' +
  'scenario / window dropdowns when set to a CAP category.  Use the ' +
  'divider to wipe between layers.',
  {fontSize: '10px', color: '#777', margin: '0 0 6px 0'}
));

/** Show/hide control rows based on the selected categories on each
 * active map.  In split mode each map has its own CAP scenario /
 * window dropdowns, and year / unit / band stay visible if EITHER
 * map needs them (only hidden when both maps are CAP). */
function syncControlVisibility() {
  var leftMeta  = categoryMeta(catDropdown.getValue());
  var rightMeta = splitActive ? categoryMeta(rightCatDropdown.getValue())
                              : null;

  var leftIsCap     = !!(leftMeta  && leftMeta.cap);
  var rightIsCap    = !!(rightMeta && rightMeta.cap);
  var leftHasUnits  = !(leftMeta  && leftMeta.units === false);
  var rightHasUnits = !(rightMeta && rightMeta.units === false);
  var leftAugmented = !!(leftMeta  && leftMeta.augmented);
  var rightAugmented = !!(rightMeta && rightMeta.augmented);

  // Year / unit / band shown if AT LEAST ONE active map needs them
  // (i.e., is non-CAP).  In single mode "active" = leftMap only.
  var anyNeedsYear = !leftIsCap || (splitActive && !rightIsCap);
  var anyHasUnits  = (!leftIsCap && leftHasUnits)
                     || (splitActive && !rightIsCap && rightHasUnits);
  var anyAugmented = (!leftIsCap && leftAugmented)
                     || (splitActive && !rightIsCap && rightAugmented);

  yearLabel.style().set('shown',  anyNeedsYear);
  yearSlider.style().set('shown', anyNeedsYear);
  unitLabel.style().set('shown',  anyHasUnits);
  unitDropdown.style().set('shown', anyHasUnits);
  bandLabel.style().set('shown',  anyAugmented);
  bandDropdown.style().set('shown', anyAugmented);

  // Left-side CAP scenario/window shown when LEFT map is CAP
  capScenarioLabel.style().set('shown',    leftIsCap);
  capScenarioDropdown.style().set('shown', leftIsCap);
  capWindowLabel.style().set('shown',      leftIsCap);
  capWindowDropdown.style().set('shown',   leftIsCap);

  // Right-map controls visible only in split mode
  rightCatLabel.style().set('shown',    splitActive);
  rightCatDropdown.style().set('shown', splitActive);

  // Right-side CAP scenario/window shown when split AND right is CAP
  rightCapScenarioLabel.style().set('shown',    splitActive && rightIsCap);
  rightCapScenarioDropdown.style().set('shown', splitActive && rightIsCap);
  rightCapWindowLabel.style().set('shown',      splitActive && rightIsCap);
  rightCapWindowDropdown.style().set('shown',   splitActive && rightIsCap);
}

// ── Overlay toggles ──────────────────────────────────────────────────
sidePanel.add(ui.Label('Overlays', {fontWeight: 'bold', margin: '12px 0 2px 0'}));
var basinToggle = ui.Checkbox({label: 'GW Basins (52, AMA/INA highlighted)',
                                value: true, onChange: refreshOverlays});
var subbasinToggle = ui.Checkbox({label: 'Sub-basins (82)', value: false,
                                   onChange: refreshOverlays});
var capAreaToggle = ui.Checkbox({label: 'CAP-eligible counties (Maricopa / Pima / Pinal)',
                                  value: true, onChange: refreshOverlays});
var wellToggle = ui.Checkbox({label: 'ADWR Wells (consumptive use only)', value: false,
                               onChange: refreshOverlays});
sidePanel.add(basinToggle);
sidePanel.add(subbasinToggle);
sidePanel.add(capAreaToggle);
sidePanel.add(wellToggle);

// Vector-overlay legend (color swatches for whatever is currently on)
var vectorLegendPanel = ui.Panel({
  style: {padding: '4px 0 0 0'},
});
sidePanel.add(vectorLegendPanel);

// ── Charts panel (populated on map click) ─────────────────────────────
sidePanel.add(ui.Label('Time series', {fontWeight: 'bold',
                                        margin: '12px 0 2px 0'}));
sidePanel.add(ui.Label('Click anywhere on the map to plot.',
                       {fontSize: '11px', color: '#555'}));
var chartsPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('vertical'),
  style: {padding: '4px'},
});
sidePanel.add(chartsPanel);

// ── Legend (below charts) ────────────────────────────────────────────
sidePanel.add(ui.Label('Legend', {fontWeight: 'bold', margin: '12px 0 2px 0'}));
var legendPanel = ui.Panel();
sidePanel.add(legendPanel);

// ── Footer ───────────────────────────────────────────────────────────
sidePanel.add(ui.Label('—', {margin: '12px 0 4px 0', color: '#aaa'}));
sidePanel.add(ui.Label(
  'Citation: Majumdar et al. (2026), Zenodo DOI 10.5281/zenodo.19057936',
  {fontSize: '10px', color: '#777'}
));
sidePanel.add(ui.Label(
  'Source: github.com/montimaj/az-hydro',
  {fontSize: '10px', color: '#777'},
  'https://github.com/montimaj/az-hydro'
));

// ui.root composition handled by setRootLayout() at initial paint
// time so the same code path covers single-map and split-pane modes.

// ═════════════════════════════════════════════════════════════════════
//  Layer management
// ═════════════════════════════════════════════════════════════════════

function bandIndex(bandKey) {
  for (var i = 0; i < AUGMENTED_BANDS.length; i++) {
    if (AUGMENTED_BANDS[i].key === bandKey) return i;
  }
  return 0;
}

function refreshLayer() {
  // Render LEFT map (always; this is the primary view)
  var leftInfo = renderLayerToMap(leftMap, catDropdown.getValue(), 'L');
  // Render RIGHT map only when comparing
  if (splitActive) {
    renderLayerToMap(rightMap, rightCatDropdown.getValue(), 'R');
  }
  refreshOverlays();
  // Legend reflects the LEFT map's layer (the primary).  In split
  // mode the right map's layer name appears in its own layer panel.
  refreshLegend(leftInfo.layerName, leftInfo.vis, leftInfo.unitLabel);
}

/** Render the data layer for one map.  Returns {layerName, vis} so
 * the caller can wire it into the legend.  Driven entirely by sidebar
 * controls — `catKey` is parameterised so the same logic powers the
 * left map (catDropdown) and the right map (rightCatDropdown) in
 * split mode. */
function renderLayerToMap(targetMap, catKey, side) {
  targetMap.layers().reset();
  var unit = unitDropdown.getValue();
  var bandKey = bandDropdown.getValue();
  var year = yearSlider.getValue();

  var meta = categoryMeta(catKey);
  var hasUnits = !(meta && meta.units === false);

  var assetId = assetIdFor(catKey);
  var ic = ee.ImageCollection(assetId);
  var img, layerName, vis, bandLabelText;

  if (meta && meta.cap) {
    // Each map has its own CAP scenario / window dropdowns.  Left
    // uses the primary capScenarioDropdown / capWindowDropdown;
    // right uses rightCapScenario* (only meaningful in split mode).
    var sc  = (side === 'R') ? rightCapScenarioDropdown.getValue()
                             : capScenarioDropdown.getValue();
    var win = (side === 'R') ? rightCapWindowDropdown.getValue()
                             : capWindowDropdown.getValue();
    var imgId = capImageId(sc, win);
    var capImg = ee.Image(ic.filter(ee.Filter.eq('system:index', imgId)).first())
                     .clip(az).select(0);
    // Mask cum-ΔGW = 0 pixels so non-impacted areas drop out cleanly
    img = capImg.updateMask(capImg.neq(0));
    var scLabel = (function() {
      for (var i = 0; i < CAP_SCENARIOS.length; i++) {
        if (CAP_SCENARIOS[i].key === sc) return CAP_SCENARIOS[i].label;
      }
      return sc;
    })();
    var winLabel = win.replace('_', '–');
    vis = defaultVis(catKey, unit, bandKey);
    bandLabelText = 'cum ΔGW (AF)';
    layerName = meta.label + ' — ' + scLabel + ', ' + winLabel;
  } else {
    // Pull the full source image once so we can mask using the
    // PREDICTION band even when the user is viewing σ / CV / SNR /
    // CI bands.  This avoids hiding pixels where the lower-CI band
    // happens to clamp to 0 even though the prediction is non-zero.
    var srcImg = imageForYear(ic, year, unit, hasUnits);
    var displayBand;
    if (meta && meta.augmented) {
      displayBand = srcImg.select(bandIndex(bandKey));
    } else {
      displayBand = srcImg.select(0);
    }
    var maskBand = srcImg.select(0);  // prediction (or sole band for OOD)
    img = displayBand.updateMask(maskBand.neq(0));
    vis = defaultVis(catKey, unit, bandKey);
    bandLabelText = (meta && meta.augmented)
        ? AUGMENTED_BANDS[bandIndex(bandKey)].name
        : 'value';
    layerName = (meta ? meta.label : catKey) +
                ' [' + bandLabelText + '] — ' + year;
  }
  // Apply user-typed colour-stretch override (if any).  Empty box or
  // non-numeric input falls back to the default vis range.
  var userMin = parseFloat(stretchMinBox.getValue());
  var userMax = parseFloat(stretchMaxBox.getValue());
  if (!isNaN(userMin) || !isNaN(userMax)) {
    vis = {
      min:     isNaN(userMin) ? vis.min : userMin,
      max:     isNaN(userMax) ? vis.max : userMax,
      palette: vis.palette,
    };
  }
  // Tag each side so the GEE layer panel can disambiguate.  Default
  // raster opacity = 0.5 so the HYBRID basemap (satellite + terrain
  // labels) shows through; users can drag the per-layer slider in
  // the Map's Layers menu to fade towards 0 (basemap only) or 1
  // (raster only).
  var sidePrefix = (side === 'R') ? '[R] ' : (splitActive ? '[L] ' : '');
  targetMap.addLayer(img, vis, sidePrefix + layerName, true, 0.5);
  // Surface the unit string for the colour-bar legend (CV / SNR /
  // OOD are dimensionless; everything else carries a pretty unit).
  var unitForLegend = '';
  if (meta && meta.cap)                 unitForLegend = 'AF (cumulative)';
  else if (meta && meta.fraction)        unitForLegend = 'fraction (0–1)';
  else if (meta && meta.units === false) unitForLegend = '';   // OOD
  else if (bandKey === 'b3' || bandKey === 'b4') unitForLegend = '';  // CV / SNR
  else                                   unitForLegend = unitConversionFromAF(unit).label;
  return {
    layerName: sidePrefix + layerName,
    vis:       vis,
    unitLabel: unitForLegend,
  };
}

function refreshOverlays() {
  // Apply overlays to every active map so the wipe-comparison stays
  // visually consistent.  Layer ordering on each map: data layer at
  // index 0, overlays above.
  var maps = getActiveMaps();
  var legendItems = [];

  for (var mi = 0; mi < maps.length; mi++) {
    var m = maps[mi];
    while (m.layers().length() > 1) {
      m.layers().remove(m.layers().get(m.layers().length() - 1));
    }

    if (basinToggle.getValue()) {
      // Three styled sub-layers so AMA / INA / regular basins are
      // visually distinct.  GEE Map layer ordering: regular first
      // (thinnest), then AMA + INA on top so their thicker strokes
      // win when basins overlap.
      m.addLayer(otherBasins.style(basinStyle), {}, 'GW Basins (other)');
      m.addLayer(amaBasins.style(amaStyle),     {}, 'AMA basins');
      m.addLayer(inaBasins.style(inaStyle),     {}, 'INA basins');
    }
    if (subbasinToggle.getValue()) {
      m.addLayer(subbasins.style(subbasinStyle), {}, 'Sub-basins');
    }
    if (capAreaToggle.getValue()) {
      // The three polygons in projects/azhydro/assets/az-wu/CAP are
      // COUNTY BOUNDARIES (Maricopa / Pinal / Pima — the three AZ
      // counties statutorily eligible for CAP delivery), not the
      // canal-fed delivery footprint.  Render as dashed outlines
      // only (no fill) so the rendering doesn't visually claim the
      // whole county is "served" — the actual CAP delivery area is
      // a much smaller subset inside those counties.
      var capNames = ['MARICOPA', 'PIMA', 'PINAL'];
      for (var i = 0; i < capNames.length; i++) {
        var name = capNames[i];
        var color = capAreaColors[name];
        var feat = capServiceArea.filter(ee.Filter.eq('NAME', name));
        m.addLayer(
          feat.style({color: color, fillColor: '00000000',
                      width: 2.8, lineType: 'dashed'}),
          {}, 'CAP-eligible county — ' + name
        );
      }
    }
    if (wellToggle.getValue()) {
      m.addLayer(wells.style(wellStyle), {}, 'ADWR Wells');
    }
  }

  // Build legend items once (color/label invariant across maps)
  if (basinToggle.getValue()) {
    legendItems.push({color: basinStyle.color, label: 'GW basin (41 other)'});
    legendItems.push({color: amaStyle.color,   label: 'AMA basin (8 active mgmt areas)'});
    legendItems.push({color: inaStyle.color,   label: 'INA basin (3 irrig. non-expansion)'});
  }
  if (subbasinToggle.getValue()) {
    legendItems.push({color: subbasinStyle.color, label: 'Sub-basin'});
  }
  if (capAreaToggle.getValue()) {
    legendItems.push({color: capAreaColors.MARICOPA, label: 'CAP-eligible county: Maricopa'});
    legendItems.push({color: capAreaColors.PIMA,     label: 'CAP-eligible county: Pima'});
    legendItems.push({color: capAreaColors.PINAL,    label: 'CAP-eligible county: Pinal'});
  }
  if (wellToggle.getValue()) {
    legendItems.push({color: wellStyle.color, label: 'ADWR well'});
  }

  refreshVectorLegend(legendItems);
}

/** Swap ui.root contents between single-map and split-pane layouts.
 * Called whenever the Compare toggle changes.
 *
 * GEE's widget model refuses to re-parent a `ui.Map` once it's been
 * inside a `ui.SplitPanel` — both wrapping it in a new Panel and
 * adding it directly to ui.root mis-render afterwards (either an
 * "Unable to set parent component" error or a half-pane-width map).
 * So instead of trying to re-attach the same map instance, we
 * RECREATE both maps on every toggle (cheap), preserving the
 * current center + zoom so visual continuity is maintained. */
var rootLayoutInitialized = false;
function setRootLayout() {
  // Capture current view state from the previous leftMap so the
  // wipe-toggle preserves visual continuity.  Skipped on the FIRST
  // call (initial paint) — at that point centerObject(az, 7) inside
  // createMaps is still in flight, so getCenter() returns the
  // uninitialized default (≈ 0/0), and feeding that back into
  // createMaps() would short-circuit the AZ centering.
  var savedCenter = null;
  var savedZoom = null;
  if (rootLayoutInitialized && leftMap) {
    try {
      var c = leftMap.getCenter();
      savedCenter = [c.coordinates().get(0).getInfo(),
                     c.coordinates().get(1).getInfo()];
      savedZoom = leftMap.getZoom();
    } catch (e) {
      savedCenter = null;
      savedZoom = null;
    }
  }

  ui.root.widgets().reset([sidePanel]);
  createMaps(savedCenter, savedZoom);
  rootLayoutInitialized = true;

  if (splitActive) {
    var sp = ui.SplitPanel({
      firstPanel:  leftMap,
      secondPanel: rightMap,
      orientation: 'horizontal',
      wipe:        true,
      style:       {stretch: 'both'},
    });
    ui.root.add(sp);
  } else {
    ui.root.add(leftMap);
  }
}

/** Rebuild the vector-overlay legend from the supplied list of
 * {color, label} entries.  Empty list = empty legend.
 *
 * The swatch is an empty ui.Label with explicit width/height +
 * backgroundColor, since a whitespace-only Label with default sizing
 * can render as a near-invisible strip in the GEE Code Editor — and
 * a heavy border makes the swatch's perceived colour skew towards
 * the border colour.  A thin pale border keeps the block crisp
 * without polluting its hue. */
function refreshVectorLegend(items) {
  vectorLegendPanel.clear();
  if (!items || items.length === 0) return;
  vectorLegendPanel.add(ui.Label('Map overlays',
    {fontSize: '11px', fontWeight: 'bold', margin: '4px 0 2px 0'}));
  for (var i = 0; i < items.length; i++) {
    var swatch = ui.Label('', {
      backgroundColor: '#' + items[i].color,
      width:           '22px',
      height:          '14px',
      margin:          '1px 8px 1px 0',
      border:          '1px solid #ccc',
    });
    var label = ui.Label(items[i].label,
      {fontSize: '11px', margin: '0 0 1px 0'});
    vectorLegendPanel.add(ui.Panel(
      [swatch, label],
      ui.Panel.Layout.Flow('horizontal')
    ));
  }
}

function refreshLegend(title, vis, unitLabel) {
  legendPanel.clear();
  // Title with unit appended in parens (when applicable) so the
  // colour-bar reads as a fully-qualified value axis.
  var fullTitle = title + (unitLabel ? '   (' + unitLabel + ')' : '');
  legendPanel.add(ui.Label(fullTitle, {fontSize: '11px',
                                        fontWeight: 'bold'}));
  if (!vis.palette) return;

  // Color ramp thumbnail
  var thumb = ui.Thumbnail({
    image: ee.Image.pixelLonLat().select('longitude')
        .multiply((vis.max - vis.min) / 100).add(vis.min),
    params: {
      bbox: [0, 0, 100, 8],
      dimensions: '300x12',
      format: 'png',
      min: vis.min, max: vis.max,
      palette: vis.palette,
    },
    style: {stretch: 'horizontal', margin: '0 0 2px 0'},
  });
  legendPanel.add(thumb);

  // Min / max labels with unit suffix (cleaner than repeating it on
  // both ends — show on the right only so the unit reads once).
  var minLabel = String(vis.min);
  var maxLabel = String(vis.max) + (unitLabel ? ' ' + unitLabel : '');
  legendPanel.add(ui.Panel(
    [ui.Label(minLabel, {fontSize: '10px'}),
     ui.Label(maxLabel, {fontSize: '10px',
                          textAlign: 'right',
                          stretch: 'horizontal'})],
    ui.Panel.Layout.Flow('horizontal')
  ));
}

// ═════════════════════════════════════════════════════════════════════
//  Click handler — pixel + basin + sub-basin time series
// ═════════════════════════════════════════════════════════════════════
// onClick handlers are attached inside createMaps() — re-bound on
// every Compare-toggle since the map instances are recreated.
// Click on leftMap uses catDropdown; click on rightMap (only shown
// in split mode) uses rightCatDropdown.

function handleClick(coords, catKey, side) {
  var pt = ee.Geometry.Point([coords.lon, coords.lat]);
  chartsPanel.clear();
  chartsPanel.add(ui.Label('Loading time series...', {color: '#777'}));

  var unit = unitDropdown.getValue();
  var bandKey = bandDropdown.getValue();
  var meta = categoryMeta(catKey);

  // CAP scenarios are cumulative (single image per scenario × window),
  // so a time-series chart isn't meaningful — show pixel + basin +
  // sub-basin totals instead.  Pass `side` so capClickReadout reads
  // the correct (left vs right) CAP scenario/window dropdowns.
  if (meta && meta.cap) {
    capClickReadout(pt, coords, side);
    return;
  }

  // Resolve meta + asset + IC, then filter to the chosen unit (when applicable)
  var hasUnits = !(meta && meta.units === false);
  var assetId = assetIdFor(catKey);
  var ic = ee.ImageCollection(assetId);
  if (hasUnits) {
    ic = ic.filter(ee.Filter.eq('unit', unit));
  }

  // Time-series ICs always show prediction + 95% CI envelope (when
  // augmented).  The band dropdown is a MAP-only control — it picks
  // which of the 6 augmented bands to render on the map.  Charts
  // always show the prediction line with a CI envelope so the
  // uncertainty context travels with every click.
  var hasCi = !!(meta && meta.augmented);
  var icSelected;
  if (hasCi) {
    // Bands: 0 = Prediction, 4 = Lower 95% CI, 5 = Upper 95% CI.
    // GEE's ui.Chart.image.* sorts series ALPHABETICALLY by band
    // name, so the rendered series indices are
    //   0 = Lower 95% CI  ('L')
    //   1 = Prediction    ('P')
    //   2 = Upper 95% CI  ('U')
    // The ciSeriesOpts() function below mirrors that ordering so
    // styling lands on the correct line — see its body.
    //
    // unmask(0) is critical: source rasters mask zero-pumping
    // pixels, so without it the chart simply skips those years and
    // the line breaks.  Filling masked → 0 makes the time series
    // continuous and shows "no activity" years as flat zeros.
    icSelected = ic.map(function(img) {
      return ee.Image(img)
        .select([0, 4, 5], ['Prediction',
                            'Lower 95% CI',
                            'Upper 95% CI'])
        .unmask(0);
    });
  } else {
    // OOD (single-band): just plot the one band as 'value', also
    // filling masked pixels with 0 so the line stays continuous.
    icSelected = ic.map(function(img) {
      return ee.Image(img).select([0]).rename('value').unmask(0);
    });
  }
  // Pretty unit label (e.g., 'mm / yr' rather than 'Depth_mm') reused
  // for chart titles and y-axis text.  unitConversionFromAF supplies
  // the same nicely-formatted label the well chart uses, so all four
  // charts read consistently.
  var prettyUnit = (meta && meta.units === false)
      ? null : unitConversionFromAF(unit).label;
  var unitLabel = prettyUnit ? ' (' + prettyUnit + ')' : '';
  var yLabel    = (hasCi ? 'Prediction' : 'value')
                  + (prettyUnit ? ' (' + prettyUnit + ')' : '');
  // CI series options — prediction is the visually dominant series
  // (thick, very dark colour, solid), with the two CI bounds shown
  // in DISTINCT lighter shades of the same hue family so they're
  // each individually identifiable.  Lower CI gets a medium shade,
  // Upper CI a paler shade — gives a "fan" effect around the
  // prediction.  Dashes kept as a hint in case the chart engine
  // renders them; colour + width contrast carries the load if not.
  function ciSeriesOpts(predColor, lowerColor, upperColor) {
    return hasCi
        ? {
            // Series indices follow GEE's alphabetical sort of band
            // names: Lower (L) → 0, Prediction (P) → 1, Upper (U) → 2.
            // We assign the BOLD prediction style to index 1 and the
            // dashed CI styles to 0 and 2 so each line renders
            // correctly on the chart.  lineDashStyle: null forces
            // the prediction solid even if a global default would
            // otherwise dash it.
            0: {color: lowerColor, lineWidth: 1, pointSize: 0,
                lineDashStyle: [4, 4]},
            1: {color: predColor,  lineWidth: 3, pointSize: 0,
                lineDashStyle: null},
            2: {color: upperColor, lineWidth: 1, pointSize: 0,
                lineDashStyle: [8, 4]},
          }
        : {0: {color: predColor,  lineWidth: 2, pointSize: 2}};
  }
  var legendPos = hasCi ? 'bottom' : 'none';

  // ── 1. Pixel time series at click point (≡ well-scale time series) ──
  var pixelChart = ui.Chart.image.series({
    imageCollection: icSelected,
    region: pt,
    reducer: ee.Reducer.first(),
    scale: 2000,  // 2 km native pipeline resolution
    xProperty: 'year',
  }).setOptions({
    title: 'Pixel value at click point — ' + meta.label + unitLabel,
    vAxis: {title: yLabel},
    hAxis: {title: 'Year', format: '####'},
    series: ciSeriesOpts('00008B', '5e92f3', '90caf9'),
    legend: {position: legendPos},
    chartArea: {width: '85%', height: '70%'},
  });

  // ── 2. Basin time series ──────────────────────────────────────────
  var basinAtClick = basins.filterBounds(pt).first();
  var basinName = ee.Algorithms.If(basinAtClick,
                                    ee.Feature(basinAtClick).get('BASIN_NAME'),
                                    'No basin');
  var basinChart = ui.Chart.image.seriesByRegion({
    imageCollection: icSelected,
    regions: ee.FeatureCollection([basinAtClick]),
    reducer: ee.Reducer.mean(),
    scale: 2000,
    seriesProperty: 'BASIN_NAME',
    xProperty: 'year',
  }).setOptions({
    title: 'Basin mean — ' + ee.String(basinName).getInfo() + unitLabel,
    vAxis: {title: yLabel + ' — basin mean'},
    hAxis: {title: 'Year', format: '####'},
    series: ciSeriesOpts('006400', '66bb6a', 'a5d6a7'),
    legend: {position: legendPos},
    chartArea: {width: '85%', height: '70%'},
  });

  // ── 3. Sub-basin time series ──────────────────────────────────────
  var subAtClick = subbasins.filterBounds(pt).first();
  var subName = ee.Algorithms.If(subAtClick,
                                  ee.Feature(subAtClick).get('SUBBASIN_N'),
                                  'No sub-basin');
  var subChart = ui.Chart.image.seriesByRegion({
    imageCollection: icSelected,
    regions: ee.FeatureCollection([subAtClick]),
    reducer: ee.Reducer.mean(),
    scale: 2000,
    seriesProperty: 'SUBBASIN_N',
    xProperty: 'year',
  }).setOptions({
    title: 'Sub-basin mean — ' + ee.String(subName).getInfo() + unitLabel,
    vAxis: {title: yLabel + ' — sub-basin mean'},
    hAxis: {title: 'Year', format: '####'},
    series: ciSeriesOpts('B22222', 'ef5350', 'ffab91'),
    legend: {position: legendPos},
    chartArea: {width: '85%', height: '70%'},
  });

  // Replace placeholder with the three charts
  chartsPanel.clear();
  chartsPanel.add(ui.Label('Click point: lat ' + coords.lat.toFixed(3) +
                            ', lon ' + coords.lon.toFixed(3),
                            {fontSize: '10px', color: '#555'}));
  chartsPanel.add(pixelChart);
  chartsPanel.add(basinChart);
  chartsPanel.add(subChart);

  // 4th panel: nearest well's published Well_Package time series
  // (capacity-disaggregated; differs from pixel value when multiple
  // wells share a 2 km cell).  Skipped for categories with no
  // well-level analog (OOD).  AF values are converted client-side to
  // the unit currently selected in the dropdown.
  addNearestWellTimeSeries(pt, catKey, unit);
}

/** CAP click readout: cumulative ΔGW (AF) at the pixel, plus the
 * total over the containing basin and sub-basin.  No time series —
 * the asset is already an integral over the chosen window.  `side`
 * picks which (left vs right) CAP scenario/window dropdowns to use,
 * so a click on the right map in split mode reflects that map's
 * scenario rather than the left's. */
function capClickReadout(pt, coords, side) {
  var sc  = (side === 'R') ? rightCapScenarioDropdown.getValue()
                           : capScenarioDropdown.getValue();
  var win = (side === 'R') ? rightCapWindowDropdown.getValue()
                           : capWindowDropdown.getValue();
  var imgId = capImageId(sc, win);
  var ic = ee.ImageCollection(assetIdFor('CAP_Scenario__Pixel_Rasters'));
  var img = ee.Image(ic.filter(ee.Filter.eq('system:index', imgId)).first())
                .select(0).rename('cum_AF');

  // Pixel value at the click point (2 km native pipeline scale)
  var pixelVal = img.reduceRegion({
    reducer: ee.Reducer.first(), geometry: pt, scale: 2000,
  }).get('cum_AF');

  // Containing basin / sub-basin (filterBounds on a point)
  var basinAtClick = ee.Feature(basins.filterBounds(pt).first());
  var subAtClick   = ee.Feature(subbasins.filterBounds(pt).first());

  // Sum cumulative ΔGW over each region (AF)
  var basinSum = img.reduceRegion({
    reducer: ee.Reducer.sum(), geometry: basinAtClick.geometry(),
    scale: 2000, maxPixels: 1e9,
  }).get('cum_AF');
  var subSum = img.reduceRegion({
    reducer: ee.Reducer.sum(), geometry: subAtClick.geometry(),
    scale: 2000, maxPixels: 1e9,
  }).get('cum_AF');

  // Pull names + values in one batch (one getInfo, not three)
  var bundle = ee.Dictionary({
    pixel:        pixelVal,
    basinName:    basinAtClick.get('BASIN_NAME'),
    basinSum:     basinSum,
    subName:      subAtClick.get('SUBBASIN_N'),
    subSum:       subSum,
  });

  bundle.evaluate(function(d) {
    chartsPanel.clear();
    chartsPanel.add(ui.Label(
      'Click point: lat ' + coords.lat.toFixed(3) +
      ', lon ' + coords.lon.toFixed(3),
      {fontSize: '10px', color: '#555'}
    ));
    function fmt(v) {
      if (v === null || v === undefined) return '—';
      // AF — group thousands with comma; one decimal for smaller magnitudes
      var n = Number(v);
      if (Math.abs(n) >= 1000) return n.toLocaleString(undefined,
        {maximumFractionDigits: 0}) + ' AF';
      return n.toFixed(1) + ' AF';
    }
    chartsPanel.add(ui.Label('Cumulative ΔGW (CAP shortfall) over ' +
      win.replace('_', '–') + ':',
      {fontWeight: 'bold', margin: '6px 0 4px 0'}));
    chartsPanel.add(ui.Label('  • Pixel (2 km): '       + fmt(d.pixel)));
    chartsPanel.add(ui.Label('  • Basin (' +
      (d.basinName || 'none') + '): '                   + fmt(d.basinSum)));
    chartsPanel.add(ui.Label('  • Sub-basin (' +
      (d.subName || 'none') + '): '                     + fmt(d.subSum)));
    chartsPanel.add(ui.Label(
      'Single-window readout above. Per-year cumulative time series for ' +
      'this basin across all 7 scenarios is below.',
      {fontSize: '10px', color: '#777', margin: '6px 0 6px 0'}
    ));
  });

  // 5th panel: per-basin cumulative ΔGW time series across all 7
  // scenarios (data: Outputs/.../CAP_Scenario/CAP_Scenario_Cumulative.csv,
  // uploaded as a non-spatial table — see gee/upload_cap_cumulative.sh).
  addCapBasinScenarioChart(pt);
}

/** CAP-mode helper: chart cumulative ΔGW for the click-basin across
 * all 7 scenarios, 2026–2099 (read from the CAP_Scenario_Cumulative
 * tabular asset).  Falls back to a friendly note if the asset isn't
 * uploaded yet. */
function addCapBasinScenarioChart(pt) {
  var cumAssetId = ASSET_ROOT + '/CAP_Scenario_Cumulative';
  var basinAtClick = ee.Feature(basins.filterBounds(pt).first());
  var basinName = basinAtClick.get('BASIN_NAME');

  var stub = ui.Label('Loading per-scenario basin time series…',
                      {color: '#777', fontSize: '11px',
                       margin: '6px 0 0 0'});
  chartsPanel.add(stub);

  // The CSV is uploaded in WIDE format: one Feature per (basin,
  // scenario) with year-stamped properties Cum_<year>.  74 year
  // columns + Basin + Scenario + (dummy geometry) per feature.
  var fc = ee.FeatureCollection(cumAssetId)
              .filter(ee.Filter.eq('Basin', basinName));
  // Pull all 7 scenarios at once
  var bundle = ee.Dictionary({
    basinName: basinName,
    rows: fc.toList(20).map(function(f) {
      return ee.Feature(f).toDictionary();
    }),
  });

  bundle.evaluate(function(d, err) {
    chartsPanel.remove(stub);
    if (err || !d) {
      chartsPanel.add(ui.Label(
        'Per-scenario chart unavailable.  Upload the CSV first:\n' +
        '  python gee/pivot_cap_cumulative.py && \\\n' +
        '  ./gee/upload_cap_cumulative.sh',
        {color: '#888', fontSize: '10px', margin: '4px 0 0 0',
         whiteSpace: 'pre'}
      ));
      return;
    }
    if (!d.rows || d.rows.length === 0) {
      chartsPanel.add(ui.Label(
        'No CAP-cumulative entries for basin "' +
        (d.basinName || 'unknown') + '".',
        {color: '#888', fontSize: '11px', margin: '4px 0 0 0'}
      ));
      return;
    }

    // Reshape to chart rows: [['Year', sc1, sc2, ...], [2026, ...], ...]
    // Year columns are named 'Cum_<year>' in the uploaded FC.
    var scNames = d.rows.map(function(r) { return r.Scenario; });
    var headerRow = ['Year'].concat(scNames);
    var data = [headerRow];
    var yearKeys = [];
    for (var k in d.rows[0]) {
      var m = k.match(/^Cum_(\d{4})$/);
      if (m) yearKeys.push({key: k, year: parseInt(m[1], 10)});
    }
    yearKeys.sort(function(a, b) { return a.year - b.year; });
    for (var yi = 0; yi < yearKeys.length; yi++) {
      var rec = yearKeys[yi];
      var row = [rec.year];
      for (var si = 0; si < d.rows.length; si++) {
        var v = d.rows[si][rec.key];
        row.push(v === null || v === undefined ? null : Number(v));
      }
      data.push(row);
    }

    var chart = ui.Chart(data, 'LineChart', {
      title: 'Cumulative ΔGW (AF) by scenario — basin: ' +
             (d.basinName || 'unknown') + ', 2026–2099',
      hAxis: {title: 'Year', format: '####'},
      vAxis: {title: 'Cumulative ΔGW (AF)'},
      legend: {position: 'bottom'},
      chartArea: {width: '85%', height: '60%'},
    });
    chartsPanel.add(chart);
    chartsPanel.add(ui.Label(
      'Source: CAP_Scenario_Cumulative table; click a different ' +
      'basin to retarget the chart.',
      {fontSize: '10px', color: '#777', margin: '0 0 4px 0'}
    ));
  });
}

/** Look up the nearest well within WELL_SEARCH_RADIUS_M of the click,
 * then chart its published Well_Package time series with a 95 % CI
 * envelope, converting AF → the unit currently selected in the
 * dropdown.  No-op for categories that have no well-level analog
 * (returns silently).  Async — appends to chartsPanel after evaluate. */
function addNearestWellTimeSeries(pt, catKey, unit) {
  var wpCat = wellPackageCategoryFor(catKey);
  if (!wpCat) return;  // OOD has no well-level analog
  var conv = unitConversionFromAF(unit);

  // Show a stub immediately so the UI feels responsive
  var stub = ui.Label('Loading nearest well…',
                      {color: '#777', fontSize: '11px',
                       margin: '6px 0 0 0'});
  chartsPanel.add(stub);

  // 1. Narrow to candidates inside the generous search radius.
  // 2. Tag each candidate with its true distance to the click.
  // 3. Sort ascending and pick the first — i.e. the truly nearest well,
  //    not just *some* well that the bounds-filter happened to return
  //    (the old `.filterBounds().first()` path had no ordering guarantee,
  //    so even when wells were nearby the chart could miss them).
  var searchArea = pt.buffer(WELL_SEARCH_RADIUS_M);
  var nearest = ee.Feature(
    wells.filterBounds(searchArea)
      .map(function(f) {
        return f.set('_dist_m', f.geometry().distance(pt, 1));
      })
      .sort('_dist_m')
      .first()
  );

  // Bundle: (a) the nearest well attributes + distance to click, (b)
  // all of its Well_Package year-stamped properties, in one server
  // roundtrip.
  var bundle = ee.Algorithms.If(
    nearest,
    ee.Dictionary({
      hasWell: true,
      registryId: nearest.get('REGISTRY_I'),
      waterUse:   nearest.get('WATER_USE'),
      distance_m: nearest.get('_dist_m'),
      lonlat:     nearest.geometry().coordinates(),
      props: ee.Feature(
        ee.FeatureCollection(assetIdFor('Well_Package__' + wpCat))
          .filter(ee.Filter.eq('REGISTRY_I', nearest.get('REGISTRY_I')))
          .first()
      ).toDictionary(),
    }),
    ee.Dictionary({hasWell: false})
  );

  ee.Dictionary(bundle).evaluate(function(d) {
    chartsPanel.remove(stub);
    if (!d || !d.hasWell) {
      chartsPanel.add(ui.Label(
        'No well within ' + (WELL_SEARCH_RADIUS_M / 1000) + ' km.',
        {color: '#888', fontSize: '11px', margin: '6px 0 0 0'}
      ));
      return;
    }
    if (!d.props) {
      chartsPanel.add(ui.Label(
        'Nearest well ' + d.registryId + ' not in Well_Package__' +
        wpCat + ' (likely uploaded but missing entry).',
        {color: '#888', fontSize: '11px', margin: '6px 0 0 0'}
      ));
      return;
    }

    // Walk the property dict and bucket year-stamped entries.  Match
    // <Cat>_AF_sigma_<year> first (longer pattern) so the prediction
    // regex doesn't swallow them.
    var years = {};   // year -> {pred, sigma}
    var sigRe = new RegExp('^' + wpCat + '_AF_sigma_(\\d{4})$');
    var predRe = new RegExp('^' + wpCat + '_AF_(\\d{4})$');
    for (var k in d.props) {
      var m = k.match(sigRe);
      if (m) {
        years[m[1]] = years[m[1]] || {};
        years[m[1]].sigma = d.props[k];
        continue;
      }
      m = k.match(predRe);
      if (m) {
        years[m[1]] = years[m[1]] || {};
        years[m[1]].pred = d.props[k];
      }
    }

    // Build the chart data array (sorted by year).  AF values are
    // converted to the user-selected unit so the well chart matches
    // the map + the other three time-series charts.
    var rows = [['Year', 'Prediction', 'Lower 95% CI', 'Upper 95% CI']];
    var sortedYears = Object.keys(years).sort();
    for (var i = 0; i < sortedYears.length; i++) {
      var y = sortedYears[i];
      var p = years[y].pred;
      var s = years[y].sigma;
      if (p === null || p === undefined) continue;
      // 95% CI = ±1.96 σ; clamp lower to 0 (water use can't be negative)
      var lower = (s != null) ? Math.max(0, p - 1.96 * s) : null;
      var upper = (s != null) ? p + 1.96 * s : null;
      rows.push([
        Number(y),
        p * conv.factor,
        lower !== null ? lower * conv.factor : null,
        upper !== null ? upper * conv.factor : null,
      ]);
    }

    var lonlat = d.lonlat || [null, null];
    var distKm = (d.distance_m != null)
        ? (Number(d.distance_m) / 1000).toFixed(2) + ' km from click'
        : null;
    var titleSuffix = ' — ' + wpCat + ' (' + conv.label +
                      ', capacity-disaggregated)';
    var chart = ui.Chart(rows, 'LineChart', {
      title: 'Well ' + d.registryId + ' [' + (d.waterUse || 'n/a') + ']' +
             (distKm ? ', ' + distKm : '') + titleSuffix,
      hAxis: {title: 'Year', format: '####'},
      vAxis: {title: conv.label},
      series: {
        // Bold dark prediction (3 px) + distinct lighter-shade bounds
        // (1 px) so prediction is the visually dominant series and
        // Lower vs Upper CI are individually identifiable.  Mirrors
        // the colour scheme used by the pixel chart.  This chart's
        // data is built from a literal 2D array so column order
        // (Pred, Lower, Upper) maps directly to series 0/1/2 — no
        // alphabetical-sort surprise here, unlike ui.Chart.image.*.
        0: {color: '1a237e', lineWidth: 3, pointSize: 0,
            lineDashStyle: null},
        1: {color: '5e92f3', lineWidth: 1, pointSize: 0,
            lineDashStyle: [4, 4]},
        2: {color: '90caf9', lineWidth: 1, pointSize: 0,
            lineDashStyle: [8, 4]},
      },
      legend: {position: 'bottom'},
      chartArea: {width: '85%', height: '70%'},
    });
    chartsPanel.add(chart);
    chartsPanel.add(ui.Label(
      'Well at lon ' + Number(lonlat[0]).toFixed(3) +
      ', lat ' + Number(lonlat[1]).toFixed(3) +
      ' — values from Well_Package__' + wpCat +
      ' (stored in AF, converted client-side to ' + conv.label + ')',
      {fontSize: '10px', color: '#777', margin: '0 0 4px 0'}
    ));
  });
}

// ═════════════════════════════════════════════════════════════════════
//  Initial paint
// ═════════════════════════════════════════════════════════════════════
// setRootLayout() recreates the maps fresh and centers them on AZ
// inside createMaps() (the no-savedCenter branch), so no top-level
// centerObject is needed here.
setRootLayout();          // installs leftMap (or splitPanel if compare on)
syncControlVisibility();
populateStretchDefaults();
refreshLayer();
