"""
Unit tests for core hydrolibs modules.

Covers:
    - sysops (utility helpers)
    - partitionops (prediction partitioning)
    - mlops (evaluation metrics / scorer functions)
    - rasterops (raster I/O round-trip)
    - uncertaintyops (UQ helper functions)
    - visualops (data-frame construction and plotting helpers)
"""
import os
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from rasterio.transform import from_bounds

from hydrolibs.sysops import (
    az_nodata,
    makedirs,
    make_proper_dir_name,
    boolean_string,
    round_to_n_nonzero,
)
from hydrolibs.partitionops import (
    CATEGORIES,
    focal_fill_irr_fraction,
    compute_sw_fraction,
    partition_predictions,
)
from hydrolibs.mlops import (
    adjusted_r2,
    normalized_rmse,
    normalized_mae,
    normalized_mbe,
    _abs_r2_score,
    _abs_adjusted_r2,
    _abs_normalized_rmse,
    _abs_normalized_mae,
    _abs_normalized_mbe,
    compute_irrigation_demand_floor,
    make_physics_floor_objective,
    PhysicsBoundsObjective,
    _build_monotone_constraints,
    _build_interaction_constraints,
    MONOTONE_CONSTRAINT_MAP,
    INTERACTION_GROUPS,
    PhysicsXGBRegressor,
    PhysicsLGBMRegressor,
    PhysicsXGBRFRegressor,
    _PHYS_COL,
)
from hydrolibs.rasterops import read_raster_as_arr, write_raster
from hydrolibs.uncertaintyops import (
    _build_pred_features,
    _predict_total,
    _compute_category_sigmas,
    _pixel_stats,
    MM_TO_FT,
    M3_TO_AF,
)
from hydrolibs.visualops import (
    build_annual_df,
    era_shaded_ts,
    ERA_PERIODS,
    ERA_COLORS,
)


# ═══════════════════════════════════════════════════════════════════════════
# sysops tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAzNodata:
    def test_returns_float(self):
        assert isinstance(az_nodata(), float)

    def test_value(self):
        assert az_nodata() == -32767.0


class TestMakedirs:
    def test_single_directory(self, tmp_path):
        d = str(tmp_path / 'test_dir')
        makedirs(d)
        assert os.path.isdir(d)

    def test_tuple_of_directories(self, tmp_path):
        dirs = (str(tmp_path / 'a'), str(tmp_path / 'b'))
        makedirs(dirs)
        assert all(os.path.isdir(d) for d in dirs)

    def test_existing_directory_no_error(self, tmp_path):
        d = str(tmp_path / 'existing')
        os.makedirs(d)
        makedirs(d)  # should not raise

    def test_none_in_list(self, tmp_path):
        makedirs((str(tmp_path / 'real'), None))


class TestMakeProperDirName:
    def test_appends_sep(self):
        result = make_proper_dir_name('/some/path')
        assert result.endswith(os.sep)

    def test_already_has_sep(self):
        result = make_proper_dir_name(f'/some/path{os.sep}')
        assert result == f'/some/path{os.sep}'

    def test_forward_slash(self):
        result = make_proper_dir_name('/some/path/')
        assert result == '/some/path/'

    def test_none_input(self):
        assert make_proper_dir_name(None) is None


class TestBooleanString:
    def test_true(self):
        assert boolean_string('True') is True

    def test_false(self):
        assert boolean_string('False') is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            boolean_string('yes')


class TestRoundToNNonzero:
    def test_zero(self):
        assert round_to_n_nonzero(0) == 0

    def test_integer_value(self):
        assert round_to_n_nonzero(123.456, 2) == 120

    def test_small_value(self):
        result = round_to_n_nonzero(0.00456, 2)
        assert abs(result - 0.0046) < 1e-10

    def test_negative(self):
        result = round_to_n_nonzero(-0.00456, 2)
        assert abs(result - (-0.0046)) < 1e-10

    def test_large_value(self):
        result = round_to_n_nonzero(98765, 3)
        assert result == 98800


# ═══════════════════════════════════════════════════════════════════════════
# partitionops tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCategories:
    def test_count(self):
        assert len(CATEGORIES) == 8

    def test_expected_keys(self):
        expected = {
            'Irrigation', 'Non_Irrigation',
            'Irrigation_GW', 'Irrigation_SW',
            'Non_Irrigation_GW', 'Non_Irrigation_SW',
            'Total_GW', 'Total_SW',
        }
        assert set(CATEGORIES) == expected


class TestFocalFillIrrFraction:
    def test_no_fill_needed(self):
        n = 25
        irr_frac = np.full(n, 0.5)
        well_dens = np.ones(n)
        valid_mask = np.ones(n, dtype=bool)
        raster_shape = (5, 5)

        result = focal_fill_irr_fraction(irr_frac, well_dens, raster_shape, valid_mask)
        np.testing.assert_array_equal(result, irr_frac)

    def test_fills_low_irr_fraction(self):
        n = 25
        irr_frac = np.full(n, 0.5)
        irr_frac[12] = 0.01  # center pixel has a very low fraction
        well_dens = np.ones(n)
        valid_mask = np.ones(n, dtype=bool)
        raster_shape = (5, 5)

        result = focal_fill_irr_fraction(irr_frac, well_dens, raster_shape, valid_mask)
        # Center should be filled with neighbourhood mean, not the original 0.01
        assert result[12] > 0.01


class TestComputeSwFraction:
    def test_zero_canal_density(self):
        n = 9
        canal_dens = np.zeros(n)
        valid_mask = np.ones(n, dtype=bool)
        raster_shape = (3, 3)

        result = compute_sw_fraction(canal_dens, raster_shape, valid_mask)
        np.testing.assert_array_equal(result, 0.0)

    def test_uniform_canal_density(self):
        n = 9
        canal_dens = np.full(n, 5.0)
        valid_mask = np.ones(n, dtype=bool)
        raster_shape = (3, 3)

        result = compute_sw_fraction(canal_dens, raster_shape, valid_mask)
        # All equal → local_max == pixel → fraction ≈ 1
        # (edge effects from padding may reduce some values)
        assert np.all(result >= 0) and np.all(result <= 1)


class TestPartitionPredictions:
    def _make_inputs(self, n=25):
        predictions = np.full(n, 100.0)
        raster_shape = (5, 5)
        valid_mask = np.ones(n, dtype=bool)
        year_df = pd.DataFrame({
            'well_density': np.ones(n),
            'annual_irr_fraction': np.full(n, 0.6),
            'annual_gw_fraction': np.full(n, 0.7),
            'canal_density': np.full(n, 0.5),
        })
        return predictions, year_df, raster_shape, valid_mask

    def test_returns_all_categories(self):
        predictions, year_df, raster_shape, valid_mask = self._make_inputs()
        result = partition_predictions(predictions, year_df, raster_shape, valid_mask)
        assert set(result.keys()) == set(CATEGORIES)

    def test_irr_plus_nonirr_equals_total(self):
        predictions, year_df, raster_shape, valid_mask = self._make_inputs()
        result = partition_predictions(predictions, year_df, raster_shape, valid_mask)
        np.testing.assert_allclose(
            result['Irrigation'] + result['Non_Irrigation'],
            predictions,
            rtol=1e-10,
        )

    def test_gw_plus_sw_equals_total(self):
        predictions, year_df, raster_shape, valid_mask = self._make_inputs()
        result = partition_predictions(predictions, year_df, raster_shape, valid_mask)
        np.testing.assert_allclose(
            result['Total_GW'] + result['Total_SW'],
            predictions,
            rtol=1e-10,
        )

    def test_irr_gw_plus_irr_sw_equals_irr(self):
        predictions, year_df, raster_shape, valid_mask = self._make_inputs()
        result = partition_predictions(predictions, year_df, raster_shape, valid_mask)
        np.testing.assert_allclose(
            result['Irrigation_GW'] + result['Irrigation_SW'],
            result['Irrigation'],
            rtol=1e-10,
        )

    def test_zero_well_density_gives_nan(self):
        n = 25
        predictions = np.full(n, 100.0)
        raster_shape = (5, 5)
        valid_mask = np.ones(n, dtype=bool)
        year_df = pd.DataFrame({
            'well_density': np.zeros(n),
            'annual_irr_fraction': np.full(n, 0.6),
            'annual_gw_fraction': np.full(n, 0.7),
            'canal_density': np.full(n, 0.5),
        })
        result = partition_predictions(predictions, year_df, raster_shape, valid_mask)
        assert np.all(np.isnan(result['Irrigation']))


# ═══════════════════════════════════════════════════════════════════════════
# mlops metric tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAdjustedR2:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert adjusted_r2(y, y, p=1) == pytest.approx(1.0)

    def test_worse_than_mean(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert adjusted_r2(y, y_pred, p=1) < 0

    def test_degenerate_n_equals_p_plus_1(self):
        y = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        assert np.isnan(adjusted_r2(y, y_pred, p=1))

    def test_more_predictors_lower_adj_r2(self):
        y = np.arange(20, dtype=float)
        y_pred = y + np.random.default_rng(42).normal(0, 0.1, 20)
        r2_p1 = adjusted_r2(y, y_pred, p=1)
        r2_p10 = adjusted_r2(y, y_pred, p=10)
        assert r2_p10 < r2_p1


class TestNormalizedRMSE:
    def test_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert normalized_rmse(y, y) == pytest.approx(0.0)

    def test_positive_error(self):
        y = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])
        assert normalized_rmse(y, y_pred) > 0

    def test_zero_mean_returns_nan(self):
        y = np.array([0.0, 0.0, 0.0])
        assert np.isnan(normalized_rmse(y, y + 0.1))


class TestNormalizedMAE:
    def test_perfect(self):
        y = np.array([5.0, 10.0, 15.0])
        assert normalized_mae(y, y) == pytest.approx(0.0)

    def test_positive_error(self):
        y = np.array([5.0, 10.0, 15.0])
        y_pred = np.array([6.0, 9.0, 14.0])
        assert normalized_mae(y, y_pred) > 0

    def test_zero_mean_returns_nan(self):
        y = np.array([0.0, 0.0, 0.0])
        assert np.isnan(normalized_mae(y, y + 0.5))


class TestNormalizedMBE:
    def test_no_bias(self):
        y = np.array([10.0, 20.0, 30.0])
        assert normalized_mbe(y, y) == pytest.approx(0.0)

    def test_over_prediction_negative_bias(self):
        y = np.array([10.0, 20.0, 30.0])
        y_pred = y + 5  # over-prediction
        assert normalized_mbe(y, y_pred) < 0

    def test_under_prediction_positive_bias(self):
        y = np.array([10.0, 20.0, 30.0])
        y_pred = y - 5  # under-prediction
        assert normalized_mbe(y, y_pred) > 0

    def test_zero_mean_returns_nan(self):
        y = np.array([-1.0, 0.0, 1.0])
        assert np.isnan(normalized_mbe(y, y))


class TestAbsScorers:
    """_abs_* wrappers apply np.abs to y_pred before scoring."""

    def test_abs_r2_negatives_become_positive(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = -y  # all negative
        assert _abs_r2_score(y, y_pred) == pytest.approx(1.0)

    def test_abs_adjusted_r2_negatives(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = -y
        assert _abs_adjusted_r2(y, y_pred, p=1) == pytest.approx(1.0)

    def test_abs_nrmse_negatives(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = -y
        assert _abs_normalized_rmse(y, y_pred) == pytest.approx(0.0)

    def test_abs_nmae_negatives(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = -y
        assert _abs_normalized_mae(y, y_pred) == pytest.approx(0.0)

    def test_abs_nmbe_negatives(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = -y
        assert _abs_normalized_mbe(y, y_pred) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# rasterops tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_raster(tmp_path):
    """Create a small GeoTIFF for round-trip tests."""
    data = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    transform_ = from_bounds(0, 0, 1, 1, 2, 2)
    path = str(tmp_path / 'test.tif')
    with rio.open(
        path, 'w', driver='GTiff',
        height=2, width=2, count=1,
        dtype='float32', crs='EPSG:4326',
        transform=transform_, nodata=-32767.0,
    ) as dst:
        out = data.copy()
        out[np.isnan(out)] = -32767.0
        dst.write(out, 1)
    return path, data, transform_


class TestReadRasterAsArr:
    def test_returns_array_and_file(self, synthetic_raster):
        path, expected, _ = synthetic_raster
        arr, rfile = read_raster_as_arr(path, get_file=True)
        assert arr.shape == (2, 2)
        assert isinstance(rfile, rio.DatasetReader)
        rfile.close()

    def test_nodata_becomes_nan(self, synthetic_raster):
        path, _, _ = synthetic_raster
        arr, rfile = read_raster_as_arr(path)
        assert np.isnan(arr[1, 1])
        rfile.close()

    def test_get_file_false(self, synthetic_raster):
        path, _, _ = synthetic_raster
        arr = read_raster_as_arr(path, get_file=False)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)

    def test_rasterio_object_input(self, synthetic_raster):
        path, _, _ = synthetic_raster
        with rio.open(path) as src:
            arr = read_raster_as_arr(src)
            assert arr.shape == (2, 2)


class TestWriteRaster:
    def test_round_trip(self, synthetic_raster, tmp_path):
        path, _, _ = synthetic_raster
        arr, rfile = read_raster_as_arr(path, get_file=True)
        out_path = str(tmp_path / 'out.tif')
        write_raster(arr, rfile, rfile.transform, out_path, -32767.0)
        rfile.close()

        arr2, rfile2 = read_raster_as_arr(out_path, get_file=True)
        np.testing.assert_array_equal(np.isnan(arr), np.isnan(arr2))
        valid = ~np.isnan(arr)
        np.testing.assert_allclose(arr[valid], arr2[valid])
        rfile2.close()

    def test_multiband_write(self, synthetic_raster, tmp_path):
        path, _, _ = synthetic_raster
        arr, rfile = read_raster_as_arr(path, get_file=True)
        data_3d = np.stack([arr, arr * 2], axis=0)
        out_path = str(tmp_path / 'multi.tif')
        write_raster(data_3d, rfile, rfile.transform, out_path, -32767.0, num_bands=2)
        rfile.close()

        with rio.open(out_path) as src:
            assert src.count == 2


# ═══════════════════════════════════════════════════════════════════════════
# uncertaintyops helper tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPredFeatures:
    def test_drops_attrs_and_target(self):
        df = pd.DataFrame({
            'feat1': [1.0, 2.0],
            'feat2': [3.0, 4.0],
            'Year': [2000, 2001],
            'gw_pumping_mm': [10.0, 20.0],
        })
        result = _build_pred_features(df, ['feat1', 'feat2'], drop_attrs=('Year',))
        assert list(result.columns) == ['feat1', 'feat2']
        assert 'Year' not in result.columns
        assert 'gw_pumping_mm' not in result.columns

    def test_missing_features_filled_with_zero(self):
        df = pd.DataFrame({'feat1': [1.0], 'gw_pumping_mm': [5.0]})
        result = _build_pred_features(df, ['feat1', 'feat2'], drop_attrs=())
        assert result['feat2'].iloc[0] == 0.0

    def test_inf_replaced_with_zero(self):
        df = pd.DataFrame({
            'feat1': [np.inf],
            'feat2': [-np.inf],
            'gw_pumping_mm': [5.0],
        })
        result = _build_pred_features(df, ['feat1', 'feat2'], drop_attrs=())
        assert result['feat1'].iloc[0] == 0.0
        assert result['feat2'].iloc[0] == 0.0

    def test_nan_replaced_with_zero(self):
        df = pd.DataFrame({
            'feat1': [np.nan],
            'gw_pumping_mm': [5.0],
        })
        result = _build_pred_features(df, ['feat1'], drop_attrs=())
        assert result['feat1'].iloc[0] == 0.0


class TestComputeCategorySigmas:
    def _make_cats(self, n_members, n_pixels, mode_val=None):
        cats = []
        rng = np.random.default_rng(42)
        for _ in range(n_members):
            vals = rng.normal(100, 10, n_pixels).astype(np.float32)
            if mode_val is not None:
                vals[:] = mode_val
            cats.append({'Irrigation': vals, 'Non_Irrigation': vals * 0.5})
        return cats

    def test_std_mode(self):
        cats = self._make_cats(5, 20)
        result = _compute_category_sigmas(cats, mode='std')
        assert 'Irrigation' in result
        assert 'Non_Irrigation' in result
        assert result['Irrigation'].dtype == np.float32
        assert np.all(result['Irrigation'] >= 0)

    def test_half_range_mode(self):
        cats = self._make_cats(2, 20)
        result = _compute_category_sigmas(cats, mode='half_range')
        expected = np.abs(cats[0]['Irrigation'] - cats[1]['Irrigation']) / 2.0
        np.testing.assert_allclose(result['Irrigation'], expected, rtol=1e-5)

    def test_identical_members_zero_sigma(self):
        cats = self._make_cats(3, 10, mode_val=50.0)
        result = _compute_category_sigmas(cats, mode='std')
        np.testing.assert_allclose(result['Irrigation'], 0.0, atol=1e-6)


class TestPixelStats:
    def test_basic_stats(self):
        vals = np.array([10.0, 20.0, 30.0])
        mm_to_m3 = 1000.0  # arbitrary scale
        result = _pixel_stats(vals, mm_to_m3)
        assert result['Mean_Depth_mm'] == pytest.approx(20.0)
        assert result['Mean_Depth_ft'] == pytest.approx(20.0 * MM_TO_FT, rel=1e-4)
        assert result['Volume_m3'] == pytest.approx(60.0 * mm_to_m3, rel=1e-4)
        assert result['Volume_AF'] == pytest.approx(60.0 * mm_to_m3 * M3_TO_AF, rel=1e-4)

    def test_empty_input(self):
        vals = np.array([])
        result = _pixel_stats(vals, 1.0)
        assert result['Mean_Depth_mm'] == 0.0
        assert result['Volume_m3'] == 0.0


class TestPredictTotal:
    def test_returns_total_and_categories(self):
        n = 25
        mock_model = MagicMock()
        mock_model.predict.return_value = np.full(n, 50.0)

        pred_features = pd.DataFrame({'f1': np.ones(n)})
        year_df = pd.DataFrame({
            'well_density': np.ones(n),
            'annual_irr_fraction': np.full(n, 0.6),
            'annual_gw_fraction': np.full(n, 0.7),
            'canal_density': np.full(n, 0.5),
        })

        import hydrolibs.partitionops as partops
        total, cats = _predict_total(
            mock_model, pred_features, year_df, partops,
            raster_shape=(5, 5), valid_mask=np.ones(n, dtype=bool),
        )
        np.testing.assert_allclose(
            total, cats['Irrigation'] + cats['Non_Irrigation'], rtol=1e-10,
        )
        assert set(cats.keys()) == set(CATEGORIES)


# ═══════════════════════════════════════════════════════════════════════════
# visualops tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildAnnualDf:
    def test_basic_pivot(self):
        yearly = {
            2000: {'BasinA': {'Volume_AF': 100}, 'BasinB': {'Volume_AF': 200}},
            2001: {'BasinA': {'Volume_AF': 110}, 'BasinB': {'Volume_AF': 210}},
        }
        df = build_annual_df(yearly, 'GW_Basin')
        assert set(df.columns) >= {'Year', 'GW_Basin', 'Volume_AF', 'Era'}
        assert len(df) == 4
        assert df.loc[df['Year'] == 2000, 'Volume_AF'].sum() == 300

    def test_era_assignment(self):
        yearly = {
            1900: {'B': {'v': 1}},
            2000: {'B': {'v': 2}},
            2050: {'B': {'v': 3}},
        }
        df = build_annual_df(yearly, 'Basin')
        eras = dict(zip(df['Year'], df['Era']))
        assert eras[1900] == 'Hindcast'
        assert eras[2000] == 'Historical'
        assert eras[2050] == 'Projection'


class TestEraShadedTs:
    def test_plots_without_error(self):
        fig, ax = plt.subplots()
        years = np.arange(1990, 2010)
        vals = np.random.default_rng(0).normal(100, 10, len(years))
        std = np.full(len(years), 5.0)
        era_shaded_ts(ax, years, vals, std, label='test')
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_none_std_skips_fill(self):
        fig, ax = plt.subplots()
        years = np.arange(2000, 2010)
        vals = np.ones(len(years))
        era_shaded_ts(ax, years, vals, std_vals=None)
        # No fill_between collections when std is None
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_era_backgrounds_drawn(self):
        fig, ax = plt.subplots()
        years = np.arange(1896, 2100)
        vals = np.ones(len(years))
        era_shaded_ts(ax, years, vals, std_vals=None)
        # axvspan creates patches; should have one per era
        assert len(ax.patches) >= len(ERA_PERIODS)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Physics-Informed ML tests
# ═══════════════════════════════════════════════════════════════════════════

class TestIrrigationDemandFloor:
    """Tests for compute_irrigation_demand_floor()."""

    def _make_data(self, n=100):
        rng = np.random.default_rng(42)
        x = pd.DataFrame({
            'annual_et_ensemble_mm': rng.uniform(200, 800, n),
            'annual_peff_mm': rng.uniform(50, 300, n),
            'annual_irr_fraction': rng.uniform(0, 1, n),
            'annual_gw_fraction': rng.uniform(0, 1, n),
            'well_density': rng.uniform(0, 10, n),
        })
        basins = pd.Series(rng.choice(['PHOENIX AMA', 'TUCSON AMA', 'PINAL AMA'], n))
        nhm_ie = {
            'mean': {'PHOENIX AMA': 0.65, 'TUCSON AMA': 0.70, 'PINAL AMA': 0.60},
            'overall_mean': 0.65,
        }
        return x, basins, nhm_ie

    def test_output_shape(self):
        x, basins, nhm_ie = self._make_data()
        wd = x['well_density'].values
        wd_p99 = np.percentile(wd[wd > 0], 99)
        y_floor = compute_irrigation_demand_floor(x, basins, nhm_ie, wd, wd_p99)
        assert y_floor.shape == (len(x),)

    def test_non_negative(self):
        x, basins, nhm_ie = self._make_data()
        wd = x['well_density'].values
        wd_p99 = np.percentile(wd[wd > 0], 99)
        y_floor = compute_irrigation_demand_floor(x, basins, nhm_ie, wd, wd_p99)
        assert np.all(y_floor >= 0)

    def test_zero_for_non_irrigated(self):
        x, basins, nhm_ie = self._make_data(10)
        x['annual_irr_fraction'] = 0.0
        wd = x['well_density'].values
        wd_p99 = np.percentile(wd[wd > 0], 99)
        y_floor = compute_irrigation_demand_floor(x, basins, nhm_ie, wd, wd_p99)
        np.testing.assert_array_equal(y_floor, 0.0)

    def test_zero_for_zero_well_density(self):
        """Floor is zero where well_density is zero."""
        x, basins, nhm_ie = self._make_data(20)
        x['well_density'] = 0.0
        wd = x['well_density'].values
        y_floor = compute_irrigation_demand_floor(x, basins, nhm_ie, wd, well_density_p99=1.0)
        np.testing.assert_array_equal(y_floor, 0.0)

    def test_unknown_basin_uses_fallback(self):
        x, _, nhm_ie = self._make_data(5)
        basins = pd.Series(['UNKNOWN'] * 5)
        wd = x['well_density'].values
        wd_p99 = np.percentile(wd[wd > 0], 99)
        y_floor = compute_irrigation_demand_floor(x, basins, nhm_ie, wd, wd_p99)
        # Should not raise; uses overall_mean IE
        assert y_floor.shape == (5,)


class TestMonotoneConstraints:
    """Tests for _build_monotone_constraints()."""

    def test_known_features(self):
        features = ['annual_et_ensemble_mm', 'well_density', 'annual_peff_mm', 'AGRI']
        mc = _build_monotone_constraints(features)
        expected = tuple(MONOTONE_CONSTRAINT_MAP.get(f, 0) for f in features)
        assert mc == expected

    def test_unconstrained_features(self):
        features = ['streamflow_mm', 'canal_density', 'soil_depth_mm']
        mc = _build_monotone_constraints(features)
        assert mc == (0, 0, 0)

    def test_mixed(self):
        features = ['annual_et_ensemble_mm', 'streamflow_mm', 'annual_peff_mm']
        mc = _build_monotone_constraints(features)
        expected = tuple(MONOTONE_CONSTRAINT_MAP.get(f, 0) for f in features)
        assert mc == expected


class TestInteractionConstraints:
    """Tests for _build_interaction_constraints()."""

    def test_returns_lists_of_names(self):
        features = ['annual_et_ensemble_mm', 'annual_peff_mm',
                     'annual_irr_fraction', 'annual_gw_fraction',
                     'well_density', 'AGRI', 'URBAN', 'annual_crop_fraction']
        ic = _build_interaction_constraints(features)
        assert ic is not None
        assert all(isinstance(g, list) for g in ic)
        # All names should be valid feature names
        feature_set = set(features)
        for group in ic:
            for name in group:
                assert name in feature_set

    def test_disabled_returns_none(self):
        features = ['annual_et_ensemble_mm', 'annual_peff_mm']
        ic = _build_interaction_constraints(features, use_interaction_constraints=False)
        assert ic is None


class TestPhysicsBoundsObjective:
    """Tests for PhysicsBoundsObjective gradient/hessian (floor + ceiling)."""

    def test_gradient_hessian_shapes(self):
        n = 50
        rng = np.random.default_rng(42)
        y_floor = rng.uniform(0, 5, n)
        y_ceil = rng.uniform(8, 15, n)
        obj = PhysicsBoundsObjective(y_floor, y_ceil, 0.1, 0.1)

        y_true = rng.uniform(0, 6, n)
        y_pred = rng.uniform(0, 6, n)
        grad, hess = obj(y_true, y_pred)
        assert grad.shape == (n,)
        assert hess.shape == (n,)

    def test_both_lambdas_zero_is_pure_mse(self):
        """When both lambdas=0, objective reduces to pure MSE gradient."""
        n = 20
        y_floor = np.ones(n) * 3.0
        y_ceil = np.ones(n) * 10.0
        obj = PhysicsBoundsObjective(y_floor, y_ceil, 0.0, 0.0)

        y_true = np.linspace(2, 4, n)
        y_pred = np.linspace(1, 5, n)
        grad, hess = obj(y_true, y_pred)

        expected_grad = y_pred - y_true
        expected_hess = np.full(n, 1.0)
        np.testing.assert_allclose(grad, expected_grad)
        np.testing.assert_allclose(hess, expected_hess)

    def test_backward_compat_floor_only(self):
        """make_physics_floor_objective still works (floor only, no ceiling)."""
        n = 20
        y_phys_log = np.ones(n) * 3.0
        obj = make_physics_floor_objective(y_phys_log, physics_lambda=0.0)

        y_true = np.linspace(2, 4, n)
        y_pred = np.linspace(1, 5, n)
        grad, hess = obj(y_true, y_pred)

        expected_grad = y_pred - y_true
        np.testing.assert_allclose(grad, expected_grad)
        np.testing.assert_allclose(hess, np.full(n, 1.0))

    def test_numerical_gradient_floor_and_ceil(self):
        """Verify gradient against finite differences with both penalties."""
        n = 30
        rng = np.random.default_rng(42)
        y_floor = rng.uniform(1, 4, n)
        y_ceil = rng.uniform(6, 10, n)
        lam_f, lam_c = 0.2, 0.15
        obj = PhysicsBoundsObjective(y_floor, y_ceil, lam_f, lam_c)

        y_true = rng.uniform(1, 4, n)
        y_pred = rng.uniform(0, 12, n)

        grad, _ = obj(y_true, y_pred)

        # Numerical gradient via central differences
        eps = 1e-5
        num_grad = np.zeros(n)
        floor_mask = (y_floor > 0).astype(np.float64)
        ceil_mask = (y_ceil > 0).astype(np.float64)
        for i in range(n):
            yp_plus = y_pred.copy(); yp_plus[i] += eps
            yp_minus = y_pred.copy(); yp_minus[i] -= eps

            def loss(yp):
                res = (yp - y_true) ** 2
                f_viol = np.clip(y_floor - yp, 0, None)
                c_viol = np.clip(yp - y_ceil, 0, None)
                return (0.5 * res
                        + 0.5 * lam_f * floor_mask * f_viol ** 2
                        + 0.5 * lam_c * ceil_mask * c_viol ** 2).sum()

            num_grad[i] = (loss(yp_plus) - loss(yp_minus)) / (2 * eps)

        np.testing.assert_allclose(grad, num_grad, atol=1e-4)

    def test_floor_one_sided_penalty(self):
        """Floor penalty only applies when prediction is below floor."""
        n = 10
        y_floor = np.full(n, 3.0)
        y_ceil = np.full(n, 10.0)
        obj = PhysicsBoundsObjective(y_floor, y_ceil, 0.3, 0.3)

        # Predictions above floor but below ceiling — no penalty
        y_true = np.full(n, 4.0)
        y_pred = np.full(n, 5.0)
        grad, hess = obj(y_true, y_pred)

        # Pure MSE grad (no floor or ceiling penalty)
        expected_grad = y_pred - y_true
        np.testing.assert_allclose(grad, expected_grad)
        np.testing.assert_allclose(hess, np.full(n, 1.0))

    def test_ceiling_penalty_activates(self):
        """Ceiling penalty activates when prediction exceeds ceiling."""
        n = 10
        y_floor = np.full(n, 1.0)
        y_ceil = np.full(n, 5.0)
        lam_c = 0.3
        obj = PhysicsBoundsObjective(y_floor, y_ceil, 0.0, lam_c)

        y_true = np.full(n, 4.0)
        y_pred = np.full(n, 8.0)  # above ceiling
        grad, hess = obj(y_true, y_pred)

        # MSE grad + ceiling penalty (pushes DOWN)
        residual = y_pred - y_true
        c_viol = y_pred - y_ceil
        expected_grad = residual + lam_c * c_viol
        np.testing.assert_allclose(grad, expected_grad)
        assert np.all(hess > 1.0)  # hessian includes ceiling term


class TestPhysicsWrappers:
    """Tests for PhysicsXGBRegressor / PhysicsLGBMRegressor / PhysicsXGBRFRegressor."""

    def _make_train_data(self, n=200):
        rng = np.random.default_rng(42)
        x = pd.DataFrame({
            f'feat_{i}': rng.uniform(0, 1, n) for i in range(5)
        })
        # easting_m / northing_m needed for per-fold pixel bounds
        x['easting_m'] = rng.choice([100.0, 200.0, 300.0, 400.0], n)
        x['northing_m'] = rng.choice([500.0, 600.0, 700.0, 800.0], n)
        x[_PHYS_COL] = rng.uniform(0, 3, n)
        y = rng.uniform(0, 5, n)
        return x, y

    def test_physics_xgb_fit_predict(self):
        x, y = self._make_train_data()
        model = PhysicsXGBRegressor(
            physics_lambda_floor=0.1, physics_lambda_ceil=0.05,
            n_estimators=10, max_depth=3,
        )
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape == (len(x),)

    def test_physics_lgbm_fit_predict(self):
        x, y = self._make_train_data()
        model = PhysicsLGBMRegressor(
            physics_lambda_floor=0.1, physics_lambda_ceil=0.05,
            n_estimators=10, max_depth=3,
            num_leaves=8, verbosity=-1,
        )
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape == (len(x),)

    def test_physics_xgbrf_fit_predict(self):
        x, y = self._make_train_data()
        model = PhysicsXGBRFRegressor(
            physics_lambda_floor=0.1, physics_lambda_ceil=0.05,
            num_parallel_tree=10, max_depth=3,
        )
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape == (len(x),)

    def test_lambdas_zero_uses_squarederror(self):
        """When both lambdas=0, wrapper falls back to reg:squarederror."""
        x, y = self._make_train_data(50)
        model = PhysicsXGBRegressor(
            physics_lambda_floor=0.0, physics_lambda_ceil=0.0,
            n_estimators=5, max_depth=3,
        )
        model.fit(x, y)
        assert model.get_params()['objective'] == 'reg:squarederror'

    def test_predict_strips_phys_columns(self):
        """Predict should work even when physics columns are in X."""
        x, y = self._make_train_data(50)
        model = PhysicsXGBRegressor(
            physics_lambda_floor=0.1, physics_lambda_ceil=0.05,
            n_estimators=5, max_depth=3,
        )
        model.fit(x, y)
        # Predict with physics column present
        preds_with = model.predict(x)
        # Predict without physics column
        preds_without = model.predict(x.drop(columns=[_PHYS_COL]))
        np.testing.assert_array_equal(preds_with, preds_without)
