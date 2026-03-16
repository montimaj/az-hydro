"""
Unit tests for core hydrolibs modules: sysops, partitionops.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

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
