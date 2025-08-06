"""
Pytest-based tests for i*_mat functions comparing Python implementations with MATLAB reference.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jaxdiffmodal.coupling import (
    i1_mat,
    i2_mat,
    i3_mat,
    i4_mat,
    i5_mat,
    i9_mat,
    i10_mat,
    i11_mat,
    i12_mat,
    i13_mat,
)


@pytest.fixture(scope="module")
def matlab_reference_data():
    """Load MATLAB reference data for comparison tests."""
    json_path = (
        Path(__file__).parent / "reference_data" / "test_imat_matlab_reference_results.json"
    )

    if not json_path.exists():
        pytest.skip(f"MATLAB reference data not found at {json_path}")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        pytest.skip(f"Failed to load MATLAB reference data: {e}")


def reshape_matlab_result(flat_result, Npsi, Nphi):
    """
    Reshape flattened MATLAB result back to 3D array.
    MATLAB uses column-major ordering, Python uses row-major.
    """
    if isinstance(flat_result, (list, tuple)):
        flat_array = np.array(flat_result)
    else:
        flat_array = flat_result

    # Reshape from flattened array to 3D: (Npsi, Nphi, Nphi)
    # MATLAB stores in column-major order, so we need to reshape accordingly
    return flat_array.reshape((Npsi, Nphi, Nphi), order="F")


def compare_3d_arrays(python_result, matlab_result, Npsi, Nphi, tolerance=1e-8):
    """
    Compare 3D arrays with appropriate tolerance handling.
    """
    # Reshape MATLAB result
    matlab_3d = reshape_matlab_result(matlab_result, Npsi, Nphi)

    # Check shapes match
    assert python_result.shape == matlab_3d.shape, (
        f"Shape mismatch: Python {python_result.shape} vs MATLAB {matlab_3d.shape}"
    )

    # Element-wise comparison with tolerance
    diff = np.abs(python_result - matlab_3d)
    max_diff = np.max(diff)

    # Use relative tolerance for large values, absolute for small values
    matlab_max = np.max(np.abs(matlab_3d))
    if matlab_max > 1e-10:
        relative_error = max_diff / matlab_max
        assert relative_error < tolerance, (
            f"Relative error {relative_error} exceeds tolerance {tolerance}. Max difference: {max_diff}"
        )
    else:
        assert max_diff < tolerance, (
            f"Absolute error {max_diff} exceeds tolerance {tolerance}"
        )


class TestI1MatMatlabComparison:
    """Test i1_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))  # All test cases
    def test_i1_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i1_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        # Get test case parameters
        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        # Get MATLAB result
        matlab_result = data["i1_results"][test_idx]

        # Skip if MATLAB returned NaN or error
        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i1_mat({Npsi}, {Nphi}, {L})")

        # Calculate Python result
        python_result = i1_mat(Npsi, Nphi, L)

        # Compare with appropriate tolerance
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI2MatMatlabComparison:
    """Test i2_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i2_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i2_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i2_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i2_mat({Npsi}, {Nphi}, {L})")

        python_result = i2_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI3MatMatlabComparison:
    """Test i3_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i3_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i3_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i3_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i3_mat({Npsi}, {Nphi}, {L})")

        python_result = i3_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI4MatMatlabComparison:
    """Test i4_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i4_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i4_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i4_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i4_mat({Npsi}, {Nphi}, {L})")

        python_result = i4_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI5MatMatlabComparison:
    """Test i5_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i5_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i5_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i5_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i5_mat({Npsi}, {Nphi}, {L})")

        python_result = i5_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI9MatMatlabComparison:
    """Test i9_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i9_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i9_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i9_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i9_mat({Npsi}, {Nphi}, {L})")

        python_result = i9_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI10MatMatlabComparison:
    """Test i10_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i10_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i10_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i10_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i10_mat({Npsi}, {Nphi}, {L})")

        python_result = i10_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI11MatMatlabComparison:
    """Test i11_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i11_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i11_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i11_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i11_mat({Npsi}, {Nphi}, {L})")

        python_result = i11_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI12MatMatlabComparison:
    """Test i12_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i12_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i12_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i12_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i12_mat({Npsi}, {Nphi}, {L})")

        python_result = i12_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestI13MatMatlabComparison:
    """Test i13_mat function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(64))
    def test_i13_mat_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare i13_mat Python implementation with MATLAB reference."""
        data = matlab_reference_data

        Npsi, Nphi, L = data["test_cases"][test_idx]
        Npsi, Nphi = int(Npsi), int(Nphi)

        matlab_result = data["i13_results"][test_idx]

        if matlab_result is None or (
            isinstance(matlab_result, float) and np.isnan(matlab_result)
        ):
            pytest.skip(f"MATLAB returned NaN/error for i13_mat({Npsi}, {Nphi}, {L})")

        python_result = i13_mat(Npsi, Nphi, L)
        compare_3d_arrays(python_result, matlab_result, Npsi, Nphi)


class TestIMatFunctionsBasic:
    """Basic tests for i*_mat functions without MATLAB dependency."""

    @pytest.mark.parametrize(
        "Npsi,Nphi,L", [(2, 2, 1.0), (3, 3, 1.5), (4, 2, 0.5), (2, 4, 2.0)]
    )
    def test_imat_functions_return_correct_shapes(self, Npsi, Nphi, L):
        """Test that all i*_mat functions return arrays with correct shapes."""
        expected_shape = (Npsi, Nphi, Nphi)

        functions = [
            i1_mat,
            i2_mat,
            i3_mat,
            i4_mat,
            i5_mat,
            i9_mat,
            i10_mat,
            i11_mat,
            i12_mat,
            i13_mat,
        ]

        for func in functions:
            result = func(Npsi, Nphi, L)
            assert result.shape == expected_shape, (
                f"{func.__name__}({Npsi}, {Nphi}, {L}) returned shape {result.shape}, expected {expected_shape}"
            )

    @pytest.mark.parametrize("Npsi,Nphi,L", [(2, 2, 1.0), (3, 3, 1.5), (5, 4, 2.0)])
    def test_imat_functions_return_finite_values(self, Npsi, Nphi, L):
        """Test that all i*_mat functions return finite values."""
        functions = [
            i1_mat,
            i2_mat,
            i3_mat,
            i4_mat,
            i5_mat,
            i9_mat,
            i10_mat,
            i11_mat,
            i12_mat,
            i13_mat,
        ]

        for func in functions:
            result = func(Npsi, Nphi, L)
            assert np.all(np.isfinite(result)), (
                f"{func.__name__}({Npsi}, {Nphi}, {L}) returned non-finite values"
            )

    def test_imat_functions_scale_with_L(self):
        """Test that i*_mat functions scale appropriately with L parameter."""
        Npsi, Nphi = 3, 3
        L1, L2 = 1.0, 2.0

        # Most functions should scale linearly with L
        functions = [i1_mat, i2_mat, i3_mat, i4_mat, i5_mat]

        for func in functions:
            result1 = func(Npsi, Nphi, L1)
            result2 = func(Npsi, Nphi, L2)

            # Check that non-zero elements scale by factor of 2
            nonzero_mask = np.abs(result1) > 1e-12
            if np.any(nonzero_mask):
                ratio = result2[nonzero_mask] / result1[nonzero_mask]
                # Allow some tolerance for numerical precision
                assert np.allclose(ratio, L2 / L1, rtol=1e-10), (
                    f"{func.__name__} does not scale linearly with L"
                )


class TestKnownIMatValues:
    """Test against manually verified values for specific cases."""

    def test_i1_mat_known_values(self):
        """Test i1_mat against known values."""
        # Test case: Npsi=2, Nphi=2, L=1.0
        result = i1_mat(2, 2, 1.0)

        # From MATLAB: i1_mat(2,2,1.0) should have specific structure
        # Only check that we get the right shape and finite values
        assert result.shape == (2, 2, 2)
        assert np.all(np.isfinite(result))
        # More specific value checks can be added based on MATLAB reference

    def test_i2_mat_known_values(self):
        """Test i2_mat against known values."""
        result = i2_mat(2, 2, 1.0)
        assert result.shape == (2, 2, 2)
        assert np.all(np.isfinite(result))
