"""
Test circular plate functions against MATLAB reference implementation.

This test suite validates the Python implementations of circular plate dynamics
functions against their MATLAB counterparts in the VK-Gong repository.

Tests cover:
- Angular integration functions (cos_cos_cos_integration, cos_sin_sin_integration)
- H-coefficient computation (hcoefficient_circular)
- H-tensor computation (H_tensor_circular)
- Circular plate eigenvalues (circ_plate_transverse_eigenvalues, circ_plate_inplane_eigenvalues)
"""

import json
from pathlib import Path

import numpy as np
import pytest

# Import functions to test
from jaxdiffmodal.coupling import (
    H_tensor_circular,
    circ_plate_inplane_eigenvalues,
    circ_plate_transverse_eigenvalues,
    cos_cos_cos_integration,
    cos_sin_sin_integration,
    hcoefficient_circular,
)
from jaxdiffmodal.ftm import circ_laplacian_eigenvalues, circ_laplacian_wavenumbers


@pytest.fixture
def matlab_reference_data():
    """Load MATLAB reference data for comparison tests."""
    json_path = (
        Path(__file__).parent / "reference_data" / "test_circular_matlab_reference_results.json"
    )

    if not json_path.exists():
        pytest.skip(f"MATLAB reference data not found at {json_path}")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        pytest.skip(f"Failed to load MATLAB reference data: {e}")


class TestAngularIntegrationMatlabComparison:
    """Test angular integration functions against MATLAB reference."""

    def test_cos_cos_cos_integration_parametrized(self, matlab_reference_data):
        """Compare cos_cos_cos_integration with MATLAB reference for all test cases."""
        test_cases = matlab_reference_data["cos_cos_cos_test_cases"]
        matlab_results = matlab_reference_data["cos_cos_cos_results"]

        for i, (k, l, m) in enumerate(test_cases):
            matlab_result = matlab_results[i]
            python_result = cos_cos_cos_integration(k, l, m)

            # Use appropriate tolerance for trigonometric integrals
            tolerance = 1e-12
            if abs(matlab_result) > 1e-12:
                assert (
                    abs(python_result - matlab_result) / abs(matlab_result) < tolerance
                )
            else:
                assert abs(python_result - matlab_result) < tolerance

    def test_cos_sin_sin_integration_parametrized(self, matlab_reference_data):
        """Compare cos_sin_sin_integration with MATLAB reference for all test cases."""
        test_cases = matlab_reference_data["cos_sin_sin_test_cases"]
        matlab_results = matlab_reference_data["cos_sin_sin_results"]

        for i, (k, l, m) in enumerate(test_cases):
            matlab_result = matlab_results[i]
            python_result = cos_sin_sin_integration(k, l, m)

            # Use appropriate tolerance for trigonometric integrals
            tolerance = 1e-12
            if abs(matlab_result) > 1e-12:
                assert (
                    abs(python_result - matlab_result) / abs(matlab_result) < tolerance
                )
            else:
                assert abs(python_result - matlab_result) < tolerance


class TestHCoefficientMatlabComparison:
    """Test H-coefficient computation against MATLAB reference."""

    def test_hcoefficient_circular_parametrized(self, matlab_reference_data):
        """Compare hcoefficient_circular with MATLAB reference for all test cases."""
        test_cases = matlab_reference_data["hcoeff_test_cases"]
        matlab_results = matlab_reference_data["hcoeff_results"]

        for i, test_case in enumerate(test_cases):
            kp, kq, cp, cq, xip, xiq, ki, ci, zeta, nu, KR, dr_H = test_case
            matlab_result = matlab_results[i]
            python_result = hcoefficient_circular(
                kp, kq, cp, cq, xip, xiq, ki, ci, zeta, nu, KR, dr_H
            )

            tolerance = 1e-6
            if abs(matlab_result) > 1e-10:
                # Use relative tolerance for non-zero values
                relative_error = abs(python_result - matlab_result) / abs(matlab_result)
                assert relative_error < tolerance, (
                    f"Case {i}: relative error {relative_error:.2e} exceeds tolerance {tolerance} "
                    f"(MATLAB: {matlab_result:.8e}, Python: {python_result:.8e})"
                )
            else:
                # Use absolute tolerance for values near zero
                assert abs(python_result - matlab_result) < tolerance, (
                    f"Case {i}: absolute error {abs(python_result - matlab_result):.2e} exceeds tolerance {tolerance} "
                    f"(MATLAB: {matlab_result:.8e}, Python: {python_result:.8e})"
                )


class TestHTensorMatlabComparison:
    """Test H-tensor computation against MATLAB reference."""

    def test_H_tensor_circular_parametrized(self, matlab_reference_data):
        """Compare H_tensor_circular with MATLAB reference for all test cases."""
        test_cases = matlab_reference_data["H_tensor_test_cases"]

        # Skip H_tensor tests since MATLAB data wasn't generated (requires mode files)
        if not test_cases:
            pytest.skip(
                "H_tensor MATLAB reference data not available - requires pre-computed mode files"
            )

        # Get MATLAB results
        matlab_H0_results = matlab_reference_data["H_tensor_H0_results"]
        matlab_H1_results = matlab_reference_data["H_tensor_H1_results"]
        matlab_H2_results = matlab_reference_data["H_tensor_H2_results"]

        for i, test_case in enumerate(test_cases):
            # Extract parameters
            Nphi = test_case["Nphi"]
            Npsi = test_case["Npsi"]
            nu = test_case["nu"]
            KR = test_case["KR"]
            if KR is None:  # Handle clamped boundary condition
                KR = np.inf
            dr_H = test_case["dr_H"]

            # Convert MATLAB mode arrays to numpy arrays
            mode_t = np.array(test_case["mode_t"])
            mode_l = np.array(test_case["mode_l"])

            # Compute Python H tensors
            python_H0, python_H1, python_H2 = H_tensor_circular(
                mode_t,
                mode_l,
                Nphi,
                Npsi,
                nu=nu,
                KR=KR,
                dr_H=dr_H,
            )

            # Get MATLAB results for this test case and reshape to (Npsi, Nphi, Nphi)
            # Use order='F' to match MATLAB's column-major flattening with H0(:)'
            matlab_H0 = np.array(matlab_H0_results[i]).reshape(
                Npsi, Nphi, Nphi, order='F'
            )
            matlab_H1 = np.array(matlab_H1_results[i]).reshape(
                Npsi, Nphi, Nphi, order='F'
            )
            matlab_H2 = np.array(matlab_H2_results[i]).reshape(
                Npsi, Nphi, Nphi, order='F'
            )

            # Compare with MATLAB results
            # After fixing the tensor indexing issue, Python should match MATLAB exactly

            max_diff_H0 = np.max(np.abs(python_H0 - matlab_H0))
            max_diff_H1 = np.max(np.abs(python_H1 - matlab_H1))
            max_diff_H2 = np.max(np.abs(python_H2 - matlab_H2))

            print(
                f"Test case {i}: Max differences - H0: {max_diff_H0:.2e}, H1: {max_diff_H1:.2e}, H2: {max_diff_H2:.2e}"
            )

            # Verify that Python implementation produces finite, reasonable results
            assert np.all(
                np.isfinite(python_H0)
            ), f"Python H0 contains non-finite values for test case {i}"
            assert np.all(
                np.isfinite(python_H1)
            ), f"Python H1 contains non-finite values for test case {i}"
            assert np.all(
                np.isfinite(python_H2)
            ), f"Python H2 contains non-finite values for test case {i}"

            # Verify tensor shapes are correct
            assert python_H0.shape == (
                Npsi,
                Nphi,
                Nphi,
            ), f"H0 shape mismatch for test case {i}"
            assert python_H1.shape == (
                Npsi,
                Nphi,
                Nphi,
            ), f"H1 shape mismatch for test case {i}"
            assert python_H2.shape == (
                Npsi,
                Nphi,
                Nphi,
            ), f"H2 shape mismatch for test case {i}"

            # Assert machine precision accuracy (within numerical tolerance)
            tolerance_H0 = 1e-12  # Tolerance for H0 (base tensor)
            tolerance_H1 = 1e-13  # Tolerance for H1 (H0/zeta^2)
            tolerance_H2 = 1e-14  # Tolerance for H2 (H0/zeta^4)

            assert (
                max_diff_H0 < tolerance_H0
            ), f"H0 max difference {max_diff_H0:.2e} exceeds tolerance {tolerance_H0:.2e} for test case {i}"
            assert (
                max_diff_H1 < tolerance_H1
            ), f"H1 max difference {max_diff_H1:.2e} exceeds tolerance {tolerance_H1:.2e} for test case {i}"
            assert (
                max_diff_H2 < tolerance_H2
            ), f"H2 max difference {max_diff_H2:.2e} exceeds tolerance {tolerance_H2:.2e} for test case {i}"

            # Count exact matches for verification
            matlab_nonzero = np.abs(matlab_H0) > 1e-10
            python_nonzero = np.abs(python_H0) > 1e-10
            both_nonzero = matlab_nonzero & python_nonzero

            if np.any(both_nonzero):
                relative_errors = np.abs(
                    (python_H0[both_nonzero] - matlab_H0[both_nonzero])
                    / matlab_H0[both_nonzero]
                )
                perfect_matches = np.sum(relative_errors < 1e-12)
                print(
                    f"  Elements with machine precision match (< 1e-12 error): {perfect_matches}/{len(relative_errors)}"
                )

            print(f"  Test case {i} completed - Python results within tolerance")


class TestCircularEigenvaluesMatlabComparison:
    """Test circular plate eigenvalue functions against MATLAB reference."""

    def test_circ_plate_transverse_eigenvalues_parametrized(
        self, matlab_reference_data
    ):
        """Compare circ_plate_transverse_eigenvalues with MATLAB reference."""
        test_cases = matlab_reference_data["transverse_eig_test_cases"]
        matlab_results = matlab_reference_data["transverse_eig_results"]

        for i, test_case in enumerate(test_cases):
            dx, xmax, BC, nu, KR, KT = test_case

            # Compute eigenvalues using Python implementation
            python_mode_t, python_zeros = circ_plate_transverse_eigenvalues(
                dx, xmax, BC, nu, KR, KT
            )

            # Get MATLAB results
            matlab_mode_t = np.array(matlab_results[i]["mode_t"])
            matlab_zeros = np.array(matlab_results[i]["zeros"])

            # Compare mode table and zeros matrix
            tolerance = 1e-8
            assert np.allclose(
                python_mode_t, matlab_mode_t, rtol=tolerance, atol=tolerance
            )

            # Handle different matrix sizes - MATLAB may have extra zero rows
            min_rows = min(python_zeros.shape[0], matlab_zeros.shape[0])
            python_zeros_trimmed = python_zeros[:min_rows, :]
            matlab_zeros_trimmed = matlab_zeros[:min_rows, :]
            assert np.allclose(
                python_zeros_trimmed,
                matlab_zeros_trimmed,
                rtol=tolerance,
                atol=tolerance,
            )

    def test_circ_plate_inplane_eigenvalues_parametrized(self, matlab_reference_data):
        """Compare circ_plate_inplane_eigenvalues with MATLAB reference."""
        test_cases = matlab_reference_data["inplane_eig_test_cases"]
        matlab_results = matlab_reference_data["inplane_eig_results"]

        for i, test_case in enumerate(test_cases):
            dx, xmax, BC, nu = test_case

            # Compute eigenvalues using Python implementation
            python_mode_l, python_zeros = circ_plate_inplane_eigenvalues(
                dx, xmax, BC, nu
            )

            # Get MATLAB results
            matlab_mode_l = np.array(matlab_results[i]["mode_l"])
            matlab_zeros = np.array(matlab_results[i]["zeros"])

            # Compare mode table and zeros matrix
            tolerance = 1e-8
            assert np.allclose(
                python_mode_l, matlab_mode_l, rtol=tolerance, atol=tolerance
            )

            # Handle different matrix sizes - MATLAB may have extra zero rows
            min_rows = min(python_zeros.shape[0], matlab_zeros.shape[0])
            python_zeros_trimmed = python_zeros[:min_rows, :]
            matlab_zeros_trimmed = matlab_zeros[:min_rows, :]
            assert np.allclose(
                python_zeros_trimmed,
                matlab_zeros_trimmed,
                rtol=tolerance,
                atol=tolerance,
            )


class TestCircularBasicFunctionality:
    """Basic functionality tests without MATLAB dependency."""

    def test_angular_integration_basic(self):
        """Test angular integration functions with known values."""
        # Test known integration results
        result = cos_cos_cos_integration(0, 0, 0)  # Should be 2*pi
        assert abs(result - 2 * np.pi) < 1e-12

        result = cos_cos_cos_integration(1, 1, 0)  # Should be pi
        assert abs(result - np.pi) < 1e-12

        result = cos_cos_cos_integration(0, 1, 0)  # Should be 0 (orthogonal)
        assert abs(result) < 1e-12

    def test_hcoefficient_circular_basic(self):
        """Test H-coefficient computation with basic functionality."""
        # Test with simple parameters
        result = hcoefficient_circular(1, 1, 1, 1, 1.0, 1.0, 1, 1, 1.0, 0.3, 0.0, 0.01)
        assert np.isfinite(result)
        assert isinstance(result, float)

    def test_H_tensor_circular_basic(self):
        """Test H-tensor computation with basic functionality."""
        # Generate simple mode data for testing
        Nphi, Npsi = 2, 3

        # Create simple mode_t (transverse modes) - simplified structure
        mode_t = np.array(
            [
                [1, 2.404, 0, 1, 1, 5.783],  # (1,0,1) mode
                [2, 5.520, 0, 2, 1, 30.47],  # (2,0,1) mode
            ]
        )

        # Create simple mode_l (in-plane modes) - simplified structure
        mode_l = np.array(
            [
                [1, 2.404, 0, 1, 1, 5.783],  # (1,0,1) mode
                [2, 5.520, 0, 2, 1, 30.47],  # (2,0,1) mode
                [3, 8.654, 0, 3, 1, 74.89],  # (3,0,1) mode
            ]
        )

        H0, H1, H2 = H_tensor_circular(
            mode_t, mode_l, Nphi, Npsi, nu=0.3, KR=0.0, dr_H=0.01
        )

        # Check outputs are well-formed
        assert all(np.all(np.isfinite(H)) for H in [H0, H1, H2])
        assert H0.shape == H1.shape == H2.shape
        assert H0.shape == (Npsi, Nphi, Nphi)  # (3, 2, 2)

    def test_circular_eigenvalues_basic(self):
        """Test circular plate eigenvalue functions with basic functionality."""
        # Test transverse eigenvalues
        mode_t, zeros_t = circ_plate_transverse_eigenvalues(
            0.01, 20.0, "free", 0.3, 0.0, 0.0
        )
        assert mode_t.shape[1] == 6  # Standard mode table format
        assert np.all(np.isfinite(mode_t))

        # Test in-plane eigenvalues
        mode_l, zeros_l = circ_plate_inplane_eigenvalues(0.01, 20.0, "free", 0.3)
        assert mode_l.shape[1] == 6  # Standard mode table format
        assert np.all(np.isfinite(mode_l))

    def test_circ_laplacian_basic(self):
        """Test circular Laplacian functions with basic functionality."""
        # Test wavenumber computation
        wavenumbers = circ_laplacian_wavenumbers(3, 4, radius=1.0)
        assert wavenumbers.shape == (3, 4)
        assert np.all(wavenumbers >= 0)

        # Test eigenvalue computation
        eigenvalues = circ_laplacian_eigenvalues(wavenumbers)
        assert eigenvalues.shape == wavenumbers.shape
        assert np.all(eigenvalues >= 0)
        assert np.allclose(eigenvalues, wavenumbers**2)


class TestCircularSymmetryProperties:
    """Test mathematical properties and symmetries."""

    def test_angular_integration_symmetry(self):
        """Test symmetry properties of angular integration."""
        # cos_cos_cos should be symmetric in first two arguments
        result1 = cos_cos_cos_integration(2, 3, 1)
        result2 = cos_cos_cos_integration(3, 2, 1)
        assert abs(result1 - result2) < 1e-14

    def test_H_tensor_finite(self):
        """Test that H-tensor values are finite and well-behaved."""
        # Generate simple mode data for testing
        Nphi, Npsi = 3, 4

        # Create mode_t (transverse modes)
        mode_t = np.array(
            [
                [1, 2.404, 0, 1, 1, 5.783],  # (1,0,1) mode
                [2, 5.520, 0, 2, 1, 30.47],  # (2,0,1) mode
                [3, 8.654, 0, 3, 1, 74.89],  # (3,0,1) mode
            ]
        )

        # Create mode_l (in-plane modes)
        mode_l = np.array(
            [
                [1, 2.404, 0, 1, 1, 5.783],  # (1,0,1) mode
                [2, 5.520, 0, 2, 1, 30.47],  # (2,0,1) mode
                [3, 8.654, 0, 3, 1, 74.89],  # (3,0,1) mode
                [4, 11.79, 0, 4, 1, 139.0],  # (4,0,1) mode
            ]
        )

        H0, H1, H2 = H_tensor_circular(
            mode_t,
            mode_l,
            Nphi,
            Npsi,
            nu=0.3,
            KR=0.0,
            dr_H=0.01,
        )

        # All values should be finite
        assert np.all(np.isfinite(H0))
        assert np.all(np.isfinite(H1))
        assert np.all(np.isfinite(H2))

        # No NaN values
        assert not np.any(np.isnan(H0))
        assert not np.any(np.isnan(H1))
        assert not np.any(np.isnan(H2))
