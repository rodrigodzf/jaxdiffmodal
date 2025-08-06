"""
Pytest-based tests for H_tensor_rectangular function,
comparing Python implementation with MATLAB reference data following the established pattern.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jaxdiffmodal.coupling import H_tensor_rectangular


@pytest.fixture
def matlab_reference_data():
    """Load MATLAB reference data for comparison tests."""
    json_path = (
        Path(__file__).parent / "reference_data" / "test_H_tensor_matlab_reference_results.json"
    )

    if not json_path.exists():
        pytest.skip(f"MATLAB reference data not found at {json_path}")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        pytest.skip(f"Failed to load MATLAB reference data: {e}")

def reshape_matlab_tensor(flat_result, S, Nphi):
    """Reshape flattened MATLAB tensor (column-major) to 2D numpy array."""
    return np.array(flat_result).reshape((S, Nphi * Nphi), order="F")


def compare_tensors(python_tensor, matlab_tensor, tolerance=1e-6):
    """Compare two tensors with relative and absolute tolerance."""
    assert python_tensor.shape == matlab_tensor.shape, (
        f"Shape mismatch: Python {python_tensor.shape} vs MATLAB {matlab_tensor.shape}"
    )

    assert np.allclose(python_tensor, matlab_tensor, rtol=tolerance, atol=tolerance), (
        f"Tensors do not match within tolerance {tolerance}"
    )


class TestHTensorRectangularMatlabComparison:
    """Tests for H_tensor_rectangular against MATLAB reference."""

    def test_H_tensor_rectangular_parametrized(self, matlab_reference_data):
        """Compare H_tensor_rectangular outputs using MATLAB coefficients as input for all test cases."""
        data = matlab_reference_data
        num_cases = len(data["test_cases"])

        for test_idx in range(num_cases):
            if test_idx >= len(data.get("H0_results", [])):
                pytest.skip(f"No H_tensor_rectangular data for case {test_idx}")

            Npsi, Nphi, Lx, Ly = [
                int(p) if i < 2 else p
                for i, p in enumerate(data["test_cases"][test_idx])
            ]
            kx_indices = data["kx_indices_results"][test_idx]
            ky_indices = data["ky_indices_results"][test_idx]

            # Check if MATLAB returned valid results
            matlab_H0_flat = data["H0_results"][test_idx]
            matlab_H1_flat = data["H1_results"][test_idx]
            matlab_H2_flat = data["H2_results"][test_idx]
            matlab_coeff0_flat = data["matlab_coeff0_inputs"][test_idx]
            matlab_coeff1_flat = data["matlab_coeff1_inputs"][test_idx]
            matlab_coeff2_flat = data["matlab_coeff2_inputs"][test_idx]

            if (
                matlab_H0_flat is None
                or (isinstance(matlab_H0_flat, float) and np.isnan(matlab_H0_flat))
                or matlab_coeff0_flat is None
                or (
                    isinstance(matlab_coeff0_flat, float)
                    and np.isnan(matlab_coeff0_flat)
                )
            ):
                pytest.skip(
                    f"MATLAB returned invalid H_tensor_rectangular results for case {test_idx} (Npsi={Npsi}, Nphi={Nphi}, Lx={Lx}, Ly={Ly})"
                )

            # Reconstruct MATLAB coefficient inputs
            # The MATLAB script computed coefficients following airy_stress_coefficients pattern
            # and applied S_final = floor(S/2) truncation
            S_final = int(
                np.sqrt(len(matlab_coeff0_flat))
            )  # Since coefficients are S_final x S_final

            matlab_coeff0 = np.array(matlab_coeff0_flat).reshape(
                (S_final, S_final), order="F"
            )
            matlab_coeff1 = np.array(matlab_coeff1_flat).reshape(
                (S_final, S_final), order="F"
            )
            matlab_coeff2 = np.array(matlab_coeff2_flat).reshape(
                (S_final, S_final), order="F"
            )

            # Use MATLAB coefficients as input to Python H_tensor_rectangular
            python_H0, python_H1, python_H2 = H_tensor_rectangular(
                matlab_coeff0,
                matlab_coeff1,
                matlab_coeff2,
                Nphi,
                Npsi,
                Lx,
                Ly,
                kx_indices,
                ky_indices,
            )

            # Reshape MATLAB H tensor results
            # MATLAB H tensors are S x (Nphi*Nphi)
            matlab_H0 = reshape_matlab_tensor(matlab_H0_flat, S_final, Nphi)
            matlab_H1 = reshape_matlab_tensor(matlab_H1_flat, S_final, Nphi)
            matlab_H2 = reshape_matlab_tensor(matlab_H2_flat, S_final, Nphi)

            # Compare H tensor results with appropriate tolerance
            tolerance = 1e-6

            try:
                compare_tensors(python_H0, matlab_H0, tolerance=tolerance)
            except AssertionError as e:
                pytest.fail(
                    f"H0 tensor comparison failed for case {test_idx} (Npsi={Npsi}, Nphi={Nphi}, Lx={Lx}, Ly={Ly}): {e}"
                )

            try:
                compare_tensors(python_H1, matlab_H1, tolerance=tolerance)
            except AssertionError as e:
                pytest.fail(
                    f"H1 tensor comparison failed for case {test_idx} (Npsi={Npsi}, Nphi={Nphi}, Lx={Lx}, Ly={Ly}): {e}"
                )

            try:
                compare_tensors(python_H2, matlab_H2, tolerance=tolerance)
            except AssertionError as e:
                pytest.fail(
                    f"H2 tensor comparison failed for case {test_idx} (Npsi={Npsi}, Nphi={Nphi}, Lx={Lx}, Ly={Ly}): {e}"
                )

    def test_H_tensor_rectangular_basic_functionality(self):
        """Test basic functionality of H_tensor_rectangular without MATLAB comparison."""
        from jaxdiffmodal.ftm import (
            plate_eigenvalues,
            plate_wavenumbers,
            select_modes_from_eigenvalues,
        )

        # Test with small problem size
        Npsi = 5
        Nphi = 3
        Lx, Ly = 1.0, 1.0

        # Use our function to get properly selected mode indices
        wnx, wny = plate_wavenumbers(Nphi, Nphi, Lx, Ly)
        lambda_mu_2d = plate_eigenvalues(wnx, wny)
        kx_indices, ky_indices, *_ = select_modes_from_eigenvalues(lambda_mu_2d, Nphi)

        # Create dummy coefficient matrices with realistic dimensions
        S = 4  # Small coefficient matrix size for testing
        coeff0 = np.random.randn(S, S) * 0.1
        coeff1 = np.random.randn(S, S) * 0.1
        coeff2 = np.random.randn(S, S) * 0.1

        # Test H_tensor_rectangular
        H0, H1, H2 = H_tensor_rectangular(
            coeff0, coeff1, coeff2, Nphi, Npsi, Lx, Ly, kx_indices, ky_indices
        )

        # Basic sanity checks
        expected_shape = (S, Nphi * Nphi)
        assert H0.shape == expected_shape, (
            f"H0 shape mismatch: expected {expected_shape}, got {H0.shape}"
        )
        assert H1.shape == expected_shape, (
            f"H1 shape mismatch: expected {expected_shape}, got {H1.shape}"
        )
        assert H2.shape == expected_shape, (
            f"H2 shape mismatch: expected {expected_shape}, got {H2.shape}"
        )

        assert np.all(np.isfinite(H0)), "H0 should contain only finite values"
        assert np.all(np.isfinite(H1)), "H1 should contain only finite values"
        assert np.all(np.isfinite(H2)), "H2 should contain only finite values"

    @pytest.mark.parametrize("Npsi,Nphi,Lx,Ly", [(5, 3, 1.0, 1.5), (10, 5, 0.8, 1.2)])
    def test_H_tensor_rectangular_parameter_variations(self, Npsi, Nphi, Lx, Ly):
        """Test H_tensor_rectangular with various parameter combinations."""
        from jaxdiffmodal.ftm import (
            plate_eigenvalues,
            plate_wavenumbers,
            select_modes_from_eigenvalues,
        )

        # Use our function to get properly selected mode indices
        wnx, wny = plate_wavenumbers(Nphi, Nphi, Lx, Ly)
        lambda_mu_2d = plate_eigenvalues(wnx, wny)
        kx_indices, ky_indices, *_ = select_modes_from_eigenvalues(lambda_mu_2d, Nphi)

        # Create small coefficient matrices
        S = min(3, Npsi)  # Keep computation manageable
        coeff0 = np.random.randn(S, S) * 0.1
        coeff1 = np.random.randn(S, S) * 0.1
        coeff2 = np.random.randn(S, S) * 0.1

        # Test function execution
        H0, H1, H2 = H_tensor_rectangular(
            coeff0, coeff1, coeff2, Nphi, Npsi, Lx, Ly, kx_indices, ky_indices
        )

        # Verify output shapes and finite values
        expected_shape = (S, Nphi * Nphi)
        assert H0.shape == expected_shape
        assert H1.shape == expected_shape
        assert H2.shape == expected_shape

        assert np.all(np.isfinite(H0))
        assert np.all(np.isfinite(H1))
        assert np.all(np.isfinite(H2))

    def test_H_tensor_rectangular_input_validation(self):
        """Test H_tensor_rectangular with edge cases and input validation."""
        from jaxdiffmodal.ftm import (
            plate_eigenvalues,
            plate_wavenumbers,
            select_modes_from_eigenvalues,
        )

        Npsi = 5
        Nphi = 3
        Lx, Ly = 1.0, 1.0

        # Use our function to get properly selected mode indices
        wnx, wny = plate_wavenumbers(Nphi, Nphi, Lx, Ly)
        lambda_mu_2d = plate_eigenvalues(wnx, wny)
        kx_indices, ky_indices, *_ = select_modes_from_eigenvalues(lambda_mu_2d, Nphi)

        # Test with minimal coefficient matrices
        S = 2
        coeff0 = np.random.randn(S, S) * 0.1
        coeff1 = np.random.randn(S, S) * 0.1
        coeff2 = np.random.randn(S, S) * 0.1

        # Should work with minimal inputs
        H0, H1, H2 = H_tensor_rectangular(
            coeff0, coeff1, coeff2, Nphi, Npsi, Lx, Ly, kx_indices, ky_indices
        )

        assert H0.shape == (S, Nphi * Nphi)
        assert H1.shape == (S, Nphi * Nphi)
        assert H2.shape == (S, Nphi * Nphi)


class TestHTensorRectangularIntegration:
    """Integration tests for H_tensor_rectangular with realistic workflows."""

    def test_integration_with_coefficient_computation(self):
        """Test H_tensor_rectangular in realistic integration with coefficient computation."""
        from scipy.linalg import eig

        from jaxdiffmodal.coupling import airy_stress_coefficients, assemble_K_and_M
        from jaxdiffmodal.ftm import (
            plate_eigenvalues,
            plate_wavenumbers,
            select_modes_from_eigenvalues,
        )

        # Small problem for integration test
        Npsi = 5
        Nphi = 3
        Lx, Ly = 1.0, 1.0

        # Use our function to get properly selected mode indices
        wnx, wny = plate_wavenumbers(Nphi, Nphi, Lx, Ly)
        lambda_mu_2d = plate_eigenvalues(wnx, wny)
        kx_indices, ky_indices, *_ = select_modes_from_eigenvalues(lambda_mu_2d, Nphi)

        # Complete workflow: K, M -> eigendecomposition -> coefficients -> H tensor
        K, M = assemble_K_and_M(Npsi, Lx, Ly)
        vals, vecs = eig(K, M)
        coeff0, coeff1, coeff2 = airy_stress_coefficients(Npsi, vals, vecs, Lx, Ly)

        # Test H_tensor_rectangular with computed coefficients
        H0, H1, H2 = H_tensor_rectangular(
            coeff0, coeff1, coeff2, Nphi, Npsi, Lx, Ly, kx_indices, ky_indices
        )

        # Verify realistic output dimensions and properties
        S = coeff0.shape[0]
        expected_shape = (S, Nphi * Nphi)

        assert H0.shape == expected_shape
        assert H1.shape == expected_shape
        assert H2.shape == expected_shape

        # H tensors should be finite and typically have some non-zero elements
        assert np.all(np.isfinite(H0))
        assert np.all(np.isfinite(H1))
        assert np.all(np.isfinite(H2))

        # At least some elements should be non-zero (unless very special case)
        assert (
            np.any(np.abs(H0) > 1e-10)
            or np.any(np.abs(H1) > 1e-10)
            or np.any(np.abs(H2) > 1e-10)
        ), "H tensors should have some non-zero elements in typical cases"
