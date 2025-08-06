"""Tests for evaluate_circular_modes function against MATLAB reference.

This module contains comprehensive tests comparing the Python implementation
of evaluate_circular_modes with MATLAB reference results.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from jaxdiffmodal.coupling import evaluate_circular_modes


@pytest.fixture
def matlab_reference_data():
    """Load MATLAB reference data for comparison tests."""
    json_path = (
        Path(__file__).parent / "reference_data" / "test_evaluate_circular_modes_matlab_reference_results.json"
    )

    if not json_path.exists():
        pytest.skip(f"MATLAB reference data not found at {json_path}")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        pytest.skip(f"Failed to load MATLAB reference data: {e}")


class TestEvaluateCircularModesMatlabComparison:
    """Tests comparing evaluate_circular_modes with MATLAB reference."""

    def test_evaluate_circular_modes_parametrized(self, matlab_reference_data):
        """Compare evaluate_circular_modes results with MATLAB reference for all test cases."""

        # Extract test parameters
        R_values = matlab_reference_data["R_values"]
        dr_values = matlab_reference_data["dr_values"]
        nu_values = matlab_reference_data["nu_values"]
        BC_values = matlab_reference_data["BC_values"][0]  # Handle JSON array wrapping
        test_points = np.array(matlab_reference_data["test_points"])

        # Mode tables
        mode_t_free = np.array(matlab_reference_data["mode_t_free"])
        mode_t_clamped = np.array(matlab_reference_data["mode_t_clamped"])
        mode_tables = [mode_t_free, mode_t_clamped]

        # Test cases and results
        test_cases = np.array(matlab_reference_data["test_cases"])
        matlab_results = matlab_reference_data["results"]
        num_cases = matlab_reference_data["num_cases"]

        tolerance = 1e-6  # Tolerance for floating-point comparison

        for case_idx in range(num_cases):
            # Skip cases with empty MATLAB results (failures)
            if not matlab_results[case_idx]:
                continue

            # Extract test parameters for this case (convert from 1-based to 0-based indexing)
            bc_idx, R_idx, dr_idx, nu_idx, pt_idx = test_cases[case_idx].astype(int) - 1

            BC = BC_values[bc_idx]
            R = R_values[R_idx]
            dr = dr_values[dr_idx]
            nu = nu_values[nu_idx]
            op = test_points[pt_idx]

            # Select appropriate mode table and KR value
            mode_t = mode_tables[bc_idx]
            KR = 0.0 if BC == "free" else np.inf

            # Get MATLAB reference result
            matlab_weights = np.array(matlab_results[case_idx])

            # Call Python implementation
            python_weights = evaluate_circular_modes(mode_t, nu, KR, op, BC, R, dr)

            # Compare results
            self._compare_weights(
                python_weights, matlab_weights, tolerance, case_idx, BC, R, dr, nu, op
            )

    def _compare_weights(
        self, python_weights, matlab_weights, tolerance, case_idx, BC, R, dr, nu, op
    ):
        """Compare Python and MATLAB weight results with appropriate tolerance."""

        assert len(python_weights) == len(matlab_weights), (
            f"Case {case_idx}: Length mismatch - Python: {len(python_weights)}, "
            f"MATLAB: {len(matlab_weights)}"
        )

        for mode_idx in range(len(python_weights)):
            python_val = python_weights[mode_idx]
            matlab_val = matlab_weights[mode_idx]

            # Handle near-zero values with absolute tolerance
            if abs(matlab_val) < 1e-10:
                error = abs(python_val - matlab_val)
                assert error < tolerance, (
                    f"Case {case_idx}, Mode {mode_idx}: Absolute error {error:.2e} "
                    f"exceeds tolerance {tolerance:.2e} for near-zero values. "
                    f"Python: {python_val:.8e}, MATLAB: {matlab_val:.8e}. "
                    f"Parameters: BC={BC}, R={R}, dr={dr}, nu={nu}, op={op}"
                )
            else:
                # Use relative tolerance for non-zero values
                relative_error = abs(python_val - matlab_val) / abs(matlab_val)
                assert relative_error < tolerance, (
                    f"Case {case_idx}, Mode {mode_idx}: Relative error {relative_error:.2e} "
                    f"exceeds tolerance {tolerance:.2e}. "
                    f"Python: {python_val:.8e}, MATLAB: {matlab_val:.8e}. "
                    f"Parameters: BC={BC}, R={R}, dr={dr}, nu={nu}, op={op}"
                )


class TestEvaluateCircularModesBasic:
    """Basic functionality tests for evaluate_circular_modes without MATLAB dependency."""

    def test_evaluate_circular_modes_basic_functionality(self):
        """Test basic functionality and input validation."""

        # Create simple test mode table
        mode_t = np.array(
            [
                [1, 3.196, 0, 1, 1, 10.214],  # (0,1) cos mode
                [2, 4.611, 1, 1, 1, 21.261],  # (1,1) cos mode
                [3, 4.611, 1, 1, 2, 21.261],  # (1,1) sin mode
            ]
        )

        nu = 0.3
        KR = np.inf  # Clamped
        op = np.array([np.pi / 4, 0.7])  # theta=45deg, r=0.7
        BC = "clamped"
        R = 1.0
        dr = 0.01

        # Call function
        weights = evaluate_circular_modes(mode_t, nu, KR, op, BC, R, dr)

        # Basic validation
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (3,)  # Should match number of modes
        assert np.all(np.isfinite(weights))  # All values should be finite

        # Test with free boundary conditions
        weights_free = evaluate_circular_modes(mode_t, nu, 0.0, op, "free", R, dr)
        assert isinstance(weights_free, np.ndarray)
        assert weights_free.shape == (3,)
        assert np.all(np.isfinite(weights_free))

        # Values should be different for different boundary conditions
        assert not np.allclose(weights, weights_free)

    def test_evaluate_circular_modes_input_validation(self):
        """Test input validation and error handling."""

        mode_t = np.array([[1, 3.196, 0, 1, 1, 10.214]])
        nu = 0.3
        KR = np.inf
        BC = "clamped"
        R = 1.0
        dr = 0.01

        # Test invalid op length
        with pytest.raises(ValueError, match="op must be a vector with 2 elements"):
            evaluate_circular_modes(mode_t, nu, KR, np.array([0.5]), BC, R, dr)

        with pytest.raises(ValueError, match="op must be a vector with 2 elements"):
            evaluate_circular_modes(
                mode_t, nu, KR, np.array([0.5, 0.7, 0.1]), BC, R, dr
            )

        # Test invalid boundary condition
        with pytest.raises(ValueError, match="Unknown boundary condition"):
            evaluate_circular_modes(
                mode_t, nu, KR, np.array([0.5, 0.7]), "invalid", R, dr
            )

    def test_evaluate_circular_modes_symmetry_properties(self):
        """Test symmetry and physical properties of the results."""

        # Create symmetric mode table with cos and sin modes
        mode_t = np.array(
            [
                [1, 3.196, 0, 1, 1, 10.214],  # (0,1) cos mode - axisymmetric
                [2, 4.611, 1, 1, 1, 21.261],  # (1,1) cos mode
                [3, 4.611, 1, 1, 2, 21.261],  # (1,1) sin mode
            ]
        )

        nu = 0.3
        KR = np.inf
        BC = "clamped"
        R = 1.0
        dr = 0.01

        # Test axisymmetric mode (k=0) should be independent of theta
        op1 = np.array([0.0, 0.5])  # theta=0
        op2 = np.array([np.pi / 2, 0.5])  # theta=90deg
        op3 = np.array([np.pi, 0.5])  # theta=180deg

        weights1 = evaluate_circular_modes(mode_t, nu, KR, op1, BC, R, dr)
        weights2 = evaluate_circular_modes(mode_t, nu, KR, op2, BC, R, dr)
        weights3 = evaluate_circular_modes(mode_t, nu, KR, op3, BC, R, dr)

        # Axisymmetric mode (index 0) should be the same for all theta values
        assert abs(weights1[0] - weights2[0]) < 1e-10
        assert abs(weights1[0] - weights3[0]) < 1e-10

        # Higher order modes should vary with theta
        assert abs(weights1[1] - weights2[1]) > 1e-6  # Should be different
        assert abs(weights1[2] - weights2[2]) > 1e-6  # Should be different

    def test_evaluate_circular_modes_boundary_conditions(self):
        """Test different boundary conditions produce different results."""

        mode_t = np.array(
            [
                [1, 3.196, 0, 1, 1, 10.214],  # Mode parameters
                [2, 4.611, 1, 1, 1, 21.261],
            ]
        )

        nu = 0.3
        op = np.array([np.pi / 4, 0.7])
        R = 1.0
        dr = 0.01

        # Test different boundary conditions
        weights_free = evaluate_circular_modes(mode_t, nu, 0.0, op, "free", R, dr)
        weights_elastic = evaluate_circular_modes(
            mode_t, nu, 10.0, op, "elastic", R, dr
        )
        weights_clamped = evaluate_circular_modes(
            mode_t, nu, np.inf, op, "clamped", R, dr
        )

        # Results should be different for different boundary conditions
        assert not np.allclose(weights_free, weights_clamped, rtol=1e-3)
        assert not np.allclose(weights_free, weights_elastic, rtol=1e-3)
        assert not np.allclose(weights_elastic, weights_clamped, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
