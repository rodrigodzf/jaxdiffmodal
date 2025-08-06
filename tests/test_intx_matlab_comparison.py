"""
Pytest-based tests for intX functions comparing Python implementations with MATLAB reference.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jaxdiffmodal.coupling import int1, int2, int4


@pytest.fixture(scope="module")
def matlab_reference_data():
    """Load MATLAB reference data for comparison tests."""
    json_path = (
        Path(__file__).parent / "reference_data" / "test_intx_matlab_reference_results.json"
    )

    if not json_path.exists():
        pytest.skip(f"MATLAB reference data not found at {json_path}")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        pytest.skip(f"Failed to load MATLAB reference data: {e}")


class TestInt1MatlabComparison:
    """Test int1 function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(144))  # All test cases
    def test_int1_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare int1 Python implementation with MATLAB reference."""
        data = matlab_reference_data

        # Get test case parameters
        L, m, p = data["test_cases"][test_idx]
        m, p = int(m), int(p)

        # Get MATLAB result
        matlab_result = data["int1_results"][test_idx]

        # Skip if MATLAB returned NaN
        if np.isnan(matlab_result):
            pytest.skip(f"MATLAB returned NaN for int1({m}, {p}, {L})")

        # Calculate Python result
        python_result = int1(m, p, L)

        # Compare with appropriate tolerance
        tolerance = 1e-8
        if abs(matlab_result) > 1e-10:
            # Use relative tolerance for non-zero values
            assert (
                abs(python_result - matlab_result) / abs(matlab_result) < tolerance
            ), f"int1({m}, {p}, {L}): Python={python_result}, MATLAB={matlab_result}"
        else:
            # Use absolute tolerance for values near zero
            assert abs(python_result - matlab_result) < tolerance, (
                f"int1({m}, {p}, {L}): Python={python_result}, MATLAB={matlab_result}"
            )


class TestInt2MatlabComparison:
    """Test int2 function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(144))  # All test cases
    def test_int2_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare int2 Python implementation with MATLAB reference."""
        data = matlab_reference_data

        # Get test case parameters
        L, m, p = data["test_cases"][test_idx]
        m, p = int(m), int(p)

        # Get MATLAB result
        matlab_result = data["int2_results"][test_idx]

        # Skip if MATLAB returned NaN
        if np.isnan(matlab_result):
            pytest.skip(f"MATLAB returned NaN for int2({m}, {p}, {L})")

        # Calculate Python result
        python_result = int2(m, p, L)

        # Compare with appropriate tolerance
        tolerance = 1e-8
        if abs(matlab_result) > 1e-10:
            # Use relative tolerance for non-zero values
            assert (
                abs(python_result - matlab_result) / abs(matlab_result) < tolerance
            ), f"int2({m}, {p}, {L}): Python={python_result}, MATLAB={matlab_result}"
        else:
            # Use absolute tolerance for values near zero
            assert abs(python_result - matlab_result) < tolerance, (
                f"int2({m}, {p}, {L}): Python={python_result}, MATLAB={matlab_result}"
            )


class TestInt4MatlabComparison:
    """Test int4 function against MATLAB reference."""

    @pytest.mark.parametrize("test_idx", range(144))  # All test cases
    def test_int4_matlab_comparison(self, matlab_reference_data, test_idx):
        """Compare int4 Python implementation with MATLAB reference."""
        data = matlab_reference_data

        # Get test case parameters
        L, m, p = data["test_cases"][test_idx]
        m, p = int(m), int(p)

        # Get MATLAB result
        matlab_result = data["int4_results"][test_idx]

        # Skip if MATLAB returned NaN
        if np.isnan(matlab_result):
            pytest.skip(f"MATLAB returned NaN for int4({m}, {p}, {L})")

        # Calculate Python result
        python_result = int4(m, p, L)

        # Compare with appropriate tolerance
        tolerance = 1e-8
        if abs(matlab_result) > 1e-10:
            # Use relative tolerance for non-zero values
            assert (
                abs(python_result - matlab_result) / abs(matlab_result) < tolerance
            ), f"int4({m}, {p}, {L}): Python={python_result}, MATLAB={matlab_result}"
        else:
            # Use absolute tolerance for values near zero
            assert abs(python_result - matlab_result) < tolerance, (
                f"int4({m}, {p}, {L}): Python={python_result}, MATLAB={matlab_result}"
            )


class TestIntXFunctionsBasic:
    """Basic tests for intX functions without MATLAB dependency."""

    def test_int1_basic_cases(self):
        """Test basic cases for int1 function."""
        # Test (0,0) case
        assert int1(0, 0, 1.0) == 720.0

        # Test zero cases
        assert int1(0, 1, 1.0) == 0.0
        assert int1(1, 0, 1.0) == 0.0

        # Test symmetry
        assert int1(1, 2, 1.0) == int1(2, 1, 1.0)
        assert int1(2, 3, 1.5) == int1(3, 2, 1.5)

    def test_int2_basic_cases(self):
        """Test basic cases for int2 function."""
        # Test (0,0) case
        assert abs(int2(0, 0, 1.0) - 10.0 / 7.0) < 1e-10

        # Test symmetry
        assert int2(1, 2, 1.0) == int2(2, 1, 1.0)
        assert int2(2, 3, 1.5) == int2(3, 2, 1.5)

    def test_int4_basic_cases(self):
        """Test basic cases for int4 function."""
        # Test (0,0) case
        assert abs(int4(0, 0, 1.0) - 120.0 / 7.0) < 1e-10

        # Test symmetry
        assert int4(1, 2, 1.0) == int4(2, 1, 1.0)
        assert int4(2, 3, 1.5) == int4(3, 2, 1.5)

    @pytest.mark.parametrize("L", [0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("m,p", [(1, 2), (2, 3), (1, 4), (3, 5)])
    def test_intx_symmetry(self, L, m, p):
        """Test that all intX functions are symmetric: f(m,p) = f(p,m)."""
        assert np.isclose(int1(m, p, L), int1(p, m, L))
        assert np.isclose(int2(m, p, L), int2(p, m, L))
        assert np.isclose(int4(m, p, L), int4(p, m, L))

    @pytest.mark.parametrize("L", [0.1, 0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("m", [0, 1, 2, 3])
    @pytest.mark.parametrize("p", [0, 1, 2, 3])
    def test_intx_finite_values(self, L, m, p):
        """Test that all intX functions return finite values."""
        result1 = int1(m, p, L)
        result2 = int2(m, p, L)
        result4 = int4(m, p, L)

        assert np.isfinite(result1), f"int1({m},{p},{L}) is not finite"
        assert np.isfinite(result2), f"int2({m},{p},{L}) is not finite"
        assert np.isfinite(result4), f"int4({m},{p},{L}) is not finite"


class TestKnownMatlabValues:
    """Test against manually verified MATLAB values."""

    @pytest.mark.parametrize(
        "L,m,p,expected_int1,expected_int2,expected_int4",
        [
            (0.5, 0, 0, 5.76000000e03, 7.14285714e-01, 3.42857143e01),
            (1.0, 0, 0, 7.20000000e02, 1.42857143e00, 1.71428571e01),
            (0.5, 0, 1, 0.00000000e00, -0.00000000e00, -0.00000000e00),
            (0.5, 1, 1, 5.63636414e00, 8.99946323e-05, 1.59371418e-02),
            (1.0, 1, 1, 7.04545517e-01, 1.79989265e-04, 7.96857088e-03),
            (0.5, 2, 2, 4.74181826e02, 2.31651282e-03, 8.12879668e-01),
            (1.0, 2, 2, 5.92727283e01, 4.63302565e-03, 4.06439834e-01),
        ],
    )
    def test_known_matlab_values(
        self, L, m, p, expected_int1, expected_int2, expected_int4
    ):
        """Test against known MATLAB values extracted from previous runs."""
        tolerance = 1e-8

        # Test int1
        result1 = int1(m, p, L)
        if abs(expected_int1) > 1e-10:
            assert abs(result1 - expected_int1) / abs(expected_int1) < tolerance
        else:
            assert abs(result1 - expected_int1) < tolerance

        # Test int2
        result2 = int2(m, p, L)
        if abs(expected_int2) > 1e-10:
            assert abs(result2 - expected_int2) / abs(expected_int2) < tolerance
        else:
            assert abs(result2 - expected_int2) < tolerance

        # Test int4
        result4 = int4(m, p, L)
        if abs(expected_int4) > 1e-10:
            assert abs(result4 - expected_int4) / abs(expected_int4) < tolerance
        else:
            assert abs(result4 - expected_int4) < tolerance
