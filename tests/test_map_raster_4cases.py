"""
Comprehensive regression tests for map_raster covering all 4 cases:

1. No antimeridian crossing, no NaN
2. No antimeridian crossing, with NaN
3. With antimeridian crossing, no NaN
4. With antimeridian crossing, with NaN

Uses fake_dataset and fake_ecmwf_0100_1h from tools_test (no I/O).
"""

import numpy as np
from tools_test import build_footprint, fake_dataset, fake_ecmwf_0100_1h

from mapraster.main import map_raster


def test_no_antimeridian_no_nan():
    """
    Case 1: No antimeridian crossing, no NaN in ECMWF.
    """
    # Create datasets
    sar_dataset = fake_dataset(cross_antimeridian=False)
    raster = fake_ecmwf_0100_1h(to180=True, with_nan=False)
    footprint = build_footprint(sar_dataset)

    # Run map_raster
    result = map_raster(
        raster_ds=raster,
        originalDataset=sar_dataset,
        footprint=footprint,
        cross_antimeridian=False,
    )

    # Basic checks
    assert set(result.data_vars) == {"U10", "V10"}, "Missing variables"
    assert result["U10"].shape == sar_dataset["longitude"].shape, "Shape mismatch"
    assert result["V10"].shape == sar_dataset["longitude"].shape, "Shape mismatch"

    # Check no full NaN
    assert not np.all(np.isnan(result["U10"].values)), "U10 is all NaN"
    assert not np.all(np.isnan(result["V10"].values)), "V10 is all NaN"

    # Reference values (from first run)
    reference_values = {
        "U10": {
            "mean": 6.778062602480738,
            "std": 0.0141060230914798,
        },
        "V10": {
            "mean": -0.8602113795325333,
            "std": 0.1465258384034195,
        },
    }

    # Compare
    u10_mean = float(np.mean(result["U10"].values))
    u10_std = float(np.std(result["U10"].values))
    v10_mean = float(np.mean(result["V10"].values))
    v10_std = float(np.std(result["V10"].values))

    tolerance = 1e-6
    assert abs(u10_mean - reference_values["U10"]["mean"]) < tolerance
    assert abs(u10_std - reference_values["U10"]["std"]) < tolerance
    assert abs(v10_mean - reference_values["V10"]["mean"]) < tolerance
    assert abs(v10_std - reference_values["V10"]["std"]) < tolerance

    print("✓ Case 1 passed: No antimeridian, no NaN")


def test_no_antimeridian_with_nan():
    """
    Case 2: No antimeridian crossing, with NaN in ECMWF.
    """
    # Create datasets
    sar_dataset = fake_dataset(cross_antimeridian=False)
    raster = fake_ecmwf_0100_1h(to180=True, with_nan=True)
    footprint = build_footprint(sar_dataset)

    # Run map_raster
    result = map_raster(
        raster_ds=raster,
        originalDataset=sar_dataset,
        footprint=footprint,
        cross_antimeridian=False,
    )

    # Basic checks
    assert set(result.data_vars) == {"U10", "V10"}, "Missing variables"
    assert result["U10"].shape == sar_dataset["longitude"].shape, "Shape mismatch"

    # In this case, the SAR region doesn't overlap with NaN zone
    # So we might not have NaN, which is fine
    # Just check not all NaN
    assert not np.all(np.isnan(result["U10"].values)), "U10 is all NaN"

    # Reference values (from first run, using nanmean/nanstd)
    reference_values = {
        "U10": {
            "nanmean": 6.778062602480738,
            "nanstd": 0.0141060230914798,
            "nan_ratio": 0.0,  # No NaN in this region
        },
        "V10": {
            "nanmean": -0.8602113795325333,
            "nanstd": 0.1465258384034195,
        },
    }

    u10_mean = float(np.nanmean(result["U10"].values))
    u10_std = float(np.nanstd(result["U10"].values))
    v10_mean = float(np.nanmean(result["V10"].values))
    v10_std = float(np.nanstd(result["V10"].values))

    # More relaxed tolerance for NaN case (interpolation differences)
    tolerance = 1e-5
    assert abs(u10_mean - reference_values["U10"]["nanmean"]) < tolerance or np.isnan(
        u10_mean
    )
    assert abs(v10_mean - reference_values["V10"]["nanmean"]) < tolerance or np.isnan(
        v10_mean
    )
    assert abs(u10_std - reference_values["U10"]["nanstd"]) < tolerance
    assert abs(v10_std - reference_values["V10"]["nanstd"]) < tolerance

    print("✓ Case 2 passed: No antimeridian, with NaN")


def test_with_antimeridian_no_nan():
    """
    Case 3: With antimeridian crossing, no NaN in ECMWF.
    """
    # Create datasets
    sar_dataset = fake_dataset(cross_antimeridian=True)
    # Use 0-360 for antimeridian
    raster = fake_ecmwf_0100_1h(to180=False, with_nan=False)
    footprint = build_footprint(sar_dataset)

    # Run map_raster
    result = map_raster(
        raster_ds=raster,
        originalDataset=sar_dataset,
        footprint=footprint,
        cross_antimeridian=True,
    )

    # Basic checks
    assert set(result.data_vars) == {"U10", "V10"}, "Missing variables"
    assert result["U10"].shape == sar_dataset["longitude"].shape, "Shape mismatch"

    # Check no full NaN
    assert not np.all(np.isnan(result["U10"].values)), "U10 is all NaN"
    assert not np.all(np.isnan(result["V10"].values)), "V10 is all NaN"

    # Reference values (from first run)
    reference_values = {
        "U10": {
            "mean": 6.778062602480738,
            "std": 0.01410602309147989,
        },
        "V10": {
            "mean": 0.1933075747771534,
            "std": 0.16163073381182927,
        },
    }

    u10_mean = float(np.mean(result["U10"].values))
    u10_std = float(np.std(result["U10"].values))
    v10_mean = float(np.mean(result["V10"].values))
    v10_std = float(np.std(result["V10"].values))

    tolerance = 1e-6
    assert abs(u10_mean - reference_values["U10"]["mean"]) < tolerance
    assert abs(u10_std - reference_values["U10"]["std"]) < tolerance
    assert abs(v10_mean - reference_values["V10"]["mean"]) < tolerance
    assert abs(v10_std - reference_values["V10"]["std"]) < tolerance

    print("✓ Case 3 passed: With antimeridian, no NaN")


def test_with_antimeridian_with_nan():
    """
    Case 4: With antimeridian crossing, with NaN in ECMWF.
    """
    # Create datasets
    sar_dataset = fake_dataset(cross_antimeridian=True)
    # Use 0-360 for antimeridian
    raster = fake_ecmwf_0100_1h(to180=False, with_nan=True)
    footprint = build_footprint(sar_dataset)

    # Run map_raster
    result = map_raster(
        raster_ds=raster,
        originalDataset=sar_dataset,
        footprint=footprint,
        cross_antimeridian=True,
    )

    # Basic checks
    assert set(result.data_vars) == {"U10", "V10"}, "Missing variables"
    assert result["U10"].shape == sar_dataset["longitude"].shape, "Shape mismatch"

    # Should have some NaN but not all
    assert not np.all(np.isnan(result["U10"].values)), "U10 is all NaN"

    # Reference values (from first run)
    reference_values = {
        "U10": {
            "nanmean": 6.780413853544618,
            "nanstd": 0.01312769010318517,
            "nan_ratio": 0.6206666666666667,  # ~62% NaN due to overlap with NaN zone
        },
        "V10": {
            "nanmean": 0.2644554730704113,
            "nanstd": 0.18536406900329389,
        },
    }

    u10_mean = float(np.nanmean(result["U10"].values))
    u10_std = float(np.nanstd(result["U10"].values))
    v10_mean = float(np.nanmean(result["V10"].values))
    v10_std = float(np.nanstd(result["V10"].values))
    nan_ratio = float(np.isnan(result["U10"].values).mean())

    tolerance = 1e-8
    assert abs(u10_mean - reference_values["U10"]["nanmean"]) < tolerance
    assert abs(v10_mean - reference_values["V10"]["nanmean"]) < tolerance
    assert abs(u10_std - reference_values["U10"]["nanstd"]) < tolerance
    assert abs(v10_std - reference_values["V10"]["nanstd"]) < tolerance
    # Check NaN ratio is reasonable (within 5% tolerance)
    assert abs(nan_ratio - reference_values["U10"]["nan_ratio"]) < 0.05

    print("✓ Case 4 passed: With antimeridian, with NaN")


def test_all_cases():
    """
    Run all 4 test cases and display summary.
    """
    print("\n" + "=" * 50)
    print("Running all 4 map_raster regression test cases")
    print("=" * 50)

    print("\n[1/4] No antimeridian, no NaN...")
    test_no_antimeridian_no_nan()

    print("\n[2/4] No antimeridian, with NaN...")
    test_no_antimeridian_with_nan()

    print("\n[3/4] With antimeridian, no NaN...")
    test_with_antimeridian_no_nan()

    print("\n[4/4] With antimeridian, with NaN...")
    test_with_antimeridian_with_nan()

    print("\n" + "=" * 50)
    print("✓ All 4 test cases passed successfully!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    test_all_cases()
