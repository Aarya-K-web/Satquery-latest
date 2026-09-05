"""
Unit tests for Spectral Check (NDWI, NDVI, Otsu Thresholding, Mask IoU)
SatQuery EvidenceSwarm (SIH26167)
"""

import time
import pytest
from pathlib import Path
import numpy as np

from src.spectral_check import (
    compute_ndwi,
    compute_ndvi,
    otsu_threshold,
    compute_overlap
)
from scripts.generate_demo_geotiffs import create_all_demo_images


@pytest.fixture(scope="session", autouse=True)
def setup_demo_images():
    """Ensure test images exist."""
    demo_dir = Path("demo/images")
    create_all_demo_images(demo_dir)


def test_ndwi_calculation():
    """Test 1: NDWI correctly identifies water bodies with positive index values."""
    s2_file = Path("demo/images/sentinel2_urban_mumbai.tif")
    ndwi = compute_ndwi(s2_file, green_band=2, nir_band=4)

    assert isinstance(ndwi, np.ndarray)
    assert ndwi.shape == (512, 512)
    assert ndwi.dtype == np.float32
    assert -1.0 <= np.nanmin(ndwi)
    assert np.nanmax(ndwi) <= 1.0

    # In our Mumbai test image, leftmost pixels (x < 100) are coastal water
    water_region = ndwi[:, :100]
    land_region = ndwi[:, 350:]
    assert np.mean(water_region) > np.mean(land_region)
    assert np.mean(water_region) > 0.2


def test_ndvi_calculation():
    """Test 2: NDVI correctly highlights vegetated pixels with high positive index values."""
    s2_file = Path("demo/images/sentinel2_urban_mumbai.tif")
    ndvi = compute_ndvi(s2_file, red_band=3, nir_band=4)

    assert isinstance(ndvi, np.ndarray)
    assert ndvi.shape == (512, 512)
    assert ndvi.dtype == np.float32
    assert -1.0 <= np.nanmin(ndvi)
    assert np.nanmax(ndvi) <= 1.0

    # Vegetated region in top-right (x > 350, y < 200)
    veg_region = ndvi[:200, 350:]
    water_region = ndvi[:, :100]
    assert np.mean(veg_region) > np.mean(water_region)
    assert np.mean(veg_region) > 0.3


def test_otsu_thresholding():
    """Test 3: Otsu thresholding creates binary mask and finds optimal threshold."""
    s2_file = Path("demo/images/sentinel2_urban_mumbai.tif")
    ndwi = compute_ndwi(s2_file)

    mask, thresh = otsu_threshold(ndwi)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == ndwi.shape
    assert mask.dtype == bool
    assert -1.0 <= thresh <= 1.0
    # There should be both water and non-water pixels
    assert np.sum(mask) > 0
    assert np.sum(~mask) > 0


def test_compute_overlap_iou():
    """Test 4: IoU overlap calculation between masks."""
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_a[20:80, 20:80] = True

    # Identical mask -> IoU = 1.0
    assert compute_overlap(mask_a, mask_a) == 1.0

    # Completely disjoint mask -> IoU = 0.0
    mask_b = np.zeros((100, 100), dtype=bool)
    mask_b[0:10, 0:10] = True
    assert compute_overlap(mask_a, mask_b) == 0.0

    # Partial overlap
    mask_c = np.zeros((100, 100), dtype=bool)
    mask_c[20:80, 20:50] = True  # half of mask_a
    overlap = compute_overlap(mask_a, mask_c)
    assert 0.4 < overlap < 0.6


def test_spectral_execution_latency():
    """Test 5: Execution time must be < 2.0 seconds per tile on CPU."""
    s2_file = Path("demo/images/sentinel2_urban_mumbai.tif")

    start_time = time.perf_counter()
    ndwi = compute_ndwi(s2_file)
    mask, thresh = otsu_threshold(ndwi)
    elapsed = time.perf_counter() - start_time

    assert elapsed < 2.0, f"Spectral check took {elapsed:.3f}s (> 2.0s limit)"
