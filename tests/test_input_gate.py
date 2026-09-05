"""
Unit tests for Input Gate and Sensor Card — SatQuery EvidenceSwarm (SIH26167)
"""

import os
import pytest
from pathlib import Path
import numpy as np

from src.input_gate import (
    validate_geotiff,
    extract_metadata,
    detect_sensor,
    estimate_uncertainty
)
from src.sensor_card import generate_card, render_card
from scripts.generate_demo_geotiffs import create_all_demo_images


@pytest.fixture(scope="session", autouse=True)
def setup_demo_images():
    """Ensure demo images are generated before running tests."""
    demo_dir = Path("demo/images")
    create_all_demo_images(demo_dir)


def test_valid_sentinel2_optical():
    """Test 1: Valid Sentinel-2 tile passes validation and extracts rich metadata."""
    s2_file = Path("demo/images/sentinel2_urban_mumbai.tif")
    assert s2_file.exists()

    val_res = validate_geotiff(s2_file)
    assert val_res["valid"] is True, f"Validation failed: {val_res['errors']}"
    assert len(val_res["errors"]) == 0

    meta = extract_metadata(s2_file)
    assert meta["valid"] is True
    assert meta["band_count"] == 4
    assert meta["width"] == 512
    assert meta["height"] == 512
    assert "32643" in meta["crs"]
    assert len(meta["bbox_wgs84"]) == 4
    assert meta["center_wgs84"]["lon"] > 70.0  # Mumbai vicinity
    assert meta["center_wgs84"]["lat"] > 18.0

    sensor = detect_sensor(s2_file, meta)
    assert "Sentinel-2" in sensor

    u_score, u_label = estimate_uncertainty(meta, sensor)
    assert 0.0 <= u_score <= 0.35
    assert u_label in ("Low", "Moderate")


def test_valid_sentinel1_sar():
    """Test 2: Valid Sentinel-1 SAR tile passes validation and detects SAR mode."""
    s1_file = Path("demo/images/sentinel1_sar_mumbai.tif")
    assert s1_file.exists()

    val_res = validate_geotiff(s1_file)
    assert val_res["valid"] is True
    assert len(val_res["errors"]) == 0

    meta = extract_metadata(s1_file)
    assert meta["band_count"] == 2

    sensor = detect_sensor(s1_file, meta)
    assert "Sentinel-1" in sensor or "SAR" in sensor


def test_valid_cartosat():
    """Test 3: Valid Cartosat high-resolution panchromatic sample passes."""
    carto_file = Path("demo/images/cartosat2s_sample_delhi.tif")
    assert carto_file.exists()

    val_res = validate_geotiff(carto_file)
    assert val_res["valid"] is True

    meta = extract_metadata(carto_file)
    assert meta["band_count"] == 1
    assert meta["resolution_m"] <= 1.5

    sensor = detect_sensor(carto_file, meta)
    assert "Cartosat" in sensor


def test_reject_renamed_png():
    """Test 4: PNG renamed to .tif must be rejected by magic header verification."""
    png_file = Path("demo/images/test_png_renamed.tif")
    assert png_file.exists()

    val_res = validate_geotiff(png_file)
    assert val_res["valid"] is False
    assert any("PNG" in err for err in val_res["errors"])


def test_reject_corrupt_file():
    """Test 5: Corrupt / truncated file must be rejected."""
    corrupt_file = Path("demo/images/test_corrupt_file.tif")
    assert corrupt_file.exists()

    val_res = validate_geotiff(corrupt_file)
    assert val_res["valid"] is False
    assert len(val_res["errors"]) > 0


def test_reject_missing_crs():
    """Test 6: GeoTIFF without spatial CRS must be rejected."""
    no_crs_file = Path("demo/images/test_missing_crs.tif")
    if no_crs_file.exists():
        val_res = validate_geotiff(no_crs_file)
        assert val_res["valid"] is False
        assert any("CRS" in err or "Coordinate" in err for err in val_res["errors"])


def test_reject_nonexistent_file():
    """Test 7: Non-existent file path returns clean error."""
    val_res = validate_geotiff(Path("demo/images/does_not_exist_99.tif"))
    assert val_res["valid"] is False
    assert any("does not exist" in err for err in val_res["errors"])


def test_sensor_card_generation_and_render():
    """Test 8: Sensor Card generation and textual rendering format."""
    s2_file = Path("demo/images/sentinel2_urban_mumbai.tif")
    card = generate_card(file_path=s2_file)

    assert "sensor_type" in card
    assert card["sensor_type"] == "Sentinel-2"
    assert "resolution_m" in card
    assert "bands" in card
    assert len(card["bands"]) == 4
    assert "crs" in card
    assert "bbox" in card
    assert "uncertainty" in card
    assert "uncertainty_label" in card

    rendered = render_card(card)
    assert "ISRO SATQUERY SENSOR CARD" in rendered
    assert "SENTINEL-2" in rendered
    assert "Resolution" in rendered
