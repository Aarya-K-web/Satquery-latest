"""
Unit Tests — Evidence Contract Module (SIH26167)
Tests all pre-flight verification rules and standardized signature refusal envelopes.
"""

from pathlib import Path
import pytest

from src.evidence_contract import evaluate_contract, calculate_bbox_iou, build_refusal_envelope
from scripts.generate_demo_geotiffs import create_all_demo_images

DEMO_DIR = Path("demo/images")


@pytest.fixture(scope="session", autouse=True)
def setup_demo_data():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    create_all_demo_images(DEMO_DIR)


def test_bbox_iou_calculation():
    """Verify 2D Bounding Box IoU arithmetic."""
    # Identical bounding boxes
    bbox1 = [72.8, 18.9, 73.0, 19.1]
    bbox2 = [72.8, 18.9, 73.0, 19.1]
    assert calculate_bbox_iou(bbox1, bbox2) == 1.0

    # Non-overlapping bounding boxes (Mumbai vs Kerala)
    bbox_mumbai = [72.825, 18.975, 73.075, 19.225]
    bbox_kerala = [76.15, 10.82, 76.35, 11.02]
    assert calculate_bbox_iou(bbox_mumbai, bbox_kerala) == 0.0

    # Partial overlap
    bbox_a = [0.0, 0.0, 2.0, 2.0]
    bbox_b = [1.0, 1.0, 3.0, 3.0]
    iou = calculate_bbox_iou(bbox_a, bbox_b)
    assert 0.1 < iou < 0.2


def test_refusal_single_image_change_detection():
    """Beat 4: Reject change detection query when only 1 image tile is supplied."""
    p1 = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    passed, refusal = evaluate_contract(
        query="Show me changes between the two dates",
        file_paths=[p1],
        beat=4
    )
    assert passed is False
    assert refusal is not None
    assert refusal["status"] == "refused"
    assert refusal["task_type"] == "change_detection"
    assert refusal["fidelity"] == "refusal_gate"
    assert "Requires two co-registered temporal GeoTIFF" in refusal["reason"]
    assert refusal["confidence"]["aggregate_score"] == 0.0
    assert "execution_trace" in refusal


def test_refusal_missing_sar_for_fusion():
    """Reject SAR fusion query when no SAR image is provided."""
    p1 = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    passed, refusal = evaluate_contract(
        query="What features are visible in SAR but obscured in optical?",
        file_paths=[p1],
        beat=3
    )
    assert passed is False
    assert refusal is not None
    assert refusal["status"] == "refused"
    assert refusal["task_type"] == "optical_sar_fusion"
    assert "Optical-SAR multi-sensor fusion requires at least two" in refusal["reason"]


def test_refusal_optical_only_for_sar_query():
    """Reject SAR query when two files are provided but both are optical."""
    p1 = DEMO_DIR / "sentinel2_flood_pre_kerala.tif"
    p2 = DEMO_DIR / "sentinel2_flood_post_kerala.tif"
    passed, refusal = evaluate_contract(
        query="Analyze SAR backscatter and radar penetration",
        file_paths=[p1, p2],
        sensor_types=["Sentinel-2 MSI", "Sentinel-2 MSI"]
    )
    assert passed is False
    assert refusal is not None
    assert "SAR modality missing" in refusal["answer_or_summary"]


def test_refusal_empty_query():
    """Reject empty or whitespace query string."""
    p1 = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    passed, refusal = evaluate_contract(
        query="   ",
        file_paths=[p1]
    )
    assert passed is False
    assert refusal is not None
    assert refusal["status"] == "refused"
    assert "empty or invalid" in refusal["reason"].lower()


def test_refusal_no_files():
    """Reject when no input files are provided."""
    passed, refusal = evaluate_contract(
        query="What is the dominant land cover?",
        file_paths=[]
    )
    assert passed is False
    assert refusal is not None
    assert "No satellite GeoTIFF files" in refusal["reason"]


def test_refusal_spatial_extent_mismatch():
    """Reject multi-image comparison when bounding boxes do not overlap at all."""
    p1 = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    p2 = DEMO_DIR / "sentinel2_flood_post_kerala.tif"
    meta1 = {"bbox_wgs84": [72.825, 18.975, 73.075, 19.225]}
    meta2 = {"bbox_wgs84": [76.15, 10.82, 76.35, 11.02]}

    passed, refusal = evaluate_contract(
        query="What changed between these scenes?",
        file_paths=[p1, p2],
        metadata_list=[meta1, meta2]
    )
    assert passed is False
    assert refusal is not None
    assert "Spatial extent mismatch" in refusal["reason"]
    assert refusal["metrics"]["spatial_iou"] == 0.0


def test_pass_valid_vqa():
    """Ensure valid single-image VQA query satisfies the contract."""
    p1 = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    passed, refusal = evaluate_contract(
        query="What land cover types are visible in this scene?",
        file_paths=[p1],
        beat=1
    )
    assert passed is True
    assert refusal is None


def test_pass_valid_change_detection():
    """Ensure valid 2-image change detection pair satisfies the contract."""
    p1 = DEMO_DIR / "sentinel2_flood_pre_kerala.tif"
    p2 = DEMO_DIR / "sentinel2_flood_post_kerala.tif"
    meta1 = {"bbox_wgs84": [76.15, 10.82, 76.35, 11.02]}
    meta2 = {"bbox_wgs84": [76.15, 10.82, 76.35, 11.02]}

    passed, refusal = evaluate_contract(
        query="What areas changed between these two dates?",
        file_paths=[p1, p2],
        beat=2,
        metadata_list=[meta1, meta2]
    )
    assert passed is True
    assert refusal is None


def test_pass_valid_optical_sar_fusion():
    """Ensure valid optical + SAR image pair satisfies the contract."""
    p1 = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    p2 = DEMO_DIR / "sentinel1_sar_mumbai.tif"
    meta1 = {"bbox_wgs84": [72.825, 18.975, 73.075, 19.225]}
    meta2 = {"bbox_wgs84": [72.825, 18.975, 73.075, 19.225]}

    passed, refusal = evaluate_contract(
        query="What features are visible in SAR but obscured in optical?",
        file_paths=[p1, p2],
        beat=3,
        sensor_types=["Sentinel-2 MSI", "Sentinel-1 C-SAR"],
        metadata_list=[meta1, meta2]
    )
    assert passed is True
    assert refusal is None
