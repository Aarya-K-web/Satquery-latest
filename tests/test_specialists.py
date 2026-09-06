"""
Unit tests for Reduced-Fidelity CPU Specialists — SatQuery EvidenceSwarm (SIH26167)
Verifies caption_grounding, change_detection, and optical_sar_fusion on CPU.
"""

from pathlib import Path
import pytest

from src.specialists.caption_grounding import run as run_caption_grounding
from src.specialists.change_detection import run as run_change_detection
from src.specialists.optical_sar_fusion import run as run_optical_sar_fusion
from scripts.generate_demo_geotiffs import create_all_demo_images


DEMO_DIR = Path("demo/images")


@pytest.fixture(scope="session", autouse=True)
def ensure_demo_images():
    """Generates all synthetic demo GeoTIFFs if not present."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    create_all_demo_images(DEMO_DIR)


def test_caption_grounding_sentinel2():
    """Verify caption_grounding specialist on Sentinel-2 optical tile."""
    img_path = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    assert img_path.exists()

    res = run_caption_grounding(img_path, query="What land cover types are visible?")
    assert res["status"] == "success"
    assert res["task_type"] == "caption_grounding"
    assert res["fidelity"] == "reduced"
    assert res["method"] == "spectral-heuristic-grounding"
    assert len(res["answer_or_summary"]) > 20
    assert res["visuals"]["primary_b64"].startswith("data:image/png;base64,")
    assert res["visuals"]["overlay_b64"].startswith("data:image/png;base64,")
    assert "metrics" in res
    assert res["metrics"]["band_count"] == 4
    assert isinstance(res["regions"], list)
    assert len(res["execution_trace"]) > 0


def test_caption_grounding_cartosat():
    """Verify caption_grounding specialist on Cartosat panchromatic tile."""
    img_path = DEMO_DIR / "cartosat2s_sample_delhi.tif"
    assert img_path.exists()

    res = run_caption_grounding(img_path)
    assert res["status"] == "success"
    assert res["fidelity"] == "reduced"
    assert res["visuals"]["primary_b64"].startswith("data:image/png;base64,")


def test_change_detection_bitemporal():
    """Verify change_detection specialist on pre/post flood pair."""
    pre_path = DEMO_DIR / "sentinel2_flood_pre_kerala.tif"
    post_path = DEMO_DIR / "sentinel2_flood_post_kerala.tif"
    assert pre_path.exists() and post_path.exists()

    res = run_change_detection(pre_path, post_path, query="What areas changed between dates?")
    assert res["status"] == "success"
    assert res["task_type"] == "change_detection"
    assert res["fidelity"] == "reduced"
    assert res["method"] == "image-differencing-otsu"
    assert "change_pct" in res["metrics"]
    assert 0.0 <= res["metrics"]["change_pct"] <= 100.0
    assert res["visuals"]["primary_b64"].startswith("data:image/png;base64,")
    assert res["visuals"]["overlay_b64"].startswith("data:image/png;base64,")
    assert len(res["execution_trace"]) > 0


def test_optical_sar_fusion():
    """Verify optical_sar_fusion specialist on optical + SAR pair."""
    opt_path = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    sar_path = DEMO_DIR / "sentinel1_sar_mumbai.tif"
    assert opt_path.exists() and sar_path.exists()

    res = run_optical_sar_fusion(opt_path, sar_path, query="What is visible in SAR?")
    assert res["status"] == "success"
    assert res["task_type"] == "optical_sar_fusion"
    assert res["fidelity"] == "reduced"
    assert res["method"] == "band-overlay-heuristic"
    assert "sar_metrics" in res or "metrics" in res
    assert "roughness_index" in res["metrics"]
    assert res["visuals"]["primary_b64"].startswith("data:image/png;base64,")
    assert res["visuals"]["overlay_b64"].startswith("data:image/png;base64,")
    assert len(res["execution_trace"]) > 0
