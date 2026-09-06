"""
Integration Test Suite — SatQuery EvidenceSwarm (SIH26167)
Tests the 4 locked demo beats, adversarial refusal scenarios, SQLite trace logging,
and records execution latency across all pipeline stages.
"""

import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.app import app
from scripts.generate_demo_geotiffs import create_all_demo_images

client = TestClient(app)
DEMO_DIR = Path("demo/images")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure demo directory and images are created."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    create_all_demo_images(DEMO_DIR)


def validate_response_envelope(res_json: dict, expected_status: str = "success"):
    """Helper validator asserting compliance with docs/api_schema.md."""
    assert "status" in res_json
    assert res_json["status"] == expected_status
    assert "fidelity" in res_json
    assert "task_type" in res_json
    assert "method" in res_json
    assert "answer_or_summary" in res_json
    assert "visuals" in res_json
    assert "metrics" in res_json
    assert "execution_trace" in res_json
    assert isinstance(res_json["execution_trace"], list)


def test_health_endpoint():
    """Verify GET /health returns online status and Phase 3 subsystem telemetry."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["mission_id"] == "SIH26167"
    assert data["version"] == "3.0.0"
    assert "subsystems" in data
    assert data["subsystems"]["evidence_contract"] == "active_strict_refusal"
    assert data["subsystems"]["trace_logger"] == "active_sqlite"


def test_sensor_card_get_endpoint():
    """Verify GET /sensor-card and GET /api/sensor-card return real Sensor Card."""
    response = client.get("/api/sensor-card?filename=sentinel2_urban_mumbai.tif")
    assert response.status_code == 200
    data = response.json()
    assert data["sensor_type"] == "Sentinel-2"
    assert "resolution_m" in data
    assert "bands" in data
    assert len(data["bands"]) == 4
    assert "uncertainty" in data
    assert "rendered_card" in data


def test_beat1_vqa_baseline():
    """Beat 1: Single-Image VQA Baseline on Sentinel-2 + latency measurement."""
    img_path = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    t0 = time.perf_counter()
    with open(img_path, "rb") as f:
        response = client.post(
            "/api/query",
            data={"query": "What land cover types are visible in this image?", "beat": 1},
            files=[("files", ("sentinel2_urban_mumbai.tif", f, "image/tiff"))]
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[LATENCY BENCHMARK] Beat 1 (VQA Baseline): {latency_ms:.2f} ms")

    assert response.status_code == 200
    data = response.json()
    validate_response_envelope(data, expected_status="success")
    assert data["fidelity"] in ["full", "base"]
    assert data["task_type"] == "vqa"
    assert "coastal" in data["answer_or_summary"].lower() or "urban" in data["answer_or_summary"].lower()
    assert "confidence" in data
    assert data["confidence"]["aggregate_score"] >= 0.85


def test_beat2_change_detection():
    """Beat 2: Bi-Temporal Change Detection (Inundation / Delta) + latency measurement."""
    pre_path = DEMO_DIR / "sentinel2_flood_pre_kerala.tif"
    post_path = DEMO_DIR / "sentinel2_flood_post_kerala.tif"
    t0 = time.perf_counter()
    with open(pre_path, "rb") as f1, open(post_path, "rb") as f2:
        response = client.post(
            "/api/query",
            data={"query": "What areas changed between these two dates?", "beat": 2},
            files=[
                ("files", ("sentinel2_flood_pre_kerala.tif", f1, "image/tiff")),
                ("files", ("sentinel2_flood_post_kerala.tif", f2, "image/tiff"))
            ]
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[LATENCY BENCHMARK] Beat 2 (Change Detection): {latency_ms:.2f} ms")

    assert response.status_code == 200
    data = response.json()
    validate_response_envelope(data, expected_status="success")
    assert data["fidelity"] == "reduced"
    assert data["task_type"] == "change_detection"
    assert "change_pct" in data["metrics"]
    assert data["metrics"]["change_pct"] > 0.0


def test_beat3_optical_sar_fusion():
    """Beat 3: Optical-SAR Multi-Sensor Cross-Modal Fusion + latency measurement."""
    opt_path = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    sar_path = DEMO_DIR / "sentinel1_sar_mumbai.tif"
    t0 = time.perf_counter()
    with open(opt_path, "rb") as f1, open(sar_path, "rb") as f2:
        response = client.post(
            "/api/query",
            data={"query": "What features are visible in SAR but obscured in optical?", "beat": 3},
            files=[
                ("files", ("sentinel2_urban_mumbai.tif", f1, "image/tiff")),
                ("files", ("sentinel1_sar_mumbai.tif", f2, "image/tiff"))
            ]
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[LATENCY BENCHMARK] Beat 3 (Optical-SAR Fusion): {latency_ms:.2f} ms")

    assert response.status_code == 200
    data = response.json()
    validate_response_envelope(data, expected_status="success")
    assert data["fidelity"] == "reduced"
    assert data["task_type"] == "optical_sar_fusion"
    assert "mean_vv_intensity" in data["metrics"]


def test_beat4_signature_refusal():
    """Beat 4: Signature Refusal Gate (Evidence Contract rejection on single tile)."""
    img_path = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    t0 = time.perf_counter()
    with open(img_path, "rb") as f:
        response = client.post(
            "/api/query",
            data={"query": "Show me changes between the two dates", "beat": 4},
            files=[("files", ("sentinel2_urban_mumbai.tif", f, "image/tiff"))]
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[LATENCY BENCHMARK] Beat 4 (Signature Refusal): {latency_ms:.2f} ms")

    assert response.status_code == 200
    data = response.json()
    validate_response_envelope(data, expected_status="refused")
    assert data["fidelity"] == "refusal_gate"
    assert "Requires two co-registered temporal GeoTIFF" in data["reason"]
    assert "suggestion" in data
    assert data["confidence"]["aggregate_score"] == 0.0


def test_adversarial_empty_query():
    """Adversarial test: Empty query string returns refusal envelope."""
    img_path = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    with open(img_path, "rb") as f:
        response = client.post(
            "/api/query",
            data={"query": ""},
            files=[("files", ("sentinel2_urban_mumbai.tif", f, "image/tiff"))]
        )
    assert response.status_code == 200
    data = response.json()
    validate_response_envelope(data, expected_status="refused")
    assert "empty or invalid" in data["reason"].lower()


def test_adversarial_corrupt_file():
    """Adversarial test: Corrupted / invalid file bytes trigger Input Gate refusal."""
    corrupt_bytes = b"NOT_A_REAL_GEOTIFF_HEADER_DATA_12345"
    response = client.post(
        "/api/query",
        data={"query": "What land cover types are visible?"},
        files=[("files", ("corrupted_sample.tif", corrupt_bytes, "image/tiff"))]
    )
    assert response.status_code == 200
    data = response.json()
    validate_response_envelope(data, expected_status="refused")
    assert data["task_type"] == "input_validation"


def test_traces_endpoint_sqlite_persistence():
    """Verify GET /api/traces retrieves persisted records from SQLite data/traces.db."""
    response = client.get("/api/traces?limit=10")
    assert response.status_code == 200
    traces = response.json()
    assert isinstance(traces, list)
    assert len(traces) > 0
    first = traces[0]
    assert "query_id" in first
    assert "timestamp" in first
    assert "task_type" in first
    assert "status" in first
    assert "trace_stages" in first
    assert isinstance(first["trace_stages"], list)
