"""
Phase 4 Test Suite — SatQuery EvidenceSwarm (SIH26167)
Tests:
1. Evidence Guard (Dual spectral + answer consistency verification)
2. Confidence Engine (Exact 4-factor formula, weights, labels)
3. Output Renderer (1-page PDF intelligence report generation using fpdf2)
4. Agentic Router (Rule overrides priority + semantic routing)
5. Export PDF API endpoint (/api/export-pdf)
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.evidence_guard import verify_evidence, compute_spectral_agreement, compute_answer_consistency
from src.confidence_engine import calculate_confidence, get_confidence_label, DEFAULT_WEIGHTS
from src.output_renderer import generate_pdf_report
from src.agentic_router import route_query
from scripts.generate_demo_geotiffs import create_all_demo_images

client = TestClient(app)
DEMO_DIR = Path("demo/images")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure demo GeoTIFFs are created."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    create_all_demo_images(DEMO_DIR)


# ═══════════════════════════════════════════════════════════════
# 1. EVIDENCE GUARD TESTS
# ═══════════════════════════════════════════════════════════════

def test_evidence_guard_spectral_optical():
    """Verify spectral check on optical Sentinel-2 tile."""
    img_p = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    res = compute_spectral_agreement([img_p], task_type="vqa")
    
    assert res["status"] == "pass"
    assert 0.70 <= res["c_spectral"] <= 1.0
    assert "metrics" in res
    assert "ndwi_mean" in res["metrics"]
    assert "ndvi_mean" in res["metrics"]


def test_evidence_guard_change_detection():
    """Verify evidence guard on bi-temporal change detection."""
    p1 = DEMO_DIR / "sentinel2_flood_pre_kerala.tif"
    p2 = DEMO_DIR / "sentinel2_flood_post_kerala.tif"
    res = verify_evidence(
        file_paths=[p1, p2],
        answer_or_summary="Inundation extent increased by 14.82% due to severe flooding.",
        query="What areas changed between these two dates?",
        task_type="change_detection",
        specialist_metrics={"change_pct": 14.82, "otsu_threshold": 48.5}
    )

    assert 0.70 <= res["c_spectral"] <= 1.0
    assert 0.70 <= res["c_guard"] <= 1.0
    assert "NDWI flood delta verified" in res["summary"] or "water" in res["summary"].lower() or "flood" in res["summary"].lower()


def test_evidence_guard_sar_fusion():
    """Verify evidence guard on multi-modal optical-SAR inputs."""
    p1 = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    p2 = DEMO_DIR / "sentinel1_sar_mumbai.tif"
    res = verify_evidence(
        file_paths=[p1, p2],
        answer_or_summary="Synthetic aperture radar reveals structures through cloud cover with roughness 0.76.",
        query="What features are visible in SAR but obscured in optical?",
        task_type="optical_sar_fusion",
        specialist_metrics={"roughness_index": 0.76, "mean_vv_intensity": 142.5}
    )

    assert 0.70 <= res["c_spectral"] <= 1.0
    assert 0.70 <= res["c_guard"] <= 1.0
    assert "roughness" in res["summary"].lower() or "backscatter" in res["summary"].lower()


# ═══════════════════════════════════════════════════════════════
# 2. CONFIDENCE ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

def test_confidence_engine_exact_formula():
    """Verify exact 4-factor weighted confidence calculation."""
    # Score = 0.15*C_sensor + 0.35*C_adapter + 0.25*C_guard + 0.25*C_spectral
    # With C_sensor=0.80, C_adapter=0.90, C_guard=0.88, C_spectral=0.84:
    # 0.15*0.80 + 0.35*0.90 + 0.25*0.88 + 0.25*0.84 = 0.12 + 0.315 + 0.22 + 0.21 = 0.865 -> 0.86 or 0.87
    res = calculate_confidence(
        c_sensor=0.80,
        c_adapter=0.90,
        c_guard=0.88,
        c_spectral=0.84
    )

    expected = round(0.15 * 0.80 + 0.35 * 0.90 + 0.25 * 0.88 + 0.25 * 0.84, 2)
    assert res["aggregate_score"] == expected
    assert res["label"] == "High Evidence Consistency"
    assert res["weights"] == DEFAULT_WEIGHTS
    assert res["breakdown"]["c_sensor"] == 0.80
    assert res["breakdown"]["c_adapter"] == 0.90
    assert res["breakdown"]["c_guard"] == 0.88
    assert res["breakdown"]["c_spectral"] == 0.84


def test_confidence_engine_labels():
    """Verify confidence threshold labels."""
    assert get_confidence_label(0.92) == "High Evidence Consistency"
    assert get_confidence_label(0.85) == "High Evidence Consistency"
    assert get_confidence_label(0.75) == "Moderate Evidence Consistency"
    assert get_confidence_label(0.65) == "Moderate Evidence Consistency"
    assert get_confidence_label(0.50) == "Low Evidence Consistency"
    assert get_confidence_label(0.0, is_refusal=True) == "Refusal / Insufficient Evidence"


def test_confidence_engine_refusal():
    """Verify refusal mode sets aggregate score to 0.0."""
    res = calculate_confidence(
        c_sensor=0.83,
        c_adapter=0.0,
        c_guard=0.0,
        c_spectral=0.0,
        is_refusal=True
    )
    assert res["aggregate_score"] == 0.0
    assert res["label"] == "Refusal / Insufficient Evidence"


# ═══════════════════════════════════════════════════════════════
# 3. OUTPUT RENDERER (PDF GENERATION) TESTS
# ═══════════════════════════════════════════════════════════════

def test_pdf_report_generation():
    """Verify fpdf2 generates valid PDF bytes with header signature."""
    report_data = {
        "task_type": "change_detection",
        "fidelity": "reduced",
        "method": "image-differencing-otsu",
        "status": "success",
        "query_text": "What areas changed between these two dates?",
        "answer_or_summary": "Bi-temporal change detection reveals a 14.82% surface flood expansion.",
        "sensor_card": {
            "sensor_type": "Sentinel-2",
            "resolution_m": 10.0,
            "crs": "EPSG:32643",
            "uncertainty": 0.17,
            "uncertainty_label": "Low"
        },
        "metrics": {
            "change_pct": 14.82,
            "changed_pixels": 38850,
            "otsu_threshold": 48.5
        },
        "confidence": {
            "aggregate_score": 0.84,
            "label": "Moderate Evidence Consistency",
            "breakdown": {
                "c_sensor": 0.83,
                "c_adapter": 0.80,
                "c_guard": 0.85,
                "c_spectral": 0.88
            }
        },
        "execution_trace": [
            {"stage": "Input Gate", "status": "pass", "time_ms": 8, "summary": "Validated 2 GeoTIFFs"},
            {"stage": "Evidence Guard", "status": "pass", "time_ms": 22, "summary": "NDWI flood delta verified"}
        ]
    }

    pdf_bytes = generate_pdf_report(report_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")


# ═══════════════════════════════════════════════════════════════
# 4. AGENTIC ROUTER TESTS (RULE OVERRIDES + SEMANTIC ROUTING)
# ═══════════════════════════════════════════════════════════════

def test_router_rule_priority():
    """Verify rule and keyword overrides take priority over embeddings."""
    p_opt = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    p_sar = DEMO_DIR / "sentinel1_sar_mumbai.tif"

    # Beat 1 override
    r1 = route_query("Any arbitrary text", [p_opt], beat=1)
    assert r1["task_type"] == "vqa"
    assert r1["routing_method"] == "rule_override"

    # Beat 2 override
    r2 = route_query("Any arbitrary text", [p_opt, p_opt], beat=2)
    assert r2["task_type"] == "change_detection"
    assert r2["routing_method"] == "rule_override"

    # SAR keyword with SAR file
    r3 = route_query("Penetrate radar cloud cover", [p_opt, p_sar])
    assert r3["task_type"] == "optical_sar_fusion"
    assert r3["routing_method"] == "rule_override"

    # Change keyword with 2 files
    r4 = route_query("What areas changed between the two dates?", [p_opt, p_opt])
    assert r4["task_type"] == "change_detection"
    assert r4["routing_method"] == "rule_override"


def test_router_semantic_general_query():
    """Verify general questions route properly to VQA with bounding box grounding."""
    p_opt = DEMO_DIR / "sentinel2_urban_mumbai.tif"
    r = route_query("Identify the coastal marine inlets and urban structures", [p_opt])
    assert r["task_type"] == "vqa"
    assert "specialist" in r
    assert r["specialist"] == "vqa_specialist"


# ═══════════════════════════════════════════════════════════════
# 5. API ENDPOINTS INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

def test_api_export_pdf_endpoint():
    """Verify POST /api/export-pdf returns valid PDF download."""
    payload = {
        "task_type": "vqa",
        "fidelity": "full",
        "status": "success",
        "query_text": "What land cover types are visible?",
        "answer_or_summary": "The scene features coastal water and urban fabric.",
        "sensor_card": {"sensor_type": "Sentinel-2", "resolution_m": 10.0, "crs": "EPSG:32643"},
        "confidence": {
            "aggregate_score": 0.89,
            "label": "High Evidence Consistency",
            "breakdown": {"c_sensor": 0.83, "c_adapter": 0.93, "c_guard": 0.90, "c_spectral": 0.88}
        }
    }

    response = client.post("/api/export-pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")


def test_health_telemetry_vqa_mode():
    """Verify GET /health includes vqa_engine_mode."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "vqa_engine_mode" in data
    assert any(mode in data["vqa_engine_mode"] for mode in ["GPU", "CPU Fallback"])
