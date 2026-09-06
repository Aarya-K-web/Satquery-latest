"""
Evidence Contract Module — SatQuery EvidenceSwarm (SIH26167)
Validates pre-flight conditions and generates standardized refusal envelopes
for all non-compliant queries, missing modalities, and geometric mismatches.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Computes 2D Intersection-over-Union (IoU) between two bounding boxes.
    Format: [min_lon, min_lat, max_lon, max_lat]
    """
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2

    inter_min_x = max(min_x1, min_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_x = min(max_x1, max_x2)
    inter_max_y = min(max_y1, max_y2)

    inter_w = max(0.0, inter_max_x - inter_min_x)
    inter_h = max(0.0, inter_max_y - inter_min_y)
    inter_area = inter_w * inter_h

    area1 = max(0.0, max_x1 - min_x1) * max(0.0, max_y1 - min_y1)
    area2 = max(0.0, max_x2 - min_x2) * max(0.0, max_y2 - min_y2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def build_refusal_envelope(
    task_type: str,
    reason: str,
    suggestion: str,
    summary: Optional[str] = None,
    method: str = "evidence_contract_preflight",
    metrics: Optional[Dict[str, Any]] = None,
    sensor_card: Optional[Dict[str, Any]] = None,
    globe_focus: Optional[Dict[str, Any]] = None,
    execution_trace: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Constructs the standardized signature refusal envelope as defined in docs/api_schema.md.
    """
    if summary is None:
        summary = f"Evidence Contract Pre-Flight Refusal: {reason}"

    if execution_trace is None:
        execution_trace = [
            {"stage": "Input Gate", "status": "pass", "time_ms": 6, "summary": "Input metadata extracted"},
            {"stage": "Evidence Contract", "status": "refused", "time_ms": 2, "summary": f"Pre-flight rejected: {reason}"}
        ]

    c_sensor_val = 0.83
    if sensor_card and "uncertainty" in sensor_card:
        try:
            c_sensor_val = round(max(0.0, 1.0 - float(sensor_card["uncertainty"])), 2)
        except (ValueError, TypeError):
            c_sensor_val = 0.83

    envelope = {
        "status": "refused",
        "task_type": task_type,
        "fidelity": "refusal_gate",
        "method": method,
        "reason": reason,
        "suggestion": suggestion,
        "answer_or_summary": summary,
        "visuals": {
            "primary_b64": "",
            "overlay_b64": ""
        },
        "metrics": metrics or {},
        "regions": [],
        "sensor_card": sensor_card or {},
        "confidence": {
            "aggregate_score": 0.0,
            "label": "Refusal / Insufficient Evidence",
            "breakdown": {
                "c_sensor": c_sensor_val,
                "c_adapter": 0.0,
                "c_guard": 0.0,
                "c_spectral": 0.0
            },
            "weights": {
                "w1_sensor": 0.15,
                "w2_adapter": 0.35,
                "w3_guard": 0.25,
                "w4_spectral": 0.25
            }
        },
        "globe_focus": globe_focus or {
            "lon": 72.95,
            "lat": 19.10,
            "bbox": [72.825, 18.975, 73.075, 19.225],
            "height_m": 25000
        },
        "execution_trace": execution_trace
    }
    return envelope


def evaluate_contract(
    query: str,
    file_paths: List[Path],
    beat: Optional[int] = None,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    sensor_types: Optional[List[str]] = None,
    primary_sensor_card: Optional[Dict[str, Any]] = None,
    primary_globe_focus: Optional[Dict[str, Any]] = None,
    stage_traces: Optional[List[Dict[str, Any]]] = None
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Evaluates evidence contract conditions for the incoming query and files.

    Returns:
        (True, None) if contract passes,
        (False, refusal_envelope) if contract fails.
    """
    q_clean = (query or "").strip().lower()
    num_files = len(file_paths)
    base_traces = stage_traces or []

    # 1. Empty or Nonsensical Query Refusal
    if not q_clean or len(q_clean) < 3:
        refusal_trace = list(base_traces) + [
            {"stage": "Evidence Contract", "status": "refused", "time_ms": 1, "summary": "Pre-flight rejected: Empty or invalid query"}
        ]
        return False, build_refusal_envelope(
            task_type="query_validation",
            reason="Geospatial query string is empty or invalid.",
            suggestion="Please provide a descriptive natural-language question regarding land cover, flood changes, or radar features.",
            summary="Evidence Contract Pre-Flight Refusal: Empty or unparseable query string.",
            sensor_card=primary_sensor_card,
            globe_focus=primary_globe_focus,
            execution_trace=refusal_trace
        )

    # 2. No Files Provided
    if num_files == 0:
        refusal_trace = list(base_traces) + [
            {"stage": "Evidence Contract", "status": "refused", "time_ms": 1, "summary": "Pre-flight rejected: No raster files provided"}
        ]
        return False, build_refusal_envelope(
            task_type="input_validation",
            reason="No satellite GeoTIFF files provided for evidence analysis.",
            suggestion="Upload at least one valid GeoTIFF scene before submitting a query.",
            summary="Evidence Contract Pre-Flight Refusal: No satellite files provided.",
            sensor_card=primary_sensor_card,
            globe_focus=primary_globe_focus,
            execution_trace=refusal_trace
        )

    # 3. Change Detection Pre-Flight Check (Beat 4 or Change Intent)
    is_change_intent = (beat == 4 or beat == 2 or any(
        kw in q_clean for kw in [
            "change", "between the two dates", "between these two", "changed",
            "differencing", "before and after", "temporal delta", "flood change"
        ]
    ))

    if is_change_intent and num_files < 2 and beat != 1 and beat != 3:
        refusal_trace = list(base_traces) + [
            {"stage": "Evidence Contract", "status": "refused", "time_ms": 2, "summary": "Pre-flight rejected: Missing T2 temporal tile for change detection"}
        ]
        return False, build_refusal_envelope(
            task_type="change_detection",
            reason="Requires two co-registered temporal GeoTIFF files (T1 pre-event and T2 post-event). Only 1 image was provided.",
            suggestion="Upload both pre-event and post-event satellite scenes to enable comparative pixel differencing.",
            summary="Evidence Contract Pre-Flight Refusal: Insufficient temporal inputs for change detection.",
            metrics={"required_tiles": 2, "provided_tiles": num_files},
            sensor_card=primary_sensor_card,
            globe_focus=primary_globe_focus,
            execution_trace=refusal_trace
        )

    # 4. Optical-SAR Fusion Pre-Flight Check (Beat 3 or SAR Intent)
    is_sar_intent = (beat == 3 or any(
        kw in q_clean for kw in ["sar", "radar", "backscatter", "penetrat", "obscured in optical", "sentinel-1"]
    ))

    if is_sar_intent:
        if num_files < 2:
            refusal_trace = list(base_traces) + [
                {"stage": "Evidence Contract", "status": "refused", "time_ms": 2, "summary": "Pre-flight rejected: Cross-modal fusion requires both Optical and SAR scenes"}
            ]
            return False, build_refusal_envelope(
                task_type="optical_sar_fusion",
                reason="Optical-SAR multi-sensor fusion requires at least two co-located sensor scenes (1 Optical scene + 1 SAR scene). Only 1 image was provided.",
                suggestion="Upload both an optical GeoTIFF (e.g. Sentinel-2) and a SAR GeoTIFF (e.g. Sentinel-1) covering the region of interest.",
                summary="Evidence Contract Pre-Flight Refusal: Incomplete multi-sensor inputs for SAR fusion.",
                metrics={"required_modalities": ["Optical", "SAR"], "provided_tiles": num_files},
                sensor_card=primary_sensor_card,
                globe_focus=primary_globe_focus,
                execution_trace=refusal_trace
            )

        # Check if SAR sensor is actually among the provided inputs
        if sensor_types:
            has_sar = any("sar" in s.lower() or "sentinel-1" in s.lower() for s in sensor_types)
            if not has_sar and not any("sar" in p.name.lower() or "s1" in p.name.lower() for p in file_paths):
                refusal_trace = list(base_traces) + [
                    {"stage": "Evidence Contract", "status": "refused", "time_ms": 2, "summary": "Pre-flight rejected: No Synthetic Aperture Radar (SAR) band detected"}
                ]
                return False, build_refusal_envelope(
                    task_type="optical_sar_fusion",
                    reason="Synthetic Aperture Radar (SAR) query requested, but no SAR imagery (Sentinel-1 C-band) was provided in the input bundle.",
                    suggestion="Ensure one of the uploaded scenes is a calibrated SAR GeoTIFF (Sentinel-1 GRD/SLC).",
                    summary="Evidence Contract Pre-Flight Refusal: SAR modality missing.",
                    metrics={"detected_sensors": sensor_types},
                    sensor_card=primary_sensor_card,
                    globe_focus=primary_globe_focus,
                    execution_trace=refusal_trace
                )

    # 5. Spatial Overlap Extent Check for Multi-Image Pairs
    if num_files >= 2 and metadata_list and len(metadata_list) >= 2:
        meta1, meta2 = metadata_list[0], metadata_list[1]
        bbox1 = meta1.get("bbox_wgs84")
        bbox2 = meta2.get("bbox_wgs84")
        if bbox1 and bbox2 and len(bbox1) == 4 and len(bbox2) == 4:
            iou = calculate_bbox_iou(bbox1, bbox2)
            if iou <= 0.0:
                refusal_trace = list(base_traces) + [
                    {"stage": "Evidence Contract", "status": "refused", "time_ms": 2, "summary": f"Pre-flight rejected: Spatial bounds do not overlap (IoU = {iou})"}
                ]
                return False, build_refusal_envelope(
                    task_type="spatial_coregistration",
                    reason=f"Spatial extent mismatch: The provided GeoTIFF scenes do not geographically overlap (Bounding Box IoU = {iou:.3f}).",
                    suggestion="Provide co-registered satellite tiles covering the same geographic region and bounding coordinates.",
                    summary="Evidence Contract Pre-Flight Refusal: Non-overlapping spatial extents.",
                    metrics={"bbox_1": bbox1, "bbox_2": bbox2, "spatial_iou": iou},
                    sensor_card=primary_sensor_card,
                    globe_focus=primary_globe_focus,
                    execution_trace=refusal_trace
                )

    # Contract satisfies all constraints
    return True, None
