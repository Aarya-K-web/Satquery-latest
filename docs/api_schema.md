# SatQuery EvidenceSwarm — API Response Schema Contract
**SIH26167 | Backend ↔ Frontend Formal Contract**
**Version:** 2.0 (Phase 2 Master Specification)

This document is the **single source of truth** defining data structures exchanged between the Python FastAPI Backend and the Vanilla HTML/JS/Cesium Frontend.

---

## 1. Top-Level Query Response Envelope (`POST /api/query` & `POST /query`)

Every query request returns a standardized JSON envelope containing status, fidelity grade, reasoning text, base64 visuals, domain metrics, sensor telemetry, 4-factor confidence scores, 3D globe coordinates, and auditable execution trace.

```json
{
  "status": "success",
  "task_type": "change_detection",
  "fidelity": "reduced",
  "method": "image-differencing-otsu",
  "answer_or_summary": "Bi-temporal change detection between T1 (sentinel2_flood_pre_kerala.tif) and T2 (sentinel2_flood_post_kerala.tif) reveals a 14.82% surface alteration across 262,144 co-registered spatial pixels. Statistical Euclidean differencing (Otsu threshold: 48.5) isolates significant surface transition, consistent with flood inundation expansion and riparian boundary displacement.",
  "visuals": {
    "primary_b64": "data:image/png;base64,iVBORw0KGgo...",
    "overlay_b64": "data:image/png;base64,iVBORw0KGgo..."
  },
  "metrics": {
    "change_pct": 14.82,
    "changed_pixels": 38850,
    "total_pixels": 262144,
    "otsu_threshold": 48.5,
    "mean_abs_diff": 22.4,
    "pre_file": "sentinel2_flood_pre_kerala.tif",
    "post_file": "sentinel2_flood_post_kerala.tif"
  },
  "regions": [],
  "sensor_card": {
    "sensor_type": "Sentinel-2",
    "resolution_m": 10.0,
    "bands": [
      "B2 (Blue)",
      "B3 (Green)",
      "B4 (Red)",
      "B8 (NIR)"
    ],
    "band_count": 4,
    "crs": "EPSG:32643",
    "bbox": [76.15, 10.82, 76.35, 11.02],
    "center": {
      "lon": 76.25,
      "lat": 10.92
    },
    "uncertainty": 0.17,
    "uncertainty_label": "Low",
    "spatial_dimensions": {
      "width": 512,
      "height": 512
    },
    "dtypes": ["uint16"],
    "nodata": 0,
    "file_size_mb": 4.0,
    "filename": "sentinel2_flood_post_kerala.tif",
    "rendered_card": "═══════════════════════════════════════════════════════════════\n  🛰️  ISRO SATQUERY SENSOR CARD :: SENTINEL-2\n..."
  },
  "confidence": {
    "aggregate_score": 0.82,
    "label": "Moderate Evidence Consistency",
    "breakdown": {
      "c_sensor": 0.83,
      "c_adapter": 0.80,
      "c_guard": 0.81,
      "c_spectral": 0.84
    },
    "weights": {
      "w1_sensor": 0.15,
      "w2_adapter": 0.35,
      "w3_guard": 0.25,
      "w4_spectral": 0.25
    }
  },
  "globe_focus": {
    "lon": 76.25,
    "lat": 10.92,
    "bbox": [76.15, 10.82, 76.35, 11.02],
    "height_m": 25000
  },
  "execution_trace": [
    {"stage": "Input Gate", "status": "pass", "time_ms": 10, "summary": "2 GeoTIFFs validated and co-registered"},
    {"stage": "Sensor Card", "status": "pass", "time_ms": 4, "summary": "Sentinel-2 metadata generated"},
    {"stage": "Evidence Contract", "status": "pass", "time_ms": 2, "summary": "Temporal overlap & CRS compatibility verified"},
    {"stage": "Agentic Router", "status": "pass", "time_ms": 12, "summary": "Dispatched to change_detection specialist (CPU)"},
    {"stage": "Bi-Temporal Ingestion", "status": "pass", "time_ms": 18, "summary": "Loaded T1 and T2 on CPU"},
    {"stage": "Pixel Differencing & Otsu", "status": "pass", "time_ms": 42, "summary": "Otsu threshold=48.5, Change ratio=14.82%"},
    {"stage": "Visual Synthesis", "status": "pass", "time_ms": 25, "summary": "Generated high-contrast change detection overlay PNG"},
    {"stage": "Evidence Guard", "status": "pass", "time_ms": 35, "summary": "NDWI flood delta verified (Otsu threshold=48.5)"},
    {"stage": "Confidence Engine", "status": "pass", "time_ms": 2, "summary": "Aggregate score 0.82 (Moderate Evidence Consistency)"}
  ]
}
```

---

## 2. Signature Refusal Envelope (Evidence Contract Rejection)

When a query fails pre-flight constraints (e.g., change detection query with only 1 file), Backend returns `status: "refused"` with actionable guidance:

```json
{
  "status": "refused",
  "task_type": "change_detection",
  "fidelity": "refusal_gate",
  "method": "evidence_contract_preflight",
  "reason": "Requires two co-registered temporal GeoTIFF files (T1 pre-event and T2 post-event). Only 1 image was provided.",
  "suggestion": "Upload both pre-event and post-event satellite scenes to enable comparative pixel differencing.",
  "answer_or_summary": "Evidence Contract Pre-Flight Refusal: Insufficient temporal inputs for change detection.",
  "visuals": {
    "primary_b64": "",
    "overlay_b64": ""
  },
  "metrics": {
    "required_tiles": 2,
    "provided_tiles": 1
  },
  "regions": [],
  "sensor_card": {
    "sensor_type": "Sentinel-2",
    "resolution_m": 10.0,
    "uncertainty": 0.17
  },
  "confidence": {
    "aggregate_score": 0.0,
    "label": "Refusal / Insufficient Evidence",
    "breakdown": {
      "c_sensor": 0.83,
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
  "globe_focus": {
    "lon": 72.95,
    "lat": 19.10,
    "bbox": [72.825, 18.975, 73.075, 19.225],
    "height_m": 25000
  },
  "execution_trace": [
    {"stage": "Input Gate", "status": "pass", "time_ms": 8, "summary": "1 GeoTIFF successfully validated"},
    {"stage": "Sensor Card", "status": "pass", "time_ms": 3, "summary": "Sentinel-2 telemetry card generated"},
    {"stage": "Evidence Contract", "status": "refused", "time_ms": 2, "summary": "Pre-flight rejected: Missing T2 temporal tile for change detection"}
  ]
}
```

---

## 3. Grounding Bounding Box Envelope (VQA / Caption Grounding)

When spatial grounding is active, the `regions` array provides both pixel coordinates and geographic WGS84 coordinates:

```json
{
  "regions": [
    {
      "label": "Water Body / Inlet Channel",
      "bbox_pixel": [128, 64, 384, 256],
      "bbox_wgs84": [72.8421, 18.9812, 72.9150, 19.0945],
      "confidence": 0.88
    },
    {
      "label": "Built-up / Urban Fabric",
      "bbox_pixel": [180, 260, 490, 500],
      "bbox_wgs84": [72.9200, 18.9950, 73.0650, 19.2100],
      "confidence": 0.92
    }
  ]
}
```

---

## 4. Optical-SAR Multi-Sensor Fusion Envelope

```json
{
  "task_type": "optical_sar_fusion",
  "fidelity": "reduced",
  "method": "band-overlay-heuristic",
  "metrics": {
    "mean_vv_intensity": 142.5,
    "std_vv_intensity": 38.2,
    "roughness_index": 0.76,
    "high_backscatter_ratio": 0.182,
    "optical_bands": 4,
    "sar_bands": 2,
    "spatial_dimensions": "512x512px"
  }
}
```

---

## 5. Sensor Card Specification (`GET /api/sensor-card`, `POST /api/sensor-card`)

```json
{
  "sensor_type": "Sentinel-2",
  "resolution_m": 10.0,
  "bands": [
    "B2 (Blue)",
    "B3 (Green)",
    "B4 (Red)",
    "B8 (NIR)"
  ],
  "band_count": 4,
  "crs": "EPSG:32643",
  "bbox": [72.825, 18.975, 73.075, 19.225],
  "center": {
    "lon": 72.95,
    "lat": 19.10
  },
  "uncertainty": 0.17,
  "uncertainty_label": "Low",
  "spatial_dimensions": {
    "width": 512,
    "height": 512
  },
  "dtypes": ["uint16"],
  "nodata": 0,
  "file_size_mb": 4.0,
  "filename": "sentinel2_urban_mumbai.tif",
  "rendered_card": "═══════════════════════════════════════════════════════════════\n  🛰️  ISRO SATQUERY SENSOR CARD :: SENTINEL-2\n..."
}
```

---

## 6. 3D Globe Fly-To Telemetry Contract

When a valid GeoTIFF is processed:
```json
{
  "globe_focus": {
    "lon": 72.95,
    "lat": 19.10,
    "bbox": [72.825, 18.975, 73.075, 19.225],
    "height_m": 25000
  }
}
```
*If no valid coordinates exist, the Frontend leaves the camera in its current orientation.*
