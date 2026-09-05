# SatQuery EvidenceSwarm — API Contract Specification
**SIH26167 | Backend ↔ Frontend Contract**
**Version:** 1.0 (Phase 1 Baseline)

---

## 1. File Validation & Metadata (`POST /api/validate`)

### Request
Multipart form data:
- `file`: Binary GeoTIFF file (`.tif` / `.tiff`)

### Response (200 OK)
```json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "metadata": {
    "filename": "sentinel2_urban_mumbai.tif",
    "file_size_bytes": 4194304,
    "file_size_mb": 4.0,
    "width": 512,
    "height": 512,
    "band_count": 4,
    "dtypes": ["uint16", "uint16", "uint16", "uint16"],
    "crs": "EPSG:32643",
    "native_bounds": [270000.0, 2090000.0, 275120.0, 2095120.0],
    "bbox_wgs84": [72.825, 18.975, 73.075, 19.225],
    "center_wgs84": {
      "lon": 72.95,
      "lat": 19.10
    },
    "resolution_m": 10.0,
    "nodata": 0.0,
    "sensor_type": "Sentinel-2",
    "uncertainty": 0.17,
    "uncertainty_label": "Low"
  }
}
```

### Rejection / Failure Response (200 OK or 400 Bad Request)
```json
{
  "valid": false,
  "errors": [
    "Invalid format: File is a PNG image renamed as GeoTIFF (.tif/.tiff). Must be a valid GeoTIFF.",
    "Missing Coordinate Reference System (CRS)."
  ],
  "warnings": [],
  "metadata": null
}
```

---

## 2. Sensor Card Data (`POST /api/sensor-card` or `GET /api/sensor-card/{file_id}`)

### Response (200 OK)
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

## 3. Spectral Index Calculation (`POST /api/spectral`)

### Request
Multipart form data or JSON:
- `file`: GeoTIFF file
- `index_type`: `"ndwi"` | `"ndvi"`
- `apply_otsu`: `true` | `false`

### Response (200 OK)
```json
{
  "index_type": "ndwi",
  "min_value": -0.842,
  "max_value": 0.721,
  "mean_value": -0.114,
  "otsu_threshold": 0.042,
  "foreground_pixels": 45120,
  "total_valid_pixels": 262144,
  "foreground_ratio": 0.172,
  "mask_preview_base64": "data:image/png;base64,iVBORw0KGgo..."
}
```

---

## 4. End-to-End Query (`POST /api/query`)

### Request
Multipart form data:
- `files`: 1 or 2 GeoTIFF files
- `query`: `"What land cover types are visible in this image?"`
- `options`: Optional execution configuration

### Response: Success Case (Full or Reduced Fidelity)
```json
{
  "query_id": "sq-91823-a1b2",
  "status": "success",
  "fidelity": "full",
  "task_type": "vqa",
  "specialist": "vqa_specialist",
  "answer": "The image captures an active coastal zone comprising dense urban fabric (buildings, transport infrastructure), an open inland water channel, and moderate vegetation patches along the northeastern perimeter.",
  "sensor_card": {
    "sensor_type": "Sentinel-2",
    "resolution_m": 10.0,
    "uncertainty": 0.17,
    "uncertainty_label": "Low"
  },
  "confidence": {
    "aggregate_score": 0.88,
    "label": "High Evidence Consistency",
    "breakdown": {
      "c_sensor": 0.83,
      "c_adapter": 0.91,
      "c_guard": 0.89,
      "c_spectral": 0.88
    },
    "weights": {
      "w1_sensor": 0.15,
      "w2_adapter": 0.35,
      "w3_guard": 0.25,
      "w4_spectral": 0.25
    }
  },
  "overlays": [
    {
      "type": "grounding_bbox",
      "label": "Urban Fabric",
      "box_wgs84": [72.84, 18.98, 73.02, 19.12],
      "confidence": 0.92
    }
  ],
  "execution_trace": [
    {"stage": "Input Gate", "status": "pass", "time_ms": 12, "summary": "Valid GeoTIFF with EPSG:32643"},
    {"stage": "Sensor Card", "status": "pass", "time_ms": 4, "summary": "Sentinel-2 MSI (10m res, C_sensor=0.83)"},
    {"stage": "Evidence Contract", "status": "pass", "time_ms": 3, "summary": "Pre-flight checks satisfied"},
    {"stage": "Agentic Router", "status": "pass", "time_ms": 18, "summary": "Routed to vqa_specialist"},
    {"stage": "Specialist Inference", "status": "pass", "time_ms": 840, "summary": "Qwen2-VL VQA generated answer"},
    {"stage": "Evidence Guard", "status": "pass", "time_ms": 65, "summary": "Spectral Otsu agreement IoU=0.88"},
    {"stage": "Confidence Engine", "status": "pass", "time_ms": 2, "summary": "Aggregate score 0.88 (High)"}
  ]
}
```

### Response: Refusal Case (Evidence Contract Rejection)
```json
{
  "query_id": "sq-91824-c3d4",
  "status": "refused",
  "fidelity": "refusal_gate",
  "task_type": "change_detection",
  "reason": "Bi-temporal change detection requires two co-registered GeoTIFF files representing different timestamps (T1 pre-event and T2 post-event). Only 1 image was provided.",
  "suggestion": "Please upload both pre-event and post-event satellite scenes to enable comparative pixel-level and spectral differencing.",
  "sensor_card": {
    "sensor_type": "Sentinel-2",
    "resolution_m": 10.0,
    "uncertainty": 0.17
  },
  "confidence": {
    "aggregate_score": 0.0,
    "label": "Refusal / Insufficient Evidence",
    "breakdown": {}
  },
  "execution_trace": [
    {"stage": "Input Gate", "status": "pass", "time_ms": 10, "summary": "Single GeoTIFF validated"},
    {"stage": "Evidence Contract", "status": "refused", "time_ms": 2, "summary": "Missing T2 temporal tile"}
  ]
}
```

---

## 5. 3D Globe Camera Navigation Contract (Center Panel)
When a GeoTIFF is validated:
```json
{
  "event": "GEO_FOCUS",
  "target": {
    "lon": 72.95,
    "lat": 19.10,
    "height_m": 25000,
    "bbox": [72.825, 18.975, 73.075, 19.225],
    "pitch_deg": -55,
    "heading_deg": 0,
    "fly_duration_sec": 2.5
  }
}
```
