"""
Captioning and Spatial Grounding Specialist (Reduced-Fidelity CPU Baseline)
SatQuery EvidenceSwarm (SIH26167)

100% CPU-only. Operates on multispectral, optical, or panchromatic GeoTIFF tiles.
Extracts spectral statistics, determines dominant terrain classes, computes
grounding bounding boxes, and generates composite overlays.
"""

import io
import time
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.warp import transform_bounds

from src.input_gate import extract_metadata, detect_sensor


def _normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalizes a 2D raster band to 0-255 uint8 range safely."""
    valid_mask = np.isfinite(band) & (band > 0)
    if not np.any(valid_mask):
        return np.zeros_like(band, dtype=np.uint8)
    
    p2 = np.percentile(band[valid_mask], 2)
    p98 = np.percentile(band[valid_mask], 98)
    
    if p98 <= p2:
        norm = np.clip(band, 0, 255).astype(np.uint8)
    else:
        norm = np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    return norm


def _find_salient_boxes(mask: np.ndarray, label: str, min_size: int = 30) -> List[Dict[str, Any]]:
    """
    Finds bounding boxes for foreground clusters in a binary mask using pure NumPy grid analysis.
    """
    h, w = mask.shape
    boxes = []
    
    # Grid search in 4x4 quadrants to find dense clusters
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = h // grid_rows, w // grid_cols
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            sub = mask[r*cell_h : (r+1)*cell_h, c*cell_w : (c+1)*cell_w]
            density = np.mean(sub)
            if density > 0.35:
                # Find active pixel bounds within this quadrant
                y_idx, x_idx = np.where(sub)
                if len(y_idx) >= min_size:
                    min_y = int(r * cell_h + np.min(y_idx))
                    max_y = int(r * cell_h + np.max(y_idx))
                    min_x = int(c * cell_w + np.min(x_idx))
                    max_x = int(c * cell_w + np.max(x_idx))
                    
                    if (max_y - min_y) >= 20 and (max_x - min_x) >= 20:
                        boxes.append({
                            "label": label,
                            "bbox_pixel": [min_y, min_x, max_y, max_x],
                            "density": round(float(density), 3),
                            "confidence": round(float(min(0.95, 0.70 + density * 0.25)), 2)
                        })
    return boxes[:3]  # Return top 3 salient boxes


def run(image_path: str | Path, query: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes reduced-fidelity CPU captioning and spatial grounding.

    Args:
        image_path: Path to the GeoTIFF image tile.
        query: Optional user question/prompt.

    Returns:
        Standardized top-level response envelope dictionary.
    """
    start_time = time.time()
    trace: List[Dict[str, Any]] = []
    
    path = Path(image_path)
    if not path.exists():
        return {
            "status": "error",
            "task_type": "caption_grounding",
            "fidelity": "reduced",
            "method": "spectral-heuristic-grounding",
            "answer_or_summary": f"File not found: {path}",
            "visuals": {"primary_b64": "", "overlay_b64": ""},
            "metrics": {},
            "regions": [],
            "execution_trace": [{"stage": "Ingestion", "status": "failed", "summary": "File not found"}]
        }

    t0 = time.time()
    metadata = extract_metadata(path)
    sensor = detect_sensor(path, metadata)
    trace.append({
        "stage": "Raster Ingestion",
        "status": "pass",
        "time_ms": round((time.time() - t0) * 1000, 2),
        "summary": f"Ingested {sensor} ({metadata.get('width')}x{metadata.get('height')}, {metadata.get('band_count')} bands)"
    })

    t1 = time.time()
    with rasterio.open(path) as src:
        band_count = src.count
        h, w = src.height, src.width
        bands = src.read()
        res_m = float(src.res[0]) if src.res else 10.0
        crs = str(src.crs) if src.crs else "EPSG:4326"
        bounds = src.bounds
        affine = src.transform

    # Base RGB construction
    if band_count >= 4:
        # Sentinel-2 style: B2(0)=Blue, B3(1)=Green, B4(2)=Red, B8(3)=NIR
        b_blue = bands[0].astype(np.float32)
        b_green = bands[1].astype(np.float32)
        b_red = bands[2].astype(np.float32)
        b_nir = bands[3].astype(np.float32)

        rgb_img_arr = np.dstack([
            _normalize_band(b_red),
            _normalize_band(b_green),
            _normalize_band(b_blue)
        ])

        # Spectral Indices
        denom_ndvi = (b_nir + b_red)
        ndvi = np.where(denom_ndvi > 0, (b_nir - b_red) / np.maximum(denom_ndvi, 1e-6), 0.0)

        denom_ndwi = (b_green + b_nir)
        ndwi = np.where(denom_ndwi > 0, (b_green - b_nir) / np.maximum(denom_ndwi, 1e-6), 0.0)

        water_mask = (ndwi > 0.05)
        veg_mask = (ndvi > 0.28) & (~water_mask)
        urban_mask = (~water_mask) & (~veg_mask)

        total_valid = max(1, int(np.sum(np.isfinite(b_red))))
        water_pct = round(float(np.sum(water_mask) / total_valid * 100), 2)
        veg_pct = round(float(np.sum(veg_mask) / total_valid * 100), 2)
        urban_pct = round(float(np.sum(urban_mask) / total_valid * 100), 2)

    elif band_count == 3:
        # Standard RGB
        rgb_img_arr = np.dstack([
            _normalize_band(bands[0]),
            _normalize_band(bands[1]),
            _normalize_band(bands[2])
        ])
        water_mask = (rgb_img_arr[:, :, 2] > rgb_img_arr[:, :, 0] + 30)
        veg_mask = (rgb_img_arr[:, :, 1] > rgb_img_arr[:, :, 0] + 20)
        urban_mask = (~water_mask) & (~veg_mask)
        total_valid = h * w
        water_pct = round(float(np.sum(water_mask) / total_valid * 100), 2)
        veg_pct = round(float(np.sum(veg_mask) / total_valid * 100), 2)
        urban_pct = round(float(np.sum(urban_mask) / total_valid * 100), 2)

    elif band_count == 2:
        # Dual-pol SAR
        vv_norm = _normalize_band(bands[0])
        vh_norm = _normalize_band(bands[1])
        ratio = _normalize_band(bands[0] / np.maximum(bands[1], 1e-6))
        rgb_img_arr = np.dstack([vv_norm, vh_norm, ratio])
        water_mask = (vv_norm < 50)
        veg_mask = (vh_norm > 120) & (~water_mask)
        urban_mask = (vv_norm >= 160)
        total_valid = h * w
        water_pct = round(float(np.sum(water_mask) / total_valid * 100), 2)
        veg_pct = round(float(np.sum(veg_mask) / total_valid * 100), 2)
        urban_pct = round(float(np.sum(urban_mask) / total_valid * 100), 2)
    else:
        # Single band Panchromatic
        pan_norm = _normalize_band(bands[0])
        rgb_img_arr = np.dstack([pan_norm, pan_norm, pan_norm])
        water_mask = (pan_norm < 40)
        veg_mask = (pan_norm >= 40) & (pan_norm < 110)
        urban_mask = (pan_norm >= 110)
        total_valid = h * w
        water_pct = round(float(np.sum(water_mask) / total_valid * 100), 2)
        veg_pct = round(float(np.sum(veg_mask) / total_valid * 100), 2)
        urban_pct = round(float(np.sum(urban_mask) / total_valid * 100), 2)

    trace.append({
        "stage": "Spectral Decomposition",
        "status": "pass",
        "time_ms": round((time.time() - t1) * 1000, 2),
        "summary": f"Land cover distribution: Urban {urban_pct}%, Water {water_pct}%, Vegetation {veg_pct}%"
    })

    t2 = time.time()
    # Find spatial grounding regions
    regions_pixel = []
    if water_pct >= 5.0:
        regions_pixel.extend(_find_salient_boxes(water_mask, "Water Body / Inlet Channel"))
    if veg_pct >= 5.0:
        regions_pixel.extend(_find_salient_boxes(veg_mask, "Vegetation / Canopy Cover"))
    if urban_pct >= 5.0:
        regions_pixel.extend(_find_salient_boxes(urban_mask, "Built-up / Urban Fabric"))

    # Convert pixel coords to WGS84
    regions = []
    bbox_wgs84_overall = metadata.get("bbox_wgs84", [72.825, 18.975, 73.075, 19.225])
    min_lon_o, min_lat_o, max_lon_o, max_lat_o = bbox_wgs84_overall

    for reg in regions_pixel:
        ymin, xmin, ymax, xmax = reg["bbox_pixel"]
        
        # Approximate geographical coordinates
        lon_min_b = min_lon_o + (xmin / float(w)) * (max_lon_o - min_lon_o)
        lon_max_b = min_lon_o + (xmax / float(w)) * (max_lon_o - min_lon_o)
        lat_max_b = max_lat_o - (ymin / float(h)) * (max_lat_o - min_lat_o)
        lat_min_b = max_lat_o - (ymax / float(h)) * (max_lat_o - min_lat_o)

        regions.append({
            "label": reg["label"],
            "bbox_pixel": [ymin, xmin, ymax, xmax],
            "bbox_wgs84": [round(lon_min_b, 4), round(lat_min_b, 4), round(lon_max_b, 4), round(lat_max_b, 4)],
            "confidence": reg["confidence"]
        })

    trace.append({
        "stage": "Grounding Region Extraction",
        "status": "pass",
        "time_ms": round((time.time() - t2) * 1000, 2),
        "summary": f"Identified {len(regions)} salient spatial grounding regions"
    })

    # Render primary RGB image
    pil_primary = Image.fromarray(rgb_img_arr)
    buf_primary = io.BytesIO()
    pil_primary.save(buf_primary, format="PNG")
    primary_b64 = "data:image/png;base64," + base64.b64encode(buf_primary.getvalue()).decode()

    # Render overlay with bounding boxes
    pil_overlay = pil_primary.copy()
    draw = ImageDraw.Draw(pil_overlay)

    color_map = {
        "Water Body / Inlet Channel": "#00f0ff",
        "Vegetation / Canopy Cover": "#10b981",
        "Built-up / Urban Fabric": "#ffaa00"
    }

    for reg in regions:
        ymin, xmin, ymax, xmax = reg["bbox_pixel"]
        col = color_map.get(reg["label"], "#ffffff")
        draw.rectangle([xmin, ymin, xmax, ymax], outline=col, width=3)
        draw.text((xmin + 4, max(0, ymin - 12)), f"{reg['label']} ({int(reg['confidence']*100)}%)", fill=col)

    buf_overlay = io.BytesIO()
    pil_overlay.save(buf_overlay, format="PNG")
    overlay_b64 = "data:image/png;base64," + base64.b64encode(buf_overlay.getvalue()).decode()

    # Generate summary caption
    area_km2 = round((w * res_m * h * res_m) / 1_000_000.0, 2)
    dominant_features = []
    if urban_pct > 20:
        dominant_features.append(f"dense built-up urban fabric ({urban_pct}%)")
    if water_pct > 10:
        dominant_features.append(f"navigable surface water channels ({water_pct}%)")
    if veg_pct > 15:
        dominant_features.append(f"canopy and vegetative cover ({veg_pct}%)")
    if not dominant_features:
        dominant_features.append("mixed land cover features")

    dom_text = ", ".join(dominant_features)
    caption = (
        f"The {res_m:.1f}m {sensor} observation covers an area of approximately {area_km2} km² in {crs}. "
        f"Automated spectral decomposition indicates {dom_text}. "
        f"Spatial bounding boxes delineate primary feature boundaries with high contrast separation."
    )

    trace.append({
        "stage": "Heuristic Captioning Synthesis",
        "status": "pass",
        "time_ms": round((time.time() - start_time) * 1000, 2),
        "summary": "Synthesized structured spatial report"
    })

    return {
        "status": "success",
        "task_type": "caption_grounding",
        "fidelity": "reduced",
        "method": "spectral-heuristic-grounding",
        "answer_or_summary": caption,
        "visuals": {
            "primary_b64": primary_b64,
            "overlay_b64": overlay_b64
        },
        "metrics": {
            "area_km2": area_km2,
            "resolution_m": res_m,
            "band_count": band_count,
            "vegetation_pct": veg_pct,
            "water_pct": water_pct,
            "urban_pct": urban_pct,
            "total_regions": len(regions)
        },
        "regions": regions,
        "execution_trace": trace
    }
