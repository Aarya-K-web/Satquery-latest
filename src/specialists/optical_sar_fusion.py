"""
Optical-SAR Multi-Sensor Fusion Specialist (Reduced-Fidelity CPU Baseline)
SatQuery EvidenceSwarm (SIH26167)

100% CPU-only. Blends optical multispectral reflectance with Sentinel-1 C-SAR
radar backscatter to penetrate clouds/haze and isolate surface roughness.
"""

import io
import time
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import rasterio

from src.input_gate import extract_metadata, detect_sensor


def _norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalizes an array to 0-255 uint8 range."""
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.uint8)
    p2, p98 = np.percentile(arr[valid], 2), np.percentile(arr[valid], 98)
    if p98 <= p2:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)


def run(
    optical_path: str | Path,
    sar_path: str | Path,
    query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes reduced-fidelity CPU Optical-SAR multi-sensor fusion.

    Args:
        optical_path: Path to optical GeoTIFF tile (e.g. Sentinel-2).
        sar_path: Path to SAR GeoTIFF tile (e.g. Sentinel-1).
        query: Optional user prompt.

    Returns:
        Standardized top-level response envelope dictionary.
    """
    start_time = time.time()
    trace: List[Dict[str, Any]] = []

    p_opt = Path(optical_path)
    p_sar = Path(sar_path)

    if not p_opt.exists() or not p_sar.exists():
        missing = []
        if not p_opt.exists():
            missing.append(str(p_opt))
        if not p_sar.exists():
            missing.append(str(p_sar))
        return {
            "status": "error",
            "task_type": "optical_sar_fusion",
            "fidelity": "reduced",
            "method": "band-overlay-heuristic",
            "answer_or_summary": f"Missing input file(s): {', '.join(missing)}",
            "visuals": {"primary_b64": "", "overlay_b64": ""},
            "metrics": {},
            "regions": [],
            "execution_trace": [{"stage": "Ingestion", "status": "failed", "summary": "Missing sensor file"}]
        }

    t0 = time.time()
    with rasterio.open(p_opt) as src_opt, rasterio.open(p_sar) as src_sar:
        bands_opt = src_opt.read().astype(np.float32)
        bands_sar = src_sar.read().astype(np.float32)
        meta_opt = extract_metadata(p_opt)
        meta_sar = extract_metadata(p_sar)

    trace.append({
        "stage": "Multi-Sensor Ingestion",
        "status": "pass",
        "time_ms": round((time.time() - t0) * 1000, 2),
        "summary": f"Ingested Optical ({meta_opt.get('sensor_type')}) + SAR ({meta_sar.get('sensor_type')})"
    })

    t1 = time.time()
    # Align grid shapes to mutual intersection
    min_h = min(bands_opt.shape[1], bands_sar.shape[1])
    min_w = min(bands_opt.shape[2], bands_sar.shape[2])

    b_opt_crop = bands_opt[:, :min_h, :min_w]
    b_sar_crop = bands_sar[:, :min_h, :min_w]

    # Construct Optical True Color RGB
    if b_opt_crop.shape[0] >= 4:
        r_opt = _norm_to_uint8(b_opt_crop[2])
        g_opt = _norm_to_uint8(b_opt_crop[1])
        b_opt = _norm_to_uint8(b_opt_crop[0])
        nir_opt = _norm_to_uint8(b_opt_crop[3])
    elif b_opt_crop.shape[0] == 3:
        r_opt = _norm_to_uint8(b_opt_crop[0])
        g_opt = _norm_to_uint8(b_opt_crop[1])
        b_opt = _norm_to_uint8(b_opt_crop[2])
        nir_opt = g_opt
    else:
        pan = _norm_to_uint8(b_opt_crop[0])
        r_opt, g_opt, b_opt, nir_opt = pan, pan, pan, pan

    rgb_opt = np.dstack([r_opt, g_opt, b_opt])

    # Construct SAR Backscatter Layer (VV co-pol + VH cross-pol)
    vv = _norm_to_uint8(b_sar_crop[0])
    vh = _norm_to_uint8(b_sar_crop[1]) if b_sar_crop.shape[0] > 1 else vv

    # Compute SAR Metrics
    mean_vv = float(np.mean(vv))
    std_vv = float(np.std(vv))
    high_backscatter_mask = (vv >= 170)
    high_backscatter_ratio = round(float(np.mean(high_backscatter_mask)), 3)
    roughness_index = round(float(std_vv / max(1.0, mean_vv)), 2)

    trace.append({
        "stage": "SAR Backscatter Telemetry",
        "status": "pass",
        "time_ms": round((time.time() - t1) * 1000, 2),
        "summary": f"Mean VV intensity: {mean_vv:.1f}, High-roughness ratio: {high_backscatter_ratio*100:.1f}%"
    })

    t2 = time.time()
    # 1. Primary Visual: Standard Optical RGB
    pil_opt = Image.fromarray(rgb_opt)
    buf_opt = io.BytesIO()
    pil_opt.save(buf_opt, format="PNG")
    primary_b64 = "data:image/png;base64," + base64.b64encode(buf_opt.getvalue()).decode()

    # 2. Fused Composite: False-Color Cross-Modal Blend
    # Channel 1: SAR VV Backscatter (structural double-bounce)
    # Channel 2: Optical NIR / Green (canopy biomass)
    # Channel 3: Optical Blue (water & contrast)
    fused_r = np.clip(0.4 * r_opt + 0.6 * vv, 0, 255).astype(np.uint8)
    fused_g = np.clip(0.7 * g_opt + 0.3 * nir_opt, 0, 255).astype(np.uint8)
    fused_b = np.clip(0.6 * b_opt + 0.4 * vh, 0, 255).astype(np.uint8)

    # Enhance high-backscatter structural returns with ISRO cyan / gold highlight
    fused_r[high_backscatter_mask] = np.clip(fused_r[high_backscatter_mask] * 0.4 + 255 * 0.6, 0, 255)
    fused_g[high_backscatter_mask] = np.clip(fused_g[high_backscatter_mask] * 0.4 + 200 * 0.6, 0, 255)
    fused_b[high_backscatter_mask] = np.clip(fused_b[high_backscatter_mask] * 0.2 + 0, 0, 255)

    fused_rgb = np.dstack([fused_r, fused_g, fused_b])
    pil_fused = Image.fromarray(fused_rgb)
    buf_fused = io.BytesIO()
    pil_fused.save(buf_fused, format="PNG")
    overlay_b64 = "data:image/png;base64," + base64.b64encode(buf_fused.getvalue()).decode()

    trace.append({
        "stage": "Cross-Modal Composite Synthesis",
        "status": "pass",
        "time_ms": round((time.time() - t2) * 1000, 2),
        "summary": "Generated fused Optical-SAR composite PNG"
    })

    # Summary phrasing
    summary = (
        f"Cross-modal multi-sensor fusion integrates Sentinel-2 optical multispectral reflectance "
        f"with Sentinel-1 C-band synthetic aperture radar (SAR) polarimetric backscatter. "
        f"SAR penetrates surface obscurations, isolating high dielectric double-bounce signatures "
        f"(roughness coefficient: {roughness_index}, high-backscatter footprint: {high_backscatter_ratio*100:.1f}%) "
        f"in built structures, transport arteries, and marine vessels that remain low-contrast in the optical channel."
    )

    trace.append({
        "stage": "Evidence Fusion Complete",
        "status": "pass",
        "time_ms": round((time.time() - start_time) * 1000, 2),
        "summary": "Multi-sensor reasoning composite compiled on CPU"
    })

    return {
        "status": "success",
        "task_type": "optical_sar_fusion",
        "fidelity": "reduced",
        "method": "band-overlay-heuristic",
        "answer_or_summary": summary,
        "visuals": {
            "primary_b64": primary_b64,
            "overlay_b64": overlay_b64
        },
        "metrics": {
            "mean_vv_intensity": round(mean_vv, 2),
            "std_vv_intensity": round(std_vv, 2),
            "roughness_index": roughness_index,
            "high_backscatter_ratio": high_backscatter_ratio,
            "optical_bands": int(bands_opt.shape[0]),
            "sar_bands": int(bands_sar.shape[0]),
            "spatial_dimensions": f"{min_w}x{min_h}px"
        },
        "regions": [],
        "execution_trace": trace
    }
