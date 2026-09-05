"""
Sensor Card Module — SatQuery EvidenceSwarm (SIH26167)
Generates structured Sensor Card metadata dictionaries and human-readable ISRO-grade dashboard representations.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from src.input_gate import extract_metadata, detect_sensor, estimate_uncertainty


def infer_band_names(sensor_type: str, band_count: int, band_descriptions: Optional[List[str]] = None) -> List[str]:
    """
    Infers standard band designators based on sensor type and band count.
    """
    if band_descriptions and len(band_descriptions) == band_count and all(b for b in band_descriptions):
        return [str(b).strip() for b in band_descriptions]

    sensor_lower = sensor_type.lower()

    if "sentinel-2" in sensor_lower:
        if band_count == 4:
            return ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)"]
        elif band_count == 3:
            return ["B4 (Red)", "B3 (Green)", "B2 (Blue)"]
        elif band_count >= 12:
            return ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"][:band_count]
        return [f"B{i+1}" for i in range(band_count)]

    if "sentinel-1" in sensor_lower or "sar" in sensor_lower or "risat" in sensor_lower:
        if band_count == 1:
            return ["VV (Co-pol backscatter)"]
        elif band_count == 2:
            return ["VV (Co-pol)", "VH (Cross-pol)"]
        return [f"Pol_{i+1}" for i in range(band_count)]

    if "cartosat" in sensor_lower:
        if band_count == 1:
            return ["PAN (Panchromatic 0.65m)"]
        elif band_count == 4:
            return ["B1 (Blue)", "B2 (Green)", "B3 (Red)", "B4 (NIR)"]
        return [f"Band_{i+1}" for i in range(band_count)]

    if "landsat" in sensor_lower:
        if band_count >= 7:
            return ["B1 (Coastal)", "B2 (Blue)", "B3 (Green)", "B4 (Red)", "B5 (NIR)", "B6 (SWIR1)", "B7 (SWIR2)"][:band_count]
        return [f"B{i+1}" for i in range(band_count)]

    if band_count == 3:
        return ["Red", "Green", "Blue"]
    elif band_count == 1:
        return ["Intensity / Greyscale"]

    return [f"Band_{i+1}" for i in range(band_count)]


def generate_card(
    file_path: Optional[str | Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    sensor_type: Optional[str] = None,
    uncertainty: Optional[float] = None,
    uncertainty_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a structured Sensor Card dictionary adhering to the SatQuery EvidenceSwarm specification.

    Args:
        file_path: Optional path to GeoTIFF file.
        metadata: Optional pre-extracted metadata dictionary.
        sensor_type: Optional pre-computed sensor string.
        uncertainty: Optional pre-computed uncertainty score.
        uncertainty_label: Optional uncertainty label ('Low', 'Moderate', etc.).

    Returns:
        Structured dictionary representing the Sensor Card.
    """
    if metadata is None and file_path is not None:
        metadata = extract_metadata(file_path)
    elif metadata is None:
        metadata = {}

    if sensor_type is None:
        if file_path is not None:
            sensor_type = detect_sensor(file_path, metadata)
        else:
            sensor_type = metadata.get("sensor_type", "Sentinel-2")

    if uncertainty is None or uncertainty_label is None:
        calc_u, calc_label = estimate_uncertainty(metadata, sensor_type)
        uncertainty = uncertainty if uncertainty is not None else calc_u
        uncertainty_label = uncertainty_label if uncertainty_label is not None else calc_label

    band_count = metadata.get("band_count", 4)
    band_names = infer_band_names(sensor_type, band_count, metadata.get("band_descriptions"))

    resolution_m = metadata.get("resolution_m", 10.0)
    crs = metadata.get("crs", "EPSG:4326")
    bbox = metadata.get("bbox_wgs84", [72.825, 18.975, 73.075, 19.225])
    center = metadata.get("center_wgs84", {
        "lon": round((bbox[0] + bbox[2]) / 2.0, 4) if len(bbox) == 4 else 72.95,
        "lat": round((bbox[1] + bbox[3]) / 2.0, 4) if len(bbox) == 4 else 19.10
    })

    file_size_bytes = metadata.get("file_size_bytes", 0)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2) if file_size_bytes > 0 else 0.0

    card = {
        "sensor_type": sensor_type,
        "resolution_m": resolution_m,
        "bands": band_names,
        "band_count": band_count,
        "crs": crs,
        "bbox": bbox,
        "center": center,
        "uncertainty": float(uncertainty),
        "uncertainty_label": uncertainty_label,
        "spatial_dimensions": {
            "width": metadata.get("width", 512),
            "height": metadata.get("height", 512)
        },
        "dtypes": metadata.get("dtypes", ["uint16"]),
        "nodata": metadata.get("nodata"),
        "file_size_mb": file_size_mb,
        "filename": metadata.get("filename", Path(file_path).name if file_path else "sample.tif")
    }

    return card


def render_card(card_dict: Dict[str, Any]) -> str:
    """
    Renders the Sensor Card in a clean, high-precision ISRO-grade textual block.
    """
    sensor = card_dict.get("sensor_type", "Unknown Sensor")
    res = card_dict.get("resolution_m", 10.0)
    crs = card_dict.get("crs", "Unknown CRS")
    bands = card_dict.get("bands", [])
    bands_str = ", ".join(bands) if bands else "N/A"
    bbox = card_dict.get("bbox", [0, 0, 0, 0])
    center = card_dict.get("center", {"lon": 0, "lat": 0})
    uncertainty = card_dict.get("uncertainty", 0.0)
    u_label = card_dict.get("uncertainty_label", "Moderate")
    dims = card_dict.get("spatial_dimensions", {})
    w, h = dims.get("width", "—"), dims.get("height", "—")
    size_mb = card_dict.get("file_size_mb", 0.0)
    filename = card_dict.get("filename", "N/A")

    # Status indicator
    badge_symbol = "🟢" if u_label == "Low" else "🟡" if u_label == "Moderate" else "🔴"

    lines = [
        "═══════════════════════════════════════════════════════════════",
        f"  🛰️  ISRO SATQUERY SENSOR CARD :: {sensor.upper()}",
        "═══════════════════════════════════════════════════════════════",
        f"  File Name           : {filename} ({size_mb} MB)",
        f"  Sensor Platform     : {sensor}",
        f"  Spatial Resolution  : {res:.2f} m/pixel",
        f"  Raster Grid Extent  : {w} × {h} px",
        f"  Coordinate System   : {crs}",
        f"  Spectral Channels   : [{len(bands)}] {bands_str}",
        f"  Bounding Box (WGS84): [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]",
        f"  Centroid (Lon, Lat) : ({center.get('lon', 0.0):.4f}°E, {center.get('lat', 0.0):.4f}°N)",
        "───────────────────────────────────────────────────────────────",
        f"  {badge_symbol} Baseline Uncertainty: {uncertainty:.2f} ({u_label} Uncertainty)",
        "═══════════════════════════════════════════════════════════════"
    ]
    return "\n".join(lines)
