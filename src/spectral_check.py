"""
Spectral Check Module — SatQuery EvidenceSwarm (SIH26167)
Pure NumPy + rasterio/tifffile CPU-only spectral calculations:
- Normalized Difference Water Index (NDWI)
- Normalized Difference Vegetation Index (NDVI)
- Otsu's Automatic Thresholding (pure NumPy implementation)
- Mask Overlap / Intersection-over-Union (IoU)
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


def _read_bands_or_array(
    source: Union[str, Path, np.ndarray],
    band_indices: Tuple[int, ...]
) -> Tuple[np.ndarray, ...]:
    """
    Helper to extract specified bands (1-indexed) from either a GeoTIFF path or numpy array.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")

        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(path) as src:
                    total_bands = src.count
                    arrays = []
                    for b_idx in band_indices:
                        actual_b = min(max(1, b_idx), total_bands)
                        arr = src.read(actual_b).astype(np.float32)
                        arrays.append(arr)
                    return tuple(arrays)
            except Exception:
                pass

        if TIFFFILE_AVAILABLE:
            raw = tifffile.imread(str(path)).astype(np.float32)
            if raw.ndim == 2:
                return (raw,) * len(band_indices)
            elif raw.ndim == 3:
                # Shape could be (C, H, W) or (H, W, C)
                if raw.shape[0] <= 16:  # (C, H, W)
                    total_bands = raw.shape[0]
                    return tuple(raw[min(max(0, b_idx - 1), total_bands - 1)] for b_idx in band_indices)
                else:  # (H, W, C)
                    total_bands = raw.shape[2]
                    return tuple(raw[:, :, min(max(0, b_idx - 1), total_bands - 1)] for b_idx in band_indices)
        
        raise RuntimeError(f"Unable to read raster data from {path}")

    elif isinstance(source, np.ndarray):
        arr = source.astype(np.float32)
        if arr.ndim == 2:
            return (arr,) * len(band_indices)
        elif arr.ndim == 3:
            if arr.shape[0] < arr.shape[2]:  # (bands, H, W)
                total_bands = arr.shape[0]
                return tuple(arr[min(max(0, b_idx - 1), total_bands - 1)] for b_idx in band_indices)
            else:  # (H, W, bands)
                total_bands = arr.shape[2]
                return tuple(arr[:, :, min(max(0, b_idx - 1), total_bands - 1)] for b_idx in band_indices)
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {arr.shape}")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")


def compute_ndwi(
    image_source: Union[str, Path, np.ndarray],
    green_band: int = 2,
    nir_band: int = 4,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes the Normalized Difference Water Index (NDWI) using McFeeters formula:
    NDWI = (Green - NIR) / (Green + NIR + eps)

    Returns:
        np.ndarray: float32 2D array with values in [-1.0, 1.0]. Water typically > 0.0.
    """
    green, nir = _read_bands_or_array(image_source, (green_band, nir_band))

    numerator = green - nir
    denominator = green + nir + eps

    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(denominator != 0, numerator / denominator, 0.0)

    ndwi = np.nan_to_num(ndwi, nan=-1.0, posinf=1.0, neginf=-1.0)
    return np.clip(ndwi, -1.0, 1.0).astype(np.float32)


def compute_ndvi(
    image_source: Union[str, Path, np.ndarray],
    red_band: int = 3,
    nir_band: int = 4,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes the Normalized Difference Vegetation Index (NDVI):
    NDVI = (NIR - Red) / (NIR + Red + eps)

    Returns:
        np.ndarray: float32 2D array with values in [-1.0, 1.0]. Dense vegetation > 0.4.
    """
    red, nir = _read_bands_or_array(image_source, (red_band, nir_band))

    numerator = nir - red
    denominator = nir + red + eps

    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(denominator != 0, numerator / denominator, 0.0)

    ndvi = np.nan_to_num(ndvi, nan=-1.0, posinf=1.0, neginf=-1.0)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def otsu_threshold(
    index_array: np.ndarray,
    num_bins: int = 256,
    clip_range: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[np.ndarray, float]:
    """
    Applies Otsu's thresholding algorithm to a continuous index array (e.g., NDWI or NDVI)
    to automatically compute an optimal binary segmentation mask in pure NumPy (CPU-only).
    """
    valid_data = index_array[np.isfinite(index_array)]
    if valid_data.size == 0:
        return np.zeros_like(index_array, dtype=bool), 0.0

    min_val, max_val = clip_range
    scaled = np.clip((valid_data - min_val) / (max_val - min_val + 1e-8) * 255.0, 0, 255).astype(np.uint8)

    hist, _ = np.histogram(scaled, bins=256, range=(0, 256))
    total = scaled.size

    if total == 0:
        return np.zeros_like(index_array, dtype=bool), 0.0

    current_max = 0.0
    best_thresh_bin = 128

    weight_bg = 0
    sum_bg = 0
    sum_total = np.dot(np.arange(256), hist)

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var_between = float(weight_bg) * float(weight_fg) * (mean_bg - mean_fg) ** 2

        if var_between > current_max:
            current_max = var_between
            best_thresh_bin = t

    optimal_threshold = float(min_val + (best_thresh_bin / 255.0) * (max_val - min_val))
    binary_mask = index_array >= optimal_threshold

    return binary_mask, optimal_threshold


def compute_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Computes Intersection-over-Union (IoU) / Jaccard Index between two binary masks.
    """
    a = (mask_a > 0).astype(bool)
    b = (mask_b > 0).astype(bool)

    if a.shape != b.shape:
        min_h = min(a.shape[0], b.shape[0])
        min_w = min(a.shape[1], b.shape[1])
        a = a[:min_h, :min_w]
        b = b[:min_h, :min_w]

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return round(float(intersection / union), 4)
