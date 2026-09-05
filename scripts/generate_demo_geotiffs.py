"""
Demo GeoTIFF Generator — SatQuery EvidenceSwarm (SIH26167)
Generates high-fidelity synthetic GeoTIFF files for demo beats and unit testing:
1. Sentinel-2 4-Band Optical Tile (Mumbai Coastline & Urban Area)
2. Sentinel-1 SAR Dual-Pol Tile (Mumbai Co-registered)
3. Pre-Flood Sentinel-2 Tile (Kerala Coast / River Basin)
4. Post-Flood Sentinel-2 Tile (Inundated Region)
5. Cartosat-2S Sub-Meter Optical Sample (High Resolution)
6. Intentionally Bad Files (Renamed PNG, Corrupt GeoTIFF, Missing CRS)
"""

import os
import sys
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

try:
    import tifffile
    TIFFFILE_OK = True
except ImportError:
    TIFFFILE_OK = False


def _write_geotiff(path: Path, data: np.ndarray, crs_str: str, bounds: tuple, tags: dict, descriptions: list = None):
    """
    Writes a multi-band GeoTIFF with spatial reference either via rasterio or tifffile.
    """
    min_x, min_y, max_x, max_y = bounds
    count, height, width = (data.shape[0], data.shape[1], data.shape[2]) if data.ndim == 3 else (1, data.shape[0], data.shape[1])

    if RASTERIO_OK:
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        with rasterio.open(
            path, "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=data.dtype,
            crs=crs_str,
            transform=transform,
            nodata=0
        ) as dst:
            if data.ndim == 3:
                dst.write(data)
            else:
                dst.write(data, 1)
            dst.update_tags(**tags)
            if descriptions:
                for idx, desc in enumerate(descriptions, start=1):
                    dst.set_band_description(idx, desc)
    elif TIFFFILE_OK:
        # Construct standard GeoTIFF tags for tifffile
        # ModelPixelScaleTag (33550): [scaleX, scaleY, scaleZ]
        pixel_scale_x = (max_x - min_x) / float(width)
        pixel_scale_y = (max_y - min_y) / float(height)
        # ModelTiepointTag (33922): [I, J, K, X, Y, Z]
        tiepoint = (0.0, 0.0, 0.0, min_x, max_y, 0.0)
        
        # GeoKeyDirectoryTag (34735): Header (1, 1, 0, num_keys) + ProjectedCSTypeGeoKey (3072, 0, 1, epsg_code)
        epsg_code = int(crs_str.split(":")[-1]) if ":" in crs_str else 32643
        geokey = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, epsg_code)

        extratags = [
            (33550, 'd', 3, (pixel_scale_x, pixel_scale_y, 0.0), True),
            (33922, 'd', 6, tiepoint, True),
            (34735, 'H', len(geokey), geokey, True),
            (270, 's', len(str(tags)), str(tags), False) # ImageDescription
        ]

        # Shape for tifffile: (bands, height, width) or (height, width)
        out_data = data
        tifffile.imwrite(
            str(path),
            out_data,
            extratags=extratags
        )


def create_all_demo_images(output_dir: Path = Path("demo/images")):
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = 512, 512

    # 1. Sentinel-2 Optical (Mumbai Coastline: EPSG:32643 - UTM Zone 43N)
    s2_path = output_dir / "sentinel2_urban_mumbai.tif"
    y, x = np.mgrid[0:height, 0:width]
    water_mask = (x < 180 + 30 * np.sin(y / 35.0))
    veg_mask = (x > 320) & (y < 260)
    urban_mask = (~water_mask) & (~veg_mask)

    b_blue = np.where(water_mask, 1600, np.where(veg_mask, 500, 1800)).astype(np.uint16)
    b_green = np.where(water_mask, 1400, np.where(veg_mask, 900, 1600)).astype(np.uint16)
    b_red = np.where(water_mask, 800, np.where(veg_mask, 400, 1900)).astype(np.uint16)
    b_nir = np.where(water_mask, 200, np.where(veg_mask, 4200, 2100)).astype(np.uint16)

    np.random.seed(42)
    noise = np.random.randint(-50, 50, (height, width))
    b_blue = np.clip(b_blue + noise, 100, 10000).astype(np.uint16)
    b_green = np.clip(b_green + noise, 100, 10000).astype(np.uint16)
    b_red = np.clip(b_red + noise, 100, 10000).astype(np.uint16)
    b_nir = np.clip(b_nir + noise, 100, 10000).astype(np.uint16)

    s2_data = np.stack([b_blue, b_green, b_red, b_nir])
    s2_bounds = (270000.0, 2090000.0, 275120.0, 2095120.0)
    _write_geotiff(
        s2_path, s2_data, "EPSG:32643", s2_bounds,
        tags={"SENSOR": "Sentinel-2B MSI", "SPATIAL_RESOLUTION": "10m", "PROCESSING_LEVEL": "Level-2A BOA"},
        descriptions=["B2 - Blue (490nm)", "B3 - Green (560nm)", "B4 - Red (665nm)", "B8 - NIR (842nm)"]
    )
    print(f"Created: {s2_path}")

    # 2. Sentinel-1 SAR Dual-Pol Tile (Co-registered with Mumbai)
    s1_path = output_dir / "sentinel1_sar_mumbai.tif"
    vv_sar = np.where(water_mask, 80, np.where(urban_mask, 1450, 450)).astype(np.uint16)
    vh_sar = np.where(water_mask, 30, np.where(urban_mask, 950, 250)).astype(np.uint16)
    speckle = np.random.exponential(scale=30.0, size=(height, width)).astype(np.uint16)
    vv_sar = np.clip(vv_sar + speckle, 10, 4000).astype(np.uint16)
    vh_sar = np.clip(vh_sar + speckle, 5, 4000).astype(np.uint16)
    s1_data = np.stack([vv_sar, vh_sar])

    _write_geotiff(
        s1_path, s1_data, "EPSG:32643", s2_bounds,
        tags={"SENSOR": "Sentinel-1A C-SAR", "MODE": "IW GRDH", "POLARIZATION": "VV+VH", "SPATIAL_RESOLUTION": "10m"},
        descriptions=["VV (Co-polarization)", "VH (Cross-polarization)"]
    )
    print(f"Created: {s1_path}")

    # 3. Pre-Flood Kerala Tile
    pre_path = output_dir / "sentinel2_flood_pre_kerala.tif"
    river_mask_pre = (np.abs(x - (256 + 40 * np.sin(y / 40.0))) < 18)
    pre_blue = np.where(river_mask_pre, 1500, 600).astype(np.uint16)
    pre_green = np.where(river_mask_pre, 1300, 1100).astype(np.uint16)
    pre_red = np.where(river_mask_pre, 700, 500).astype(np.uint16)
    pre_nir = np.where(river_mask_pre, 200, 4500).astype(np.uint16)
    pre_data = np.stack([pre_blue, pre_green, pre_red, pre_nir])
    kerala_bounds = (640000.0, 1090000.0, 645120.0, 1095120.0)

    _write_geotiff(
        pre_path, pre_data, "EPSG:32643", kerala_bounds,
        tags={"SENSOR": "Sentinel-2A MSI", "ACQUISITION_DATE": "2024-07-10", "SCENE": "Kerala River Basin (Pre-event)"}
    )
    print(f"Created: {pre_path}")

    # 4. Post-Flood Kerala Tile
    post_path = output_dir / "sentinel2_flood_post_kerala.tif"
    river_mask_post = (np.abs(x - (256 + 40 * np.sin(y / 40.0))) < 90)
    post_blue = np.where(river_mask_post, 1500, 600).astype(np.uint16)
    post_green = np.where(river_mask_post, 1300, 1100).astype(np.uint16)
    post_red = np.where(river_mask_post, 700, 500).astype(np.uint16)
    post_nir = np.where(river_mask_post, 200, 4500).astype(np.uint16)
    post_data = np.stack([post_blue, post_green, post_red, post_nir])

    _write_geotiff(
        post_path, post_data, "EPSG:32643", kerala_bounds,
        tags={"SENSOR": "Sentinel-2B MSI", "ACQUISITION_DATE": "2024-08-15", "SCENE": "Kerala River Basin (Post-event)"}
    )
    print(f"Created: {post_path}")

    # 5. Cartosat-2S Sub-Meter Optical Sample
    carto_path = output_dir / "cartosat2s_sample_delhi.tif"
    carto_pan = (np.sin(x / 5.0) * 128 + np.cos(y / 5.0) * 128 + 128).astype(np.uint8)
    carto_bounds = (715000.0, 3160000.0, 715512.0, 3160512.0)
    _write_geotiff(
        carto_path, carto_pan, "EPSG:32643", carto_bounds,
        tags={"SENSOR": "Cartosat-2S PAN", "SPATIAL_RESOLUTION": "0.65m", "AGENCY": "ISRO"}
    )
    print(f"Created: {carto_path}")

    # 6. Negative / Refusal Test Files
    corrupt_path = output_dir / "test_corrupt_file.tif"
    with open(corrupt_path, "wb") as f:
        f.write(b"II*\x00\x08\x00\x00\x00CORRUPTED_BYTES_GARBAGE_DATA_HERE")
    print(f"Created: {corrupt_path}")

    png_path = output_dir / "test_png_renamed.tif"
    with open(png_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x06\x00\x00\x00")
    print(f"Created: {png_path}")

    no_crs_path = output_dir / "test_missing_crs.tif"
    if RASTERIO_OK:
        with rasterio.open(
            no_crs_path, "w",
            driver="GTiff",
            height=64,
            width=64,
            count=1,
            dtype=np.uint8
        ) as dst:
            dst.write(np.zeros((64, 64), dtype=np.uint8), 1)
    elif TIFFFILE_OK:
        tifffile.imwrite(str(no_crs_path), np.zeros((64, 64), dtype=np.uint8))
    print(f"Created: {no_crs_path}")


if __name__ == "__main__":
    create_all_demo_images()
