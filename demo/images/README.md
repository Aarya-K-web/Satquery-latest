# Demo GeoTIFF Sample Catalog — SatQuery EvidenceSwarm (SIH26167)

This directory contains test and demo satellite imagery used across Phase 1–5 pipelines.

## Target Imagery Manifest

| Filename | Platform / Sensor | Spatial Res | Bands / Channels | Purpose | Source / Public Reference |
|---|---|---|---|---|---|
| `sentinel2_urban_mumbai.tif` | Sentinel-2B MSI | 10 m/px | B2, B3, B4, B8 (Blue, Green, Red, NIR) | Beat 1 VQA & Beat 3 Fusion optical base | ESA Copernicus Open Access Hub / BigEarthNet |
| `sentinel1_sar_mumbai.tif` | Sentinel-1A C-SAR | 10 m/px | VV, VH (Co-pol & Cross-pol backscatter) | Beat 3 Multi-sensor SAR Fusion | ESA Copernicus Open Access Hub |
| `sentinel2_flood_pre_kerala.tif` | Sentinel-2A MSI | 10 m/px | 4-Band Multispectral | Beat 2 Bi-temporal change detection (pre) | Copernicus Sentinel-2 Level-2A |
| `sentinel2_flood_post_kerala.tif` | Sentinel-2B MSI | 10 m/px | 4-Band Multispectral | Beat 2 Bi-temporal change detection (post) | Copernicus Sentinel-2 Level-2A |
| `cartosat2s_sample_delhi.tif` | Cartosat-2S PAN | 0.65 m/px | Single-band Panchromatic | High-resolution ISRO sensor validation | ISRO Bhuvan Open Data Archive |
| `test_corrupt_file.tif` | Corrupt Byte Stream | N/A | Corrupted headers | Input Gate refusal verification | Synthetically generated |
| `test_png_renamed.tif` | PNG renamed as .tif | N/A | PNG file signature | Format spoofing rejection | Synthetically generated |
| `test_missing_crs.tif` | GeoTIFF without CRS | N/A | 1-band without projection | Missing spatial metadata rejection | Synthetically generated |

## Generation & Verification
Run the generator script to populate synthetic test tiles matching these exact dimensions, bands, and geotransforms:
```bash
python scripts/generate_demo_geotiffs.py
```
