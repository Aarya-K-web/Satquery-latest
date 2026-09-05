# Team Execution Guide — SIH26167 SatQuery EvidenceSwarm
### Role-Based Phase-by-Phase Playbook
**Derived from:** Execution_Plan_SIH26167.md & Solution_SIH26167 (2).md
**Date:** September 2, 2026

---

## Team Roles

| Role | Label | Hardware | Scope |
|---|---|---|---|
| 🔴 **GPU Lead** | `GPU` | 8GB VRAM laptop (the only GPU machine) | Model fine-tuning, VQA inference, tunnel serving, all GPU-dependent work |
| 🔵 **Backend Dev(s)** | `BACK` | CPU-only laptop(s) — 1 or 2 people | FastAPI backend, agentic router, Evidence Contract, Evidence Guard, placeholder specialists, API wiring |
| 🟢 **Frontend Dev** | `FRONT` | CPU-only laptop | UI/UX — upload flow, Sensor Card display, results rendering, confidence badge, PDF download |
| 🟡 **Flex / QA** | `FLEX` | CPU-only laptop | Data sourcing, test images, integration testing, demo prep, documentation, pitch support, fills gaps |

> **Communication rule:** At the end of every phase, all four roles sync up, demo their deliverables to each other, and confirm the phase exit checklist before moving on.

---

## Quick Reference — Who Builds What

```
src/
├── app.py                       ← BACK
├── input_gate.py                ← BACK
├── sensor_card.py               ← BACK
├── evidence_contract.py         ← BACK
├── agentic_router.py            ← BACK
├── evidence_guard.py            ← BACK
├── confidence_engine.py         ← BACK
├── output_renderer.py           ← BACK + FRONT (collaborate)
├── spectral_check.py            ← BACK
├── trace_logger.py              ← BACK
└── specialists/
    ├── vqa_specialist.py        ← GPU
    ├── caption_grounding.py     ← BACK
    ├── change_detection.py      ← BACK
    └── optical_sar_fusion.py    ← BACK

scripts/
└── finetune_vqa.py              ← GPU

data/
├── vqa_train.jsonl              ← GPU + FLEX
├── vqa_val.jsonl                ← GPU + FLEX
└── vqa_test.jsonl               ← GPU + FLEX

demo/
├── images/                      ← FLEX
├── demo_cases.json              ← FLEX
├── speaker_notes.md             ← FLEX
└── backup_beat*.mp4             ← FLEX + GPU

deploy/
├── Dockerfile                   ← BACK
└── tunnel_config.yml            ← GPU

frontend/                        ← FRONT
```

---
---

# PHASE 1 — Data Engineering & Input Pipeline

---

## 🔴 GPU Lead — Phase 1

**Your mission:** Get the model running on your machine and prepare all training data. Everything in Phase 2 depends on you finishing this cleanly.

### Step-by-step:

**Step 1 — Download & verify model weights**
1. Install `transformers`, `bitsandbytes`, `accelerate`, `torch` (CUDA-enabled).
2. Download `Qwen/Qwen2-VL-2B-Instruct` from Hugging Face:
   ```bash
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', device_map='auto', load_in_4bit=True)"
   ```
3. Confirm the model loads in 4-bit without OOM. Note exact VRAM usage (should be ~3–3.5GB).
4. Run one test inference — pass a sample image and a question, confirm you get a text response.
5. Document exact VRAM usage, load time, and inference time in `results/gpu_baseline.md`.

**Step 2 — Download BigEarthNet v2.0**
1. Download BigEarthNet v2.0 dataset (Sentinel-1/2 multi-label patches).
2. Download supplementary VRSBench and/or RSVQA datasets.
3. Organize raw data into `data/raw/bigearth/`, `data/raw/vrsbench/`.

**Step 3 — Build QA pair generation script**
1. Write `scripts/generate_qa_pairs.py`:
   - Read BigEarthNet land-cover labels from metadata CSVs.
   - For each image patch, generate 2–3 natural-language QA pairs. Example templates:
     - *"What land cover types are visible?"* → *"Urban fabric, arable land"*
     - *"Is there water in this image?"* → *"Yes, there is an inland water body visible"* / *"No water is visible"*
     - *"Describe the vegetation coverage"* → *"Dense broadleaf forest covers approximately 60% of the area"*
   - Include VRSBench/RSVQA samples that already have QA pairs — filter and convert format.
2. Output unified JSONL format:
   ```json
   {"image": "path/to/patch.tif", "question": "What land cover types are visible?", "answer": "Urban fabric, arable land"}
   ```
3. Generate train/val/test splits: 90% train, 8% val, 2% test (hold out 50–100 for manual eval).

**Step 4 — Validate training data**
1. Run a schema check: every record has `image`, `question`, `answer`.
2. Spot-check 20 random samples: open the image, read the QA, confirm it makes sense.
3. Count total: must have ≥ 5,000 train pairs.

### 📋 Phase 1 Deliverables for GPU Lead:
- [ ] Qwen2-VL-2B loads in 4-bit on your GPU — VRAM usage documented
- [ ] Test inference works (image → question → answer)
- [ ] `data/vqa_train.jsonl` — ≥ 5,000 QA pairs
- [ ] `data/vqa_val.jsonl` — ~500 QA pairs
- [ ] `data/vqa_test.jsonl` — 50–100 held-out QA pairs
- [ ] `scripts/generate_qa_pairs.py` — working data generation script
- [ ] `results/gpu_baseline.md` — VRAM, load time, inference time documented

### 🚫 Blocker you must resolve before Phase 2:
> If the model does NOT load in 4-bit on your 8GB card, immediately notify the team. Fallback: try `load_in_8bit=True` or test a smaller Qwen model. Do NOT proceed to Phase 2 until you have confirmed a working model load.

---

## 🔵 Backend Dev(s) — Phase 1

**Your mission:** Build the input validation layer and spectral check. These are CPU-only components that the entire pipeline depends on.

> *If there are 2 of you:* Person B1 takes Steps 1–2 (Input Gate + Sensor Card), Person B2 takes Step 3 (Spectral Check). If solo, do them sequentially.

### Step-by-step:

**Step 1 — Input Gate (`src/input_gate.py`)**
1. Install: `pip install rasterio GDAL pyproj geopandas`
2. Write `input_gate.py` with these functions:
   - `validate_geotiff(file_path)` → `{"valid": bool, "errors": [str]}`
     - Check file is GeoTIFF (not PNG, JPEG, etc.)
     - Check file is not corrupt (can be opened by `rasterio`)
     - Check CRS exists and is parseable
   - `extract_metadata(file_path)` → `dict`
     - Extract: CRS, band count, resolution (m/px), spatial extent (bounding box), nodata value, file size
   - `detect_sensor(file_path)` → `str`
     - Heuristic detection from metadata/filename: "Sentinel-2", "Sentinel-1 SAR", "Cartosat-2S", "RISAT", "Unknown"
   - `estimate_uncertainty(metadata)` → `float` (0–1)
     - Higher uncertainty for lower resolution, fewer bands, unknown sensor
3. Write `tests/test_input_gate.py` with 5+ test cases:
   - Valid Sentinel-2 GeoTIFF → passes
   - Valid SAR GeoTIFF → passes
   - PNG file → rejected with clear error
   - Corrupt GeoTIFF → rejected
   - Missing CRS → rejected with suggestion

**Step 2 — Sensor Card (`src/sensor_card.py`)**
1. Write `sensor_card.py`:
   - `generate_card(metadata, sensor_type, uncertainty)` → `dict`
     - Returns structured JSON:
       ```json
       {
         "sensor_type": "Sentinel-2",
         "resolution_m": 10,
         "bands": ["B2", "B3", "B4", "B8"],
         "crs": "EPSG:32643",
         "bbox": [72.8, 18.9, 73.1, 19.2],
         "uncertainty": 0.17,
         "uncertainty_label": "Low"
       }
       ```
   - `render_card(card_dict)` → formatted string for display
2. Test against 2–3 real GeoTIFF files — verify output accuracy.

**Step 3 — Spectral Check (`src/spectral_check.py`)**
1. Write `spectral_check.py` — all NumPy, CPU-only, no GPU dependency:
   - `compute_ndwi(image_path)` → `np.ndarray` (float32 index map)
     - NDWI = (Green – NIR) / (Green + NIR)
   - `compute_ndvi(image_path)` → `np.ndarray`
     - NDVI = (NIR – Red) / (NIR + Red)
   - `otsu_threshold(index_array)` → `np.ndarray` (binary mask)
   - `compute_overlap(mask_a, mask_b)` → `float` (IoU)
2. Write `tests/test_spectral.py`:
   - Test NDWI on a tile with known water body → mask should highlight water
   - Test NDVI on a tile with known vegetation → mask should highlight green areas
   - Execution time must be < 2 seconds per tile

### 📋 Phase 1 Deliverables for Backend:
- [ ] `src/input_gate.py` — validates/rejects GeoTIFFs correctly
- [ ] `src/sensor_card.py` — generates structured metadata JSON
- [ ] `src/spectral_check.py` — CPU-only NDWI/NDVI with Otsu thresholding
- [ ] `tests/test_input_gate.py` — all 5+ tests pass
- [ ] `tests/test_spectral.py` — all tests pass, < 2s per tile
- [ ] All three modules importable with a clean Python API

---

## 🟢 Frontend Dev — Phase 1

**Your mission:** Design and scaffold the UI. By the end of this phase you should have a working upload interface and Sensor Card display — even if the backend isn't connected yet, use mock data.

### Step-by-step:

**Step 1 — Choose framework and scaffold**
1. Pick your tool: **Gradio** (fastest), **Streamlit** (mid), or **custom HTML/JS** (most flexible).
   - Recommended: **Gradio** for fastest time-to-demo — `pip install gradio`
2. Create `frontend/app.py` (or `frontend/index.html` if custom).
3. Set up basic layout:
   - Header: "SatQuery EvidenceSwarm" + project branding
   - Main content area split into panels

**Step 2 — Upload interface**
1. Build a file upload area that accepts 1 or 2 GeoTIFF files.
2. Display upload confirmation: filename, file size, thumbnail preview (if possible — GeoTIFFs may need band selection for RGB rendering).
3. Add a natural-language query input box below the upload area.
4. Add a "Submit Query" button.

**Step 3 — Sensor Card display panel**
1. Build a Sensor Card display component.
2. For now, use **mock data** (hardcoded JSON) to design the layout:
   ```json
   {
     "sensor_type": "Sentinel-2",
     "resolution_m": 10,
     "bands": ["B2", "B3", "B4", "B8"],
     "crs": "EPSG:32643",
     "bbox": [72.8, 18.9, 73.1, 19.2],
     "uncertainty": 0.17,
     "uncertainty_label": "Low"
   }
   ```
3. Make it visually clean: card-style layout with clear labels, color-coded uncertainty badge (green/yellow/red).

**Step 4 — Results placeholder panel**
1. Build a results display area (empty for now, will be populated in Phase 3–4):
   - Answer text area
   - Image overlay area (for visual evidence)
   - Confidence badge area
   - Execution trace accordion/collapsible section
   - PDF download button (disabled for now)
2. Build a **refusal display** component — distinct red/warning styling:
   - Shows when the Evidence Contract refuses a query
   - Displays reason and suggestion
3. Build a **reduced-fidelity badge** — small banner/tag that says "⚠ Reduced-Fidelity Path" (used for placeholder specialist outputs in Beats 2–3).

**Step 5 — Design system**
1. Define your color palette, fonts, spacing.
2. Ensure the UI looks professional and demo-ready — this is what judges see first.

### 📋 Phase 1 Deliverables for Frontend:
- [ ] Working upload interface (accepts 1–2 files + text query)
- [ ] Sensor Card display renders mock data beautifully
- [ ] Results panel scaffolded (answer, overlay, confidence, trace, download)
- [ ] Refusal display component designed (red/warning styling)
- [ ] Reduced-fidelity badge component designed
- [ ] UI runs locally and looks polished
- [ ] **Interface contract agreed with Backend:** What JSON shapes will the backend send? Agree on the response schema now.

---

## 🟡 Flex / QA — Phase 1

**Your mission:** Source all test images, help with dataset work, and start preparing demo materials. You are the team's safety net — if anyone is blocked, you help them.

### Step-by-step:

**Step 1 — Source demo and test GeoTIFF images**
1. Find and download the following (all must be free/public):
   - 2–3 **Sentinel-2 optical** tiles (different regions — urban, agricultural, water body)
   - 1 **Sentinel-1 SAR** tile (co-registered with one of the optical tiles if possible)
   - 1 **Pre/post-event Sentinel-2 pair** (e.g., before/after a flood, fire, or urban expansion — check Copernicus Open Access Hub)
   - 1 **Cartosat-2S sample** or **RISAT sample** (if publicly available — for domain shift testing)
   - 1 **Intentionally bad file** — PNG renamed to .tif, or a corrupt file (for refusal testing)
2. Save to `demo/images/` with clear naming:
   ```
   demo/images/sentinel2_urban_mumbai.tif
   demo/images/sentinel2_agri_punjab.tif
   demo/images/sentinel2_flood_pre_kerala.tif
   demo/images/sentinel2_flood_post_kerala.tif
   demo/images/sentinel1_sar_mumbai.tif
   demo/images/test_corrupt_file.tif
   demo/images/test_png_renamed.tif
   ```

**Step 2 — Help GPU Lead with dataset (parallel)**
1. Help download and organize BigEarthNet v2.0 patches if the download is large.
2. Help review/spot-check the generated QA pairs — open 20+ image-question-answer triples and flag any that look wrong.
3. Help curate additional VRSBench/RSVQA samples if GPU Lead needs more data.

**Step 3 — Draft demo case manifest**
1. Create `demo/demo_cases.json`:
   ```json
   [
     {
       "beat": 1,
       "name": "Single-Image VQA Baseline",
       "images": ["sentinel2_urban_mumbai.tif"],
       "query": "What land cover types are visible in this image?",
       "expected": "VQA answer with spatial grounding",
       "specialist": "vqa_specialist",
       "fidelity": "full"
     },
     {
       "beat": 2,
       "name": "Bi-Temporal Change Detection",
       "images": ["sentinel2_flood_pre_kerala.tif", "sentinel2_flood_post_kerala.tif"],
       "query": "What areas changed between these two dates?",
       "expected": "Change mask with percentage",
       "specialist": "change_detection",
       "fidelity": "reduced"
     },
     {
       "beat": 3,
       "name": "Optical-SAR Fusion",
       "images": ["sentinel2_urban_mumbai.tif", "sentinel1_sar_mumbai.tif"],
       "query": "What features are visible in SAR but obscured in the optical image?",
       "expected": "Fused composite with summary",
       "specialist": "optical_sar_fusion",
       "fidelity": "reduced"
     },
     {
       "beat": 4,
       "name": "The Signature Refusal",
       "images": ["sentinel2_urban_mumbai.tif"],
       "query": "Show me changes between the two dates",
       "expected": "Refusal — only one image for change detection",
       "specialist": "none",
       "fidelity": "n/a"
     }
   ]
   ```

**Step 4 — Set up project structure**
1. Create the full folder structure on a shared repo (GitHub/Git):
   ```
   SatQuery-AI/
   ├── src/specialists/
   ├── models/
   ├── data/
   ├── demo/images/
   ├── deploy/
   ├── scripts/
   ├── tests/
   ├── results/
   ├── frontend/
   └── requirements.txt
   ```
2. Initialize `requirements.txt` with all known dependencies:
   ```
   fastapi
   uvicorn
   rasterio
   GDAL
   pyproj
   geopandas
   numpy
   sentence-transformers
   transformers
   bitsandbytes
   accelerate
   peft
   torch
   Pillow
   fpdf2
   gradio
   ```

### 📋 Phase 1 Deliverables for Flex:
- [ ] `demo/images/` — at least 6 test GeoTIFFs sourced and organized
- [ ] `demo/demo_cases.json` — 4 demo beat cases defined
- [ ] Project folder structure created in shared repo
- [ ] `requirements.txt` — complete dependency list
- [ ] 20+ QA pairs spot-checked and verified for GPU Lead's dataset
- [ ] Pre/post event image pair confirmed to have overlapping spatial extent

---

### 🤝 Phase 1 Sync — All Roles

Before moving to Phase 2, the team meets and confirms:

- [ ] GPU Lead: Model loads ✅, training data ready ✅
- [ ] Backend: Input Gate + Sensor Card + Spectral Check all working ✅
- [ ] Frontend: Upload UI + Sensor Card display working with mock data ✅
- [ ] Flex: Test images sourced, demo cases drafted, repo structured ✅
- [ ] **API contract agreed**: Backend and Frontend have agreed on the exact JSON shapes for Sensor Card and query responses

---
---

# PHASE 2 — Model Training & Specialist Implementation

---

## 🔴 GPU Lead — Phase 2

**Your mission:** Fine-tune Qwen2-VL-2B to become the VQA specialist. This is the single most critical task in the entire project — the PS explicitly requires a fine-tuned RS model.

### Step-by-step:

**Step 1 — Set up fine-tuning script**
1. Install Unsloth: `pip install unsloth`
   - If Unsloth fails on your setup, fallback to LLaMA-Factory or Swift
2. Write `scripts/finetune_vqa.py`:
   - Load base model in 4-bit via `bitsandbytes`
   - Configure LoRA: `r=8`, `lora_alpha=16`, target attention modules
   - Set training args:
     - `learning_rate=2e-4`
     - `per_device_train_batch_size=1`
     - `gradient_accumulation_steps=4`
     - `num_train_epochs=3`
     - `max_seq_length=2048`
     - `save_steps=500`
     - `output_dir="models/qwen2vl_vqa_lora"`
3. **Dry run first**: Train on 10 samples only. Confirm:
   - No OOM
   - Loss is printed and decreasing
   - VRAM stays ≤ 7.5GB
   - Checkpoint saves correctly

**Step 2 — Full training run**
1. Launch full training on `data/vqa_train.jsonl`.
2. Monitor every 30 minutes: check loss curve, VRAM usage, disk space.
3. If OOM occurs:
   - First try: `gradient_accumulation_steps=8`
   - Second try: `max_seq_length=1024`
   - Last resort: subsample training data to 3,000 pairs
4. Training will likely take several hours — use this time to write `src/specialists/vqa_specialist.py` (the inference wrapper).

**Step 3 — Write VQA inference wrapper**
1. While training runs, write `src/specialists/vqa_specialist.py`:
   - `load_model()` → loads base model + fine-tuned LoRA adapter
   - `infer(image_path, question)` → `{"answer": str, "confidence": float, "grounding": optional, "fidelity": "full"}`
   - Handle errors gracefully — if model fails to load, return a fallback response using the base model
2. Make sure this module can be imported by the Backend team's FastAPI app.

**Step 4 — Validate fine-tuned model**
1. Load the fine-tuned adapter.
2. Run inference on all 50–100 held-out test samples.
3. Compute accuracy (exact match or semantic similarity).
4. **Compare against base model** (no fine-tuning) on the same test set.
5. Write `results/vqa_eval_report.md`:
   - Table: base model accuracy vs. fine-tuned accuracy
   - 5 qualitative examples (image + question + base answer + fine-tuned answer)
   - Inference latency per sample

**Step 5 — Package for Backend integration**
1. Confirm `vqa_specialist.py` works as a standalone module:
   ```python
   from src.specialists.vqa_specialist import load_model, infer
   model = load_model()
   result = infer(model, "demo/images/sentinel2_urban.tif", "What is visible?")
   print(result)
   ```
2. Share this interface spec with Backend so they can integrate it.

### 📋 Phase 2 Deliverables for GPU Lead:
- [ ] `scripts/finetune_vqa.py` — working fine-tuning script
- [ ] `models/qwen2vl_vqa_lora/` — saved adapter weights
- [ ] Fine-tuned model outperforms base model on held-out test set
- [ ] `src/specialists/vqa_specialist.py` — inference wrapper with clean API
- [ ] `results/vqa_eval_report.md` — metrics + qualitative examples
- [ ] Inference latency ≤ 8 seconds per query on your GPU
- [ ] Interface spec shared with Backend team

---

## 🔵 Backend Dev(s) — Phase 2

**Your mission:** Build the three reduced-fidelity placeholder specialists. These are CPU-only and don't need a GPU. Each must return structured output with a `"fidelity": "reduced"` flag.

> *If there are 2 of you:* B1 takes Caption/Grounding + Change Detection, B2 takes Optical-SAR Fusion + starts the FastAPI skeleton (head start for Phase 3).

### Step-by-step:

**Step 1 — Captioning / Grounding placeholder (`src/specialists/caption_grounding.py`)**
1. Write the module:
   - `run(image_path, query)` → `dict`
   - Logic:
     - Open image with `rasterio`, compute band statistics (mean, std per band).
     - Determine dominant features from statistics (e.g., high NIR = vegetation, high blue = water).
     - Generate template caption:
       ```
       "This {resolution}m {sensor_type} image covers an area of approximately {area_km2} km².
       Band analysis suggests {dominant_feature_description}."
       ```
     - For grounding: run simple edge/contour detection (OpenCV `findContours`) → return bounding boxes of largest regions.
   - Return:
     ```json
     {
       "caption": "This 10m Sentinel-2 image covers approximately 12 km². Band analysis suggests dominant vegetation with scattered urban patches.",
       "regions": [{"bbox": [10, 20, 200, 300], "label": "region_1"}],
       "fidelity": "reduced",
       "method": "rule-based"
     }
     ```

**Step 2 — Change Detection placeholder (`src/specialists/change_detection.py`)**
1. Write the module:
   - `run(image_path_pre, image_path_post, query)` → `dict`
   - Logic:
     - Load both images with `rasterio`.
     - If CRS or extent differs, reproject/clip to overlap region.
     - Compute pixel-wise absolute difference (use a representative band or average of RGB).
     - Apply Otsu thresholding (`spectral_check.otsu_threshold`) to get binary change mask.
     - Calculate: `change_pct = changed_pixels / total_pixels * 100`
   - Return:
     ```json
     {
       "change_mask": "<base64 encoded PNG>",
       "change_pct": 14.7,
       "summary": "Approximately 14.7% of the overlapping area shows detectable change.",
       "fidelity": "reduced",
       "method": "image-differencing"
     }
     ```

**Step 3 — Optical-SAR Fusion placeholder (`src/specialists/optical_sar_fusion.py`)**
1. Write the module:
   - `run(optical_path, sar_path, query)` → `dict`
   - Logic:
     - Load both images, co-register if CRS differs.
     - Normalize both to 0–1 range.
     - Create composite: take RGB from optical, use SAR backscatter as intensity/alpha overlay.
     - Generate summary from SAR statistics: high backscatter = rough surface/urban, low = smooth/water.
   - Return:
     ```json
     {
       "composite": "<base64 encoded PNG>",
       "summary": "SAR backscatter indicates high surface roughness in the northern region, consistent with urban structures. The southern region shows low backscatter, suggesting smooth surfaces or water.",
       "fidelity": "reduced",
       "method": "band-overlay-heuristic"
     }
     ```

**Step 4 — Verify all three work with Flex's test images**
1. Run each specialist against the images in `demo/images/`.
2. Confirm all return valid structured JSON with `fidelity: reduced`.
3. Confirm all run on CPU only — no GPU calls.

**Step 5 (if 2 people — B2 head start) — FastAPI skeleton**
1. Start writing `src/app.py` (the main backend application).
2. Set up `GET /health` endpoint.
3. Stub out `POST /query` and `GET /sensor-card` endpoints with placeholder responses.
4. This gives Phase 3 a head start.

### 📋 Phase 2 Deliverables for Backend:
- [ ] `src/specialists/caption_grounding.py` — working, returns structured JSON
- [ ] `src/specialists/change_detection.py` — working, returns change mask + percentage
- [ ] `src/specialists/optical_sar_fusion.py` — working, returns composite + summary
- [ ] All three return `"fidelity": "reduced"` in their output
- [ ] All three run on CPU only
- [ ] All three tested against `demo/images/` files
- [ ] (Bonus) FastAPI skeleton started

---

## 🟢 Frontend Dev — Phase 2

**Your mission:** Build out the results display components. By the end of Phase 2, the UI should be able to render every possible output type — even if you're still using mock data.

### Step-by-step:

**Step 1 — VQA results display**
1. Build a component that shows:
   - Answer text (large, prominent)
   - Image with overlay/bounding boxes (grounding visualization)
   - Confidence score badge (colored: green ≥ 0.85, yellow 0.65–0.84, red < 0.65)
2. Use mock data:
   ```json
   {
     "answer": "Urban fabric and arable land are visible, with a water body in the southeast.",
     "confidence": 0.87,
     "confidence_label": "High Evidence Consistency",
     "fidelity": "full"
   }
   ```

**Step 2 — Change detection results display**
1. Build a component that shows:
   - Change mask image (binary mask overlaid on the pre-image)
   - Change percentage (large number, e.g., "14.7% changed")
   - Summary text
   - Reduced-fidelity badge: "⚠ Reduced-Fidelity: Image Differencing"
2. Use mock data.

**Step 3 — Optical-SAR fusion results display**
1. Build a component that shows:
   - Fused composite image
   - Summary text
   - Reduced-fidelity badge: "⚠ Reduced-Fidelity: Band Overlay"
2. Use mock data.

**Step 4 — Execution trace accordion**
1. Build a collapsible/accordion component that shows the execution trace:
   ```
   [1] Input Gate          ✅ Passed  (0.3s)  — Sentinel-2, 10m, EPSG:32643
   [2] Evidence Contract   ✅ Passed  (0.1s)  — Single-image VQA compatible
   [3] Agentic Router      ✅ Routed  (0.2s)  — Task: vqa (confidence: 0.94)
   [4] VQA Specialist      ✅ Done    (4.2s)  — Answer generated
   [5] Evidence Guard      ✅ Passed  (0.8s)  — Guard: 0.91, Spectral: 0.86
   [6] Confidence Engine   ✅ Done    (0.1s)  — Score: 0.87 (High)
   ```
2. Use mock trace data for now.

**Step 5 — PDF download button**
1. Add a "Download Report" button.
2. For now it can be disabled or download a placeholder PDF.
3. The actual PDF generation will come from Backend in Phase 4.

### 📋 Phase 2 Deliverables for Frontend:
- [ ] VQA results display — answer, overlay, confidence badge
- [ ] Change detection results display — mask, percentage, reduced-fidelity badge
- [ ] Optical-SAR fusion results display — composite, summary, reduced-fidelity badge
- [ ] Execution trace accordion — renders mock trace data
- [ ] PDF download button (placeholder)
- [ ] All components render with mock data and look polished
- [ ] **Agreed response JSON schema** with Backend documented in `docs/api_schema.md`

---

## 🟡 Flex / QA — Phase 2

**Your mission:** Test everything that's being built, prepare the integration test harness, and start on documentation.

### Step-by-step:

**Step 1 — Test placeholder specialists**
1. As Backend delivers each specialist, run it against all relevant test images:
   - Caption/Grounding → run on each single-image GeoTIFF
   - Change Detection → run on the pre/post flood pair
   - Optical-SAR Fusion → run on the optical + SAR pair
2. Log results: does the output make sense? Is the JSON valid? Is `fidelity: reduced` present?
3. Report bugs/issues to Backend immediately.

**Step 2 — Review GPU Lead's eval results**
1. When GPU Lead finishes fine-tuning validation, review `results/vqa_eval_report.md`.
2. Check: does the fine-tuned model clearly beat the base model?
3. Verify the 5 qualitative examples are sensible.

**Step 3 — Write integration test harness**
1. Create `tests/test_integration.py`:
   - This will be used in Phase 3 to test the full pipeline.
   - For now, write test function stubs for each demo beat:
     ```python
     def test_beat1_vqa(): pass  # Fill in Phase 3
     def test_beat2_change(): pass
     def test_beat3_fusion(): pass
     def test_beat4_refusal(): pass
     ```
2. Write helper functions: `load_test_image()`, `send_query()`, `validate_response_schema()`.

**Step 4 — Start pitch preparation**
1. Create `demo/speaker_notes.md` with a skeleton:
   - Beat 1 intro → what we'll show → expected output → key talking point
   - Beat 2 intro → what we'll show → note reduced-fidelity → talking point
   - Beat 3 intro → what we'll show → note reduced-fidelity → talking point
   - Beat 4 intro → why this matters → refusal demonstrates trust
2. Start drafting the "scope-reduction framing" slide content:
   - "Working agentic architecture with one production-quality specialist and three architecturally-complete reduced-fidelity paths"
   - "Deliberate engineering trade-off under 8GB VRAM / 6-day constraints"

### 📋 Phase 2 Deliverables for Flex:
- [ ] All 3 placeholder specialists tested against real images, bugs reported
- [ ] GPU Lead's eval report reviewed and confirmed
- [ ] `tests/test_integration.py` — stubs ready for Phase 3
- [ ] `demo/speaker_notes.md` — skeleton with all 4 beats outlined
- [ ] Scope-reduction framing text drafted

---

### 🤝 Phase 2 Sync — All Roles

- [ ] GPU Lead: VQA specialist fine-tuned and validated ✅
- [ ] Backend: All 3 placeholder specialists working ✅
- [ ] Frontend: All result display components rendering with mock data ✅
- [ ] Flex: Everything tested, integration harness ready ✅
- [ ] **Key question resolved:** Does VQA inference latency fit within budget? If not, what adjustments?

---
---

# PHASE 3 — Pipeline Integration

---

## 🔴 GPU Lead — Phase 3

**Your mission:** Make your VQA specialist callable as an API endpoint so the Backend can integrate it. Help test the full pipeline.

### Step-by-step:

**Step 1 — VQA inference endpoint**
1. Write a lightweight FastAPI app `src/vqa_server.py` that runs on your GPU laptop:
   - `POST /infer` — accepts image (file upload) + question (string) → returns VQA result
   - Loads the fine-tuned model once on startup, keeps it in memory
   - Returns the same JSON format as `vqa_specialist.py`
2. Test it locally: `uvicorn src.vqa_server:app --port 8001`
3. Confirm Backend can call this endpoint from their laptop over the local network (same WiFi).

**Step 2 — Help test full pipeline**
1. When Backend has the end-to-end pipeline wired, test all 4 demo beats through it.
2. Specifically verify that the VQA path goes through the full chain: Input Gate → Contract → Router → your VQA model → Evidence Guard → Confidence → Output.
3. Check inference latency and report back.

### 📋 Phase 3 Deliverables for GPU Lead:
- [ ] `src/vqa_server.py` — VQA inference accessible via HTTP endpoint
- [ ] Backend can call your endpoint and get VQA responses
- [ ] Full pipeline tested with your VQA model in the loop

---

## 🔵 Backend Dev(s) — Phase 3

**Your mission:** This is your biggest phase. Wire everything into a working end-to-end pipeline.

### Step-by-step:

**Step 1 — FastAPI application (`src/app.py`)**
1. Complete the FastAPI app:
   - `POST /query` — the main pipeline endpoint
   - `GET /sensor-card` — returns Sensor Card for an uploaded image
   - `GET /health` — liveness check
2. Implement the full pipeline inside `POST /query`:
   ```
   Image(s) + Query
       ↓
   input_gate.validate_geotiff()
       ↓
   sensor_card.generate_card()
       ↓
   evidence_contract.check()         ←── if fails, return refusal immediately
       ↓
   agentic_router.classify()         ←── for now, use keyword-based routing (full router in Phase 4)
       ↓
   specialist.run()                  ←── dispatch to VQA (via GPU Lead's endpoint) or placeholder
       ↓
   evidence_guard.check()            ←── for now, just run spectral check
       ↓
   confidence_engine.compute()       ←── for now, use simple average
       ↓
   Return JSON response
   ```

**Step 2 — Evidence Contract (`src/evidence_contract.py`)**
1. Implement fully:
   - `check(query, images_metadata, router_result)` → `{"status": "pass" | "refused", "reason": str, "suggestion": str}`
   - Rules:
     - Change detection requires 2 images with overlapping extent → else refuse
     - Optical-SAR fusion requires 1 optical + 1 SAR → else refuse
     - VQA/Captioning requires at least 1 valid image → else refuse
2. Write tests: `tests/test_evidence_contract.py`

**Step 3 — Trace Logger (`src/trace_logger.py`)**
1. Build execution trace logging:
   - Each pipeline stage logs: `{stage_name, status, duration_ms, input_summary, output_summary}`
   - Store in SQLite: `data/traces.db`
   - Return the full trace array as part of the `/query` response
2. Frontend will display this in the execution trace accordion.

**Step 4 — Temporary router (keyword-based)**
1. Write a simple keyword-based router in `src/agentic_router.py` (placeholder for Phase 4's full router):
   - "change" / "difference" / "before after" → `change_detection`
   - "SAR" / "radar" / "backscatter" → `optical_sar_fusion`
   - "caption" / "describe" / "grounding" → `caption_grounding`
   - Everything else → `vqa`
2. This lets you test the full pipeline now; Phase 4 will add the sentence-transformer classifier.

**Step 5 — Test all 4 demo cases end-to-end**
1. Run each demo case from `demo/demo_cases.json` through the API.
2. Verify correct routing, correct specialist invocation, correct response format.
3. Verify Beat 4 triggers a clean refusal.

### 📋 Phase 3 Deliverables for Backend:
- [ ] `src/app.py` — FastAPI app with `/query`, `/sensor-card`, `/health` endpoints
- [ ] `src/evidence_contract.py` — all refusal scenarios working
- [ ] `src/trace_logger.py` — execution trace logged to SQLite
- [ ] Temporary keyword router dispatches correctly
- [ ] All 4 demo beats work end-to-end through the API
- [ ] Response JSON matches the schema agreed with Frontend
- [ ] `tests/test_evidence_contract.py` — all tests pass

---

## 🟢 Frontend Dev — Phase 3

**Your mission:** Connect the UI to the real backend API. Replace all mock data with live API calls.

### Step-by-step:

**Step 1 — Connect to Backend API**
1. Replace mock data with real API calls to Backend's FastAPI server.
2. Upload flow: upload file(s) → call `/sensor-card` → display real Sensor Card.
3. Query flow: submit query → call `/query` → display real results.

**Step 2 — Handle all response types**
1. Detect `fidelity` field in response and show/hide the reduced-fidelity badge accordingly.
2. Detect `status: "refused"` and render the refusal display instead of results.
3. Render the execution trace from the real trace data returned by the API.

**Step 3 — Error handling**
1. Handle network errors gracefully (Backend not reachable, timeout, etc.)
2. Show loading spinner while waiting for response (VQA can take 5–8 seconds).
3. Handle edge cases: empty response, malformed JSON, missing fields.

**Step 4 — Test with all 4 demo beats**
1. Run all 4 beats through the UI:
   - Beat 1: Upload single image → ask VQA question → see answer + confidence
   - Beat 2: Upload 2 images → ask about changes → see change mask + reduced-fidelity badge
   - Beat 3: Upload optical + SAR → ask fusion question → see composite + reduced-fidelity badge
   - Beat 4: Upload 1 image → ask change question → see refusal display

### 📋 Phase 3 Deliverables for Frontend:
- [ ] UI connected to real Backend API — no more mock data
- [ ] All 4 demo beats render correctly through the UI
- [ ] Reduced-fidelity badge appears for Beats 2 and 3
- [ ] Refusal renders correctly for Beat 4
- [ ] Execution trace renders real data
- [ ] Loading states and error handling work

---

## 🟡 Flex / QA — Phase 3

**Your mission:** Run integration tests on the assembled pipeline. Break things. Find bugs before the demo.

### Step-by-step:

**Step 1 — Run full integration tests**
1. Complete `tests/test_integration.py`:
   - `test_beat1_vqa()` — upload image, send VQA query, verify response schema
   - `test_beat2_change()` — upload 2 images, verify change mask in response
   - `test_beat3_fusion()` — upload optical + SAR, verify composite in response
   - `test_beat4_refusal()` — upload 1 image, send change query, verify refusal
2. Run all tests, log failures.

**Step 2 — Adversarial testing**
1. Try to break the system with edge cases:
   - Upload a very large GeoTIFF (100MB+) — does it timeout?
   - Upload 3 images when only 2 are expected — what happens?
   - Submit an empty query string — does it crash?
   - Submit a query in Hindi — does it handle gracefully?
   - Upload a JPEG — does Input Gate reject it properly?
2. Log every bug and report to the responsible role.

**Step 3 — Latency audit**
1. Time every demo beat end-to-end.
2. Fill in the latency table:
   | Beat | Total Time | Within Budget? |
   |---|---|---|
   | Beat 1 (VQA) | ___s | ≤ 10s? |
   | Beat 2 (Change) | ___s | ≤ 5s? |
   | Beat 3 (Fusion) | ___s | ≤ 5s? |
   | Beat 4 (Refusal) | ___s | ≤ 2s? |

### 📋 Phase 3 Deliverables for Flex:
- [ ] `tests/test_integration.py` — all 4 tests passing
- [ ] Adversarial test results logged — bugs filed with responsible roles
- [ ] Latency audit complete — all beats within budget or issues flagged
- [ ] Edge case behavior documented

---

### 🤝 Phase 3 Sync — All Roles

- [ ] Full pipeline works end-to-end for all 4 demo beats ✅
- [ ] Frontend shows live data from Backend ✅
- [ ] GPU Lead's VQA model is callable from the pipeline ✅
- [ ] All known bugs fixed or documented ✅
- [ ] **CRITICAL:** If anything is broken, this is the last chance to fix core functionality before Phase 4 adds deployment complexity

---
---

# PHASE 4 — Agentic Wiring & Deployment

---

## 🔴 GPU Lead — Phase 4

**Your mission:** Set up the tunnel so your GPU laptop serves VQA inference to the deployed system.

### Step-by-step:

**Step 1 — Install and configure tunnel**
1. Choose tunnel tool: `ngrok` (easier) or `cloudflare tunnel` (more stable).
2. Install and authenticate:
   ```bash
   # ngrok
   ngrok http 8001
   
   # or cloudflare tunnel
   cloudflared tunnel --url http://localhost:8001
   ```
3. Get the public URL (e.g., `https://abc123.ngrok.io`).
4. Share this URL with Backend so they can configure the deployed app to call it.

**Step 2 — Ensure VQA server is robust**
1. Add error handling to `src/vqa_server.py`:
   - Timeout handling (max 15 seconds per request)
   - Memory monitoring (if VRAM usage spikes, log a warning)
   - Auto-restart capability if the model crashes
2. Test: send 10 queries rapidly — does it handle them without crashing?

**Step 3 — Test the full deployed chain**
1. Backend deploys to Railway/Render → you expose your laptop via tunnel.
2. Test: public URL → Railway → tunnel → your laptop → VQA response → back to user.
3. Measure round-trip latency. If > 15 seconds, investigate bottleneck.

**Step 4 — Stretch: serverless GPU**
1. Only if tunnel is stable and time permits.
2. Try deploying the VQA model to Modal or RunPod serverless.
3. If it works, share the endpoint URL with Backend as a backup for the tunnel.

### 📋 Phase 4 Deliverables for GPU Lead:
- [ ] Tunnel is live and stable — public URL works
- [ ] VQA server handles errors gracefully
- [ ] Full deployed chain tested end-to-end
- [ ] Tunnel URL shared with Backend
- [ ] Round-trip latency documented

---

## 🔵 Backend Dev(s) — Phase 4

**Your mission:** Upgrade the router to use sentence-transformers, add the Evidence Guard, build the output renderer, and deploy to Railway/Render.

### Step-by-step:

**Step 1 — Upgrade Agentic Router**
1. Upgrade `src/agentic_router.py`:
   - Load `all-MiniLM-L6-v2` sentence-transformer (~80MB, CPU-only).
   - Pre-compute embeddings for 10–15 canonical query templates per task type.
   - On query: embed → cosine similarity → select best task.
   - Keep the keyword rule overrides as a first-pass filter (from Phase 3).
   - Return: `{"task_type": str, "confidence": float, "method": "rule_override" | "embedding_similarity"}`
2. Test with 20+ diverse queries — verify accuracy.

**Step 2 — Evidence Guard (`src/evidence_guard.py`)**
1. Implement fully:
   - For VQA path: CLIP similarity + spectral cross-check (call `spectral_check.py`)
   - For reduced-fidelity paths: spectral cross-check only
   - Return: `{"C_guard": float, "C_spectral": float}`
2. Integrate into the pipeline (after specialist, before confidence engine).

**Step 3 — Confidence Engine (`src/confidence_engine.py`)**
1. Implement the 4-factor formula:
   ```python
   score = 0.15 * C_sensor + 0.35 * C_adapter + 0.25 * C_guard + 0.25 * C_spectral
   ```
2. Assign label: ≥ 0.85 = "High Evidence Consistency", 0.65–0.84 = "Moderate", < 0.65 = "Low"

**Step 4 — Output Renderer (`src/output_renderer.py`)**
1. Visual overlay: draw bounding boxes / change masks on the image using Pillow.
2. JSON execution trace: already built (trace_logger).
3. PDF report: use `fpdf2` to generate a 1-page summary with image, answer, confidence breakdown.
4. Collaborate with Frontend on how overlays and PDFs are delivered (base64 in JSON? file download URL?).

**Step 5 — Deploy to Railway/Render**
1. Write `deploy/Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   COPY requirements-cpu.txt .
   RUN pip install -r requirements-cpu.txt
   COPY src/ ./src/
   CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. Ensure `requirements-cpu.txt` excludes `torch` GPU dependencies.
3. Configure the app to call GPU Lead's tunnel URL for VQA inference.
4. Deploy and verify health endpoint is reachable.

### 📋 Phase 4 Deliverables for Backend:
- [ ] `src/agentic_router.py` — sentence-transformer classifier + rule overrides
- [ ] `src/evidence_guard.py` — dual verification working
- [ ] `src/confidence_engine.py` — 4-factor score computed correctly
- [ ] `src/output_renderer.py` — overlays, PDF report, JSON trace
- [ ] App deployed to Railway/Render — public URL live
- [ ] All 4 demo beats work through the deployed public URL

---

## 🟢 Frontend Dev — Phase 4

**Your mission:** Connect to the deployed public URL, add the PDF download feature, and polish the visual experience.

### Step-by-step:

**Step 1 — Point UI to deployed URL**
1. Update the API base URL to the deployed Railway/Render URL.
2. Verify all 4 demo beats work through the deployed system.

**Step 2 — PDF download**
1. Connect the "Download Report" button to the PDF endpoint from Backend.
2. Trigger download when user clicks — should open/save a clean 1-page PDF.

**Step 3 — Confidence breakdown display**
1. Render the 4-factor confidence breakdown (not just the aggregate score):
   ```
   Sensor certainty:      0.83  ███████████░░
   Adapter confidence:    0.88  ████████████░
   Guard survival rate:   0.91  █████████████
   Spectral agreement:    0.86  ████████████░
   ──────────────────────────────────────────
   Overall:               0.87  High Evidence Consistency
   ```

**Step 4 — Visual polish pass**
1. Final color and spacing refinements.
2. Ensure all badge colors are consistent (green/yellow/red for confidence, orange for reduced-fidelity).
3. Make sure the refusal display is visually distinct and impossible to miss.

### 📋 Phase 4 Deliverables for Frontend:
- [ ] UI works against deployed public URL
- [ ] PDF download works
- [ ] 4-factor confidence breakdown renders beautifully
- [ ] Visual polish complete — demo-ready appearance

---

## 🟡 Flex / QA — Phase 4

**Your mission:** Test the deployed system, prepare the final demo flow, and make sure fallbacks work.

### Step-by-step:

**Step 1 — Test deployed system**
1. Access the public URL from a completely different device/network.
2. Run all 4 demo beats. Log any failures.
3. Test tunnel stability: does the VQA path still work after 30 minutes? After the laptop sleeps and wakes?

**Step 2 — Stress test**
1. Send 5 VQA queries in rapid succession — does the system handle them?
2. Try submitting queries while the tunnel is momentarily down — does the system fail gracefully?

**Step 3 — Verify fallback layers**
1. **Layer 1 test**: Ask GPU Lead to temporarily disconnect the fine-tuned adapter. Does the system fall back to the base model?
2. **Layer 2 test**: Test the CPU-only / 4-bit quantization path.
3. **Layer 3 prep**: If the tunnel is unstable, start recording backup videos NOW. Do not wait for Phase 5.

**Step 4 — Finalize demo flow**
1. Update `demo/demo_cases.json` with the final query wordings and expected outputs.
2. Update `demo/speaker_notes.md` with exact timing and transitions.
3. Do a practice run-through of the entire 4-beat demo script alone (no live system needed — just read through the script and practice the narrative).

### 📋 Phase 4 Deliverables for Flex:
- [ ] Deployed system tested from external device — all 4 beats work
- [ ] Stress test results documented
- [ ] Fallback Layer 1 and Layer 2 verified
- [ ] Demo flow finalized — speaker notes updated
- [ ] **CRITICAL:** If tunnel is unstable, backup videos must be recorded in this phase

---

### 🤝 Phase 4 Sync — All Roles

- [ ] Deployed system fully operational ✅
- [ ] All 4 demo beats work through the public URL ✅
- [ ] Tunnel is stable (or backup videos recorded) ✅
- [ ] All fallback layers verified ✅
- [ ] Demo script finalized ✅

---
---

# PHASE 5 — Polish & Delivery

---

## 🔴 GPU Lead — Phase 5

### Step-by-step:
1. **Keep the tunnel alive.** Your laptop must be online and serving during all rehearsals and the actual demo.
2. **Optimize VQA latency** — if inference > 5 seconds:
   - Reduce `max_new_tokens` (e.g., 150 → 100)
   - Enable flash attention if supported
   - Pre-load the image in memory instead of reading from disk each time
3. **Help record backup videos** — screen-record each demo beat with clean output for Layer-3 fallback.
4. **Participate in full dress rehearsal** — run all 4 beats live, timed.
5. **Monitor GPU during rehearsal** — watch for VRAM leaks or thermal throttling.

### 📋 Phase 5 Deliverables for GPU Lead:
- [ ] Tunnel stable for 2+ hours continuously
- [ ] VQA inference optimized (≤ 5 seconds per query)
- [ ] Backup videos recorded (your screen showing VQA responses)
- [ ] 3+ full dress rehearsals completed

---

## 🔵 Backend Dev(s) — Phase 5

### Step-by-step:
1. **Fix any remaining bugs** from Phase 4 testing.
2. **Latency optimization**:
   - Profile the pipeline — find the slowest non-GPU step
   - Cache sentence-transformer model on startup (don't reload per request)
   - Pre-compute anything that can be pre-computed
3. **PDF report polish** — make the generated PDF look professional:
   - Clean layout, proper fonts, SatQuery branding
   - Include: query, answer, confidence breakdown, execution trace summary, image thumbnail
4. **Final API stability check** — restart the deployed server and verify it comes back clean.
5. **Code freeze** after all fixes are in.

### 📋 Phase 5 Deliverables for Backend:
- [ ] All known bugs fixed
- [ ] Pipeline latency ≤ 10 seconds (total)
- [ ] PDF report looks professional
- [ ] Deployed server stable after restart
- [ ] **CODE FREEZE**

---

## 🟢 Frontend Dev — Phase 5

### Step-by-step:
1. **Final visual polish pass**:
   - Typography, spacing, alignment
   - Ensure every interactive element has hover/click feedback
   - Test on different screen sizes (demo will likely be on a projector — test at 1080p)
2. **Loading UX**:
   - Smooth loading spinner during VQA inference (5–8 second wait)
   - Consider a progress stepper: "Validating input... Routing query... Running specialist..."
3. **Demo-specific tweaks**:
   - Make sure the demo flow is frictionless — minimize clicks between beats
   - Pre-fill query text for each demo beat if possible (to save typing time during live demo)
4. **Screenshot the UI** — capture clean screenshots of all states for the pitch deck if needed.

### 📋 Phase 5 Deliverables for Frontend:
- [ ] UI fully polished — looks premium on a 1080p display
- [ ] Loading states are smooth and informative
- [ ] Demo flow is frictionless (minimal clicks per beat)
- [ ] Screenshots captured for pitch deck

---

## 🟡 Flex / QA — Phase 5

### Step-by-step:
1. **Record all 4 backup videos** (Layer-3 fallback):
   - Screen-record each demo beat in 1080p
   - Clean, edited, no cursor fumbling
   - Save as `demo/backup_beat1.mp4` through `demo/backup_beat4.mp4`
   - These are the absolute last resort — they must look perfect
2. **Finalize speaker notes** (`demo/speaker_notes.md`):
   - Exact words for each beat transition
   - Time budget per beat (aim for 2 minutes each, 8 minutes total)
   - Key talking points for judges:
     - Beat 1: "This is the real fine-tuned model — RS-adapted, not generic"
     - Beat 2: "Full agentic pipeline — reduced-fidelity specialist by design"
     - Beat 3: "Same architecture — demonstrates cross-modal routing"
     - Beat 4: "Evidence Contract proves the system knows what it doesn't know"
3. **Run the final dress rehearsal**:
   - All 4 roles present
   - Run all 4 beats live, timed
   - Practice handling failures (what do you do if the tunnel drops mid-demo?)
   - Practice the scope-reduction pitch: "This is a deliberate engineering trade-off..."
4. **Prepare the judge Q&A cheat sheet**:
   - Likely question: "Why are only some paths fine-tuned?" → Answer: 8GB VRAM constraint, deliberate trade-off
   - Likely question: "How would you scale this?" → Answer: Replace placeholders with trained models, add GPU resources
   - Likely question: "What's the evidence that fine-tuning worked?" → Answer: Show eval report metrics
5. **GO / NO-GO checklist** — final confirmation:

### 📋 Phase 5 Deliverables for Flex:
- [ ] `demo/backup_beat1.mp4` through `demo/backup_beat4.mp4` — recorded, clean, 1080p
- [ ] `demo/speaker_notes.md` — finalized with exact script
- [ ] `demo/judge_qa_cheatsheet.md` — prepared
- [ ] Full dress rehearsal completed with all team members
- [ ] Failure recovery plan documented (what to do when things break live)

---

### 🤝 Phase 5 Final Sync — GO / NO-GO

The entire team confirms:

- [ ] 🔴 GPU: Tunnel stable, model serving, latency optimized ✅
- [ ] 🔵 Backend: API deployed, stable, code frozen ✅
- [ ] 🟢 Frontend: UI polished, connected to deployed API ✅
- [ ] 🟡 Flex: Backup videos recorded, speaker notes finalized ✅
- [ ] **All 4 demo beats work live** ✅
- [ ] **All 3 fallback layers tested** ✅
- [ ] **Speaker notes rehearsed** ✅
- [ ] **Judge Q&A prepared** ✅

### ✅ **GO — Ready to present.**
