# Team Execution Guide — SIH26167 SatQuery EvidenceSwarm
### Role-Based Phase-by-Phase Playbook
**Derived from:** Execution_Plan_SIH26167.md & Solution_SIH26167 (2).md
**Deployment Strategy:** Railway / Render (Main Orchestration App) + Modal (Serverless GPU for VQA Specialist)
**Date:** September 2, 2026

---

## Team Roles

| Role | Label | Hardware | Scope |
|---|---|---|---|
| 🔴 **GPU Lead** | `GPU` | 8GB VRAM laptop | Model fine-tuning, adapter validation, Modal serverless GPU packaging & deployment (`modal_vqa.py`), response formatting & cold-start optimization, local test server / tunnel backup |
| 🔵 **Backend Dev(s)** | `BACK` | CPU-only laptop(s) — 1 or 2 people | FastAPI backend, agentic router, Evidence Contract, Evidence Guard, placeholder specialists, Railway/Render deployment, Modal API integration (`VQA_SERVER_URL`), automatic fallback logic |
| 🟢 **Frontend Dev** | `FRONT` | CPU-only laptop | UI/UX — upload flow, Sensor Card display, results rendering, confidence badge, PDF download, public cloud endpoint integration |
| 🟡 **Flex / QA** | `FLEX` | CPU-only laptop | Data sourcing, test images, integration testing, demo prep, documentation, Layer-3 backup demo video recording (all 4 beats), pitch support |

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
├── vqa_server.py                ← GPU (local dev test server / backup)
└── specialists/
    ├── vqa_specialist.py        ← GPU + BACK (calls Modal with local fallback)
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
└── backup_beat*.mp4             ← FLEX + GPU (critical safety net)

deploy/
├── Dockerfile                   ← BACK (for Railway / Render)
├── modal_vqa.py                 ← GPU (Modal serverless GPU endpoint)
└── tunnel_config.yml            ← GPU (dev / emergency backup only)

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
- [ ] **Interface contract agreed with Backend:** Agree on the JSON response schema now.

---

## 🟡 Flex / QA — Phase 1

**Your mission:** Source all test images, help with dataset work, and start preparing demo materials. You are the team's safety net — if anyone is blocked, you help them.

### Step-by-step:

**Step 1 — Source demo and test GeoTIFF images**
1. Find and download the following (all must be free/public):
   - 2–3 **Sentinel-2 optical** tiles (different regions — urban, agricultural, water body)
   - 1 **Sentinel-1 SAR** tile (co-registered with one of the optical tiles if possible)
   - 1 **Pre/post-event Sentinel-2 pair** (e.g., before/after a flood, fire, or urban expansion)
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
2. Initialize `requirements.txt` and `requirements-cpu.txt` with dependencies.

### 📋 Phase 1 Deliverables for Flex:
- [ ] `demo/images/` — at least 6 test GeoTIFFs sourced and organized
- [ ] `demo/demo_cases.json` — 4 demo beat cases defined
- [ ] Project folder structure created in shared repo
- [ ] `requirements.txt` & `requirements-cpu.txt` created
- [ ] 20+ QA pairs spot-checked and verified for GPU Lead's dataset
- [ ] Pre/post event image pair confirmed to have overlapping spatial extent

---

### 🤝 Phase 1 Sync — All Roles

Before moving to Phase 2, the team meets and confirms:

- [ ] GPU Lead: Model loads ✅, training data ready ✅
- [ ] Backend: Input Gate + Sensor Card + Spectral Check all working ✅
- [ ] Frontend: Upload UI + Sensor Card display working with mock data ✅
- [ ] Flex: Test images sourced, demo cases drafted, repo structured ✅
- [ ] **API contract agreed**: Backend and Frontend agree on response schemas

---
---

# PHASE 2 — Model Training & Specialist Implementation

---

## 🔴 GPU Lead — Phase 2

**Your mission:** Fine-tune Qwen2-VL-2B to become the VQA specialist.

### Step-by-step:

**Step 1 — Set up fine-tuning script**
1. Install Unsloth: `pip install unsloth` (or fallback to LLaMA-Factory / Swift).
2. Write `scripts/finetune_vqa.py`:
   - Load base model in 4-bit via `bitsandbytes`
   - Configure LoRA: `r=8`, `lora_alpha=16`, target attention modules
   - Set training args: `learning_rate=2e-4`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, `num_train_epochs=3`, `max_seq_length=2048`, `output_dir="models/qwen2vl_vqa_lora"`
3. **Dry run first**: Train on 10 samples to confirm no OOM and loss decreases.

**Step 2 — Full training run**
1. Launch full training on `data/vqa_train.jsonl`.
2. Monitor training loss, validation loss, and GPU memory.
3. Save checkpoints and final weights in `models/qwen2vl_vqa_lora/`.

**Step 3 — Write VQA inference wrapper**
1. While training runs, write `src/specialists/vqa_specialist.py`:
   - `load_model()` → loads base model + fine-tuned LoRA adapter
   - `infer(image_path, question)` → `{"answer": str, "confidence": float, "grounding": optional, "fidelity": "full"}`
   - Provide clean error handling and base model fallback.

**Step 4 — Validate fine-tuned model**
1. Load fine-tuned adapter and run inference on 50–100 held-out test samples.
2. Compute metrics (accuracy, semantic similarity, grounding IoU).
3. **Compare against base model** on the same test set.
4. Write `results/vqa_eval_report.md` with comparative metrics and 5 qualitative examples.

**Step 5 — Package for Backend integration**
1. Confirm `vqa_specialist.py` works as a standalone module.
2. Share interface spec with Backend team.

### 📋 Phase 2 Deliverables for GPU Lead:
- [ ] `scripts/finetune_vqa.py` — working fine-tuning script
- [ ] `models/qwen2vl_vqa_lora/` — saved adapter weights
- [ ] Fine-tuned model outperforms base model on held-out test set
- [ ] `src/specialists/vqa_specialist.py` — inference wrapper with clean API
- [ ] `results/vqa_eval_report.md` — metrics + qualitative examples
- [ ] Inference latency ≤ 8 seconds per query locally

---

## 🔵 Backend Dev(s) — Phase 2

**Your mission:** Build the three reduced-fidelity placeholder specialists (CPU-only).

### Step-by-step:

**Step 1 — Captioning / Grounding placeholder (`src/specialists/caption_grounding.py`)**
1. Write module: compute band statistics, generate template caption, extract simple contours for bounding boxes, return `fidelity: reduced`.

**Step 2 — Change Detection placeholder (`src/specialists/change_detection.py`)**
1. Write module: load pre/post images, compute absolute difference, apply Otsu thresholding for binary change mask, compute `change_pct`, return `fidelity: reduced`.

**Step 3 — Optical-SAR Fusion placeholder (`src/specialists/optical_sar_fusion.py`)**
1. Write module: load optical and SAR images, create composite overlay (RGB + SAR intensity), summarize surface roughness from backscatter, return `fidelity: reduced`.

**Step 4 — Verify all three work with Flex's test images**
1. Run each against `demo/images/` and confirm valid structured output on CPU.

**Step 5 — FastAPI skeleton head start**
1. Start `src/app.py` with `/health`, `/sensor-card`, `/query` endpoint stubs.

### 📋 Phase 2 Deliverables for Backend:
- [ ] `src/specialists/caption_grounding.py` — working, returns structured JSON
- [ ] `src/specialists/change_detection.py` — working, returns change mask + percentage
- [ ] `src/specialists/optical_sar_fusion.py` — working, returns composite + summary
- [ ] All three return `"fidelity": "reduced"` and run on CPU only
- [ ] FastAPI skeleton started

---

## 🟢 Frontend Dev — Phase 2

**Your mission:** Build out the results display components to render all output types.

### Step-by-step:
1. **VQA results display**: Answer text, bounding box overlay, colored confidence badge.
2. **Change detection results display**: Change mask overlay, change percentage badge, reduced-fidelity warning.
3. **Optical-SAR fusion results display**: Composite image, roughness summary, reduced-fidelity warning.
4. **Execution trace accordion**: Collapsible view of pipeline stages and timestamps.
5. **PDF download button**: UI action hook ready for backend integration.

### 📋 Phase 2 Deliverables for Frontend:
- [ ] All result display components rendering with mock data
- [ ] Reduced-fidelity badge implemented
- [ ] Execution trace accordion built
- [ ] API schema agreement documented

---

## 🟡 Flex / QA — Phase 2

**Your mission:** Test all specialist modules, review eval reports, and prepare the integration test harness.

### Step-by-step:
1. Test all 3 placeholder specialists against real images; report edge cases.
2. Review GPU Lead's `results/vqa_eval_report.md` to confirm fine-tuning lift.
3. Write `tests/test_integration.py` test stubs for the 4 demo beats.
4. Draft pitch outline and speaker notes in `demo/speaker_notes.md`.

### 📋 Phase 2 Deliverables for Flex:
- [ ] Placeholder specialists verified
- [ ] `tests/test_integration.py` harness ready
- [ ] `demo/speaker_notes.md` initial draft completed

---

### 🤝 Phase 2 Sync — All Roles

- [ ] GPU Lead: VQA specialist fine-tuned and validated ✅
- [ ] Backend: All 3 placeholder specialists working ✅
- [ ] Frontend: All UI components rendering mock data ✅
- [ ] Flex: Test harness and speaker notes prepared ✅

---
---

# PHASE 3 — Pipeline Integration

---

## 🔴 GPU Lead — Phase 3

**Your mission:** Make your VQA specialist callable via an HTTP endpoint for local development integration and prepare it for Modal serverless deployment.

### Step-by-step:

**Step 1 — Local VQA inference endpoint (`src/vqa_server.py`)**
1. Write a lightweight FastAPI app `src/vqa_server.py` that runs on your local GPU:
   - `POST /infer` — accepts image file + query string → returns VQA JSON.
   - Loads base model + LoRA adapter in 4-bit on startup.
2. Test locally: `uvicorn src.vqa_server:app --port 8001`.
3. Confirm Backend can call this endpoint over the local network during integration.

**Step 2 — Help test full pipeline integration**
1. Verify that incoming queries from the main pipeline invoke the VQA model and receive valid grounded answers.
2. Benchmark local inference latency and log baseline metrics.

### 📋 Phase 3 Deliverables for GPU Lead:
- [ ] `src/vqa_server.py` — local VQA HTTP server functional
- [ ] Backend able to query local VQA endpoint
- [ ] Full pipeline tested end-to-end with VQA model in the loop

---

## 🔵 Backend Dev(s) — Phase 3

**Your mission:** Wire every component into a unified end-to-end FastAPI pipeline with the Evidence Contract and trace logging.

### Step-by-step:

**Step 1 — Implement `/query` pipeline in `src/app.py`**
1. Wire sequence: Input Gate → Sensor Card → Evidence Contract → Router → Specialist → Evidence Guard → Confidence Engine → Response.
2. Implement `GET /sensor-card` and `GET /health`.

**Step 2 — Evidence Contract (`src/evidence_contract.py`)**
1. Enforce strict pre-flight validation rules:
   - Bi-temporal change requires 2 overlapping images → else refuse with clear guidance.
   - Optical-SAR fusion requires 1 optical + 1 SAR image → else refuse.
   - Single-image VQA requires valid optical imagery → else refuse.
2. Write unit tests in `tests/test_evidence_contract.py`.

**Step 3 — Trace Logger (`src/trace_logger.py`)**
1. Record timestamp, stage name, duration, and status for each pipeline step into SQLite (`data/traces.db`).
2. Include trace in `/query` JSON response.

**Step 4 — Test all 4 demo cases end-to-end**
1. Run `demo/demo_cases.json` queries through the API and verify all outputs (including the Beat 4 signature refusal).

### 📋 Phase 3 Deliverables for Backend:
- [ ] `src/app.py` — fully functioning API
- [ ] `src/evidence_contract.py` — all refusal rules working
- [ ] `src/trace_logger.py` — SQLite execution trace logging
- [ ] All 4 demo beats working through the API

---

## 🟢 Frontend Dev — Phase 3

**Your mission:** Connect UI to the live backend API and replace mock data with real responses.

### Step-by-step:
1. Connect upload and query flows to `/sensor-card` and `/query`.
2. Render real answers, overlays, confidence badges, and execution traces.
3. Render distinct refusal display for Beat 4.
4. Add loading spinner and network error handling.

### 📋 Phase 3 Deliverables for Frontend:
- [ ] UI connected to live Backend API
- [ ] All 4 demo beats rendered with real data
- [ ] Loading and refusal states working cleanly

---

## 🟡 Flex / QA — Phase 3

**Your mission:** Run full integration tests, conduct adversarial edge-case testing, and perform a latency audit.

### Step-by-step:
1. Complete `tests/test_integration.py` for all 4 beats.
2. Adversarial tests: corrupt files, non-overlapping tiles, invalid formats.
3. Perform latency audit across all beats and log findings.

### 📋 Phase 3 Deliverables for Flex:
- [ ] `tests/test_integration.py` passing
- [ ] Adversarial test log completed
- [ ] Latency audit documented

---

### 🤝 Phase 3 Sync — All Roles

- [ ] End-to-end pipeline operational for all 4 beats ✅
- [ ] UI displaying live backend responses ✅
- [ ] Evidence Contract cleanly refusing invalid queries ✅
- [ ] **Ready for Phase 4 cloud deployment** ✅

---
---

# PHASE 4 — Agentic Wiring & Deployment

---

## 🔴 GPU Lead — Phase 4

**Your mission:** Deploy the fine-tuned VQA specialist to **Modal (Serverless GPU)** as the primary serving path, optimize cold starts and response formatting, and maintain the local tunnel purely as a development / backup option.

### Step-by-step:

**Step 1 — Create and deploy Modal VQA endpoint (`deploy/modal_vqa.py`)**
1. Install Modal CLI: `pip install modal` and authenticate (`modal setup`).
2. Write `deploy/modal_vqa.py`:
   - Define a Modal App (`satquery-vqa`).
   - Create a Debian-based container image with `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, and `Pillow`.
   - Mount model weights or download base `Qwen/Qwen2-VL-2B-Instruct` with the fine-tuned LoRA adapter (`models/qwen2vl_vqa_lora/`).
   - Configure a GPU-accelerated serverless function (T4 / A10G / L4).
   - Use `@modal.web_endpoint(method="POST")` or FastAPI app inside Modal to expose `/infer`.
   - Optimize cold starts: set `keep_warm=1` during demo windows and cache model loading in memory.
3. Deploy the function:
   ```bash
   modal deploy deploy/modal_vqa.py
   ```
4. Test the deployed Modal URL directly with a test script: pass a sample image and question, confirm response JSON.
5. Provide the live Modal endpoint URL to the Backend Lead.

**Step 2 — Ensure response format compatibility & error handling**
1. Guarantee that the Modal endpoint returns the exact schema expected by `vqa_specialist.py`:
   ```json
   {
     "answer": "...",
     "confidence": 0.88,
     "grounding": {"bbox": [120, 80, 450, 600]},
     "fidelity": "full",
     "method": "qwen2-vl-2b-qlora-modal"
   }
   ```
2. Handle image decoding, base64 payloads, and GeoTIFF band conversions cleanly.

**Step 3 — Configure local laptop tunnel as secondary / emergency backup**
1. Keep `src/vqa_server.py` and `deploy/tunnel_config.yml` ready.
2. If Modal is unreachable or for offline local testing, run ngrok / Cloudflare tunnel:
   ```bash
   ngrok http 8001
   ```
3. Share the backup tunnel URL format with Backend for instant toggle if ever needed.

**Step 4 — Measure roundtrip latency with Backend**
1. Measure roundtrip inference time from Railway/Render to Modal GPU (< 5s target).
2. Tweak token generation limits (`max_new_tokens=120`) to stay strictly within budget.

### 📋 Phase 4 Deliverables for GPU Lead:
- [ ] `deploy/modal_vqa.py` — working Modal serverless GPU deployment
- [ ] Modal endpoint live, deployed, and tested (`POST /infer`)
- [ ] Response schema validated and 100% compatible with backend
- [ ] Cold start optimized (< 5s inference latency)
- [ ] Modal URL shared with Backend Lead
- [ ] Local tunnel config documented as backup option

---

## 🔵 Backend Dev(s) — Phase 4

**Your mission:** Deploy the main orchestration app to Railway/Render, integrate the deployed Modal GPU endpoint via environment variables, implement automatic fallback handling, upgrade the router with sentence-transformers, and integrate the Evidence Guard.

### Step-by-step:

**Step 1 — Upgrade Agentic Router (`src/agentic_router.py`)**
1. Integrate `all-MiniLM-L6-v2` sentence-transformer (~80MB, CPU-only).
2. Embed canonical query templates and perform cosine similarity routing with deterministic rule overrides.

**Step 2 — Evidence Guard & Confidence Engine**
1. Complete `src/evidence_guard.py` (learned grounding / CLIP + CPU spectral NDWI/NDVI check).
2. Complete `src/confidence_engine.py` (4-factor weighted score calculation).
3. Complete `src/output_renderer.py` (overlay generation, PDF report via `fpdf2`, JSON trace).

**Step 3 — Deploy Main App to Railway / Render**
1. Create `deploy/Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements-cpu.txt .
   RUN pip install --no-cache-dir -r requirements-cpu.txt
   COPY . .
   EXPOSE 8000
   CMD sh -c "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"
   ```
2. Deploy to **Railway** or **Render** as a web service.
3. Set environment variable:
   ```env
   VQA_SERVER_URL=https://<your-modal-workspace>--satquery-vqa-infer.modal.run
   ```

**Step 4 — Implement Clean Fallback Handling**
1. In `src/specialists/vqa_specialist.py`:
   - Send HTTP request to `VQA_SERVER_URL` with a 10-second timeout.
   - If Modal returns an error, times out, or is unreachable:
     - Log warning in the execution trace: `"Modal GPU endpoint unreachable — falling back to CPU heuristics"`.
     - Automatically execute fallback baseline heuristic response.
     - Never crash the pipeline or return a 500 error to the user.

**Step 5 — Verify deployed application**
1. Test all 4 beats through the public Railway/Render URL.

### 📋 Phase 4 Deliverables for Backend:
- [ ] `src/agentic_router.py` — sentence-transformer router with rule overrides
- [ ] `src/evidence_guard.py` & `src/confidence_engine.py` integrated
- [ ] `src/output_renderer.py` — PDF report and visual overlays
- [ ] Main app deployed on Railway/Render with public HTTPS URL
- [ ] `VQA_SERVER_URL` environment variable configured to Modal
- [ ] Automatic fallback logic tested and verified
- [ ] All 4 demo beats working through public URL

---

## 🟢 Frontend Dev — Phase 4

**Your mission:** Point the frontend to the deployed Railway/Render URL, verify PDF report downloads, and polish the visual presentation.

### Step-by-step:
1. Update API base URL to the public Railway/Render endpoint.
2. Verify all 4 demo beats through the deployed cloud system.
3. Wire the "Download Report" button to the live PDF generation endpoint.
4. Render the 4-factor confidence breakdown with clean progress bars and badges.
5. Polish color schemes, typography, and refusal state visibility.

### 📋 Phase 4 Deliverables for Frontend:
- [ ] Frontend operational against public Railway/Render URL
- [ ] PDF report download working seamlessly
- [ ] 4-factor confidence breakdown visually rendered
- [ ] UI polished and demo-ready

---

## 🟡 Flex / QA — Phase 4

**Your mission:** Extensively test the deployed cloud system, verify fallback layers, and record high-quality backup videos of all 4 beats (Layer 3 fallback).

### Step-by-step:

**Step 1 — Test deployed cloud system from external devices**
1. Access the Railway/Render URL from independent laptops, phones, and networks.
2. Run all 4 beats. Verify end-to-end execution without reliance on any local server.

**Step 2 — Test fallback layers**
1. **Layer 1 test**: Temporarily invalidate `VQA_SERVER_URL` in test config — verify the app gracefully falls back to CPU heuristics with an execution trace warning.
2. **Layer 2 test**: Verify the backup local tunnel configuration (`src/vqa_server.py` + ngrok).
3. **Layer 3 recording (CRITICAL)**:
   - Record high-definition (1080p) screen captures of each beat running successfully.
   - Save to `demo/backup_beat1.mp4` through `demo/backup_beat4.mp4`.

**Step 3 — Finalize demo script & speaker notes**
1. Update `demo/speaker_notes.md` with finalized timings and narrative cues.

### 📋 Phase 4 Deliverables for Flex:
- [ ] Deployed split-cloud system verified from external networks
- [ ] Automatic fallback tested and confirmed
- [ ] High-definition backup videos recorded for all 4 beats (`demo/backup_beat*.mp4`)
- [ ] `demo/speaker_notes.md` finalized

---

### 🤝 Phase 4 Sync — All Roles

- [ ] Main app live on Railway/Render ✅
- [ ] VQA specialist deployed on Modal serverless GPU ✅
- [ ] Split cloud communication working seamlessly ✅
- [ ] Automatic fallback and backup videos verified ✅
- [ ] All 4 demo beats functional on public URL ✅

---
---

# PHASE 5 — Polish & Delivery

---

## 🔴 GPU Lead — Phase 5

### Step-by-step:
1. **Monitor Modal endpoint health** — run periodic warm-up pings before rehearsals.
2. **Optimize inference latency**:
   - Ensure Modal container caching is active.
   - Keep token generation concise (`max_new_tokens=100–120`).
3. **Assist with dress rehearsals**:
   - Verify VQA response quality on all demo questions.
   - Confirm cold-start latency is < 5 seconds.
4. **Maintain backup tunnel ready**:
   - Have local `vqa_server.py` running in the background on your machine as an instant fallback.

### 📋 Phase 5 Deliverables for GPU Lead:
- [ ] Modal serverless GPU endpoint responsive with low latency (< 5s)
- [ ] Inference parameters optimized
- [ ] Backup local server on standby
- [ ] 3+ full dress rehearsals completed

---

## 🔵 Backend Dev(s) — Phase 5

### Step-by-step:
1. **Fix any remaining edge-case bugs** identified during QA testing.
2. **Latency optimization**:
   - Cache router model and pre-computed embeddings on container startup.
   - Ensure rasterio operations are optimized.
3. **PDF report polish**:
   - Verify layout, fonts, confidence table, and SatQuery header branding.
4. **Cloud stability verification**:
   - Restart the Railway/Render service and verify it boots up cleanly and reconnects to Modal.
5. **Enforce Code Freeze**.

### 📋 Phase 5 Deliverables for Backend:
- [ ] All known bugs resolved
- [ ] Total pipeline latency ≤ 8–10 seconds
- [ ] PDF report generated cleanly
- [ ] Cloud service verified stable across restarts
- [ ] **CODE FREEZE**

---

## 🟢 Frontend Dev — Phase 5

### Step-by-step:
1. **Final visual polish pass**:
   - Verify responsiveness on 1080p presentation displays.
   - Ensure clear visual contrast on confidence badges and refusal alerts.
2. **Loading UX**:
   - Smooth loading spinner with descriptive step indicators ("Analyzing Imagery → Routing Query → Running VQA Specialist → Verifying Physics...").
3. **Frictionless demo UX**:
   - Quick-load buttons or pre-filled queries for the 4 demo beats to prevent typing mistakes during live evaluation.
4. **Capture UI screenshots** for presentation slides.

### 📋 Phase 5 Deliverables for Frontend:
- [ ] UI visual polish complete for 1080p display
- [ ] Informative loading progression indicator
- [ ] Quick-select demo presets implemented
- [ ] Presentation screenshots captured

---

## 🟡 Flex / QA — Phase 5

### Step-by-step:
1. **Verify all 4 backup videos** (Layer-3 fallback):
   - Confirm `demo/backup_beat1.mp4` through `demo/backup_beat4.mp4` are 1080p, crisp, and stored locally on the presentation laptop.
2. **Finalize pitch script & judge Q&A cheat sheet**:
   - Prepare answers for architectural questions:
     - *"How is the model deployed?"* → Split-cloud architecture: Railway/Render for orchestration + Modal serverless GPU for VQA inference.
     - *"What happens if the GPU endpoint fails?"* → Automatic fallback to CPU heuristics, plus local tunnel standby and pre-rendered videos.
     - *"Why are change detection and SAR fusion reduced-fidelity?"* → Deliberate engineering prioritization under hackathon constraints, fully complete agentic routing architecture.
3. **Coordinate full dress rehearsals**:
   - Run 3+ timed rehearsals with all team members present.
   - Practice smooth transitions between live demo and narrative points.

### 📋 Phase 5 Deliverables for Flex:
- [ ] Layer-3 backup videos confirmed ready on presentation device
- [ ] `demo/speaker_notes.md` finalized with exact 8-minute timing
- [ ] `demo/judge_qa_cheatsheet.md` prepared
- [ ] 3+ timed dress rehearsals completed

---

### 🤝 Phase 5 Final Sync — GO / NO-GO

The entire team confirms:

- [ ] 🔴 GPU: Modal VQA endpoint deployed, warm, latency < 5s ✅
- [ ] 🔵 Backend: Railway/Render app online, fallback tested, code frozen ✅
- [ ] 🟢 Frontend: UI polished, quick presets ready, 1080p verified ✅
- [ ] 🟡 Flex: Backup videos on deck, speaker notes finalized, rehearsals complete ✅
- [ ] **All 4 demo beats work live on public URL** ✅
- [ ] **All 3 fallback layers verified** ✅
- [ ] **Judge Q&A answers prepared** ✅

### ✅ **GO — Ready to present.**
