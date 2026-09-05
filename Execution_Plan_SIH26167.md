# Execution Plan — SIH26167 SatQuery EvidenceSwarm
### Phased Build Sprint
**Derived from:** Solution_SIH26167 (2).md (Revised)
**Hardware constraint:** 8GB VRAM laptop
**Model:** Qwen2-VL-2B-Instruct (QLoRA fine-tuned)
**Date:** September 2, 2026

---

## Execution Overview

```
Phase 1               Phase 2                Phase 3                 Phase 4                 Phase 5
Data Engineering       Model Training         Pipeline Integration    Agentic Wiring          Polish & Delivery
────────────────────  ─────────────────────  ──────────────────────  ──────────────────────  ──────────────────
• Dataset curation     • QLoRA fine-tune      • End-to-end wiring     • Router ↔ 4 paths      • Visual polish
• Input Gate           • VQA validation       • Evidence Contract     • Split deployment      • Latency tuning
• Sensor Card          • 3 placeholders       • Lock demo cases       • Evidence Guard         • Demo rehearsal
• Spectral script      • Eval metrics                                 • Tunnel setup           • Backup videos
```

---

## Phase 1 — Data Engineering & Input Pipeline

**Goal:** Prepare all training data and build the input validation layer so fine-tuning can start in Phase 2 with zero blockers.

### Task 1.1 — VQA Training Data Curation

| Detail | Specification |
|---|---|
| **Source dataset** | BigEarthNet v2.0 (Multispectral & Sentinel-1/2 land-cover) |
| **Supplementary** | VRSBench & RSVQA (remote sensing VQA and grounding) |
| **Output format** | QLoRA-compatible instruction-tuning JSONL (image path, question, answer, metadata) |
| **Target volume** | Minimum 5,000 QA pairs; aim for 10,000 if time allows |

**Steps:**
1. Download BigEarthNet v2.0 multi-label patches and metadata CSVs.
2. Write a Python script to generate natural-language QA pairs from land-cover labels (e.g., *"What land cover types are visible?"* → *"Urban fabric, arable land"*).
3. Supplement with VRSBench/RSVQA samples that already have QA pairs — filter for relevant categories.
4. Convert all samples to unified JSONL format compatible with Unsloth/LLaMA-Factory/Swift.
5. Perform a 90/10 train/validation split. Hold out 50–100 samples as a manual test set.

**Deliverable:** `data/vqa_train.jsonl`, `data/vqa_val.jsonl`, `data/vqa_test.jsonl`

**Verification:**
- [ ] Run a `jsonl` schema validator — every record has `image`, `question`, `answer` fields.
- [ ] Spot-check 20 random samples visually (open image, read QA pair, confirm correctness).
- [ ] Confirm total count ≥ 5,000 train pairs.

---

### Task 1.2 — Input Gate & Sensor Card

| Detail | Specification |
|---|---|
| **Tech** | Python + `rasterio` + `GDAL` + `pyproj` |
| **Input** | 1 or 2 GeoTIFF files uploaded by user |
| **Output** | Structured Sensor Card JSON + pass/fail validation status |

**Steps:**
1. Build `input_gate.py`:
   - Parse GeoTIFF headers using `rasterio` — extract CRS, band count, resolution, spatial extent, nodata values.
   - Detect sensor type from metadata or filename heuristics (Sentinel-2, Cartosat-2S, RISAT, etc.).
   - Estimate baseline sensor uncertainty based on resolution and band availability.
   - Validate format: reject non-GeoTIFF, corrupt files, or missing CRS.
2. Build `sensor_card.py`:
   - Generate a structured JSON/dict with: sensor type, spatial resolution, spectral bands, CRS, bounding box, uncertainty score.
   - Provide a `render_card()` function that formats this as a human-readable display block for the UI.
3. Write 5+ unit tests covering: valid Sentinel-2 tile, valid Cartosat sample, corrupt file, non-GeoTIFF, missing CRS.

**Deliverable:** `src/input_gate.py`, `src/sensor_card.py`, `tests/test_input_gate.py`

**Verification:**
- [ ] All unit tests pass.
- [ ] Run against 3 real GeoTIFF samples — valid Sentinel-2, valid SAR, intentionally corrupt file.
- [ ] Sensor Card output is correct for each.

---

### Task 1.3 — Standalone Spectral Index Script

| Detail | Specification |
|---|---|
| **Tech** | NumPy (CPU-only), `rasterio` for band extraction |
| **Indices** | NDWI (Normalized Difference Water Index), NDVI (Normalized Difference Vegetation Index) |
| **Method** | Otsu thresholding to produce binary masks |

**Steps:**
1. Build `spectral_check.py`:
   - `compute_ndwi(image_path)` → float32 array
   - `compute_ndvi(image_path)` → float32 array
   - `otsu_threshold(index_array)` → binary mask
   - `compute_overlap(mask_a, mask_b)` → IoU / overlap ratio (float)
2. Ensure all computation is CPU-only NumPy — no GPU dependency.
3. Test on 3 Sentinel-2 tiles with known water/vegetation features.

**Deliverable:** `src/spectral_check.py`, `tests/test_spectral.py`

**Verification:**
- [ ] NDWI mask correctly highlights known water bodies in test tiles.
- [ ] NDVI mask correctly highlights vegetation in test tiles.
- [ ] Execution time < 2 seconds per tile on CPU.

---

### Phase 1 Exit Checklist

- [ ] VQA training JSONL ready (≥ 5,000 pairs)
- [ ] Input Gate validates/rejects GeoTIFFs correctly
- [ ] Sensor Card generates accurate metadata JSON
- [ ] Spectral check script produces NDWI/NDVI masks on CPU
- [ ] All unit tests pass
- [ ] **BLOCKER CHECK:** Confirm Qwen2-VL-2B-Instruct model weights downloaded and loadable on the 8GB VRAM laptop

---

## Phase 2 — Model Training & Specialist Implementation

**Goal:** Fine-tune the VQA specialist to production quality. Implement the three reduced-fidelity placeholder paths in parallel.

### Task 2.1 — QLoRA Fine-Tuning Setup

| Detail | Specification |
|---|---|
| **Base model** | `Qwen/Qwen2-VL-2B-Instruct` from Hugging Face |
| **Method** | QLoRA (4-bit quantization + LoRA adapters) |
| **Tooling** | Unsloth (primary), LLaMA-Factory or Swift (fallback) |
| **Hardware** | 8GB VRAM laptop (~3–3.5GB model headroom after quantization) |

**Steps:**
1. Install Unsloth and dependencies. Verify GPU detection and VRAM availability.
2. Load `Qwen2-VL-2B-Instruct` in 4-bit quantized mode via `bitsandbytes`.
3. Configure LoRA: `r=8`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]` (adjust based on Qwen2-VL architecture).
4. Set up training config:
   - Learning rate: `2e-4`
   - Batch size: `1` (gradient accumulation = `4`)
   - Epochs: `3`
   - Max sequence length: `2048`
5. Do a 10-sample dry run to confirm no OOM errors.

**Deliverable:** Working training script `scripts/finetune_vqa.py`

**Verification:**
- [ ] Dry run completes without OOM.
- [ ] GPU utilization stays under 7.5GB VRAM.
- [ ] Loss is decreasing on the 10-sample test.

---

### Task 2.2 — Full VQA Fine-Tuning Run

**Steps:**
1. Launch full training on `data/vqa_train.jsonl`.
2. Monitor training loss, validation loss, and GPU memory every 30 minutes.
3. Save checkpoints every 500 steps.
4. If OOM occurs: reduce batch size to 1 with gradient accumulation = 8, or reduce max sequence length to 1024.

**Deliverable:** Fine-tuned adapter weights in `models/qwen2vl_vqa_lora/`

**Verification:**
- [ ] Training completes without crashes.
- [ ] Final validation loss is lower than initial loss.
- [ ] Adapter weights saved and loadable.

---

### Task 2.3 — VQA Specialist Validation

**Steps:**
1. Load fine-tuned adapter onto base model.
2. Run inference on the 50–100 held-out test samples.
3. Compute metrics:
   - **Accuracy**: Exact-match or semantic similarity score.
   - **Grounding quality**: If bounding boxes are generated, compute IoU against ground truth.
4. Compare against base model (no fine-tuning) on the same test set — confirm measurable improvement.
5. Run 5 qualitative samples: display image + question + model answer side-by-side.

**Deliverable:** `results/vqa_eval_report.md` with metrics and qualitative examples

**Verification:**
- [ ] Fine-tuned model outperforms base model on held-out set.
- [ ] Qualitative outputs are grounded and sensible.
- [ ] Inference runs within 8–10 second latency budget.

---

### Task 2.4 — Reduced-Fidelity Placeholder Specialists (Parallel Track)

Build all three placeholder specialists. Each follows the same pattern: receive routed input → run simple logic → return structured output with `"fidelity": "reduced"` flag.

#### 2.4.1 — Captioning / Grounding Placeholder

| Detail | Specification |
|---|---|
| **Method** | Rule-based template captioning + basic contour detection (OpenCV) |
| **Input** | Single GeoTIFF + query |
| **Output** | Template caption based on band statistics + crude bounding regions |

**Steps:**
1. Build `src/specialists/caption_grounding.py`:
   - Compute band statistics (mean, std, histogram) from the image.
   - Generate a template caption: *"This {resolution}m {sensor_type} image shows {dominant_band_description} patterns across the {extent_description} area."*
   - Run simple edge/contour detection for crude spatial grounding.
   - Return `{"caption": "...", "regions": [...], "fidelity": "reduced", "method": "rule-based"}`.

#### 2.4.2 — Bi-Temporal Change Detection Placeholder

| Detail | Specification |
|---|---|
| **Method** | Image differencing + Otsu thresholding |
| **Input** | Two co-registered GeoTIFF tiles (pre/post) |
| **Output** | Binary change mask + change percentage |

**Steps:**
1. Build `src/specialists/change_detection.py`:
   - Load both images, align to same CRS/extent if needed.
   - Compute pixel-wise absolute difference.
   - Apply Otsu thresholding to generate binary change mask.
   - Calculate change percentage: `changed_pixels / total_pixels`.
   - Return `{"change_mask": array, "change_pct": float, "fidelity": "reduced", "method": "image-differencing"}`.

#### 2.4.3 — Optical-SAR Fusion Placeholder

| Detail | Specification |
|---|---|
| **Method** | Basic band-overlay heuristic |
| **Input** | One optical GeoTIFF + one SAR GeoTIFF |
| **Output** | Fused composite + simple analysis summary |

**Steps:**
1. Build `src/specialists/optical_sar_fusion.py`:
   - Load optical and SAR images, co-register if CRS differs.
   - Normalize both to 0–1 range.
   - Create composite overlay: RGB from optical + SAR backscatter as alpha/intensity channel.
   - Generate summary based on band statistics: *"SAR backscatter indicates {high/low} surface roughness in {region}."*
   - Return `{"composite": array, "summary": "...", "fidelity": "reduced", "method": "band-overlay-heuristic"}`.

**Deliverables:** `src/specialists/caption_grounding.py`, `src/specialists/change_detection.py`, `src/specialists/optical_sar_fusion.py`

**Verification:**
- [ ] Each placeholder accepts input and returns structured output with `fidelity: reduced` flag.
- [ ] Change detection produces a visible change mask on two different-date Sentinel-2 tiles.
- [ ] All three run on CPU only with no GPU dependency.

---

### Phase 2 Exit Checklist

- [ ] QLoRA fine-tuning complete, adapter saved
- [ ] VQA specialist outperforms base model on eval
- [ ] All 3 placeholder specialists produce structured output
- [ ] Every specialist output includes `fidelity` field
- [ ] **BLOCKER CHECK:** VQA inference latency ≤ 8–10 seconds on laptop GPU

---

## Phase 3 — Pipeline Integration

**Goal:** Wire every component into a single end-to-end pipeline. Implement the Evidence Contract & Refusal Gate. Lock the 4 demo test cases.

### Task 3.1 — FastAPI Backend Skeleton

| Detail | Specification |
|---|---|
| **Tech** | FastAPI + Uvicorn |
| **Endpoints** | `POST /query`, `GET /health`, `GET /sensor-card` |

**Steps:**
1. Create `src/app.py` — FastAPI application.
2. Implement `POST /query`:
   - Accept: image file(s) + natural-language query string.
   - Pipeline: Input Gate → Evidence Contract → Agentic Router → Specialist → Evidence Guard → Confidence Engine → Response.
   - Return: JSON with answer, confidence scores, execution trace, and any overlays (base64 or URL).
3. Implement `GET /sensor-card`:
   - Accept: image file.
   - Return: Sensor Card JSON from Task 1.2.
4. Implement `GET /health` — simple liveness check.

**Deliverable:** `src/app.py` with all three endpoints functional

---

### Task 3.2 — Evidence Contract & Refusal Gate

| Detail | Specification |
|---|---|
| **Purpose** | Pre-flight check: can the uploaded data physically support the query? |
| **Location in pipeline** | After Input Gate, before Router |

**Steps:**
1. Build `src/evidence_contract.py`:
   - **Single-image query**: Verify image is valid and has the required bands for the query type.
   - **Bi-temporal query**: Verify two images are provided, have overlapping spatial extent, and represent different timestamps.
   - **Optical-SAR query**: Verify one optical and one SAR image are provided, with overlapping spatial extent.
   - **Refusal logic**: If any check fails → return `{"status": "refused", "reason": "...", "suggestion": "..."}` with clear, actionable feedback.
2. Write test cases for each refusal scenario:
   - Single image submitted for change detection → refuse.
   - Two non-overlapping images → refuse.
   - SAR-only submitted for optical VQA → refuse.
   - Valid single optical image for VQA → pass.

**Deliverable:** `src/evidence_contract.py`, `tests/test_evidence_contract.py`

**Verification:**
- [ ] All 4 refusal scenarios correctly trigger refusal with appropriate feedback.
- [ ] All valid scenarios pass through to the router.

---

### Task 3.3 — End-to-End Pipeline Wiring

**Steps:**
1. Connect all components in sequence inside the `/query` endpoint:
   ```
   Input Gate → Sensor Card → Evidence Contract → Router → Specialist → Evidence Guard → Confidence → Output
   ```
2. Add execution trace logging at each stage — record timestamp, component name, input/output summary, and status (pass/fail/skip).
3. Store traces in SQLite via `src/trace_logger.py`.
4. Test the full pipeline with one sample query for each of the 4 task types.

**Deliverable:** Working end-to-end `/query` endpoint

---

### Task 3.4 — Lock 4 Demo Test Cases

Select and freeze the exact image files and queries for each demo beat:

| Beat | Image(s) | Query | Expected Output |
|---|---|---|---|
| **Beat 1 — VQA** | Single Sentinel-2 optical tile | *"What land cover types are visible in this image?"* | VQA answer with grounding |
| **Beat 2 — Change Detection** | Pre/post-event Sentinel-2 pair | *"What areas changed between these two dates?"* | Change mask + percentage (reduced-fidelity) |
| **Beat 3 — Optical-SAR Fusion** | Sentinel-2 optical + Sentinel-1 SAR | *"Identify features visible in SAR but obscured in optical"* | Fused composite + summary (reduced-fidelity) |
| **Beat 4 — Refusal** | Single optical image | *"Show me changes between the two dates"* | Clean refusal with explanation |

**Deliverable:** `demo/` folder with frozen test images and a `demo_cases.json` manifest

**Verification:**
- [ ] All 4 demo cases run through the full pipeline without errors.
- [ ] Beat 4 triggers a clear refusal.
- [ ] End-to-end latency ≤ 10 seconds for VQA beat.

---

### Phase 3 Exit Checklist

- [ ] FastAPI server starts and serves all endpoints
- [ ] Full pipeline runs for all 4 task types
- [ ] Evidence Contract correctly refuses invalid inputs
- [ ] Execution trace is logged to SQLite for every query
- [ ] 4 demo cases locked and passing
- [ ] **BLOCKER CHECK:** No runtime errors on full pipeline traversal

---

## Phase 4 — Agentic Wiring & Deployment

**Goal:** Integrate the agentic router, set up split deployment, and integrate the Evidence Guard.

### Task 4.1 — Agentic Router Integration

| Detail | Specification |
|---|---|
| **Tech** | `sentence-transformers` + rule overrides |
| **Classification targets** | `vqa`, `caption_grounding`, `change_detection`, `optical_sar_fusion` |

**Steps:**
1. Build `src/agentic_router.py`:
   - Load a lightweight sentence-transformer model (e.g., `all-MiniLM-L6-v2`, ~80MB, CPU-only).
   - Pre-compute embeddings for canonical query templates per task type.
   - On incoming query: compute embedding → cosine similarity → select highest-scoring task type.
   - **Rule overrides**: hard-coded keyword patterns that bypass the embedding classifier:
     - *"change"*, *"before and after"*, *"difference"* → `change_detection`
     - *"SAR"*, *"radar"*, *"all-weather"*, *"backscatter"* → `optical_sar_fusion`
     - *"caption"*, *"describe"*, *"grounding"*, *"locate"* → `caption_grounding`
   - Return: `{"task_type": "...", "confidence": float, "method": "rule_override" | "embedding_similarity"}`.
2. Validate input count: if task requires 2 images and only 1 provided, defer to Evidence Contract (which should have already caught this — but double-check).
3. Dispatch to the appropriate specialist function.

**Deliverable:** `src/agentic_router.py`

**Verification:**
- [ ] 10 test queries correctly classified to the right task type.
- [ ] Rule overrides fire deterministically on keyword-match queries.
- [ ] Router latency < 200ms.

---

### Task 4.2 — Evidence Guard Integration

| Detail | Specification |
|---|---|
| **Component 1** | Qwen2-VL grounding / CLIP spatial bounds (for VQA path) |
| **Component 2** | Otsu NDWI/NDVI spectral cross-check (from Phase 1 script) |

**Steps:**
1. Build `src/evidence_guard.py`:
   - **For VQA path (full verification)**:
     - Extract spatial attention / grounding from Qwen2-VL output if available.
     - Compute CLIP similarity between the model's textual answer and the image region.
     - Run `spectral_check.py` on the same image region — compute overlap between model's highlighted region and NDWI/NDVI mask.
     - Return guard scores: `C_guard` (spatial alignment) and `C_spectral` (spectral agreement).
   - **For reduced-fidelity paths (basic validation)**:
     - Run only the spectral cross-check (NDWI/NDVI) as a sanity check.
     - Return basic validation score.
2. Integrate into the pipeline — Evidence Guard runs after specialist execution, before Confidence Engine.

**Deliverable:** `src/evidence_guard.py`

---

### Task 4.3 — Confidence & Output Engine

| Detail | Specification |
|---|---|
| **Formula** | `Score_conf = w1·C_sensor + w2·C_adapter + w3·C_guard + w4·C_spectral` |
| **Weights** | `w1=0.15, w2=0.35, w3=0.25, w4=0.25` (tunable) |
| **Labels** | ≥0.85 = "High Evidence Consistency", 0.65–0.84 = "Moderate", <0.65 = "Low" |

**Steps:**
1. Build `src/confidence_engine.py`:
   - Collect scores from Sensor Card (`C_sensor`), specialist (`C_adapter`), Evidence Guard (`C_guard`, `C_spectral`).
   - Compute weighted aggregate.
   - Assign label.
2. Build `src/output_renderer.py`:
   - Render visual overlay on the image (highlight regions, bounding boxes, change masks).
   - Generate JSON execution trace.
   - Generate downloadable 1-page PDF summary report (use `reportlab` or `fpdf2`).

**Deliverable:** `src/confidence_engine.py`, `src/output_renderer.py`

---

### Task 4.4 — Split Deployment Setup

**Steps:**
1. **Orchestration layer (CPU-only) → Railway or Render:**
   - Create `Dockerfile` for the FastAPI app (exclude GPU dependencies).
   - Ensure all placeholder specialists run on CPU.
   - Deploy to Railway or Render as a web service.
   - Verify health endpoint is reachable from public URL.

2. **VQA model inference (GPU) → Laptop via tunnel:**
   - Install `ngrok` or `cloudflare tunnel` on the laptop.
   - Create a lightweight FastAPI endpoint `POST /infer` on the laptop that accepts an image + query and returns VQA output.
   - Expose this endpoint via tunnel.
   - Configure the deployed orchestration layer to call the tunnel URL for VQA inference.
   - Test the full chain: public URL → Railway → tunnel → laptop GPU → response.

3. **Stretch goal — Serverless GPU (Modal/RunPod):**
   - Only attempt if deployment is stable and buffer time remains.
   - Package model + inference code as a Modal function or RunPod serverless endpoint.

**Deliverable:** Deployed public URL, working tunnel setup, `Dockerfile`, `deploy/` config files

**Verification:**
- [ ] Public URL returns health check OK.
- [ ] Full query through public URL → laptop tunnel → VQA response works.
- [ ] Latency through tunnel is acceptable (< 15 seconds total including network).

---

### Phase 4 Exit Checklist

- [ ] Router correctly classifies and dispatches all 4 query types
- [ ] Evidence Guard runs on VQA output and produces guard scores
- [ ] Confidence Engine computes and labels scores correctly
- [ ] Deployment is live — public URL works
- [ ] Tunnel from laptop GPU is stable
- [ ] All 4 demo beats work through the deployed system
- [ ] **BLOCKER CHECK:** If tunnel is unstable, record demo videos immediately (Layer 3 fallback)

---

## Phase 5 — Polish & Delivery

**Goal:** Polish the UI, optimize latency, rehearse the demo, and record backup videos. This phase is also a buffer — if earlier phases slipped, catch up here.

### Task 5.1 — UI / Visual Polish

**Steps:**
1. Build or polish the frontend interface (Gradio / Streamlit / custom HTML):
   - Clean upload area for 1–2 GeoTIFF files.
   - Natural-language query input box.
   - Sensor Card display panel.
   - Results area: answer text, visual overlay image, confidence badge, execution trace accordion.
   - Download button for PDF report.
2. Ensure the refusal path renders clearly (distinct red/warning styling, actionable suggestion).
3. Add reduced-fidelity path indicator — a visible badge/banner when a reduced-fidelity specialist is used.

**Deliverable:** Polished, demo-ready UI

---

### Task 5.2 — Latency Optimization

| Target | Budget |
|---|---|
| Input Gate + Sensor Card | < 1 second |
| Evidence Contract | < 0.5 seconds |
| Agentic Router | < 0.5 seconds |
| VQA Specialist (GPU) | < 5 seconds |
| Placeholder Specialists (CPU) | < 2 seconds |
| Evidence Guard | < 1 second |
| Confidence + Rendering | < 1 second |
| **Total pipeline** | **≤ 8–10 seconds** |

**Steps:**
1. Profile the full pipeline end-to-end. Identify bottleneck.
2. If VQA inference is too slow: reduce `max_new_tokens`, enable flash attention if supported.
3. If spectral check is slow: pre-compute indices on upload rather than at query time.
4. Cache sentence-transformer embeddings for the router.

---

### Task 5.3 — Demo Rehearsal & Backup Recording

**Steps:**
1. **Full rehearsal**: Run all 4 demo beats through the live system, end-to-end, at least 3 times.
2. **Time each beat**: Ensure each beat fits within a 2-minute pitch segment.
3. **Script the narrative**: Write exact speaker notes for each beat transition.
4. **Record Layer-3 backup videos**:
   - Screen-record each of the 4 beats in high definition.
   - Save as `demo/backup_beat1.mp4`, `demo/backup_beat2.mp4`, etc.
   - These are the absolute last-resort fallback if everything goes wrong live.
5. **Stress-test the refusal path**: Try 5+ adversarial queries to make sure the Evidence Contract holds.

**Deliverable:** `demo/speaker_notes.md`, `demo/backup_beat*.mp4`

---

### Task 5.4 — Final Verification & Freeze

**Steps:**
1. Run the complete demo flow one final time, live, as if presenting to judges.
2. Verify:
   - [ ] All 4 beats execute correctly through the deployed URL.
   - [ ] Refusal beat triggers cleanly.
   - [ ] Confidence scores render correctly.
   - [ ] PDF report downloads and looks professional.
   - [ ] Execution trace is visible and readable.
   - [ ] Reduced-fidelity badge displays for Beats 2 and 3.
3. **Code freeze**: No more changes after this point.
4. Ensure fallback layers are ready:
   - [ ] Layer 1: Base model fallback works if adapter fails.
   - [ ] Layer 2: 4-bit quantization + CPU-only mode works.
   - [ ] Layer 3: All 4 backup videos recorded and accessible.

---

### Phase 5 Exit Checklist (FINAL)

- [ ] UI is polished and demo-ready
- [ ] Full pipeline latency ≤ 10 seconds
- [ ] All 4 demo beats rehearsed ≥ 3 times
- [ ] Speaker notes finalized
- [ ] Backup videos recorded (all 4 beats)
- [ ] All 3 fallback layers tested and confirmed working
- [ ] Code frozen
- [ ] **GO / NO-GO decision confirmed**

---

## Project File Structure

```
SatQuery-AI/
├── src/
│   ├── app.py                        # FastAPI main application
│   ├── input_gate.py                 # GeoTIFF validation & metadata extraction
│   ├── sensor_card.py                # Sensor Card generator
│   ├── evidence_contract.py          # Pre-flight query-data compatibility check
│   ├── agentic_router.py             # Intent classification & task dispatch
│   ├── evidence_guard.py             # Dual verification (neural + spectral)
│   ├── confidence_engine.py          # 4-factor confidence score computation
│   ├── output_renderer.py            # Visual overlay + PDF/JSON report generation
│   ├── spectral_check.py             # CPU-only NDWI/NDVI Otsu thresholding
│   ├── trace_logger.py               # SQLite execution trace logger
│   └── specialists/
│       ├── vqa_specialist.py          # [FINE-TUNED] Qwen2-VL-2B QLoRA inference
│       ├── caption_grounding.py       # [REDUCED-FIDELITY] Rule-based captioning
│       ├── change_detection.py        # [REDUCED-FIDELITY] Image differencing
│       └── optical_sar_fusion.py      # [REDUCED-FIDELITY] Band-overlay heuristic
├── models/
│   └── qwen2vl_vqa_lora/             # Fine-tuned QLoRA adapter weights
├── data/
│   ├── vqa_train.jsonl                # Training data
│   ├── vqa_val.jsonl                  # Validation data
│   └── vqa_test.jsonl                 # Held-out test data
├── demo/
│   ├── demo_cases.json                # Frozen demo test case manifest
│   ├── speaker_notes.md               # Pitch script
│   ├── images/                        # Demo GeoTIFF files
│   └── backup_beat*.mp4               # Layer-3 fallback videos
├── deploy/
│   ├── Dockerfile                     # CPU-only orchestration container
│   └── tunnel_config.yml              # ngrok/Cloudflare tunnel config
├── scripts/
│   └── finetune_vqa.py                # QLoRA fine-tuning script
├── tests/
│   ├── test_input_gate.py
│   ├── test_evidence_contract.py
│   ├── test_spectral.py
│   └── test_router.py
├── results/
│   └── vqa_eval_report.md             # Fine-tuning evaluation results
├── requirements.txt
└── README.md
```

---

## Risk Checkpoints (Built Into Each Phase)

| Phase | Critical Risk | Checkpoint Action |
|---|---|---|
| **Phase 1** | Model weights won't load on 8GB card | Test load Qwen2-VL-2B in 4-bit before Phase 2 |
| **Phase 2** | OOM during fine-tuning | Dry run with 10 samples first; reduce batch/seq length if OOM |
| **Phase 2** | Fine-tuned model doesn't outperform base | If < 5% improvement, adjust data/hyperparams and re-train |
| **Phase 3** | Pipeline integration errors | Test each component in isolation before wiring |
| **Phase 4** | Tunnel is unreliable for live demo | Immediately record backup videos (Layer 3 fallback) |
| **Phase 5** | System crashes during rehearsal | Confirm all 3 fallback layers are operational |

---

## Open Decision Required

> **Judge-testable URL vs. live local demo:** The deployment plan assumes a tunnel-based approach for GPU inference. If the SIH internal round requires a persistent public URL that judges can test independently (without the laptop being online), the team must allocate Phase 4 buffer time to deploy the VQA model to a serverless GPU host (Modal/RunPod) instead of relying on the laptop tunnel. This decision should be resolved **before Phase 3**.
