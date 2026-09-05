# Solution Blueprint — SIH26167 SatQuery EvidenceSwarm

## 1. Solution Title & One-Liner

### Solution Title
**SatQuery EvidenceSwarm — Lean Adaptive Agentic Architecture for Multimodal Remote Sensing Analysis**

### One-Sentence Description
A lean, latency-budgeted agentic remote-sensing assistant that runs a fine-tuned Qwen2-VL-2B-Instruct VQA specialist alongside three architecturally-complete reduced-fidelity pipeline paths, cross-checks every claim against both learned grounding and classical spectral physics, and enforces an explicit Evidence Contract to refuse unsupported queries — delivering verified, confidence-calibrated, and auditable satellite insights on an 8GB VRAM laptop.

---

## 2. Problem–Solution Fit & Core Value Proposition

### Restatement of Core Problem
India’s Earth-observation (EO) satellites (Cartosat-2S, RISAT, Sentinel, etc.) capture critical data for agriculture, disaster response, urban planning, forestry, and water management. However, extracting actionable answers requires navigating fragmented GIS tools, selecting complex task-specific models, and interpreting physical sensor characteristics (optical vs. SAR backscatter). Generic Vision-Language Models (VLMs) fail due to domain shift (overhead perspective, multi-spectral bands, spatial scale variation), while traditional AI tools lack auditability and often hallucinate answers on ambiguous or incomplete inputs.

### Direct Address of Root Causes

| Root Cause | How SatQuery EvidenceSwarm Resolves It |
|---|---|
| **Fragmented single-task tools** | Integrates an **Agentic Router** with a fully fine-tuned **VQA specialist** and three **reduced-fidelity pipeline paths** (Captioning/Grounding, Bi-Temporal Change, Optical-SAR Fusion) under a unified natural-language interface. |
| **Domain shift in generic VLMs** | Employs a **Domain-Adapted Qwen2-VL-2B-Instruct** fine-tuned via QLoRA on BigEarthNet-derived QA data, with native grounding support as a stretch capability. |
| **Sensor generalization risk (Cartosat-2S / RISAT)** | Features an **Input Gate & Sensor Card** that detects sensor characteristics, normalizes formats, and reports sensor uncertainty upfront. |
| **Model hallucination & unverified claims** | Implements an **Evidence Contract & Refusal Gate** (stops execution if inputs cannot support the query) paired with an **Evidence Guard** that cross-checks neural predictions against classical spectral physics (NDWI/NDVI). |
| **Black-box predictions & lack of auditability** | Generates a **4-Factor Calibrated Confidence Score**, visual evidence overlays, observable execution traces, and downloadable PDF/JSON audit reports. |

### Unique Value Proposition
Unlike brittle VLM wrappers that guess when faced with incomplete data, **SatQuery EvidenceSwarm** introduces an **evidence-first execution model**. It defines what must be proven *before* invoking models, cross-verifies neural outputs against independent physical laws (classical spectral indices), and proves trust by explicitly refusing to answer when data is insufficient. Under current hardware and timeline constraints (8GB VRAM, 6-day build), the system delivers one production-quality fine-tuned specialist (VQA) alongside three architecturally-complete reduced-fidelity pipeline paths — a deliberate engineering trade-off that preserves the full agentic framework while concentrating real ML depth where it has the highest evaluation impact.

---

## 3. System Architecture & Workflow

### High-Level Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Query + Input Imagery                       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Input Gate & Sensor Card                          │
│   • Format & CRS Validation (rasterio/GDAL)                            │
│   • Sensor Type & Resolution Detection                                  │
│   • Baseline Sensor Uncertainty Estimation                              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Evidence Contract & Refusal Gate                      │
│   • Evaluates query compatibility vs. input metadata                     │
│   • Checks image overlap, spatial coverage, and temporal alignment      │
│   • [REFUSAL PATH]: Refuses with clear feedback if criteria fail        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ (Contract Satisfied)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Agentic Router                               │
│   • Intent classification via sentence-transformers & rule overrides    │
│   • Routes to VQA specialist or reduced-fidelity pipeline paths        │
└───┬──────────────────┬──────────────────┬──────────────────┬────────────┘
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
┌────────────────┐┌────────────────┐┌────────────────┐┌────────────────┐
│ VQA Specialist ││ Caption /      ││ Bi-Temporal    ││ Optical-SAR    │
│ [FINE-TUNED]   ││ Grounding      ││ Change         ││ Fusion         │
│ Qwen2-VL-2B   ││ [REDUCED-      ││ [REDUCED-      ││ [REDUCED-      │
│ QLoRA          ││  FIDELITY]     ││  FIDELITY]     ││  FIDELITY]     │
└───────┬────────┘└───────┬────────┘└───────┬────────┘└───────┬────────┘
        │                 │                 │                 │
        └─────────────────┴─────────────────┴─────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             Evidence Guard                              │
│   1. Learned Grounding Verification (Qwen2-VL / CLIP Spatial Bounds)    │
│   2. Classical Spectral Verification (Otsu-NDWI / NDVI overlapping mask) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Confidence & Output Engine                         │
│   • Calculates 4-Factor Calibrated Confidence Score                     │
│   • Renders visual overlays & bounding masks on imagery                 │
│   • Generates observable JSON execution trace & downloadable PDF report │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

1. **Input Gate & Sensor Card**: Validates GeoTIFF headers, Coordinate Reference Systems (CRS), and band availability. Instantly displays a "Sensor Card" with metadata, resolution, and baseline sensor uncertainty.
2. **Evidence Contract & Refusal Gate**: Acting as an early safety check, this component verifies whether the inputs can physically support the question (e.g., verifying bi-temporal coverage before attempting change analysis). If inputs fail criteria, it triggers a clean refusal with actionable feedback instead of hallucinating.
3. **Agentic Router**: Uses sentence embeddings and rule overrides to direct queries to the appropriate specialist adapters in under 1 second.
4. **Specialist Paths**: One fully fine-tuned specialist and three reduced-fidelity pipeline paths:
   - *VQA Specialist* (**Fully Fine-Tuned**): QLoRA fine-tuned Qwen2-VL-2B-Instruct on BigEarthNet-derived QA data for single-image visual questioning.
   - *Captioning / Grounding Path* (**Reduced-Fidelity**): Router-dispatched with a rule-based / classical CV placeholder specialist.
   - *Bi-Temporal Change Path* (**Reduced-Fidelity**): Router-dispatched with an image differencing/thresholding placeholder specialist.
   - *Optical-SAR Fusion Path* (**Reduced-Fidelity**): Router-dispatched with a basic band-overlay heuristic placeholder specialist.
5. **Evidence Guard**: Dual verification layer combining neural grounding with classical spectral math (NumPy-based CPU Otsu thresholding for NDWI/NDVI overlap). Applies fully to the VQA path; reduced-fidelity paths receive basic validation only.
6. **Confidence & Output Engine**: Synthesizes verification outputs into a 4-factor confidence badge, renders map overlays, and exports structured reports.

---

## 4. Technical Architecture & Tech Stack

### System Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Backend API** | FastAPI + Uvicorn | Lightweight async model serving and pipeline orchestration |
| **Geospatial Processing** | `rasterio`, `GDAL`, `pyproj`, `geopandas` | Industry-standard handling of multi-band GeoTIFFs, CRS reprojection, and spatial masks |
| **RS-VLM Backbone** | **Qwen2-VL-2B-Instruct** (QLoRA fine-tuned) | Best VRAM headroom (~3–3.5GB on 8GB card), mature QLoRA tooling, native grounding support |
| **Model Adaptation** | **Unsloth / LLaMA-Factory / Swift** + QLoRA | Most mature fine-tuning tooling for Qwen2-VL; single VQA adapter with maximum iteration room |
| **Quantization** | `bitsandbytes` (4-bit / 8-bit) | Enables full pipeline execution within an **8GB VRAM** laptop footprint (~3–3.5GB model headroom) |
| **Agentic Router** | `sentence-transformers` + rule overrides | High-speed, deterministic intent routing with low latency |
| **Evidence Guard** | Qwen2-VL grounding + CLIP similarity + Otsu NDWI/NDVI | Dual-verification combining neural spatial attention with classical CPU spectral physics |
| **Storage & Logging** | Local filesystem + SQLite | Zero-dependency local persistence for execution logs and report generation |
| **Deployment** | Railway/Render (orchestration, CPU-only) + laptop-via-tunnel / Modal/RunPod (GPU inference) | Split architecture driven by 8GB VRAM constraint; serverless GPU is stretch goal |

### Deployment Architecture

The system uses a split deployment model driven by the 8GB VRAM laptop constraint:

- **Orchestration Layer** (CPU-only): GUI, agentic router, and reduced-fidelity placeholder specialists deploy to **Railway or Render** as standard web services.
- **VQA Model Inference** (GPU-dependent): The fine-tuned Qwen2-VL-2B-Instruct model runs on the **local laptop exposed via tunnel** (ngrok/Cloudflare Tunnel) as the primary plan, requiring no additional cost. A **serverless GPU host** (Modal/RunPod) is a stretch goal only if Day 5–6 buffer time allows.

> **Open item:** Confirm whether the SIH internal round requires a judge-testable public URL or is satisfied by a live local demo — this determines how much buffer time should go toward hardening the tunnel vs. polishing the demo script.

---

## 5. Confidence Calibration Model

Rather than asserting arbitrary accuracy metrics, **SatQuery EvidenceSwarm** computes an **Evidence-Consistency Score** ($\text{Score}_{\text{conf}}$) aggregated across four independent factors:

$$\text{Score}_{\text{conf}} = w_1 \cdot C_{\text{sensor}} + w_2 \cdot C_{\text{adapter}} + w_3 \cdot C_{\text{guard}} + w_4 \cdot C_{\text{spectral}}$$

Where:
- $C_{\text{sensor}}$: Sensor-normalizer certainty (evaluates metadata completeness and resolution suitability).
- $C_{\text{adapter}}$: Adapter grounding confidence (model's soft attention over targeted visual regions).
- $C_{\text{guard}}$: Evidence-guard survival rate (spatial alignment between prediction and image features).
- $C_{\text{spectral}}$: Spectral cross-check agreement (overlap ratio between VLM prediction and classical NDWI/NDVI mask).

### Illustrative Confidence Output
```
Sensor-normalizer certainty:   0.83
Adapter/grounding confidence:  0.88
Evidence-guard survival rate:  0.91
Spectral cross-check agreement: 0.86
--------------------------------------
Aggregated Evidence Score:     0.87  (Labeled: "High Evidence Consistency")
```

---

## 6. Functional Capabilities & Scope

### Must-Have Features (MVP Scope)
- **GeoTIFF Upload & Compatibility Validation**: Automated header parsing and spatial alignment check.
- **Interactive Sensor Card**: Instant display of sensor type, resolution, spectral bands, and sensor uncertainty.
- **Evidence Contract & Refusal Engine**: Formal constraint check that halts execution with clear explanations on incompatible inputs.
- **Single-Image VQA & Bounding** *(Fully Fine-Tuned Specialist)*: Spatial grounding for feature identification, powered by QLoRA fine-tuned Qwen2-VL-2B-Instruct on BigEarthNet-derived QA data.
- **Bi-Temporal Change Detection** *(Reduced-Fidelity Path)*: Architecturally complete pipeline (routed, validated, output-formatted) with a classical CV placeholder specialist (image differencing/thresholding) flagged as reduced-fidelity.
- **Optical-SAR Fusion Analysis** *(Reduced-Fidelity Path)*: Architecturally complete pipeline with a basic band-overlay heuristic placeholder specialist flagged as reduced-fidelity.
- **Captioning / Grounding** *(Reduced-Fidelity Path)*: Architecturally complete pipeline with a rule-based / classical CV placeholder specialist flagged as reduced-fidelity.
- **Agentic Adapter Routing**: Low-latency query routing across the VQA specialist and three reduced-fidelity pipeline paths.
- **Dual Verification (Evidence Guard)**: Neural grounding + CPU-based Otsu NDWI/NDVI spectral cross-checking.
- **4-Factor Confidence Calibration**: Transparent confidence breakdown display.
- **Auditability & Export**: Observable execution trace log and downloadable 1-page PDF/JSON summary report.
- **Strict Latency Budget**: $\sim 8\text{--}10$ second total execution pipeline.

### Nice-to-Have Features (Deferred Post-MVP)
- **Fast-Draft Mode**: Lower resolution preview option for quick exploratory queries.
- **Side-by-Side Sensor Comparator**: Visual split-screen comparing raw optical vs. SAR bands.
- **Session History & Caching**: Local caching of previous query results.

### Out of Scope (Future Roadmap)
- Full vernacular voice-to-text NLU pipeline.
- Distributed Docker/Kubernetes cluster deployment.
- Live satellite data API integration.

---

## 7. Data & Training Strategy

### Core Benchmark Datasets
- **Adaptation Anchor**: *BigEarthNet v2.0* (Multispectral & Sentinel-1/2 land-cover anchor).
- **Single-Image VQA**: *VRSBench* & *RSVQA* (Remote sensing visual question answering and grounding).
- **Bi-Temporal Change**: *CDVQA* & *LEVIR-CC* (Change detection and captioning).
- **Cross-Modal Fusion**: *BigEarthNet-MM* & *SARLANG-1M* (Co-registered optical and SAR pairs).
- **Indian Satellite Robustness**: Public Cartosat-2S and RISAT sample tiles for domain shift testing.

---

## 8. User Experience & Live Demo Flow

### Step-by-Step User Journey
1. **Data Ingestion**: User uploads 1 or 2 satellite GeoTIFF files.
2. **Sensor Inspection**: Sensor Card displays sensor metadata, spatial resolution, and band details.
3. **Query Submission**: User types a natural language question (e.g., *"Identify flooded agricultural land between these two dates"*).
4. **Contract Verification**: Evidence Contract validates if uploaded imagery matches query requirements (triggers Refusal Path if invalid).
5. **Specialist Execution**: Agentic Router invokes required LoRA adapter(s).
6. **Physics Verification**: Evidence Guard runs learned grounding and Otsu NDWI spectral cross-check.
7. **Result Delivery**: UI renders highlighted overlay, 4-factor confidence badge, execution trace, and downloadable PDF report.

### 4-Beat Live Demonstration Script
1. **Beat 1 (Single-Image VQA Baseline)**: Standard object identification and spatial bounding on optical imagery — powered by the fully fine-tuned Qwen2-VL-2B-Instruct VQA specialist.
2. **Beat 2 (Bi-Temporal Change Detection)**: Pre/post-flood analysis generating a pixel-level change mask and confidence report — demonstrates the full agentic pipeline with a reduced-fidelity placeholder specialist (image differencing/thresholding).
3. **Beat 3 (Optical-SAR Fusion)**: Querying cloud-covered flood zones using combined Sentinel-1 SAR and optical imagery — demonstrates the full agentic pipeline with a reduced-fidelity placeholder specialist (band-overlay heuristic).
4. **Beat 4 (The Signature Refusal)**: Submitting an incompatible query (e.g., asking for change detection on non-overlapping images). System triggers the Evidence Contract and refuses to answer, demonstrating reliability and trust.

### 3-Layer Fallback Strategy
- **Layer 1 (VQA Specialist Fallback)**: If the fine-tuned Qwen2-VL-2B model fails to load, pipeline falls back to the base model with generic prompt templates; reduced-fidelity paths continue operating as designed.
- **Layer 2 (CPU/Limited GPU Fallback)**: Runs quantization down to 4-bit and processes spectral checks strictly on CPU.
- **Layer 3 (Video/Pre-rendered Backup)**: Full recorded demo videos of all 4 beats available for live pitch presentation.

---

## 9. Development Roadmap & Execution Phases

| Phase | Milestone & Focus Deliverables |
|---|---|
| **Day 1 — Data Engineering** | • Curate and format BigEarthNet-derived VQA training data.<br>• Implement Input Gate, Sensor Card, and standalone spectral check script. |
| **Days 2–3 — Fine-Tuning** | • Fine-tune **Qwen2-VL-2B-Instruct** via QLoRA on BigEarthNet-derived VQA data.<br>• Validate VQA specialist on held-out test set.<br>• Implement reduced-fidelity placeholder specialists (classical CV / rule-based). |
| **Day 4 — Pipeline Integration** | • Wire pipeline end-to-end.<br>• Implement Evidence Contract and Refusal Gate logic.<br>• Lock 4 primary demo test cases. |
| **Day 5 — Agentic Wiring & Deployment** | • Integrate agentic router with all four pipeline paths.<br>• Set up split deployment (Railway/Render + laptop tunnel).<br>• Integrate Evidence Guard (NDWI/NDVI CPU cross-check). |
| **Day 6 — Buffer & Polish** | • Polish visual overlays, execution trace display, and PDF exporter.<br>• Perform latency optimization and stress-test refusal paths.<br>• Rehearse 4-beat demo script and record Layer-3 backup videos. |

---

## 10. Risk Assessment & Risk Mitigation Matrix

| Identified Risk | Risk Severity | Proposed Mitigation Strategy |
|---|---|---|
| **Domain Shift (Unseen Cartosat/RISAT data)** | Medium | Sensor Card reports uncertainty upfront; apply spatial/contrast normalization at the Input Gate. |
| **Spectral Verification Latency Overhead** | Low | Implement spectral index computation as CPU-only parallel NumPy calls so it does not consume GPU execution time. |
| **Router Misclassification** | Medium | Maintain deterministic rule overrides for key phrasing alongside vector embeddings. |
| **Refusal Gate Over-Conservatism** | Medium | Provide an admin UI toggle to adjust contract sensitivity thresholds during testing. |
| **GPU Memory Overhead (OOM)** | High | Use 4-bit `bitsandbytes` quantization to maintain an **≤8GB VRAM** footprint; single VQA adapter eliminates multi-adapter memory contention. |
| **Live Pitch System Failure** | High | Enforce 3-layer fallback hierarchy ending in recorded high-definition demo clips. |

---

## 11. Strategic Summary: Key Execution Priorities

1. **VQA Fine-Tuning**: Begin **Qwen2-VL-2B-Instruct** QLoRA fine-tuning on BigEarthNet-derived VQA data immediately (Days 1–3).
2. **Contract & Refusal Testing**: Validate the Evidence Contract and Otsu NDWI/NDVI spectral cross-check scripts on 5–10 real image pairs.
3. **Demo Practice**: Script, verify, and record all 4 demo beats early to ensure zero-downtime presentation during evaluation.
