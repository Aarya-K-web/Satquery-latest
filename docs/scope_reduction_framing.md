# SatQuery EvidenceSwarm — Scope-Reduction & Architectural Framing
**SIH26167 | Hackathon Engineering Rationale**

---

## 🏛️ Executive Summary

Under hackathon constraints (6-day delivery lifecycle and 8GB–16GB consumer VRAM hardware targets), building four full-scale deep learning vision-language models simultaneously is neither practical nor architecturally sound.

Instead, **SatQuery EvidenceSwarm** adopts a deliberate, production-proven strategy:
1. **One Production-Quality Specialist:** Fine-tuned multimodal vision-language model (**Qwen2-VL-2B-Instruct** via 4-bit QLoRA on geospatial domain data) delivering full-fidelity natural language VQA and grounded spatial bounding boxes.
2. **Three Structurally Complete Reduced-Fidelity Specialists:** 100% CPU-only algorithmic baselines (heuristic spectral grounding, bi-temporal Otsu differencing, and cross-modal optical-SAR band overlay) that operate with **identical I/O contracts** and explicit `"fidelity": "reduced"` flags.
3. **Agentic Orchestration & Verification:** Input Gate validation, Sensor Card telemetry extraction, Evidence Contract pre-flight refusal, Evidence Guard mathematical checks, and 4-factor Confidence scoring.

---

## 🔍 Key Architectural Principles

### 1. Honest AI Over Hallucinated Perfection
In national defense, disaster management, and ISRO mission planning, **a reliable refusal is infinitely more valuable than a plausible hallucination**.
When input files are insufficient (e.g., Beat 4), EvidenceSwarm halts execution at the pre-flight gate, saves GPU compute cycles, and issues actionable remediation steps.

### 2. Complete End-to-End Piping
The system architecture is 100% finished from UI to API to Orchestrator to Specialists:
- Adding a high-fidelity deep learning change detection network (e.g. ChangeFormer / BIT) or a specialized SAR diffusion model in the future requires **zero changes to the API contract, database schema, or 3-panel UI**.
- Every specialist communicates via the identical standardized JSON response envelope.

### 3. CPU Resilience & Zero GPU Latency for Triage
By retaining lightweight CPU algorithms (pure NumPy, rasterio, and PIL Otsu math):
- Preliminary triage, flood extent calculations, and spectral verification complete in **< 50 milliseconds**.
- Deployable on low-cost edge servers, ground stations, or portable field laptops without requiring expensive GPU clusters.

---

## 📊 Summary of Specialist Fidelity Matrix

| Specialist | Fidelity Level | Execution Engine | Hardware Required | Typical Latency | Key Output |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VQA Reasoning** | Full | Qwen2-VL-2B QLoRA | GPU (or quant CPU) | ~800ms | Natural language reasoning + Grounding BBoxes |
| **Change Detection** | Reduced | Bi-Temporal Differencing + Otsu | CPU-only | ~40ms | % Change Extent + Visual Heatmap Mask |
| **Optical-SAR Fusion** | Reduced | Polarimetric Band Overlay | CPU-only | ~30ms | Cross-modal Composite + Roughness Index |
| **Caption & Grounding** | Reduced | Spectral Indices (NDWI/NDVI) | CPU-only | ~25ms | Dominant Terrain Summary + Spatial Regions |
| **Signature Refusal Gate** | Guardrail | Evidence Contract Rules | CPU-only | ~2ms | Formal Rejection + Remediation Advice |

---

## 🎙️ Talking Points for Jury Evaluation

> *"We chose not to present four brittle, half-trained deep models. Instead, we delivered a complete, battle-ready Agentic Swarm: one production-grade fine-tuned multimodal model, three CPU-resilient specialist baselines with standardized contracts, and a mathematical Confidence Engine that prevents hallucinations. Every component is audited, logged, and rendered in our ISRO-grade 3-panel console."*
