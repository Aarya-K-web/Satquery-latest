# SatQuery EvidenceSwarm — 4-Beat Live Pitch Speaker Script
**SIH26167 :: Mission Presentation Script**

---

## 🎯 Executive Setup (0:00 – 0:30)
* **Stage Action:** Open full-screen 3-panel UI on the projector.
* **Speaker:**
  > "Respected Jury, welcome to **SatQuery EvidenceSwarm**. Satellite data is complex, multi-modal, and often treated as simple RGB images by generic vision-language models—leading to hallucinations and missed operational insights.
  >
  > We present an **ISRO-grade Agentic Swarm** that ingests raw multispectral GeoTIFFs, enforces strict spatial Coordinate Reference Systems, injects sensor physics telemetry through Sensor Cards, routes queries to domain-specialist models, and mathematically bounds model outputs with a 4-factor Evidence Score."

---

## 🛰️ BEAT 1: Single-Image VQA Baseline (0:30 – 1:30)
* **Action:** Click **BEAT 1** button on the Left Panel.
* **Visual Highlights:**
  - Center Globe smoothly rotates and zooms to Mumbai coastal coordinates.
  - Right Panel instantly populates the **Sentinel-2 Sensor Card** with 10m resolution, 4 bands, EPSG:32643 CRS, and Low Uncertainty (0.17).
  - Left Panel renders grounded VQA reasoning with high-contrast bounding boxes outlining the water inlet and urban fabric.
* **Speaker:**
  > "In Beat 1, we ingest a 4-band Sentinel-2 tile over Mumbai. Notice how our **Input Gate** extracts real sensor metadata without hallucinating.
  >
  > The query is routed to our fine-tuned **Qwen2-VL VQA Specialist**. Unlike black-box LLMs, every claim is spatially grounded with bounding boxes verified against CPU spectral Otsu checks (NDWI/NDVI). Our Confidence Engine outputs an aggregate evidence score of **0.89 (High)**."

---

## 🌊 BEAT 2: Bi-Temporal Change Detection (1:30 – 2:30)
* **Action:** Click **BEAT 2** button.
* **Visual Highlights:**
  - Center Globe flies to Kerala flood river basin.
  - Left Panel displays the **Change Detection Card** with a prominent **14.82% Extent Delta** callout and side-by-side heatmaps.
  - Right Panel shows the updated execution trace with Otsu thresholding convergence.
* **Speaker:**
  > "Beat 2 tests bi-temporal reasoning during disaster events. We upload pre-flood and post-flood acquisitions.
  >
  > The Swarm executes our **Change Detection Specialist**, calculating Euclidean spectral differencing and Otsu boundary isolation on CPU in under 50 milliseconds. The system flags this as a **Reduced-Fidelity path** with full architectural transparency—delivering actionable inundation extent without GPU latency."

---

## ⚡ BEAT 3: Optical-SAR Multi-Sensor Fusion (2:30 – 3:30)
* **Action:** Click **BEAT 3** button.
* **Visual Highlights:**
  - Globe focuses on Mumbai harbor.
  - Left Panel displays the **Fused Composite** blending Optical NIR/Red with Sentinel-1 C-band synthetic aperture radar backscatter.
  - Right Panel displays dual-sensor telemetry (6 channels, VV/VH polarizations).
* **Speaker:**
  > "Beat 3 demonstrates cross-modal sensor fusion. Optical imagery often suffers from cloud cover, haze, or low contrast.
  >
  > By combining Sentinel-2 optical reflectance with Sentinel-1 SAR backscatter, our **Optical-SAR Fusion Specialist** isolates high-roughness metallic and structural returns (mean VV intensity: 142.5). This exposes port infrastructure and ships penetrating through optical haze."

---

## ⛔ BEAT 4: The Signature Refusal Gate (3:30 – 4:30)
* **Action:** Click **BEAT 4** button ("Show me changes between the two dates" with only 1 file).
* **Visual Highlights:**
  - Left Panel flashes the bold red **EVIDENCE CONTRACT REFUSAL GATE** alert card.
  - Right Panel shows aggregate score dropping to **0.0 (Refusal)** and Evidence Contract marked as Refused.
* **Speaker:**
  > "Finally, Beat 4 is our signature feature: **The Refusal Gate**.
  >
  > When an analyst asks for change detection but only provides a single satellite image, generic AI will hallucinate a fake comparison. **EvidenceSwarm refuses.**
  >
  > Our pre-flight Evidence Contract detects missing temporal evidence, halts execution at zero GPU cost, logs the audit trace, and provides actionable remediation guidance: *'Upload both pre-event and post-event satellite scenes to enable comparative differencing.'*
  >
  > **This is how AI earns military and ISRO operational trust.**"

---

## 🏆 Conclusion & Q&A (4:30 – 5:00)
* **Speaker:**
  > "SatQuery EvidenceSwarm combines agentic routing, pure-CPU fallback specialists, ISRO sensor metadata injection, and mathematical guardrails into a robust, deployable geospatial reasoning console. Thank you, and we look forward to your questions."
