# SatQuery EvidenceSwarm (SIH26167) — VQA GPU Inference Architecture & Setup

This document outlines the VQA inference deployment hierarchy for SatQuery EvidenceSwarm.

---

## 🚀 Primary Production Architecture — Modal Serverless GPU

The primary production deployment runs the Qwen2-VL-2B-Instruct QLoRA specialist on **Modal Serverless GPU** (NVIDIA T4 / A10G), providing automated scaling, zero-idle cost, and robust cloud-to-cloud low-latency execution with Railway/Render.

### 1. Deploy to Modal
```bash
# From workspace root
modal deploy src/specialists/modal_vqa.py
# → Output URL: https://<your-workspace>--satquery-vqa-infer.modal.run
```

### 2. Configure Production Backend (Railway / Render)
Set the environment variable on the deployed backend:
```env
VQA_SERVER_URL=https://<your-workspace>--satquery-vqa-infer.modal.run/infer
```

### 3. Verification
```bash
curl https://<your-workspace>--satquery-vqa-infer.modal.run/health
curl -X POST https://<your-workspace>--satquery-vqa-infer.modal.run/infer \
  -F "file=@demo/images/sentinel2_urban_mumbai.tif" \
  -F "question=Is urban area present?"
```
*Note: The backend orchestrator automatically skips JSON file-path attempts for remote Modal endpoints, directly streaming multipart payload with a 25-second cold-start tolerance.*

---

## 🛠️ Local Development & Offline Fallbacks

If testing offline or running a dedicated local GPU laptop alongside backend services, use tunnels or local bindings:

### Fallback Option A — Direct Local GPU Server
When running backend and GPU server on the same physical workstation:
```bash
uvicorn src.vqa_server:app --host 127.0.0.1 --port 8001
# Backend default: VQA_SERVER_URL=http://127.0.0.1:8001/infer (utilizes JSON fast-path)
```

### Fallback Option B — ngrok Tunnel (Development / Local Laptop)
```bash
# 1. Start VQA server locally
uvicorn src.vqa_server:app --host 0.0.0.0 --port 8001

# 2. Expose via ngrok
ngrok http 8001
# → Forwarding https://abc123.ngrok-free.app -> http://localhost:8001

# 3. Set environment variable on Railway/Render for testing
VQA_SERVER_URL=https://abc123.ngrok-free.app/infer
```

### Fallback Option C — Cloudflare Tunnel (Development / Local Laptop)
```bash
cloudflared tunnel --url http://localhost:8001
# → https://random-words-1234.trycloudflare.com
# Set VQA_SERVER_URL=https://random-words-1234.trycloudflare.com/infer
```

---

## ⚡ Fallback & Reliability Guardrails
1. **Modal / Tunnel Cold Starts**: 25-second multipart request timeout accommodates serverless container provisioning.
2. **Offline / Outage Tolerance**: If the remote GPU endpoint times out or is unreachable, the orchestrator automatically invokes the CPU baseline heuristic specialist (`vqa_specialist`). Never throws an unhandled 500 error.
3. **Telemetry**: Backend telemetry automatically detects Modal endpoints and displays `VQA: Modal GPU` with a dedicated badge in the ISRO UI.
