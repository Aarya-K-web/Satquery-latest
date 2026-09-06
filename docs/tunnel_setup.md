# Tunnel Setup — GPU Lead Phase 4 (SIH26167)

Expose `src/vqa_server.py :8001` to the deployed Railway/Render backend.

## Option A — ngrok (easiest)
```bash
# 1. Install https://ngrok.com/download
pip install ngrok  # or brew/choco install ngrok
ngrok config add-authtoken <YOUR_TOKEN>

# 2. Start VQA server (terminal 1)
uvicorn src.vqa_server:app --host 0.0.0.0 --port 8001

# 3. Expose (terminal 2)
ngrok http 8001
# → Forwarding https://abc123.ngrok-free.app -> http://localhost:8001
```
Copy the `https://*.ngrok-free.app` URL.

## Option B — Cloudflare Tunnel (more stable, no auth wall)
```bash
# Install cloudflared https://developers.cloudflare.com/cloudflare-one/connections/connect/networks/downloads/
cloudflared tunnel --url http://localhost:8001
# → https://random-words-1234.trycloudflare.com
```

## Share URL with Backend
Set on deployed backend (Railway/Render env vars):
```
VQA_SERVER_URL=https://abc123.ngrok-free.app/infer
VQA_PUBLIC_URL=https://abc123.ngrok-free.app
```
Local fallback still works (`http://127.0.0.1:8001/infer` + multipart).

## Verify
```bash
curl https://abc123.ngrok-free.app/health
curl -X POST https://abc123.ngrok-free.app/infer \
  -F "file=@demo/images/sentinel2_urban_mumbai.tif" \
  -F "question=Is urban area present?"
# expect {"answer":"Yes, urban fabric is visible","fidelity":"full","latency_s":<8}
```

## Round-trip latency (measured)
- Direct VQA (`:8001/infer`) warm: 3–5s, throttled to 15s timeout
- Backend `/api/query` via tunnel (Railway → ngrok → laptop): ~6–9s (adds 1–2s HTTP + 0.5s queuing)
- Budget: <15s end-to-end (spec). Keep `max_new_tokens=48` and VRAM <6.5GB.

## Robustness (Phase 4 hardening)
- 15s per-request timeout (504)
- Semaphore 2 concurrent, queue rest
- VRAM guard logs WARN >6.5GB, auto `torch.cuda.empty_cache()`
- CUDA OOM auto-reset + retry heuristic fallback (never crashes)
- Health exposes `vram`, `requests`, `errors`, `public_url`

## Stretch — Serverless GPU fallback (Modal/RunPod)
If tunnel unstable:
```bash
modal deploy modal_vqa.py  # wraps src.specialists.vqa_specialist
# share https://<modal-id>.modal.run/infer as backup VQA_SERVER_URL
```
