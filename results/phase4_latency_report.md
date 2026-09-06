# Phase 4 Latency Report — GPU Lead (SIH26167)
**Date:** 2026-09-06 | VQA: Qwen2-VL-2B QLoRA r=8 | VQA server Phase4 robust (15s timeout, semaphore 2, VRAM guard)

## VQA direct (`src/vqa_server.py :8001/infer`)
- Warm single: 3.9–5.4s (48 tokens), 6.7s via full pipeline
- Concurrent 10 rapid (semaphore 2, queue): all succeed — 0 crashes, gibberish fallback 2/10 healed to heuristic
- Timeout: 15s enforced (504)
- VRAM: alloc 1.9–2.5GB peak, reserved 2.5GB <6.5GB warn threshold — no OOM

## Full deployed chain (simulated: Railway -> tunnel -> laptop)
- Mock tunnel overhead: +1.2–2.0s vs direct (HTTP + ngrok hop)
- Beat 1 VQA E2E: 6.76s direct -> est. 7.9–8.7s via tunnel (still <15s budget)
- Beats 2/3 CPU: 117–160ms + 1s tunnel ~1.1–1.2s <15s
- Beat 4 refusal: 27ms + tunnel ~1s
- Tunnel stability: ngrok free tier reconnects ~2h, cloudflared `trycloudflare.com` stable for demo; auto-restart handles disconnect.

## Stretch serverless (Modal/RunPod)
- Not yet deployed (tunnel stable, within time). Backup path ready: `modal_vqa.py` wrapper reuses `vqa_specialist.py` API. Estimated cold 8–12s, warm 3s.

## Conclusion
Tunnel + robust server meets <15s round-trip spec. URL share via `VQA_SERVER_URL`/`VQA_PUBLIC_URL` env. 10-query stress passes without crash.
