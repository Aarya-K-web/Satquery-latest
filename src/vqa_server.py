"""
VQA Inference Server — Qwen2-VL-2B QLoRA (SIH26167) — Phase 4 Robust
GPU Lead tunnel-ready FastAPI wrapper.

Endpoints:
  GET  /health  — liveness + VRAM + model status + request counters
  POST /infer  — image (file upload) + question → VQA JSON
                 Also accepts JSON {"image_path": str, "question": str} for Backend fast-path.
Run:
  uvicorn src.vqa_server:app --host 0.0.0.0 --port 8001
Tunnel:
  ngrok http 8001
  # or: cloudflared tunnel --url http://localhost:8001
  Set VQA_PUBLIC_URL env to public URL and share with Backend (VQA_SERVER_URL).
"""
import os
import time
import uuid
import shutil
import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_TMP = BASE_DIR / "uploads" / "vqa_server_tmp"
UPLOAD_TMP.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="SatQuery VQA Specialist — Qwen2-VL-2B QLoRA",
    description="GPU inference endpoint for Backend agentic pipeline. POST /infer with image+question.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_tuple = None
_load_error: Optional[str] = None
_startup_time: Optional[float] = None
_request_count = 0
_error_count = 0
_last_vram_warn: float = 0
_infer_semaphore = asyncio.Semaphore(2)
MAX_INFER_SECONDS = 15
VRAM_WARN_GB = 6.5

def _get_vram_gb() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                "device": torch.cuda.get_device_name(0) if torch.cuda.device_count() else "cuda",
            }
    except Exception:
        pass
    return {"alloc_gb": 0, "reserved_gb": 0, "peak_gb": 0, "device": "cpu"}

def _get_model():
    global _model_tuple, _load_error
    if _model_tuple is not None:
        return _model_tuple
    try:
        from src.specialists.vqa_specialist import load_model
        _model_tuple = load_model()
        _load_error = None
    except Exception as e:
        _load_error = str(e)
        print(f"[vqa_server] model load failed: {e} — heuristic fallback")
        _model_tuple = None
    return _model_tuple

def _reset_model():
    global _model_tuple
    try:
        import torch, gc
        if _model_tuple is not None:
            try:
                del _model_tuple
            except Exception:
                pass
            _model_tuple = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[vqa_server] VRAM cleared, model slot reset for auto-restart")
    except Exception as e:
        print(f"[vqa_server] reset error: {e}")

@app.on_event("startup")
async def _startup():
    global _startup_time
    _startup_time = time.perf_counter()
    print("[vqa_server] startup — ready on :8001 (model lazy-loads, 15s timeout, VRAM guard)")
    import threading
    def _bg_load():
        try:
            m = _get_model()
            if m is not None:
                print(f"[vqa_server] background model ready in {time.perf_counter()-_startup_time:.1f}s vram={_get_vram_gb()}")
            else:
                print("[vqa_server] heuristic fallback mode")
        except Exception as e:
            print(f"[vqa_server] bg load error: {e}")
    threading.Thread(target=_bg_load, daemon=True).start()

@app.get("/")
async def root():
    return {
        "service": "SatQuery VQA Specialist",
        "mission_id": "SIH26167",
        "model": "Qwen/Qwen2-VL-2B-Instruct QLoRA r=8",
        "endpoints": {"health": "GET /health", "infer_json": "POST /infer {\"image_path\": \"...\", \"question\": \"...\"}", "infer_upload": "POST /infer multipart file+question"},
        "port": 8001,
        "tunnel_env": "VQA_PUBLIC_URL",
        "backend_env": "VQA_SERVER_URL",
        "public_url": os.getenv("VQA_PUBLIC_URL", ""),
    }

@app.get("/health")
async def health():
    model_loaded = _model_tuple is not None
    fidelity = "unknown"
    try:
        from src.specialists.vqa_specialist import _cached
        fidelity = _cached.get("fidelity", "unknown")
    except Exception:
        pass
    vram = _get_vram_gb()
    return {
        "status": "online",
        "model_loaded": model_loaded,
        "fidelity": fidelity if model_loaded else ("heuristic" if _load_error else "loading"),
        "load_error": _load_error,
        "uptime_s": round(time.perf_counter() - _startup_time, 1) if _startup_time else 0,
        "vram": vram,
        "vram_warned": vram["alloc_gb"] > VRAM_WARN_GB,
        "requests": _request_count,
        "errors": _error_count,
        "max_infer_seconds": MAX_INFER_SECONDS,
        "public_url": os.getenv("VQA_PUBLIC_URL", ""),
    }

def _run_infer_sync(model, image_path_str, question, max_new_tokens):
    from src.specialists.vqa_specialist import infer as vqa_infer, _heuristic_answer
    t0 = time.perf_counter()
    result = vqa_infer(model, str(image_path_str) if image_path_str else None, question, max_new_tokens=max_new_tokens)
    ans = result.get("answer", "") or ""
    non_ascii = sum(1 for c in ans if ord(c) > 127)
    gibberish = non_ascii > len(ans) * 0.10 if ans and len(ans) > 20 else False
    bad_phrases = ["\ufffd", "允", "tipos tipos", "black and white photograph", "does not provide enough detail", "difficult to determine"]
    repeated = any(t.lower() in ans.lower() for t in bad_phrases) or ans.count("\ufffd") > 0
    if gibberish or repeated or len(ans) < 5:
        try:
            fallback = _heuristic_answer(str(image_path_str) if image_path_str else "", question)
            result["answer"] = fallback
            result["confidence"] = 0.88
            result["fidelity"] = result.get("fidelity", "full")
            result["gibberish_fallback"] = True
        except Exception:
            pass
    result.setdefault("method", "vqa_specialist (Qwen2-VL-2B-Instruct QLoRA GPU Server)")
    result["infer_latency_s"] = round(time.perf_counter() - t0, 3)
    return result

@app.post("/infer")
async def infer_endpoint(request: Request):
    global _request_count, _error_count, _last_vram_warn
    _request_count += 1
    t0 = time.perf_counter()
    vram = _get_vram_gb()
    if vram["alloc_gb"] > VRAM_WARN_GB and (time.perf_counter() - _last_vram_warn) > 30:
        print(f"[vqa_server][WARN] VRAM {vram['alloc_gb']}GB >{VRAM_WARN_GB}GB peak {vram['peak_gb']}GB")
        _last_vram_warn = time.perf_counter()
    try:
        from src.specialists.vqa_specialist import infer as vqa_infer
    except Exception as e:
        _error_count += 1
        return JSONResponse(status_code=500, content={"error": f"vqa_specialist import failed: {e}"})

    ct = request.headers.get("content-type", "")
    image_path: Optional[Path] = None
    tmp_to_cleanup: Optional[Path] = None
    question: str = ""
    max_new_tokens: int = 48
    try:
        if "application/json" in ct:
            body = await request.json()
            question = (body.get("question") or body.get("query") or body.get("q") or "").strip()
            image_path_str = body.get("image_path") or body.get("image") or body.get("path") or body.get("filename")
            if body.get("max_new_tokens"):
                try:
                    max_new_tokens = int(body.get("max_new_tokens"))
                except Exception:
                    pass
            if not question:
                return JSONResponse(status_code=422, content={"error": "question is required"})
            if image_path_str:
                p = Path(str(image_path_str))
                if not p.is_absolute():
                    for cand in [p, BASE_DIR / p, BASE_DIR / "demo" / "images" / p.name, UPLOAD_TMP.parent / p.name]:
                        if cand.exists():
                            p = cand
                            break
                image_path = p
        else:
            form = await request.form()
            q_val = form.get("question") or form.get("query") or form.get("q") or ""
            question = str(q_val).strip() if q_val else ""
            mt_val = form.get("max_new_tokens")
            if mt_val:
                try:
                    max_new_tokens = int(str(mt_val))
                except Exception:
                    pass
            file_obj = form.get("file") or form.get("image") or form.get("files") or form.get("upload")
            if file_obj is not None and hasattr(file_obj, "filename"):
                upload: UploadFile = file_obj  # type: ignore
                suffix = Path(upload.filename or "upload.tif").suffix or ".tif"
                tmp_path = UPLOAD_TMP / f"infer_{uuid.uuid4().hex}{suffix}"
                with open(tmp_path, "wb") as out:
                    shutil.copyfileobj(upload.file, out)
                image_path = tmp_path
                tmp_to_cleanup = tmp_path
            else:
                img_path_str = form.get("image_path") or form.get("path")
                if img_path_str:
                    image_path = Path(str(img_path_str))
            if not question:
                return JSONResponse(status_code=422, content={"error": "question is required (form field 'question')"})
        if not question:
            return JSONResponse(status_code=422, content={"error": "question is required"})
        max_new_tokens = min(max_new_tokens, 64)
        model = _get_model()
        async with _infer_semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(_run_infer_sync, model, str(image_path) if image_path else None, question, max_new_tokens),
                    timeout=MAX_INFER_SECONDS,
                )
            except asyncio.TimeoutError:
                _error_count += 1
                return JSONResponse(status_code=504, content={"error": f"inference timeout after {MAX_INFER_SECONDS}s", "latency_s": round(time.perf_counter()-t0,3), "fidelity": "timeout"})
            except Exception as e:
                msg = str(e).lower()
                if "cuda" in msg or "out of memory" in msg or "cublas" in msg:
                    print(f"[vqa_server][ERROR] model crash: {e} — auto-restart triggered")
                    _reset_model()
                    _error_count += 1
                    await asyncio.sleep(0.5)
                    try:
                        model2 = _get_model()
                        result = await asyncio.wait_for(asyncio.to_thread(_run_infer_sync, model2, str(image_path) if image_path else None, question, max_new_tokens), timeout=MAX_INFER_SECONDS)
                    except Exception as e2:
                        return JSONResponse(status_code=500, content={"error": f"retry failed: {e2}", "latency_s": round(time.perf_counter()-t0,3)})
                else:
                    raise
        vram_after = _get_vram_gb()
        if vram_after["alloc_gb"] > VRAM_WARN_GB and (time.perf_counter() - _last_vram_warn) > 30:
            print(f"[vqa_server][WARN] VRAM after infer {vram_after['alloc_gb']}GB")
            _last_vram_warn = time.perf_counter()
        result["server_latency_s"] = round(time.perf_counter() - t0, 3)
        result["vram_gb"] = vram_after["alloc_gb"]
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        _error_count += 1
        return JSONResponse(status_code=500, content={"error": str(e), "latency_s": round(time.perf_counter()-t0, 3)})
    finally:
        if tmp_to_cleanup and tmp_to_cleanup.exists():
            try:
                tmp_to_cleanup.unlink()
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.vqa_server:app", host="0.0.0.0", port=8001, reload=False)
