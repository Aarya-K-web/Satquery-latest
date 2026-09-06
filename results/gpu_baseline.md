# GPU Baseline - Qwen2-VL-2B-Instruct 4-bit

**Date:** 2026-09-02
**GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, 8151 MiB)
**Driver:** 596.49 | CUDA 13.2 | torch 2.15.0+cu130
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Quantization:** bitsandbytes 4-bit (NF4, compute dtype float16, device_map=auto)
**Transformers:** 5.12.1 | accelerate 1.14.0 | bitsandbytes 0.50.2

## Load Metrics
- Processor load time: 9.6s
- Model load time (4-bit): 26.6s
- VRAM allocated (torch): 1.51 GB
- VRAM reserved (torch): 1.56 GB
- Peak VRAM (torch): 1.55 GB
- nvidia-smi used: 1708 MiB / 8151 MiB
- Expected range: 3.0-3.5 GB (spec estimate) - PASS (actual 1.51 GB allocated / 1.71 GB nvidia-smi, more efficient than estimate, well within 8GB budget)
- OOM: No - loads cleanly on 8GB card
- Fallback needed: No (load_in_8bit not required)

## Inference Test
- Input: 512x512 synthetic satellite patch (green vegetation + urban grey + water blue)
- Question: "What land cover types are visible?"
- Response: "The image shows three land cover types: a green square, a blue circle, and a gray rectangle."
- Inference time (cold, 80 tokens): 7.77s
- Inference time (warm, 30 tokens): 5.49s
- Status: Text response generated successfully

## Reproduce
```bash
python -c "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig; import torch; m=Qwen2VLForConditionalGeneration.from_pretrained('''Qwen/Qwen2-VL-2B-Instruct''', quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map='''auto''')"
```

## Notes
- Model cached at: C:/Users/VEDANT/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c (~4.2GB shards)
- No OOM observed; ~4.5GB headroom remaining on 8GB card for training adapters
- Next step: QLoRA fine-tuning feasible within 8GB budget
