"""
Agentic Router Module — SatQuery EvidenceSwarm (SIH26167)
Routes user queries and input satellite files to the appropriate specialist:
- VQA Specialist (Qwen2-VL LoRA GPU endpoint with fallback)
- Bi-Temporal Change Detection (CPU Differencing + Otsu)
- Optical-SAR Fusion (Multi-sensor Cross-Modal Heuristic)

Implements Dual-Routing:
1. High-Priority Rule & Keyword Overrides (First priority, deterministic)
2. Semantic Embedding Similarity via sentence-transformers (all-MiniLM-L6-v2) with zero-dependency fallback
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Global cache for sentence-transformers model
_EMBED_MODEL = None
_EMBED_MODEL_LOADED = False

# Task prototype anchors for semantic similarity
TASK_PROTOTYPES = {
    "vqa": [
        "what land cover types and urban features are visible in this satellite scene",
        "describe the spatial objects, waterways, buildings, and vegetation in this image",
        "single-tile satellite visual question answering and scene reasoning"
    ],
    "change_detection": [
        "what areas changed between these two temporal dates and before after flood extent",
        "bi-temporal surface difference, inundation expansion, and displacement detection",
        "compare pre-event and post-event satellite scenes to isolate damage delta"
    ],
    "optical_sar_fusion": [
        "what features are visible in SAR radar but obscured by clouds in optical imagery",
        "multi-modal optical and synthetic aperture radar backscatter fusion",
        "all-weather radar cloud penetration and dielectric roughness index analysis"
    ]
}


def _get_embedding_model():
    """Lazily loads the sentence-transformers model if explicitly enabled and available."""
    global _EMBED_MODEL, _EMBED_MODEL_LOADED
    if _EMBED_MODEL_LOADED:
        return _EMBED_MODEL

    import os
    if os.getenv("ENABLE_HEAVY_EMBEDDINGS", "0") != "1":
        _EMBED_MODEL = None
        _EMBED_MODEL_LOADED = True
        return None

    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        _EMBED_MODEL_LOADED = True
    except Exception:
        _EMBED_MODEL = None
        _EMBED_MODEL_LOADED = True

    return _EMBED_MODEL


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D numpy arrays."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _semantic_route_similarity(query: str) -> Dict[str, Any]:
    """
    Computes semantic similarity against task prototypes using sentence-transformers.
    Falls back to token Jaccard similarity if sentence-transformers is unavailable.
    """
    model = _get_embedding_model()
    q_text = (query or "").strip()
    
    if model is not None and q_text:
        try:
            q_emb = model.encode(q_text)
            best_task = "vqa"
            best_score = -1.0
            
            for task, prototypes in TASK_PROTOTYPES.items():
                p_embs = model.encode(prototypes)
                for p_emb in p_embs:
                    sim = _cosine_similarity(q_emb, p_emb)
                    if sim > best_score:
                        best_score = sim
                        best_task = task
                        
            return {
                "task_type": best_task,
                "similarity": round(float(best_score), 4),
                "engine": "sentence-transformers/all-MiniLM-L6-v2"
            }
        except Exception:
            pass

    # Lightweight Token-Set Jaccard Semantic Fallback
    q_tokens = set(q_text.lower().split())
    best_task = "vqa"
    best_jaccard = 0.0
    
    for task, prototypes in TASK_PROTOTYPES.items():
        for proto in prototypes:
            p_tokens = set(proto.lower().split())
            intersection = len(q_tokens.intersection(p_tokens))
            union = len(q_tokens.union(p_tokens))
            jaccard = intersection / max(1, union)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_task = task

    return {
        "task_type": best_task,
        "similarity": round(float(best_jaccard), 4),
        "engine": "token_semantic_overlap"
    }


def route_query(
    query: str,
    file_paths: List[Path],
    beat: Optional[int] = None,
    sensor_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Determines the target specialist, task type, fidelity, and execution strategy
    based on explicit demo beat overrides, natural language keywords, file modalities,
    and semantic embedding similarity.

    Rules and keyword overrides ALWAYS take first priority.

    Returns:
        {
            "task_type": str,
            "specialist": str,
            "method": str,
            "fidelity": str,
            "confidence": float,
            "routing_method": "rule_override" | "embedding_similarity",
            "reasoning": str
        }
    """
    q_lower = (query or "").strip().lower()
    num_files = len(file_paths)
    sensor_types = sensor_types or []
    file_names = [p.name.lower() for p in file_paths]

    # 1. Explicit Beat Overrides (Rule Priority 1)
    if beat == 1:
        return {
            "task_type": "vqa",
            "specialist": "vqa_specialist",
            "method": "vqa_specialist (Qwen2-VL-2B-Instruct QLoRA)",
            "fidelity": "full",
            "confidence": 0.98,
            "routing_method": "rule_override",
            "reasoning": "Beat 1 override: Dispatched to VQA reasoning specialist (Single-tile Sentinel-2 VQA baseline)."
        }

    if beat == 2:
        return {
            "task_type": "change_detection",
            "specialist": "change_detection",
            "method": "image-differencing-otsu",
            "fidelity": "reduced",
            "confidence": 0.95,
            "routing_method": "rule_override",
            "reasoning": "Beat 2 override: Dispatched to Bi-temporal Change Detection specialist (CPU differencing)."
        }

    if beat == 3:
        return {
            "task_type": "optical_sar_fusion",
            "specialist": "optical_sar_fusion",
            "method": "band-overlay-heuristic",
            "fidelity": "reduced",
            "confidence": 0.96,
            "routing_method": "rule_override",
            "reasoning": "Beat 3 override: Dispatched to Optical-SAR Cross-Modal Fusion specialist."
        }

    if beat == 4:
        return {
            "task_type": "change_detection",
            "specialist": "change_detection",
            "method": "evidence_contract_preflight",
            "fidelity": "refusal_gate",
            "confidence": 0.99,
            "routing_method": "rule_override",
            "reasoning": "Beat 4 override: Signature refusal gate for change detection."
        }

    # 2. Modality & Sensor Evidence Check (Rule Priority 2)
    has_sar_file = any("sar" in fn or "sentinel1" in fn or "s1" in fn for fn in file_names)
    has_sar_sensor = any("sar" in s.lower() or "sentinel-1" in s.lower() for s in sensor_types)
    is_multi_modal = (has_sar_file or has_sar_sensor) and num_files >= 2

    # 3. Keyword Pattern Matching (Rule Priority 3)
    sar_keywords = ["sar", "radar", "backscatter", "penetrat", "obscured", "all-weather", "cloud penetration", "c-band"]
    change_keywords = ["change", "between these two", "between the two", "differencing", "changed", "before and after", "flood inundation", "temporal delta", "flood damage"]

    has_sar_kw = any(k in q_lower for k in sar_keywords)
    has_change_kw = any(k in q_lower for k in change_keywords)

    if (has_sar_kw or is_multi_modal) and (num_files >= 2 or has_sar_file):
        return {
            "task_type": "optical_sar_fusion",
            "specialist": "optical_sar_fusion",
            "method": "band-overlay-heuristic",
            "fidelity": "reduced",
            "confidence": 0.91,
            "routing_method": "rule_override",
            "reasoning": f"Optical-SAR fusion routed based on {'multi-modal file presence' if is_multi_modal else 'SAR query keywords'}."
        }

    if has_change_kw and num_files >= 2:
        return {
            "task_type": "change_detection",
            "specialist": "change_detection",
            "method": "image-differencing-otsu",
            "fidelity": "reduced",
            "confidence": 0.92,
            "routing_method": "rule_override",
            "reasoning": f"Bi-temporal change detection routed based on query intent '{query}' and {num_files} co-registered tiles."
        }

    # If change was requested with single file, route to change_detection so contract evaluates refusal
    if has_change_kw and num_files < 2:
        return {
            "task_type": "change_detection",
            "specialist": "change_detection",
            "method": "evidence_contract_preflight",
            "fidelity": "refusal_gate",
            "confidence": 0.95,
            "routing_method": "rule_override",
            "reasoning": "Change detection intent identified with insufficient temporal inputs."
        }

    # 4. Semantic Embedding Routing (When no direct keyword match triggers)
    semantic_res = _semantic_route_similarity(query)
    sem_task = semantic_res.get("task_type", "vqa")
    sem_sim = semantic_res.get("similarity", 0.0)

    # If semantic similarity strongly indicates change or SAR and file conditions match
    if sem_task == "change_detection" and num_files >= 2 and sem_sim > 0.40:
        return {
            "task_type": "change_detection",
            "specialist": "change_detection",
            "method": "image-differencing-otsu",
            "fidelity": "reduced",
            "confidence": round(float(0.85 + (sem_sim * 0.1)), 2),
            "routing_method": "embedding_similarity",
            "reasoning": f"Semantic routing ({semantic_res['engine']}, sim={sem_sim}) matched Change Detection."
        }

    if sem_task == "optical_sar_fusion" and (num_files >= 2 or has_sar_file) and sem_sim > 0.40:
        return {
            "task_type": "optical_sar_fusion",
            "specialist": "optical_sar_fusion",
            "method": "band-overlay-heuristic",
            "fidelity": "reduced",
            "confidence": round(float(0.85 + (sem_sim * 0.1)), 2),
            "routing_method": "embedding_similarity",
            "reasoning": f"Semantic routing ({semantic_res['engine']}, sim={sem_sim}) matched Optical-SAR Fusion."
        }

    # 5. Default / Fallback Routing: VQA Specialist
    routing_method = "embedding_similarity" if sem_sim > 0.35 else "rule_override"
    return {
        "task_type": "vqa",
        "specialist": "vqa_specialist",
        "method": "vqa_specialist (Qwen2-VL-2B-Instruct QLoRA)",
        "fidelity": "full",
        "confidence": 0.88,
        "routing_method": routing_method,
        "reasoning": f"Natural language query '{query}' routed to VQA specialist with grounding bounding boxes ({semantic_res['engine']})."
    }
