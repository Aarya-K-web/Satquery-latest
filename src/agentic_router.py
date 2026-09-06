"""
Agentic Router Module — SatQuery EvidenceSwarm (SIH26167)
Routes user queries and input satellite files to the appropriate specialist:
- VQA Specialist (Qwen2-VL LoRA GPU endpoint with fallback)
- Bi-Temporal Change Detection (CPU Differencing + Otsu)
- Optical-SAR Fusion (Multi-sensor Cross-Modal Heuristic)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional


def route_query(
    query: str,
    file_paths: List[Path],
    beat: Optional[int] = None,
    sensor_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Determines the target specialist, task type, fidelity, and execution strategy
    based on explicit demo beat overrides, natural language keywords, and file modalities.

    Returns:
        {
            "task_type": str,
            "specialist": str,
            "method": str,
            "fidelity": str,
            "confidence": float,
            "reasoning": str
        }
    """
    q_lower = (query or "").strip().lower()
    num_files = len(file_paths)
    sensor_types = sensor_types or []
    file_names = [p.name.lower() for p in file_paths]

    # 1. Explicit Beat Overrides
    if beat == 1:
        return {
            "task_type": "vqa",
            "specialist": "vqa_specialist",
            "method": "vqa_specialist (Qwen2-VL-2B-Instruct QLoRA)",
            "fidelity": "full",
            "confidence": 0.98,
            "reasoning": "Beat 1 override: Dispatched to VQA reasoning specialist (Single-tile Sentinel-2 VQA baseline)."
        }

    if beat == 2:
        return {
            "task_type": "change_detection",
            "specialist": "change_detection",
            "method": "image-differencing-otsu",
            "fidelity": "reduced",
            "confidence": 0.95,
            "reasoning": "Beat 2 override: Dispatched to Bi-temporal Change Detection specialist (CPU differencing)."
        }

    if beat == 3:
        return {
            "task_type": "optical_sar_fusion",
            "specialist": "optical_sar_fusion",
            "method": "band-overlay-heuristic",
            "fidelity": "reduced",
            "confidence": 0.96,
            "reasoning": "Beat 3 override: Dispatched to Optical-SAR Cross-Modal Fusion specialist."
        }

    if beat == 4:
        return {
            "task_type": "change_detection",
            "specialist": "change_detection",
            "method": "evidence_contract_preflight",
            "fidelity": "refusal_gate",
            "confidence": 0.99,
            "reasoning": "Beat 4 override: Signature refusal gate for change detection."
        }

    # 2. Modality & Sensor Evidence Check
    has_sar_file = any("sar" in fn or "sentinel1" in fn or "s1" in fn for fn in file_names)
    has_sar_sensor = any("sar" in s.lower() or "sentinel-1" in s.lower() for s in sensor_types)
    is_multi_modal = (has_sar_file or has_sar_sensor) and num_files >= 2

    # 3. Keyword Pattern Matching
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
            "reasoning": f"Optical-SAR fusion routed based on {'multi-modal file presence' if is_multi_modal else 'SAR query keywords'}."
        }

    if has_change_kw and num_files >= 2:
        return {
            "task_type": "change_detection",
            "specialist": "change_detection",
            "method": "image-differencing-otsu",
            "fidelity": "reduced",
            "confidence": 0.92,
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
            "reasoning": "Change detection intent identified with insufficient temporal inputs."
        }

    # 4. Default / Fallback Routing: VQA Specialist
    return {
        "task_type": "vqa",
        "specialist": "vqa_specialist",
        "method": "vqa_specialist (Qwen2-VL-2B-Instruct QLoRA)",
        "fidelity": "full",
        "confidence": 0.88,
        "reasoning": f"Natural language query '{query}' routed to VQA specialist with grounding bounding boxes."
    }
