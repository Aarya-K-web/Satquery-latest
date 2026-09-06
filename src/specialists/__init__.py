"""
SatQuery EvidenceSwarm — Reduced-Fidelity CPU Specialists (Phase 2)
100% CPU-only specialists for Captioning/Grounding, Change Detection, and Optical-SAR Fusion.
"""

from src.specialists.caption_grounding import run as run_caption_grounding
from src.specialists.change_detection import run as run_change_detection
from src.specialists.optical_sar_fusion import run as run_optical_sar_fusion

__all__ = [
    "run_caption_grounding",
    "run_change_detection",
    "run_optical_sar_fusion",
]
