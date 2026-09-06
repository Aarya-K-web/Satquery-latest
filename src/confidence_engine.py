"""
Confidence Engine Module — SatQuery EvidenceSwarm (SIH26167)
Implements the exact 4-Factor Confidence Scoring Formula:
Score_conf = 0.15·C_sensor + 0.35·C_adapter + 0.25·C_guard + 0.25·C_spectral
Assigns qualitative labels: "High Evidence Consistency", "Moderate Evidence Consistency", "Low Evidence Consistency".
"""

from typing import Dict, Any, Optional

DEFAULT_WEIGHTS = {
    "w1_sensor": 0.15,
    "w2_adapter": 0.35,
    "w3_guard": 0.25,
    "w4_spectral": 0.25
}


def get_confidence_label(score: float, is_refusal: bool = False) -> str:
    """
    Returns the ISRO-grade qualitative confidence label based on aggregate score.
    """
    if is_refusal or score <= 0.0:
        return "Refusal / Insufficient Evidence"
    if score >= 0.85:
        return "High Evidence Consistency"
    if score >= 0.65:
        return "Moderate Evidence Consistency"
    return "Low Evidence Consistency"


def calculate_confidence(
    c_sensor: float,
    c_adapter: float,
    c_guard: float,
    c_spectral: float,
    is_refusal: bool = False,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Computes the 4-factor confidence score envelope:
    Score = 0.15·C_sensor + 0.35·C_adapter + 0.25·C_guard + 0.25·C_spectral

    Returns:
        {
            "aggregate_score": float,
            "label": str,
            "breakdown": {
                "c_sensor": float,
                "c_adapter": float,
                "c_guard": float,
                "c_spectral": float
            },
            "weights": {
                "w1_sensor": 0.15,
                "w2_adapter": 0.35,
                "w3_guard": 0.25,
                "w4_spectral": 0.25
            }
        }
    """
    w = weights or DEFAULT_WEIGHTS
    w_sensor = w.get("w1_sensor", 0.15)
    w_adapter = w.get("w2_adapter", 0.35)
    w_guard = w.get("w3_guard", 0.25)
    w_spectral = w.get("w4_spectral", 0.25)

    # Clamp subscores to [0.0, 1.0]
    c_sensor_clamped = max(0.0, min(1.0, float(c_sensor)))
    c_adapter_clamped = max(0.0, min(1.0, float(c_adapter)))
    c_guard_clamped = max(0.0, min(1.0, float(c_guard)))
    c_spectral_clamped = max(0.0, min(1.0, float(c_spectral)))

    if is_refusal:
        score = 0.0
        label = "Refusal / Insufficient Evidence"
    else:
        raw_score = (
            (w_sensor * c_sensor_clamped) +
            (w_adapter * c_adapter_clamped) +
            (w_guard * c_guard_clamped) +
            (w_spectral * c_spectral_clamped)
        )
        score = round(float(raw_score), 2)
        label = get_confidence_label(score, is_refusal=False)

    return {
        "aggregate_score": score,
        "label": label,
        "breakdown": {
            "c_sensor": round(c_sensor_clamped, 2),
            "c_adapter": round(c_adapter_clamped, 2),
            "c_guard": round(c_guard_clamped, 2),
            "c_spectral": round(c_spectral_clamped, 2)
        },
        "weights": {
            "w1_sensor": w_sensor,
            "w2_adapter": w_adapter,
            "w3_guard": w_guard,
            "w4_spectral": w_spectral
        }
    }
