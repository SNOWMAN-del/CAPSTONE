from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IrrigationPolicy:
    """Field-configurable advisory policy; values require agronomist validation."""

    effective_rainfall_fraction: float = 0.8
    application_efficiency: float = 0.85
    maximum_daily_irrigation_mm: float = 12.0
    stress_ks_threshold: float = 0.8


def irrigation_advisory(eta_mm: float, ks: float, rainfall_mm: float = 0.0, policy: IrrigationPolicy = IrrigationPolicy()) -> dict:
    if eta_mm < 0 or rainfall_mm < 0:
        raise ValueError("ETa and rainfall must be non-negative.")
    net_crop_demand = max(0.0, eta_mm - rainfall_mm * policy.effective_rainfall_fraction)
    recommended_mm = min(policy.maximum_daily_irrigation_mm, net_crop_demand / policy.application_efficiency)
    if ks < policy.stress_ks_threshold:
        priority = "high"
    elif recommended_mm > 0:
        priority = "normal"
    else:
        priority = "none"
    return {
        "recommendation": "irrigate" if recommended_mm > 0 else "no_irrigation_required",
        "priority": priority,
        "recommended_irrigation_mm": round(recommended_mm, 3),
        "net_crop_demand_mm": round(net_crop_demand, 3),
        "predicted_ks": round(ks, 3),
        "policy": asdict(policy),
        "safety_note": "Advisory only. Validate policy thresholds, ETa units, rainfall measurement, and field capacity with an agronomist before operational use.",
    }
