"""Homeostatic wellbeing signal: drives within comfort zones.

Consolidates the drive-regulation logic previously duplicated in
``scripts/run_micro_fouloide.py`` and ``scripts/demo_fouloides_front.py``,
and extends it with per-drive comfort zones for the homeostatic reward.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


def drive_regulation(drives: Mapping[str, Any]) -> float:
    """Mean of clamped drive levels (legacy signal, no comfort zones)."""
    values = [
        float(drives.get("energy", 0.0)),
        float(drives.get("hydration", 0.0)),
        1.0 - abs(float(drives.get("temperature", 0.5)) - 0.5) * 2.0,
        float(drives.get("health", 0.0)),
    ]
    return float(np.mean([max(0.0, min(1.0, v)) for v in values]))


def _zone_score(value: float, low: float, high: float) -> float:
    """1.0 inside [low, high], linearly decreasing to 0.0 at the [0, 1] bounds."""
    value = max(0.0, min(1.0, float(value)))
    if value < low:
        return 1.0 - (low - value) / low if low > 0.0 else 1.0
    if value > high:
        return 1.0 - (value - high) / (1.0 - high) if high < 1.0 else 1.0
    return 1.0


def wellbeing(drives: Mapping[str, Any], comfort: Optional[Mapping[str, Any]] = None) -> float:
    """Wellbeing in [0, 1]: 1.0 when every drive sits in its comfort zone.

    ``comfort`` maps drive names to zones: ``{low, high}`` for level drives
    (energy, hydration, health) or ``{target, tolerance}`` for temperature.
    Falls back to :func:`drive_regulation` when no comfort config is given.
    """
    if not comfort:
        return drive_regulation(drives)

    scores = []
    for name in ("energy", "hydration", "health"):
        zone = comfort.get(name, {})
        scores.append(_zone_score(
            float(drives.get(name, 0.0)),
            low=float(zone.get("low", 0.0)),
            high=float(zone.get("high", 1.0)),
        ))
    temp_zone = comfort.get("temperature", {})
    target = float(temp_zone.get("target", 0.5))
    tolerance = float(temp_zone.get("tolerance", 0.5))
    scores.append(_zone_score(
        float(drives.get("temperature", 0.5)),
        low=max(0.0, target - tolerance),
        high=min(1.0, target + tolerance),
    ))
    return float(np.mean(scores))
