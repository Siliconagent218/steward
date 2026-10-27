"""Category inference from measured geometry and colour, against the robot's shape priors.

A measured box can be smaller than the object (only part of it was in view) but rarely larger, so
an observed extent below the prototype costs little and one above it costs a lot. Whether the
object stands on the floor or on something raised separates furniture from tableware. The result
is a probability per category; the estimator accumulates these over observations.
"""

from __future__ import annotations

import math

import numpy as np

# Robot knowledge: typical full extents (long side, short side, height) per category, a scale range
# for categories that come in several sizes, where they stand, and an optional colour test.
SHAPES: dict[str, dict] = {
    "wall": {"dims": (4.0, 0.10, 2.6), "scale": (0.3, 1.6), "floor": True, "min_top": 1.2},
    "counter": {"dims": (1.85, 0.62, 0.90), "floor": True},
    "table": {"dims": (1.20, 0.80, 0.75), "floor": True},
    "sideboard": {"dims": (1.42, 0.49, 0.85), "floor": True},
    "cabinet": {"dims": (0.62, 0.52, 0.80), "floor": True},
    "chair": {"dims": (0.47, 0.46, 0.90), "floor": True},
    "drawer": {"dims": (0.56, 0.44, 0.22), "floor": False, "min_bottom": 0.35},
    "bowl": {"dims": (0.17, 0.17, 0.06), "scale": (0.65, 1.85), "floor": False},
    "cup": {"dims": (0.08, 0.08, 0.10), "floor": False},
    "plate": {"dims": (0.24, 0.24, 0.02), "floor": False},
    "wine_glass": {"dims": (0.08, 0.08, 0.19), "floor": False, "color": "clear"},
    "mug": {"dims": (0.11, 0.085, 0.095), "floor": False, "color": "dark"},
    "teddy_bear": {"dims": (0.11, 0.105, 0.18), "floor": False, "color": "brown"},
    "scissors": {"dims": (0.16, 0.07, 0.012), "floor": False},
}

UNDER = (0.55, 1.2, 0.55)  # tolerance (log ratio) per axis for measuring less than the prototype; the
# short side is usually the depth seen from the front, so it may come out much thinner
OVER = 0.25  # tolerance for measuring more (1 cm voxels, depth noise, overhangs)
UNKNOWN = "unknown"
UNKNOWN_COST = 7.0  # a region that fits no prototype this badly is something the robot has no prior for;
# one broken placement rule (6) on an otherwise fitting shape still names the category


def _color_score(rule: str | None, color: np.ndarray, invalid_frac: float) -> float:
    r, g, b = (float(x) for x in color)
    value, low = max(r, g, b), min(r, g, b)
    saturation = 0.0 if value == 0 else (value - low) / value
    if rule == "clear":
        return 1.0 if invalid_frac > 0.15 or (saturation < 0.12 and value > 0.6) else -2.0
    if rule == "dark":
        return 0.0 if value < 0.3 else -3.0
    if rule == "brown":
        return 0.0 if (r > g > b and 0.25 < value < 0.8 and saturation > 0.3) else -4.0
    return 0.0


def classify(size: np.ndarray, bottom: float, color: np.ndarray, invalid_frac: float = 0.0,
             temperature: float = 1.0, bottom_seen: bool = True) -> dict[str, float]:
    """Probability per category for a measured box (full extents, long side first) whose lowest
    point is ``bottom`` metres above the floor; :data:`UNKNOWN` takes the mass when no prototype fits.

    With ``bottom_seen`` false the lowest point is an image edge or an occluder, not the base: it is
    no evidence of standing on the floor or not, and a floor-standing prototype is compared with
    the visible top instead of the visible height (a table seen from close shows only its top)."""
    seen = np.array([max(size[0], size[1]), min(size[0], size[1]), size[2]], float)
    seen = np.maximum(seen, 0.004)
    scores = {}
    on_floor = bottom < 0.06
    top = bottom + seen[2]
    for name, shape in SHAPES.items():
        proto = np.array(shape["dims"], float)
        obs = seen if bottom_seen or not shape["floor"] else np.array([seen[0], seen[1], max(seen[2], top)])
        lo, hi = shape.get("scale", (1.0, 1.0))
        # One scale for the whole object, set by the axis that fills the prototype most: a partial
        # view only shrinks the other axes, which costs little, while a flat wide region cannot pass
        # for a small deep object scaled per axis.
        scale = min(max(float(np.max(obs / proto)), lo), hi)
        cost = 0.0
        for k in range(3):
            r = math.log(obs[k] / (proto[k] * scale))
            cost += (r / UNDER[k]) ** 2 if r < 0 else (r / OVER) ** 2
        if bottom_seen and shape["floor"] != on_floor:
            cost += 6.0
        if bottom_seen and bottom < shape.get("min_bottom", -1.0):
            cost += 6.0
        if bottom + obs[2] < shape.get("min_top", -1.0):
            cost += 6.0
        cost -= _color_score(shape.get("color"), color, invalid_frac)
        scores[name] = -cost / temperature
    scores[UNKNOWN] = -UNKNOWN_COST / temperature
    top = max(scores.values())
    weights = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items() if v / total > 0.01}


def prototype(category: str) -> tuple[float, float, float] | None:
    shape = SHAPES.get(category)
    return shape["dims"] if shape and "scale" not in shape else None
