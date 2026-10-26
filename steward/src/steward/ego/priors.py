"""Category priors the robot brings to perception: affordances, how likely the object is fixed,
and fragility. They are robot knowledge, not World State facts."""

from __future__ import annotations

CATEGORY_PRIORS: dict[str, dict] = {
    "wall": {"role": "environment", "root": {"fixed": 1.0}, "affordances": [], "fragility": 0.0},
    "counter": {"role": "object", "root": {"fixed": 0.9, "free": 0.1}, "affordances": ["support-surface"], "fragility": 0.0},
    "table": {"role": "object", "root": {"free": 0.7, "fixed": 0.3}, "affordances": ["support-surface"], "fragility": 0.0},
    "cabinet": {"role": "object", "root": {"fixed": 0.9, "free": 0.1}, "affordances": ["support-surface", "container"],
                "fragility": 0.0},
    "cup": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.8},
    "bowl": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.6},
    "plate": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "support-surface"], "fragility": 0.6},
}

UNKNOWN_PRIOR = {"role": "object", "root": {"free": 0.5, "fixed": 0.5}, "affordances": [], "fragility": 0.5}

COMPATIBLE: dict[str, set[str]] = {"cup": {"cup", "bowl"}, "bowl": {"bowl", "cup"}}


def prior(category: str) -> dict:
    return CATEGORY_PRIORS.get(category, UNKNOWN_PRIOR)


def compatible(a: str, b: str) -> bool:
    return a == b or b in COMPATIBLE.get(a, set())
