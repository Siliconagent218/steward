"""Scene-level reading of what each perceived object is (its role), for tasks that name objects.

One track's best category can be wrong for furniture seen in part (a sideboard taken for a counter,
a cabinet for a table) and for small look-alikes (a mug taken for a cup). A natural-language task
names objects ("the sideboard's drawer"), so such a confusion stalls every later step. Roles are
therefore resolved over the whole scene from each object's full category distribution, a drawer
held in a piece of furniture as evidence for the furniture kinds that have drawers, and the
robot's knowledge of its home: the categories of which the home has exactly one. Objects keep the
perceived ``category``; ``role`` is the label tasks use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DRAWER_FURNITURE = ("sideboard", "cabinet")


@dataclass
class RoleConfig:
    """Robot knowledge used to resolve roles; deployment-specific (it describes one home)."""

    singletons: tuple[str, ...] = ("table", "sideboard", "cabinet", "counter")
    """Categories of which the home has exactly one: two objects never share such a role."""
    confusable: tuple[tuple[str, ...], ...] = (("table", "sideboard", "cabinet", "counter"), ("mug", "cup"))
    """Groups of categories that partial views confuse; roles move only within a group."""
    min_probability: float = 0.08
    """A category less likely than this is not considered as the object's role."""
    drawer_evidence: float = 2.5
    """Factor on the drawer-bearing categories of furniture that holds a perceived drawer."""
    min_existence: float = 0.5
    """Objects less likely to exist than this take no part in the assignment."""


def assign_roles(objects: list[dict], config: RoleConfig) -> dict[str, tuple[str, float]]:
    """Role and its probability within its confusion group for every object of an Ego API summary.

    Within each group, (object, category) pairs are taken from the most to the least probable; a
    pair is accepted when the object has no role yet and the category is not a singleton already
    taken. An object outside every group, or whose group categories are all taken, keeps its
    perceived category.

    :param objects: summary objects with ``id``, ``category``, ``categories``, ``existence``,
        ``position``, ``yawDeg`` and ``extentM`` or ``sizeM``.
    :param config: the robot's knowledge of its home.
    :returns: ``{object id: (role, probability)}``.
    """
    probs = {o["id"]: dict(o["categories"]) for o in objects}
    drawers = [o for o in objects if o["category"] == "drawer" and o["existence"] >= config.min_existence]
    for o in objects:
        if o["category"] != "drawer" and any(_holds(o, d) for d in drawers):
            p = probs[o["id"]]
            for c in DRAWER_FURNITURE:
                if c in p:
                    p[c] *= config.drawer_evidence
            total = sum(p.values()) or 1.0
            probs[o["id"]] = {k: v / total for k, v in p.items()}
    roles = {o["id"]: (o["category"], float(o["categories"].get(o["category"], 1.0))) for o in objects}
    for group in config.confusable:
        members = [o["id"] for o in objects if o["existence"] >= config.min_existence
                   and max(probs[o["id"]], key=probs[o["id"]].get) in group]
        pairs = sorted(((probs[oid].get(c, 0.0), oid, c) for oid in members for c in group
                        if probs[oid].get(c, 0.0) >= config.min_probability), reverse=True)
        taken: set[str] = set()
        chosen: dict[str, str] = {}
        for _, oid, c in pairs:
            if oid in chosen or (c in config.singletons and c in taken):
                continue
            chosen[oid] = c
            taken.add(c)
        for oid in members:
            c = chosen.get(oid) or max((k for k in group if k in probs[oid]), key=probs[oid].get)
            in_group = sum(probs[oid].get(k, 0.0) for k in group) or 1.0
            roles[oid] = (c, probs[oid].get(c, 0.0) / in_group)
    return roles


def _holds(furniture: dict, drawer: dict) -> bool:
    """Whether a drawer lies within a piece of furniture's footprint (with a margin for an open
    drawer) and below its top."""
    extent = furniture.get("extentM") or furniture["sizeM"]
    yaw = math.radians(furniture["yawDeg"])
    dx = drawer["position"][0] - furniture["position"][0]
    dy = drawer["position"][1] - furniture["position"][1]
    lx, ly = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    top = furniture["position"][2] + extent[2] / 2
    return abs(lx) <= extent[0] / 2 + 0.2 and abs(ly) <= extent[1] / 2 + 0.2 and drawer["position"][2] < top
