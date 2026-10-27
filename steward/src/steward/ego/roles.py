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
TABLEWARE = ("bowl", "cup", "mug", "plate", "wine_glass")
UNKNOWN = "unknown"


@dataclass
class RoleConfig:
    """Robot knowledge used to resolve roles; deployment-specific (it describes one home)."""

    singletons: tuple[str, ...] = ("table", "sideboard", "cabinet", "counter")
    """Categories of which the home has exactly one: two objects never share such a role."""
    confusable: tuple[tuple[str, ...], ...] = (("table", "sideboard", "cabinet", "counter"), ("mug", "cup"), ("bowl", "plate"))
    """Groups of categories that partial views confuse; roles move only within a group."""
    min_probability: float = 0.08
    """A category less likely than this is not considered as the object's role."""
    drawer_evidence: float = 2.5
    """Factor on the drawer-bearing categories of furniture that holds a perceived drawer."""
    min_existence: float = 0.5
    """Objects less likely to exist than this take no part in the assignment."""
    nested_evidence: float = 3.0
    """Factor on ``bowl`` for a flat object resting on a bowl: a nested bowl seen only by its rim."""
    stored_in_drawer: float = 0.2
    """Factor on the tableware categories of an object resting in a drawer: in this home tableware
    is not kept in drawers, so a flat toy there is not a bowl a task would carry to the table."""
    stickiness: float = 1.5
    """Factor on the role an object had in the previous assignment, so that roles do not swap
    between two similar objects from one view to the next (a robot sent back and forth)."""


def assign_roles(objects: list[dict], config: RoleConfig, previous: dict[str, str] | None = None) -> dict[str, tuple[str, float]]:
    """Role and its probability within its confusion group for every object of an Ego API summary.

    Within each group, (object, category) pairs are taken from the most to the least probable; a
    pair is accepted when the object has no role yet and the category is not a singleton already
    taken. An object outside every group takes its likeliest category once the evidence is applied
    (drawers in furniture, nested bowls, tableware not stored in drawers); one whose possible group
    categories are all one-of-a-kind and taken gets its likeliest category outside the group, or
    ``unknown``, so that a task never picks it as the one sideboard or table of the home.

    :param objects: summary objects with ``id``, ``category``, ``categories``, ``existence``,
        ``position``, ``yawDeg`` and ``extentM`` or ``sizeM``.
    :param config: the robot's knowledge of its home.
    :param previous: the roles of the previous assignment, which are kept unless the evidence
        against them is clearly stronger.
    :returns: ``{object id: (role, probability)}``.
    """
    previous = previous or {}
    probs = {o["id"]: dict(o["categories"]) for o in objects}
    bowls = {o["id"] for o in objects if max(o["categories"], key=o["categories"].get) == "bowl"}
    for o in objects:
        p = probs[o["id"]]
        if o.get("on") in bowls and "plate" in p:
            p["bowl"] = p.get("bowl", 0.0) + p["plate"] * (config.nested_evidence - 1.0)
            total = sum(p.values())
            probs[o["id"]] = {k: v / total for k, v in p.items()}
    drawers = [o for o in objects if o["category"] == "drawer" and o["existence"] >= config.min_existence]
    for o in objects:
        if o["category"] != "drawer" and any(_holds(o, d) for d in drawers):
            p = probs[o["id"]]
            for c in DRAWER_FURNITURE:
                if c in p:
                    p[c] *= config.drawer_evidence
            total = sum(p.values()) or 1.0
            probs[o["id"]] = {k: v / total for k, v in p.items()}
    drawer_ids = {o["id"] for o in objects if o["category"] == "drawer"}
    for o in objects:
        p = probs[o["id"]]
        if o.get("on") in drawer_ids and any(c in p for c in TABLEWARE):
            for c in TABLEWARE:
                if c in p:
                    p[c] *= config.stored_in_drawer
            total = sum(p.values()) or 1.0
            probs[o["id"]] = {k: v / total for k, v in p.items()}
    roles = {}
    for o in objects:
        best = max(probs[o["id"]], key=probs[o["id"]].get) if probs[o["id"]] else o["category"]
        roles[o["id"]] = (best, float(probs[o["id"]].get(best, 1.0)))
    for group in config.confusable:
        members = [o["id"] for o in objects if o["existence"] >= config.min_existence
                   and max(probs[o["id"]], key=probs[o["id"]].get) in group]
        held = lambda oid, c: config.stickiness if previous.get(oid) == c else 1.0  # noqa: E731
        pairs = sorted(((probs[oid].get(c, 0.0) * held(oid, c), oid, c) for oid in members for c in group
                        if probs[oid].get(c, 0.0) >= config.min_probability), reverse=True)
        taken: set[str] = set()
        chosen: dict[str, str] = {}
        for _, oid, c in pairs:
            if oid in chosen or (c in config.singletons and c in taken):
                continue
            chosen[oid] = c
            taken.add(c)
        for oid in members:
            if oid in chosen:
                in_group = sum(probs[oid].get(k, 0.0) for k in group) or 1.0
                roles[oid] = (chosen[oid], probs[oid][chosen[oid]] / in_group)
                continue
            free = [k for k in group if k in probs[oid] and k not in config.singletons]
            if free:
                c = max(free, key=probs[oid].get)
                roles[oid] = (c, probs[oid][c])
                continue
            # Every one-of-a-kind category it could be is taken by a likelier object: by the robot's
            # knowledge of its home it is none of them.
            outside = {k: v for k, v in probs[oid].items() if k not in group and v >= config.min_probability}
            roles[oid] = max(outside.items(), key=lambda kv: kv[1]) if outside else (UNKNOWN, 0.0)
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
