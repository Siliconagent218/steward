"""Physical validation of asset templates in an isolated MuJoCo instance.

Checks: the template is a valid World State entity; it compiles; moving bodies have mass;
movable assets dropped 2 cm above a floor come to rest without tipping, exploding or sinking;
prismatic and revolute joints declare limits. The report is JSON-serialisable and is stored
with the published asset.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from steward.assets.generators import TEMPLATE_ID
from steward.math3d import quat_angle_between
from steward.scene.extent import entity_obb
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, RigidTree, SceneState, exact, pose, primitive
from steward.scene.validate import validate_world

_FLOOR_ID = "w:validation-floor"


def _floor() -> Entity:
    from steward.scene.schema import Link

    link = Link(id="floor", name="floor", geometry={"collision": [primitive("box", [20, 20, 0.1], pose([0, 0, -0.05]))]})
    return Entity(id=_FLOOR_ID, role="environment", category=exact("floor"), pose=exact(pose()),
                  body=RigidTree(root=exact("fixed"), links=[link]))


def validate_template(template: dict, settle_seconds: float = 2.0) -> dict:
    report: dict = {"ok": False, "errors": [], "warnings": [], "metrics": {}}
    if template.get("id") != TEMPLATE_ID:
        report["errors"].append(f"template id must be {TEMPLATE_ID!r}")
        return report
    schema_errors = validate_world({"mode": "world", "entities": [template]})
    if schema_errors:
        report["errors"] += schema_errors
        return report
    ent = Entity.from_dict(template)
    if not isinstance(ent.body, RigidTree):
        report["errors"].append("Phase 0 validates rigid assets only")
        return report
    tree = ent.body
    for joint in tree.joints:
        if joint.type.value in ("prismatic", "revolute") and joint.limits is None:
            report["warnings"].append(f"joint {joint.id} has no limits")
    movable = tree.root.value in ("free", "planar")
    ent.id = "w:asset"
    if movable and tree.root.value == "free":
        ent.pose = exact(pose([0, 0, 0.02]))
    scene = SceneState(mode="world", entities=[_floor(), ent] if ent.role != "environment" else [ent])
    try:
        build = build_mjcf(scene, timestep=0.002)
        model = build.compile()
    except Exception as exc:  # noqa: BLE001 - the compiler error is the report content
        report["errors"].append(f"MuJoCo compile failed: {exc}")
        return report
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    names = build.names["w:asset"]
    total_mass = 0.0
    for link_id, body_name in names.bodies.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        total_mass += float(model.body_mass[bid])
        if tree.root.value != "fixed" or link_id != tree.links[0].id:
            if model.body_mass[bid] <= 0 and (link_id != tree.links[0].id or movable):
                report["errors"].append(f"link {link_id} has no mass")
    obb = entity_obb(model, data, names)
    report["metrics"]["mass_kg"] = round(total_mass, 4)
    report["metrics"]["size_m"] = [round(s, 4) for s in obb.size]
    if movable and tree.root.value == "free":
        q0 = list(data.xquat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names.bodies[tree.links[0].id])])
        steps = int(settle_seconds / model.opt.timestep)
        min_dist = 0.0
        for _ in range(steps):
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                report["errors"].append("simulation diverged (non-finite state)")
                return report
            for c in range(data.ncon):
                min_dist = min(min_dist, float(data.contact[c].dist))
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names.bodies[tree.links[0].id])
        speed = float(np.linalg.norm(data.cvel[bid][3:]))
        tilt = quat_angle_between(q0, list(data.xquat[bid]))
        report["metrics"].update(final_speed=round(speed, 5), tilt_rad=round(tilt, 4),
                                 max_penetration_m=round(-min_dist, 5))
        if speed > 0.05:
            report["errors"].append(f"asset does not come to rest (speed {speed:.3f} m/s)")
        if tilt > math.radians(10):
            report["errors"].append(f"asset tips over when placed upright ({math.degrees(tilt):.1f} deg)")
        if -min_dist > 0.01:
            report["errors"].append(f"penetration {-min_dist:.3f} m exceeds 0.01 m")
    report["ok"] = not report["errors"]
    return report
