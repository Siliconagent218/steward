"""Asset and scene authoring operations used by the World Harness and the CLI.

Scenes are composed from published assets. A placement either gives a floor pose
(``at``: x, y, yaw) or puts the asset on top of an already placed support entity (``on`` plus an
``offset`` in the support frame). A composed scene is validated by compiling it, rejecting
initial interpenetration between entities and letting it settle for one second.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from steward.assets.generators import GENERATORS
from steward.assets.library import AssetLibrary, instantiate
from steward.assets.preview import render_scene, render_template
from steward.assets.validator import validate_template
from steward.math3d import pose_compose, quat_from_yaw
from steward.scene.extent import entity_obb
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState, SpatialLayers, pose
from steward.scene.validate import validate_world


class AuthoringError(ValueError):
    pass


class AuthoringService:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.library = AssetLibrary(self.root / "assets")
        (self.root / "scenes").mkdir(parents=True, exist_ok=True)
        (self.root / "scenarios").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- assets
    def generators(self) -> list[str]:
        return sorted(GENERATORS)

    def create_asset(self, generator: str | None = None, params: dict | None = None,
                     template: dict | None = None) -> dict:
        """Build a template from a generator or accept an authored template, then validate it."""
        if (generator is None) == (template is None):
            raise AuthoringError("give exactly one of generator or template")
        if generator is not None:
            if generator not in GENERATORS:
                raise AuthoringError(f"unknown generator {generator!r}; known: {', '.join(sorted(GENERATORS))}")
            try:
                template = GENERATORS[generator](**(params or {})).to_dict()
            except TypeError as exc:
                raise AuthoringError(f"invalid parameters for {generator}: {exc}") from None
        return {"template": template, "validation": validate_template(template)}

    def preview_asset(self, template: dict, stem: str) -> list[str]:
        out = self.root / "previews"
        return [str(p) for p in render_template(template, out, stem)]

    def publish_asset(self, name: str, template: dict, meta: dict | None = None) -> dict:
        validation = validate_template(template)
        if not validation["ok"]:
            raise AuthoringError(f"asset {name!r} failed validation: {validation['errors']}")
        version = self.library.publish(name, template, meta=meta, validation=validation)
        previews = render_template(template, self.library.root / "previews", version[:16])
        return {"name": name, "version": version, "validation": validation, "previews": [str(p) for p in previews]}

    # ----------------------------------------------------------------- scenes
    def compose_scene(self, spec: dict) -> dict:
        name = spec["name"]
        scene = SceneState(mode="world", layers=SpatialLayers(regions=copy.deepcopy(spec.get("regions", []))))
        for placement in spec["entities"]:
            record = self.library.get(placement["asset"], placement.get("version"))
            at = self._placement_pose(scene, placement)
            ent = instantiate(record["template"], placement["id"], at,
                              asset={"id": placement["asset"], "version": record["version"]})
            if scene.find(ent.id) is not None:
                raise AuthoringError(f"duplicate entity id {ent.id!r}")
            scene.entities.append(ent)
        scene_dict = scene.to_dict()
        errors = validate_world(scene_dict)
        if errors:
            raise AuthoringError(f"composed scene is not a valid World State: {errors[:5]}")
        report = self.check_scene(scene_dict)
        if not report["ok"]:
            raise AuthoringError(f"scene {name!r} failed physical checks: {report['errors']}")
        path = self.root / "scenes" / f"{name}.json"
        path.write_text(json.dumps(scene_dict, indent=2))
        return {"name": name, "path": str(path), "report": report, "entities": len(scene.entities)}

    def _placement_pose(self, scene: SceneState, placement: dict) -> dict:
        if "at" in placement:
            at = placement["at"]
            return pose([at["x"], at["y"], at.get("z", 0.0)], quat_from_yaw(at.get("yaw", 0.0)))
        if "on" in placement:
            support = scene.find(placement["on"])
            if support is None:
                raise AuthoringError(f"support {placement['on']!r} must be placed before {placement['id']!r}")
            probe = SceneState(mode="world", entities=[support])
            build = build_mjcf(probe)
            model = build.compile()
            data = mujoco.MjData(model)
            build.apply_initial(model, data)
            obb = entity_obb(model, data, build.names[support.id])
            top = obb.center[2] + obb.size[2] / 2
            dx, dy = placement.get("offset", [0.0, 0.0])
            sp = support.pose.value
            local = pose_compose(sp, pose([dx, dy, 0.0]))
            yaw = placement.get("yaw", 0.0)
            return pose([local["p"][0], local["p"][1], top + 0.001], quat_from_yaw(yaw))
        raise AuthoringError(f"placement {placement['id']!r} needs 'at' or 'on'")

    def check_scene(self, scene_dict: dict, settle_seconds: float = 1.0) -> dict:
        report = {"ok": False, "errors": [], "metrics": {}}
        scene = SceneState.from_dict(scene_dict)
        build = build_mjcf(scene)
        model = build.compile()
        data = mujoco.MjData(model)
        build.apply_initial(model, data)
        body_entity = {}
        for eid, names in build.names.items():
            for body in names.bodies.values():
                body_entity[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)] = eid
        for c in range(data.ncon):
            con = data.contact[c]
            e1 = body_entity.get(int(model.geom_bodyid[con.geom1]))
            e2 = body_entity.get(int(model.geom_bodyid[con.geom2]))
            if e1 != e2 and con.dist < -0.01:
                report["errors"].append(f"{e1} and {e2} interpenetrate by {-con.dist:.3f} m")
        free = [e.id for e in scene.entities if e.rigid.root.value == "free"]
        start = {eid: np.array(entity_obb(model, data, build.names[eid]).center) for eid in free}
        for _ in range(int(settle_seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            report["errors"].append("simulation diverged while settling")
            return report
        moved = {}
        for eid in free:
            d = float(np.linalg.norm(np.array(entity_obb(model, data, build.names[eid]).center) - start[eid]))
            moved[eid] = round(d, 4)
            if d > 0.03:
                report["errors"].append(f"{eid} moved {d:.3f} m while settling")
        report["metrics"]["settle_displacement_m"] = moved
        report["ok"] = not report["errors"]
        return report

    def render_scene_views(self, scene_name: str) -> list[str]:
        scene = SceneState.from_dict(self.load_scene(scene_name))
        width, depth, cx, cy = 6.0, 5.0, 0.0, 0.0
        for ent in scene.entities:
            if ent.category.value == "building":
                floor = ent.rigid.links[0].geometry["collision"][0]["size"]
                width, depth = floor[0], floor[1]
                cx, cy = ent.pose.value["p"][0], ent.pose.value["p"][1]
        span = max(width, depth)
        cams = [
            {"name": "preview_overview", "pos": [cx, cy - depth * 0.35, span * 1.1], "target": [cx, cy + depth * 0.05, 0.0],
             "fovy": 60},
            {"name": "preview_corner", "pos": [cx + width / 2 - 0.3, cy - depth / 2 + 0.3, 2.0],
             "target": [cx - width * 0.15, cy + depth * 0.15, 0.6], "fovy": 75},
        ]
        return [str(p) for p in render_scene(scene, cams, self.root / "previews", f"scene-{scene_name}")]

    def load_scene(self, name: str) -> dict:
        path = self.root / "scenes" / f"{name}.json"
        if not path.exists():
            raise AuthoringError(f"unknown scene {name!r}")
        return json.loads(path.read_text())

    def list_scenes(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "scenes").glob("*.json"))

    # -------------------------------------------------------------- scenarios
    def save_scenario(self, scenario: dict) -> dict:
        for key in ("name", "scene", "robots", "task"):
            if key not in scenario:
                raise AuthoringError(f"scenario needs {key!r}")
        scene = self.load_scene(scenario["scene"])
        ids = {e["id"] for e in scene["entities"]}
        for rid in scenario["robots"]:
            if rid not in ids:
                raise AuthoringError(f"scenario robot {rid!r} is not in scene {scenario['scene']!r}")
        record = {"perturbationPolicy": {"allowed": [], "budget": 0}, **scenario}
        path = self.root / "scenarios" / f"{scenario['name']}.json"
        path.write_text(json.dumps(record, indent=2))
        return {"name": scenario["name"], "path": str(path)}

    def load_scenario(self, name: str) -> dict:
        path = self.root / "scenarios" / f"{name}.json"
        if not path.exists():
            raise AuthoringError(f"unknown scenario {name!r}")
        scenario = json.loads(path.read_text())
        return scenario | {"sceneState": self.load_scene(scenario["scene"])}

    def list_scenarios(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "scenarios").glob("*.json"))


def kitchen_spec() -> dict:
    """The Phase 0 kitchen: counter with a cup and a bowl, a dining table with a plate, a drawer
    cabinet, and one robot."""
    return {
        "name": "phase0-kitchen",
        "entities": [
            {"id": "w:room", "asset": "kitchen-room", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:counter-1", "asset": "counter", "at": {"x": 0.0, "y": 2.18}},
            {"id": "w:cabinet-1", "asset": "drawer-cabinet", "at": {"x": 2.70, "y": 0.6, "yaw": math.pi}},
            {"id": "w:table-1", "asset": "dining-table", "at": {"x": 0.6, "y": -0.9}},
            {"id": "w:cup-1", "asset": "cup", "on": "w:counter-1", "offset": [-0.5, 0.0]},
            {"id": "w:bowl-1", "asset": "bowl", "on": "w:counter-1", "offset": [0.45, 0.02]},
            {"id": "w:plate-1", "asset": "plate", "on": "w:table-1", "offset": [-0.3, 0.1]},
            {"id": "w:robot-1", "asset": "steward-bot", "at": {"x": -1.8, "y": -1.2, "yaw": 0.0}},
        ],
        "regions": [{"id": "kitchen", "label": "kitchen", "polygon": [[-3, -2.5], [3, -2.5], [3, 2.5], [-3, 2.5]]}],
    }


KITCHEN_ASSETS = {
    "kitchen-room": ("building", {"width": 6.0, "depth": 5.0}),
    "counter": ("counter", {}),
    "drawer-cabinet": ("cabinet_with_drawer", {}),
    "dining-table": ("dining_table", {}),
    "cup": ("cup", {}),
    "bowl": ("bowl", {}),
    "plate": ("plate", {}),
    "steward-bot": ("steward_bot", {}),
}


def build_default_kitchen(service: AuthoringService) -> dict:
    """Generate, validate and publish the kitchen assets, compose the scene and save the scenario."""
    published = {}
    for name, (generator, params) in KITCHEN_ASSETS.items():
        created = service.create_asset(generator=generator, params=params)
        if not created["validation"]["ok"]:
            raise AuthoringError(f"generated asset {name} failed validation: {created['validation']['errors']}")
        published[name] = service.publish_asset(name, created["template"], meta={"generator": generator, "params": params})
    scene = service.compose_scene(kitchen_spec())
    scenario = service.save_scenario({
        "name": "phase0-kitchen",
        "scene": "phase0-kitchen",
        "robots": ["w:robot-1"],
        "task": "把杯子放到桌上。",
    })
    return {"assets": {k: v["version"][:12] for k, v in published.items()}, "scene": scene, "scenario": scenario}
