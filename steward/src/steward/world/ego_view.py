"""Render a robot's Ego-State as a 3D scene for the Steward console.

The Ego-State is converted with the same SceneState → MJCF converter that ESim uses, so the
picture is the robot's own model of the world: only perceived entities, coloured by the colour it
perceived (or by category),
fainter when their existence probability is low, the robot body at its own estimated pose, and a
neutral floor for orientation. Walls on the camera side are left out so the camera looks into the
room.

A compiled model is reused while the set of rendered entities stays the same: each new Ego-State
version only updates poses, sizes, colours and joint positions in place. All calls must run on
the main thread (OpenGL); the Steward host routes them there.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from steward.glrender import close_renderer, encode_image, free_camera
from steward.math3d import yaw_from_quat
from steward.scene.mjcf import MjcfBuild, build_mjcf
from steward.scene.schema import SceneState

CATEGORY_RGB = {
    "table": [0.79, 0.64, 0.42], "counter": [0.54, 0.42, 0.29], "cabinet": [0.63, 0.47, 0.31],
    "sideboard": [0.69, 0.53, 0.38], "cup": [0.50, 0.83, 1.00], "bowl": [0.91, 0.91, 0.92],
    "plate": [0.95, 0.95, 0.95], "wall": [0.55, 0.58, 0.62], "drawer": [0.80, 0.67, 0.50],
    "wine_glass": [0.80, 0.92, 0.98], "mug": [0.12, 0.12, 0.13], "teddy_bear": [0.58, 0.38, 0.20],
    "scissors": [0.70, 0.72, 0.75],
}
COLOR_RGB = {
    "red": [0.85, 0.15, 0.15], "green": [0.15, 0.70, 0.25], "blue": [0.15, 0.35, 0.90], "black": [0.12, 0.12, 0.13],
    "white": [0.93, 0.93, 0.94], "clear": [0.80, 0.92, 0.98], "yellow": [0.90, 0.85, 0.20],
}
DEFAULT_RGB = [0.62, 0.68, 0.75]
FLOOR_ID = "e:render-floor"


def ego_render_scene(ego: dict, toward_camera: np.ndarray, room_center: list[float]) -> SceneState:
    """A renderable copy of an Ego-State: colours, a floor, and camera-side walls removed."""
    data = copy.deepcopy(ego)
    entities = []
    for ent in data["entities"]:
        if ent["id"] != data.get("self"):
            if ent["role"] == "environment":
                c = ent["pose"]["value"]["p"]
                offset = np.array([c[0] - room_center[0], c[1] - room_center[1]])
                norm = float(np.linalg.norm(offset))
                if norm > 1e-6 and float(offset @ toward_camera) / norm > 0.35:
                    continue
            seen = (ent.get("stateVars", {}).get("color") or {}).get("value")
            rgb = COLOR_RGB.get(seen) or CATEGORY_RGB.get(ent["category"]["value"], DEFAULT_RGB)
            alpha = 1.0 if ent["role"] == "environment" else 0.35 + 0.65 * float(ent.get("existence", 1.0))
            for link in ent["body"]["links"]:
                for geom in link["geometry"].get("collision", []):
                    geom["rgba"] = [*rgb, alpha]
        entities.append(ent)
    entities.append({
        "id": FLOOR_ID, "role": "environment", "category": {"value": "floor"},
        "pose": {"value": {"p": [room_center[0], room_center[1], 0.0], "q": [1, 0, 0, 0]}},
        "body": {"kind": "rigid", "root": {"value": "fixed"}, "links": [{"id": "floor", "name": "floor", "geometry": {
            "collision": [{"kind": "primitive", "shape": "box", "size": [12, 12, 0.02],
                           "pose": {"p": [0, 0, -0.01], "q": [1, 0, 0, 0]}, "rgba": [0.24, 0.26, 0.29, 1]}]}}]},
    })
    data["entities"] = entities
    data["relations"] = []
    data["mode"] = "world"
    return SceneState.from_dict(data)


@dataclass
class _Compiled:
    structure: tuple
    build: MjcfBuild
    model: mujoco.MjModel
    data: mujoco.MjData
    renderers: dict[tuple[int, int], mujoco.Renderer] = field(default_factory=dict)

    def close(self) -> None:
        for renderer in self.renderers.values():
            close_renderer(renderer)
        self.renderers.clear()


class EgoViewRenderer:
    def __init__(self) -> None:
        self._compiled: dict[str, _Compiled] = {}

    def render(self, robot_id: str, ego: dict, camera: dict, width: int, height: int, fmt: str = "png") -> bytes:
        az = math.radians(float(camera["azimuth"]))
        toward_camera = np.zeros(2) if float(camera["elevation"]) < -80 else np.array([-math.cos(az), -math.sin(az)])
        room_center = [float(camera["lookat"][0]), float(camera["lookat"][1])]
        scene = ego_render_scene(ego, toward_camera, room_center)
        structure = tuple((e.id, e.rigid.root.value, len(e.rigid.links)) for e in scene.entities)
        compiled = self._compiled.get(robot_id)
        if compiled is None or compiled.structure != structure:
            if compiled is not None:
                compiled.close()
            build = build_mjcf(scene)
            model = build.compile()
            data = mujoco.MjData(model)
            build.apply_initial(model, data)
            compiled = _Compiled(structure, build, model, data)
            self._compiled[robot_id] = compiled
        else:
            self._update(compiled, scene)
        if (width, height) not in compiled.renderers:
            compiled.renderers[(width, height)] = mujoco.Renderer(compiled.model, height, width)
        renderer = compiled.renderers[(width, height)]
        renderer.update_scene(compiled.data, camera=free_camera(camera))
        return encode_image(renderer.render(), fmt)

    @staticmethod
    def _update(compiled: _Compiled, scene: SceneState) -> None:
        model, data = compiled.model, compiled.data
        for ent in scene.entities:
            names = compiled.build.names[ent.id]
            p, q = ent.pose.value["p"], ent.pose.value["q"]
            root = ent.rigid.root.value
            root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names.bodies[ent.rigid.links[0].id])
            if root == "free":
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names.root_joints[0])
                adr = model.jnt_qposadr[jid]
                data.qpos[adr : adr + 3] = p
                data.qpos[adr + 3 : adr + 7] = q
            elif root == "planar":
                for jname, value in zip(names.root_joints, (p[0], p[1], yaw_from_quat(q)), strict=True):
                    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)]] = value
            else:
                model.body_pos[root_bid] = p
                model.body_quat[root_bid] = q
            for joint in ent.rigid.joints:
                if joint.id in names.joints and joint.q.value:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names.joints[joint.id])
                    data.qpos[model.jnt_qposadr[jid]] = joint.q.value[0]
            if ent.role != "robot":
                for k, geom in enumerate(ent.rigid.links[0].geometry.get("collision", [])):
                    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{names.bodies[ent.rigid.links[0].id]}::c{k}")
                    if geom["shape"] == "box":
                        model.geom_size[gid] = [s / 2 for s in geom["size"]]
                    if "rgba" in geom:
                        model.geom_rgba[gid] = geom["rgba"]
        mujoco.mj_forward(model, data)

    def close(self) -> None:
        for compiled in self._compiled.values():
            compiled.close()
        self._compiled.clear()
