"""WSim: the authoritative World State simulation.

WorldSim owns the MuJoCo model built from a World State scene, advances it at a fixed timestep
(paced against wall time by ``speed``, or stepped explicitly), executes robot actuator commands,
performs Phase 0 ideal grasping, generates each robot's sensor readings, and reads the physics
state back as a World State ``SceneState``.

Phase 0 perception is an *ideal detector*: for every entity (or environment link) visible from a
robot camera, it reports the category and a noisy oriented bounding box in the camera mount
frame. It never reports World State identifiers.
"""

from __future__ import annotations

import copy
import io
import math
import queue
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from PIL import Image

from steward.math3d import matrix_from_quat, pose_compose, pose_inverse, quat_from_yaw, quat_mul, quat_normalize
from steward.scene.extent import entity_body_ids, entity_obb
from steward.scene.mjcf import MjcfBuild, build_mjcf
from steward.scene.schema import Relation, RigidTree, SceneState, exact

_COLLISION_GROUPS = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)


@dataclass
class PerceptionConfig:
    max_range: float = 8.0
    pos_sigma_base: float = 0.01
    pos_sigma_per_m: float = 0.02
    size_sigma_rel: float = 0.03
    yaw_sigma: float = 0.05
    odom_pos_sigma: float = 0.003
    odom_yaw_sigma: float = 0.003
    confusion: dict[str, list[str]] = field(default_factory=lambda: {"cup": ["bowl"], "bowl": ["cup"]})
    confusion_prob: float = 0.03


@dataclass
class GraspConfig:
    capture_radius: float = 0.06


@dataclass
class _RenderRequest:
    camera: str
    width: int
    height: int
    done: threading.Event = field(default_factory=threading.Event)
    png: bytes = b""
    error: Exception | None = None


@dataclass
class _RobotState:
    entity_id: str
    closed: bool = False
    attached: str | None = None
    rel: dict | None = None
    saved_contacts: dict[int, tuple[int, int]] = field(default_factory=dict)


class WorldSim:
    def __init__(self, scene: dict, *, timestep: float = 0.002, speed: float = 1.0, seed: int = 0,
                 perception: PerceptionConfig | None = None, grasp: GraspConfig | None = None,
                 overview_camera: dict | None = None):
        self._initial_scene = copy.deepcopy(scene)
        self.timestep = timestep
        self.speed = speed
        self.perception = perception or PerceptionConfig()
        self.grasp = grasp or GraspConfig()
        self._seed = seed
        self._overview = overview_camera
        self.lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False
        self.paused = False
        self.events: list[dict] = []
        self._render_requests: queue.Queue[_RenderRequest] = queue.Queue()
        self._load(self._initial_scene)

    # ------------------------------------------------------------------ setup
    def _load(self, scene_dict: dict) -> None:
        self.template = SceneState.from_dict(scene_dict)
        cameras = [self._overview or self._default_overview()]
        self.build: MjcfBuild = build_mjcf(self.template, timestep=self.timestep, extra_cameras=cameras)
        self.model = self.build.compile()
        self.data = mujoco.MjData(self.model)
        self.build.apply_initial(self.model, self.data)
        self.rng = np.random.default_rng(self._seed)
        self.robots = {e.id: _RobotState(e.id) for e in self.template.entities if e.role == "robot"}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._body_entity: dict[int, str] = {}
        for eid, names in self.build.names.items():
            for bid in entity_body_ids(self.model, names):
                self._body_entity[bid] = eid
        self.step_count = 0

    def _default_overview(self) -> dict:
        span = 6.0
        for ent in self.template.entities:
            if ent.category.value == "building":
                xs = [abs(j.origin.value["p"][0]) for j in ent.rigid.joints]
                ys = [abs(j.origin.value["p"][1]) for j in ent.rigid.joints]
                span = max(max(xs, default=3) * 2, max(ys, default=3) * 2)
        return {"name": "overview", "pos": [0.0, -span * 0.42, span * 1.05], "target": [0.0, 0.15, 0.0], "fovy": 60}

    def reset(self) -> None:
        with self.lock:
            self._load(copy.deepcopy(self._initial_scene))
            self.events.append({"t": 0.0, "type": "episode/reset"})

    # --------------------------------------------------------------- stepping
    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, n: int = 1) -> None:
        with self.lock:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)
                self._follow_attached()
                self.step_count += 1

    def step_seconds(self, seconds: float) -> None:
        self.step(max(1, int(round(seconds / self.timestep))))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="wsim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self.lock:
            for r in self._renderers.values():
                r.close()
            self._renderers.clear()

    def _loop(self) -> None:
        wall0 = time.perf_counter()
        sim0 = self.time
        while self._running:
            if self.paused:
                time.sleep(0.01)
                wall0 = time.perf_counter()
                sim0 = self.time
                continue
            target = sim0 + (time.perf_counter() - wall0) * self.speed
            behind = target - self.time
            if behind > self.timestep:
                self.step(min(int(behind / self.timestep), 200))
            else:
                time.sleep(0.001)

    # --------------------------------------------------------------- commands
    def command(self, robot_id: str, cmd: dict) -> dict:
        """Apply a robot actuator command: base [x, y, yaw], lift, reach, gripper "open"/"close"."""
        with self.lock:
            rs = self._robot(robot_id)
            names = self.build.names[robot_id]
            if "base" in cmd:
                for key, value in zip(("root_x", "root_y", "root_yaw"), cmd["base"], strict=True):
                    self._set_ctrl(names.actuators[key], value)
            for joint in ("lift", "reach"):
                if joint in cmd:
                    self._set_ctrl(names.actuators[joint], cmd[joint])
            if "gripper" in cmd:
                if cmd["gripper"] == "close" and not rs.closed:
                    rs.closed = True
                    self._try_attach(rs)
                elif cmd["gripper"] == "open" and rs.closed:
                    rs.closed = False
                    self._release(rs)
            return {"ok": True, "t": self.time}

    def _set_ctrl(self, actuator: str, value: float) -> None:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        lo, hi = self.model.actuator_ctrlrange[aid]
        if self.model.actuator_ctrllimited[aid]:
            value = min(max(value, lo), hi)
        self.data.ctrl[aid] = value

    def _robot(self, robot_id: str) -> _RobotState:
        if robot_id not in self.robots:
            raise KeyError(f"unknown robot {robot_id!r}")
        return self.robots[robot_id]

    def gripper_pose(self, robot_id: str) -> dict:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.build.names[robot_id].bodies["gripper"])
        return {"p": self.data.xpos[bid].tolist(), "q": self.data.xquat[bid].tolist()}

    def _try_attach(self, rs: _RobotState) -> None:
        grip = np.asarray(self.gripper_pose(rs.entity_id)["p"])
        best, best_d = None, self.grasp.capture_radius
        for ent in self.template.entities:
            if ent.role != "object" or ent.rigid.root.value != "free":
                continue
            obb = entity_obb(self.model, self.data, self.build.names[ent.id])
            d = float(np.linalg.norm(np.asarray(obb.center) - grip))
            if d <= best_d:
                best, best_d = ent.id, d
        if best is None:
            self.events.append({"t": self.time, "type": "grasp/miss", "robot": rs.entity_id})
            return
        body = self._root_body(best)
        obj_pose = {"p": self.data.xpos[body].tolist(), "q": self.data.xquat[body].tolist()}
        rs.attached = best
        rs.rel = pose_compose(pose_inverse(self.gripper_pose(rs.entity_id)), obj_pose)
        for gid in range(self.model.ngeom):
            if self._body_entity.get(int(self.model.geom_bodyid[gid])) == best:
                rs.saved_contacts[gid] = (int(self.model.geom_contype[gid]), int(self.model.geom_conaffinity[gid]))
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0
        self.events.append({"t": self.time, "type": "grasp/attach", "robot": rs.entity_id, "entity": best})

    def _release(self, rs: _RobotState) -> None:
        if rs.attached is None:
            return
        for gid, (ct, ca) in rs.saved_contacts.items():
            self.model.geom_contype[gid] = ct
            self.model.geom_conaffinity[gid] = ca
        jadr = self._free_joint_dof(rs.attached)
        self.data.qvel[jadr : jadr + 6] = 0.0
        self.events.append({"t": self.time, "type": "grasp/release", "robot": rs.entity_id, "entity": rs.attached})
        rs.attached, rs.rel, rs.saved_contacts = None, None, {}

    def _follow_attached(self) -> None:
        for rs in self.robots.values():
            if rs.attached is None:
                continue
            target = pose_compose(self.gripper_pose(rs.entity_id), rs.rel)
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.build.names[rs.attached].root_joints[0])
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr : qadr + 3] = target["p"]
            self.data.qpos[qadr + 3 : qadr + 7] = target["q"]
            dadr = self.model.jnt_dofadr[jid]
            self.data.qvel[dadr : dadr + 6] = 0.0

    def _root_body(self, entity_id: str) -> int:
        names = self.build.names[entity_id]
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, next(iter(names.bodies.values())))

    def _free_joint_dof(self, entity_id: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.build.names[entity_id].root_joints[0])
        return int(self.model.jnt_dofadr[jid])

    # ------------------------------------------------------------ world state
    def world_state(self) -> dict:
        """The current World State as a SceneState dictionary (exact values, ``w:`` ids)."""
        with self.lock:
            scene = copy.deepcopy(self.template)
            for ent in scene.entities:
                names = self.build.names[ent.id]
                tree: RigidTree = ent.rigid
                root_type = tree.root.value
                if root_type == "free":
                    bid = self._root_body(ent.id)
                    ent.pose = exact({"p": self.data.xpos[bid].tolist(), "q": quat_normalize(self.data.xquat[bid].tolist())})
                elif root_type == "planar":
                    x, y, yaw = (self._joint_q(j) for j in names.root_joints)
                    z = ent.pose.value["p"][2]
                    ent.pose = exact({"p": [x, y, z], "q": quat_from_yaw(yaw)})
                for joint in tree.joints:
                    if joint.id in names.joints:
                        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, names.joints[joint.id])
                        joint.q = exact([float(self.data.qpos[self.model.jnt_qposadr[jid]])])
                        joint.qd = exact([float(self.data.qvel[self.model.jnt_dofadr[jid]])])
                if ent.id in self.robots:
                    ent.state_vars["gripper"] = exact("closed" if self.robots[ent.id].closed else "open")
            scene.relations = self._relations()
            scene.version = self.step_count
            scene.stamp = self.time
            return scene.to_dict()

    def _joint_q(self, name: str) -> float:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return float(self.data.qpos[self.model.jnt_qposadr[jid]])

    def _relations(self) -> list[Relation]:
        rels: list[Relation] = []
        for rs in self.robots.values():
            if rs.attached is not None:
                rels.append(Relation(type="attached", a={"entity": rs.attached}, b={"entity": rs.entity_id, "link": "gripper"},
                                     mode="rigid", derived=False))
        seen = set()
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            e1 = self._body_entity.get(int(self.model.geom_bodyid[con.geom1]))
            e2 = self._body_entity.get(int(self.model.geom_bodyid[con.geom2]))
            if e1 is None or e2 is None or e1 == e2:
                continue
            normal_z = float(con.frame[2])
            if abs(normal_z) < 0.7:
                continue
            upper, lower = (e2, e1) if normal_z > 0 else (e1, e2)
            if (upper, lower) in seen or self._entity_role(upper) != "object":
                continue
            seen.add((upper, lower))
            rels.append(Relation(type="supported-by", a={"entity": upper}, b={"entity": lower}, derived=True))
        return rels

    def _entity_role(self, entity_id: str) -> str:
        return self.template.entity(entity_id).role

    def entity_obb(self, entity_id: str, links: list[str] | None = None):
        with self.lock:
            return entity_obb(self.model, self.data, self.build.names[entity_id], links)

    # ---------------------------------------------------------------- sensors
    def camera_pose(self, robot_id: str, sensor_id: str = "head_camera") -> tuple[dict, dict]:
        """World pose of a robot camera mount frame (looks along +x) and its sensor description."""
        ent = self.template.entity(robot_id)
        for link in ent.rigid.links:
            for sensor in link.sensors:
                if sensor["id"] == sensor_id:
                    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.build.names[robot_id].bodies[link.id])
                    body_pose = {"p": self.data.xpos[bid].tolist(), "q": self.data.xquat[bid].tolist()}
                    return pose_compose(body_pose, sensor["mount"]), sensor
        raise KeyError(f"robot {robot_id} has no sensor {sensor_id}")

    def sensors(self, robot_id: str) -> dict:
        """One sensor frame for a robot: odometry, joint encoders, gripper and ideal detections."""
        with self.lock:
            rs = self._robot(robot_id)
            names = self.build.names[robot_id]
            x, y, yaw = (self._joint_q(j) for j in names.root_joints)
            p = self.perception
            odom = {
                "x": x + self.rng.normal(0, p.odom_pos_sigma),
                "y": y + self.rng.normal(0, p.odom_pos_sigma),
                "yaw": yaw + self.rng.normal(0, p.odom_yaw_sigma),
                "sigma": {"pos": p.odom_pos_sigma, "yaw": p.odom_yaw_sigma},
            }
            joints = {jid: self._joint_q(jname) for jid, jname in names.joints.items()}
            cam_pose, sensor = self.camera_pose(robot_id)
            return {
                "stamp": self.time,
                "odometry": odom,
                "joints": joints,
                "gripper": {"state": "closed" if rs.closed else "open", "holding": rs.attached is not None},
                "camera": {"id": sensor["id"], "intrinsics": sensor["intrinsics"]},
                "detections": self._detections(robot_id, cam_pose, sensor),
            }

    def _detection_targets(self, robot_id: str):
        for ent in self.template.entities:
            if ent.id == robot_id:
                continue
            if ent.role == "environment":
                for link in ent.rigid.links:
                    if link.id == "floor":
                        continue
                    yield ent, [link.id], "wall"
            else:
                yield ent, None, ent.category.value

    def _detections(self, robot_id: str, cam_pose: dict, sensor: dict) -> list[dict]:
        p = self.perception
        fovy = math.radians(sensor["intrinsics"]["fovy"])
        aspect = sensor["intrinsics"]["width"] / sensor["intrinsics"]["height"]
        half_v = fovy / 2
        half_h = math.atan(math.tan(half_v) * aspect)
        cam_inv = pose_inverse(cam_pose)
        cam_p = np.asarray(cam_pose["p"])
        robot_base = self._root_body(robot_id)
        dets = []
        for ent, links, category in self._detection_targets(robot_id):
            obb = entity_obb(self.model, self.data, self.build.names[ent.id], links)
            target_bodies = set(entity_body_ids(self.model, self.build.names[ent.id], links))
            if not self._visible(obb, cam_inv, cam_p, half_h, half_v, robot_base, target_bodies):
                continue
            rel = pose_compose(cam_inv, obb.pose())
            rng_m = float(np.linalg.norm(rel["p"]))
            sigma = p.pos_sigma_base + p.pos_sigma_per_m * rng_m
            noisy_p = [rel["p"][i] + self.rng.normal(0, sigma) for i in range(3)]
            noisy_q = quat_normalize(quat_mul(rel["q"], quat_from_yaw(self.rng.normal(0, p.yaw_sigma))))
            size = [max(0.005, s * (1 + self.rng.normal(0, p.size_sigma_rel))) for s in obb.size]
            label = category
            if category in p.confusion and self.rng.random() < p.confusion_prob:
                label = str(self.rng.choice(p.confusion[category]))
            dets.append({"category": label, "confidence": round(float(0.9 - 0.05 * rng_m / p.max_range), 3),
                         "pose": {"p": noisy_p, "q": noisy_q}, "size": size, "range": rng_m})
        return dets

    def _visible(self, obb, cam_inv: dict, cam_p: np.ndarray, half_h: float, half_v: float,
                 robot_base: int, target_bodies: set[int]) -> bool:
        """True when any OBB sample point lies in the camera frustum with an unobstructed ray.

        Rays test collision geometry only (groups 0 and 3), so visual-only parts such as the
        robot's own arm never occlude; the robot base body is excluded explicitly.
        """
        center = np.asarray(obb.center)
        rot = matrix_from_quat(obb.quat)
        half = np.asarray(obb.size) / 2 * 0.9
        samples = [center] + [center + rot @ (half * np.array([sx, sy, sz]))
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        for pt in samples:
            local = pose_compose(cam_inv, {"p": pt.tolist(), "q": [1, 0, 0, 0]})["p"]
            fwd = local[0]
            if fwd < 0.1 or fwd > self.perception.max_range:
                continue
            if abs(math.atan2(local[1], fwd)) > half_h or abs(math.atan2(local[2], fwd)) > half_v:
                continue
            vec = pt - cam_p
            dist = float(np.linalg.norm(vec))
            geomid = np.array([-1], dtype=np.int32)
            hit = mujoco.mj_ray(self.model, self.data, cam_p, vec / dist, _COLLISION_GROUPS, 1, robot_base, geomid)
            if hit < 0 or hit >= dist - 0.02:
                return True
            if int(self.model.geom_bodyid[geomid[0]]) in target_bodies:
                return True
        return False

    # --------------------------------------------------------------- rendering
    def render(self, camera: str, width: int = 640, height: int = 480, timeout: float = 10.0) -> bytes:
        """Render an RGB PNG from a named MuJoCo camera.

        OpenGL rendering must run on the main thread on macOS. Calls from other threads queue a
        request and wait for the main thread to serve it through :meth:`pump`; without a pumping
        main thread they raise ``TimeoutError`` instead of blocking the simulation.
        """
        if threading.current_thread() is threading.main_thread():
            return self._render_now(camera, width, height)
        request = _RenderRequest(camera, width, height)
        self._render_requests.put(request)
        if not request.done.wait(timeout):
            raise TimeoutError("rendering needs the main thread to call WorldSim.pump()")
        if request.error is not None:
            raise request.error
        return request.png

    def pump(self, timeout: float = 0.05) -> None:
        """Serve queued render requests on the calling (main) thread, waiting up to ``timeout``."""
        try:
            request = self._render_requests.get(timeout=timeout)
        except queue.Empty:
            return
        while True:
            try:
                request.png = self._render_now(request.camera, request.width, request.height)
            except Exception as exc:  # noqa: BLE001 - re-raised on the requesting thread
                request.error = exc
            request.done.set()
            try:
                request = self._render_requests.get_nowait()
            except queue.Empty:
                return

    def _render_now(self, camera: str, width: int, height: int) -> bytes:
        with self.lock:
            key = (width, height)
            if key not in self._renderers:
                self._renderers[key] = mujoco.Renderer(self.model, height, width)
            renderer = self._renderers[key]
            renderer.update_scene(self.data, camera=camera)
            img = renderer.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    def robot_camera_name(self, robot_id: str, sensor_id: str = "head_camera") -> str:
        return self.build.names[robot_id].cameras[sensor_id]

    def status(self) -> dict:
        with self.lock:
            return {
                "t": self.time,
                "paused": self.paused,
                "speed": self.speed,
                "entities": [{"id": e.id, "category": e.category.value, "role": e.role} for e in self.template.entities],
                "robots": {rid: {"holding": rs.attached, "gripper": "closed" if rs.closed else "open"}
                           for rid, rs in self.robots.items()},
                "events": self.events[-20:],
            }
