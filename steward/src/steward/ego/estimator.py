"""Ego-State estimation: fuse detections into robot-owned object tracks and publish the ES.

Each detection is transformed from the camera mount frame into the robot's ``map`` frame with
the robot's own odometry and body model, associated to an existing track (nearest compatible
category within a gate) and fused with a per-axis Kalman update, so position variance shrinks
as observations accumulate. Unmatched detections start new tracks with robot-assigned ``e:``
ids. A track that should have been visible but was not detected loses existence probability
and is dropped below a threshold.

Phase 0 assumption: the ``map`` frame coincides with the world frame because odometry is
nearly ideal; the Observer still estimates the alignment explicitly.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from steward.ego.planner import Box2D, protrusion
from steward.ego.priors import compatible, prior
from steward.math3d import pose_compose, pose_inverse, quat_from_yaw, wrap_angle, yaw_from_quat
from steward.scene.schema import Entity, Est, Link, Relation, RigidTree, SceneState, primitive

SELF_ID = "e:self"


@dataclass
class Track:
    id: str
    category_counts: dict[str, float]
    pos: np.ndarray
    var: np.ndarray
    yaw: float
    yaw_var: float
    size: np.ndarray
    observations: int
    first_seen: float
    last_seen: float
    existence: float = 0.5
    held: bool = False
    held_offset: np.ndarray | None = None
    color_counts: dict[str, float] = field(default_factory=dict)

    @property
    def category(self) -> str:
        return max(self.category_counts, key=self.category_counts.get)

    @property
    def color(self) -> str | None:
        return max(self.color_counts, key=self.color_counts.get) if self.color_counts else None

    @property
    def footprint(self) -> float:
        return float(max(self.size[0], self.size[1]))

    @property
    def sigma(self) -> float:
        return float(math.sqrt(float(np.mean(self.var[:2]))))

    def top(self) -> float:
        return float(self.pos[2] + self.size[2] / 2)

    def bottom(self) -> float:
        return float(self.pos[2] - self.size[2] / 2)


@dataclass
class EstimatorConfig:
    pos_sigma_base: float = 0.01
    pos_sigma_per_m: float = 0.02
    process_noise: float = 1e-6
    gate_m: float = 0.30
    size_gate_ratio: float = 0.25
    size_match_log: float = 0.15
    missed_decay: float = 0.97
    drop_below: float = 0.05
    max_range: float = 8.0


@dataclass
class EgoEstimator:
    robot_model: dict
    config: EstimatorConfig = field(default_factory=EstimatorConfig)

    def __post_init__(self) -> None:
        self.tracks: dict[str, Track] = {}
        self._ids = itertools.count(1)
        self.version = 0
        self.stamp = 0.0
        self.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
        self.joints: dict[str, float] = {}
        self.gripper = {"state": "open", "holding": False}
        self.held_id: str | None = None
        self._self_template = self._make_self_template()
        self._head_mount, self._camera = self._camera_model()

    # ------------------------------------------------------------- body model
    def _make_self_template(self) -> dict:
        data = copy.deepcopy(self.robot_model)
        data["id"] = SELF_ID
        data.pop("asset", None)
        return data

    def _camera_model(self) -> tuple[dict, dict]:
        tree = self.robot_model["body"]
        for link in tree["links"]:
            for sensor in link.get("sensors", []):
                if sensor["kind"] in ("rgb", "depth", "rgbd"):
                    mount_joint = next(j for j in tree["joints"] if j["child"] == link["id"])
                    return mount_joint["origin"]["value"], sensor
        raise ValueError("robot model has no camera")

    def base_pose(self) -> dict:
        o = self.odometry
        return {"p": [o["x"], o["y"], 0.0], "q": quat_from_yaw(o["yaw"])}

    def camera_pose(self) -> dict:
        return pose_compose(pose_compose(self.base_pose(), self._head_mount), self._camera["mount"])

    def gripper_position(self) -> np.ndarray:
        """Gripper position from the robot's own kinematics (base, lift, reach)."""
        tree = self.robot_model["body"]
        lift = next(j for j in tree["joints"] if j["id"] == "lift")
        grip = next(j for j in tree["joints"] if j["id"] == "gripper_mount")
        local_x = lift["origin"]["value"]["p"][0] + self.joints.get("reach", 0.0) + grip["origin"]["value"]["p"][0]
        local = {"p": [local_x, 0.0, self.joints.get("lift", 0.0)], "q": [1, 0, 0, 0]}
        return np.asarray(pose_compose(self.base_pose(), local)["p"])

    # ----------------------------------------------------------------- update
    def update(self, frame: dict) -> None:
        self.stamp = float(frame["stamp"])
        self.odometry = frame["odometry"]
        self.joints = frame["joints"]
        self.gripper = frame["gripper"]
        if not self.gripper.get("holding"):
            self._clear_held()
        cam = self.camera_pose()
        measurements = []
        for det in frame.get("detections", []):
            world = pose_compose(cam, det["pose"])
            rng = float(det.get("range", np.linalg.norm(det["pose"]["p"])))
            sigma = self.config.pos_sigma_base + self.config.pos_sigma_per_m * rng
            measurements.append((det, np.asarray(world["p"]), yaw_from_quat(world["q"]), sigma))
        matched: set[str] = set()
        for det, pos, yaw, sigma in sorted(measurements, key=lambda m: m[3]):
            track = self._associate(det["category"], pos, sigma, matched, float(max(det["size"][0], det["size"][1])))
            if track is None:
                track = self._new_track(det, pos, yaw, sigma)
            else:
                self._fuse(track, det, pos, yaw, sigma)
            matched.add(track.id)
        self._decay_unseen(cam, matched)
        self._merge_duplicates()
        self._follow_held()
        self.version += 1

    def _gate(self, track: Track, sigma: float) -> float:
        """Association gate: measurement and track uncertainty, widened for large objects whose
        detected box centre moves with the visible portion."""
        gate = max(self.config.gate_m, 3.0 * math.sqrt(float(np.mean(track.var)) + sigma**2),
                   self.config.size_gate_ratio * float(max(track.size[0], track.size[1])))
        return max(gate, 0.15) if track.held else gate

    def _size_match(self, a: float, b: float) -> bool:
        """Footprints of the same object agree within the detector's size noise; nested bowls do not."""
        return abs(math.log(max(a, 1e-3) / max(b, 1e-3))) <= self.config.size_match_log

    def _associate(self, category: str, pos: np.ndarray, sigma: float, taken: set[str],
                   footprint: float | None = None) -> Track | None:
        best, best_d = None, None
        for track in self.tracks.values():
            if track.id in taken or not compatible(category, track.category):
                continue
            if footprint is not None and not track.held and not self._size_match(footprint, track.footprint):
                continue
            d = float(np.linalg.norm(track.pos - pos))
            if d <= self._gate(track, sigma) and (best_d is None or d < best_d):
                best, best_d = track, d
        return best

    def _merge_duplicates(self) -> None:
        """Merge compatible tracks whose centres lie inside each other's uncertainty-expanded
        footprint; the track with more observations keeps its id."""
        ids = sorted(self.tracks, key=lambda t: -self.tracks[t].observations)
        for i, keep_id in enumerate(ids):
            keep = self.tracks.get(keep_id)
            if keep is None or keep.held:
                continue
            for drop_id in ids[i + 1:]:
                drop = self.tracks.get(drop_id)
                if drop is None or drop.held or not compatible(drop.category, keep.category):
                    continue
                if not self._size_match(drop.footprint, keep.footprint):
                    continue
                z_tol = max(0.5 * max(keep.size[2], drop.size[2], 0.1), 3.0 * _sigma_z(keep, drop))
                if abs(drop.pos[2] - keep.pos[2]) > z_tol:
                    continue
                if not (_inside(keep, drop.pos, 3.0 * keep.sigma) or _inside(drop, keep.pos, 3.0 * drop.sigma)):
                    continue
                w_keep, w_drop = 1.0 / keep.var, 1.0 / drop.var
                keep.pos = (keep.pos * w_keep + drop.pos * w_drop) / (w_keep + w_drop)
                keep.var = 1.0 / (w_keep + w_drop)
                for cat, n in drop.category_counts.items():
                    keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                for col, n in drop.color_counts.items():
                    keep.color_counts[col] = keep.color_counts.get(col, 0.0) + n
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                keep.last_seen = max(keep.last_seen, drop.last_seen)
                del self.tracks[drop_id]

    def _new_track(self, det: dict, pos: np.ndarray, yaw: float, sigma: float) -> Track:
        track = Track(
            id=f"e:{next(self._ids)}",
            category_counts={det["category"]: 1.0},
            pos=pos.copy(),
            var=np.full(3, sigma**2),
            yaw=yaw,
            yaw_var=0.05**2,
            size=np.asarray(det["size"], float),
            observations=1,
            first_seen=self.stamp,
            last_seen=self.stamp,
            color_counts={det["color"]: 1.0} if det.get("color") else {},
        )
        self.tracks[track.id] = track
        return track

    def _fuse(self, track: Track, det: dict, pos: np.ndarray, yaw: float, sigma: float) -> None:
        track.category_counts[det["category"]] = track.category_counts.get(det["category"], 0.0) + 1.0
        if det.get("color"):
            track.color_counts[det["color"]] = track.color_counts.get(det["color"], 0.0) + 1.0
        track.observations += 1
        track.last_seen = self.stamp
        track.existence = min(1.0, track.existence + 0.2 * (1.0 - track.existence))
        if track.held:
            return
        r = sigma**2
        prior_var = track.var + self.config.process_noise
        gain = prior_var / (prior_var + r)
        track.pos = track.pos + gain * (pos - track.pos)
        track.var = (1.0 - gain) * prior_var
        k_yaw = track.yaw_var / (track.yaw_var + 0.05**2)
        track.yaw = wrap_angle(track.yaw + k_yaw * wrap_angle(yaw - track.yaw))
        track.yaw_var = (1.0 - k_yaw) * track.yaw_var
        n = track.observations
        track.size = track.size + (np.asarray(det["size"], float) - track.size) / n

    def _decay_unseen(self, cam: dict, matched: set[str]) -> None:
        fovy = math.radians(self._camera["intrinsics"]["fovy"])
        aspect = self._camera["intrinsics"]["width"] / self._camera["intrinsics"]["height"]
        half_v = fovy / 2 * 0.8
        half_h = math.atan(math.tan(fovy / 2) * aspect) * 0.8
        inv = pose_inverse(cam)
        for tid in list(self.tracks):
            track = self.tracks[tid]
            if tid in matched or track.held or self._in_closed_drawer(track):
                continue
            local = pose_compose(inv, {"p": track.pos.tolist(), "q": [1, 0, 0, 0]})["p"]
            if not (0.3 < local[0] < self.config.max_range * 0.8):
                continue
            if abs(math.atan2(local[1], local[0])) > half_h or abs(math.atan2(local[2], local[0])) > half_v:
                continue
            track.existence *= self.config.missed_decay
            if track.existence < self.config.drop_below:
                del self.tracks[tid]

    # ---------------------------------------------------------------- holding
    def mark_held(self, track_id: str) -> None:
        track = self.tracks[track_id]
        track.held = True
        track.held_offset = track.pos - self.gripper_position()
        self.held_id = track_id

    def _clear_held(self) -> None:
        if self.held_id and self.held_id in self.tracks:
            track = self.tracks[self.held_id]
            track.held = False
            track.held_offset = None
            track.var = np.maximum(track.var, 0.02**2)
        self.held_id = None

    def _follow_held(self) -> None:
        if self.held_id and self.held_id in self.tracks:
            track = self.tracks[self.held_id]
            track.pos = self.gripper_position() + track.held_offset

    # ------------------------------------------------------------------ output
    def support_of(self, track: Track) -> Track | None:
        """What holds a track: the innermost receptacle (drawer, larger bowl) whose floor it rests
        on, else the highest support surface whose top it rests on."""
        inside, best = None, None
        for other in self.tracks.values():
            if other.id == track.id or other.held:
                continue
            affordances = prior(other.category)["affordances"]
            z_tol = max(0.06, 2.5 * _sigma_z(track, other))
            local = _to_local_xy(other, track.pos)
            if "receptacle" in affordances and track.footprint < other.footprint:
                within = abs(local[0]) <= other.size[0] / 2 and abs(local[1]) <= other.size[1] / 2
                if within and -0.03 <= track.bottom() - other.bottom() <= min(z_tol, 0.5 * other.size[2]):
                    if inside is None or other.bottom() > inside.bottom():
                        inside = other
                    continue
            if "support-surface" not in affordances or abs(other.top() - track.bottom()) > z_tol:
                continue
            margin = max(0.02, 2.0 * track.sigma)
            if abs(local[0]) <= other.size[0] / 2 + margin and abs(local[1]) <= other.size[1] / 2 + margin:
                if best is None or other.top() > best.top():
                    best = other
        return inside or best

    def _in_closed_drawer(self, track: Track) -> bool:
        holder = self.support_of(track)
        return holder is not None and holder.category == "drawer" and self.drawer_opening(holder) < 0.08

    def drawer_opening(self, drawer: Track) -> float:
        """How far a drawer sticks out of the furniture holding it, along the drawer's front axis."""
        furniture = [_box2d(t) for t in self.tracks.values()
                     if t.id != drawer.id and "support-surface" in prior(t.category)["affordances"]]
        return protrusion(_box2d(drawer), furniture)

    def shift(self, track_id: str, delta: np.ndarray) -> list[str]:
        """Move a track the robot itself moved (a drawer it pulled) together with what it holds."""
        track = self.tracks.get(track_id)
        if track is None:
            return []
        carried = [t.id for t in self.tracks.values() if t.id != track_id and not t.held and self.support_of(t) is track]
        for tid in [track_id, *carried]:
            self.tracks[tid].pos = self.tracks[tid].pos + np.asarray(delta, float)
        self.version += 1
        return carried

    def to_scene(self) -> SceneState:
        entities = [self._self_entity()]
        relations: list[Relation] = []
        for track in self.tracks.values():
            entities.append(self._track_entity(track))
            if track.held:
                relations.append(Relation(type="attached", a={"entity": track.id}, b={"entity": SELF_ID, "link": "gripper"},
                                          mode="rigid", derived=False, confidence=1.0))
            elif prior(track.category)["root"].get("fixed", 0.0) < 0.9:
                support = self.support_of(track)
                if support is not None:
                    relations.append(Relation(type="supported-by", a={"entity": track.id}, b={"entity": support.id},
                                              derived=True, confidence=round(min(track.existence, support.existence), 3)))
        return SceneState(mode="ego", entities=entities, relations=relations, version=self.version, stamp=self.stamp,
                          self_id=SELF_ID)

    def _self_entity(self) -> Entity:
        ent = Entity.from_dict(self._self_template)
        o = self.odometry
        pos_var = o["sigma"]["pos"] ** 2
        yaw_var = o["sigma"]["yaw"] ** 2
        ent.pose = Est(value=self.base_pose(), unc=_gaussian6([pos_var, pos_var, 1e-8, 1e-8, 1e-8, yaw_var]),
                       t=self.stamp, src="observed")
        tree: RigidTree = ent.body
        for joint in tree.joints:
            if joint.id in self.joints:
                joint.q = Est(value=[self.joints[joint.id]], unc={"kind": "gaussian", "cov": [1e-6]}, t=self.stamp,
                              src="observed")
        ent.state_vars["gripper"] = Est(value=self.gripper["state"], t=self.stamp, src="observed")
        ent.state_vars["holding"] = Est(value=self.gripper.get("holding", False), t=self.stamp, src="observed")
        return ent

    def _track_entity(self, track: Track) -> Entity:
        p = prior(track.category)
        total = sum(track.category_counts.values())
        cat_unc = {"kind": "categorical", "p": {k: round(v / total, 4) for k, v in track.category_counts.items()}}
        src = "fused" if track.observations > 1 else "observed"
        inflate = 0.0 if track.held else 2.0 * track.sigma
        size = [float(track.size[0] + inflate), float(track.size[1] + inflate), float(track.size[2])]
        link = Link(
            id="body", name=track.category,
            geometry={"collision": [primitive("box", size)]},
            material=Est(value={"fragility": p["fragility"]}, src="prior"),
            affordances=[Est(value=a, src="prior") for a in p["affordances"]],
        )
        root_value = max(p["root"], key=p["root"].get)
        return Entity(
            id=track.id,
            role=p["role"],
            category=Est(value=track.category, unc=cat_unc, t=track.last_seen, src=src),
            pose=Est(value={"p": track.pos.tolist(), "q": quat_from_yaw(track.yaw)},
                     unc=_gaussian6([*track.var.tolist(), 1e-4, 1e-4, track.yaw_var]), t=track.last_seen, src=src),
            body=RigidTree(root=Est(value=root_value, unc={"kind": "categorical", "p": p["root"]}, src="prior"),
                           links=[link]),
            existence=round(track.existence, 4),
            state_vars={"color": Est(value=track.color, src=src)} if track.color else {},
        )


def _box2d(track: Track) -> Box2D:
    return Box2D(track.id, float(track.pos[0]), float(track.pos[1]), track.yaw, track.size[0] / 2, track.size[1] / 2)


def _gaussian6(diag: list[float]) -> dict:
    cov = [0.0] * 36
    for i, v in enumerate(diag):
        cov[i * 7] = float(v)
    return {"kind": "gaussian", "cov": cov}


def _to_local_xy(track: Track, point: np.ndarray) -> tuple[float, float]:
    dx, dy = point[0] - track.pos[0], point[1] - track.pos[1]
    c, s = math.cos(-track.yaw), math.sin(-track.yaw)
    return (c * dx - s * dy, s * dx + c * dy)


def _sigma_z(a: Track, b: Track) -> float:
    """Standard deviation of the height difference between two tracks."""
    return math.sqrt(float(a.var[2] + b.var[2]))


def _inside(track: Track, point: np.ndarray, margin: float) -> bool:
    lx, ly = _to_local_xy(track, point)
    return abs(lx) <= track.size[0] / 2 + margin and abs(ly) <= track.size[1] / 2 + margin
