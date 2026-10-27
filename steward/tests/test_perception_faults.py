"""Gross perception faults: concave corners must not become solid boxes, and the world watches for
phantoms and a robot standing inside an object."""

from __future__ import annotations

import numpy as np

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.perception import Segment
from steward.observer.observer import ANOMALY_REPEATS, Observer, _anomalies
from steward.world.server import StewardHost


def _wall_points(rng, xs, ys):
    n = 1500
    return np.c_[rng.uniform(*xs, n), rng.uniform(*ys, n), rng.uniform(0.05, 1.1, n)]


def test_the_second_wall_of_a_corner_is_not_fused_into_the_same_track():
    rng = np.random.default_rng(0)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    first = _wall_points(rng, (-3.0, -1.4), (-2.51, -2.49))
    track = Track(id="e:1", category_counts={"unknown": 5.0}, pos=np.array([-2.2, -2.5, 0.6]), var=np.full(3, 1e-4),
                  yaw=0.0, yaw_var=1e-3, size=np.array([1.6, 0.05, 1.1]), observations=6, first_seen=0.0, last_seen=0.0,
                  existence=1.0)
    track.voxels = np.unique(np.floor(first / 0.01).astype(np.int32), axis=0)
    est.tracks[track.id] = track
    other = _wall_points(rng, (-3.01, -2.99), (-2.5, -1.2))
    seg = Segment(points=other, color=np.array([0.9, 0.9, 0.9]), pixels=4000, invalid_frac=0.0, border=False, range=2.0)
    m = {"seg": seg, "centre": np.array([-3.0, -1.85, 0.6]), "size": np.array([1.3, 0.02, 1.1]), "yaw": 1.57,
         "probs": {"unknown": 1.0}, "footprint": 1.3}
    est._fuse_cloud(track, m, np.array([0.0, 0.0, 1.4]))
    assert max(track.size[:2]) < 1.8 and min(track.size[:2]) < 0.2, "the corner is not filled in as a cube"


def _ent(eid, cat, centre, size, role="object"):
    return {"id": eid, "role": role, "category": {"value": cat},
            "pose": {"value": {"p": centre, "q": [1, 0, 0, 0]}},
            "body": {"links": [{"geometry": {"collision": [{"size": size}]}}]}}


GT = [{"id": "w:room/wall_west", "category": "wall", "center": [-3.05, 0.0, 1.3], "size": [0.1, 5.0, 2.6], "env": True},
      {"id": "w:room/wall_south", "category": "wall", "center": [0.0, -2.55, 1.3], "size": [6.2, 0.1, 2.6], "env": True},
      {"id": "w:chair-1", "category": "chair", "center": [-2.6, -1.4, 0.45], "size": [0.46, 0.46, 0.9], "env": False}]


def test_a_phantom_corner_box_and_a_robot_inside_it_are_explained():
    corner = _ent("e:398", "unknown", [-2.11, -1.83, 0.74], [1.9, 1.48, 1.48])
    found = {a["kind"]: a for a in _anomalies([corner], set(), GT, {"p": [-1.5, -1.24, 0.0]})}
    assert set(found) == {"phantom", "robot-inside"}
    assert "凹进去的墙角被当成了凸出的立方体" in found["phantom"]["analysis"]
    assert "chair-1" in found["phantom"]["analysis"] and "导航与停靠规划会全部失败" in found["robot-inside"]["analysis"]
    table = _ent("e:6", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.76])
    assert _anomalies([table], {"e:6"}, GT, {"p": [-1.5, -1.24, 0.0]}) == [], "a matched object far from the robot is fine"


def test_a_fault_is_reported_once_after_it_persists():
    reports = []
    observer = Observer(lambda: None, on_anomaly=lambda robot, a: reports.append((robot, a["key"])))
    fault = [{"key": "phantom:e:398", "kind": "phantom", "entity": "e:398", "message": "m", "analysis": "a"}]
    for _ in range(ANOMALY_REPEATS - 1):
        observer._watch("w:robot-1", fault)
    assert reports == []
    observer._watch("w:robot-1", fault)
    observer._watch("w:robot-1", fault)
    assert reports == [("w:robot-1", "phantom:e:398")]


def test_a_perception_fault_pauses_the_world_until_acknowledged(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path, pause_on_perception_fault=True)
    try:
        host._perception_fault("w:robot-1", {"kind": "phantom", "entity": "e:398", "message": "幽灵物体", "analysis": "墙角"})
        assert host.sim.paused and [a["message"] for a in host.active_perception_alerts()] == ["幽灵物体"]
        printed = [m["text"] for m in host.chat.state()["messages"] if m["role"] == "error"]
        assert printed and "世界侧感知监测：幽灵物体" in printed[-1], "the operator reads the fault in the chat"
        host.acknowledge_alerts()
        assert not host.sim.paused and host.active_perception_alerts() == []
    finally:
        host.stop()
