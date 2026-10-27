"""Phase levels and the Phase 1 perception pipeline (render → segment → estimate)."""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.classify import classify
from steward.ego.perception import box_of
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.net import call
from steward.phases import PHASES, phase_modules
from steward.world.console import console_state
from steward.world.server import StewardHost


def _pump_until(host: StewardHost, done, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while not done() and time.time() < deadline:
        host.pump(0.05)
    assert done(), "timed out"


def test_box_fit_recovers_the_extent_and_heading_of_a_rotated_box():
    rng = np.random.default_rng(0)
    local = rng.uniform([-0.3, -0.1, 0.0], [0.3, 0.1, 0.2], (800, 3))
    yaw = 0.6
    c, s = math.cos(yaw), math.sin(yaw)
    points = np.stack([local[:, 0] * c - local[:, 1] * s + 1.0, local[:, 0] * s + local[:, 1] * c - 2.0, local[:, 2]], 1)
    centre, size, fitted = box_of(points)
    assert centre[:2] == pytest.approx([1.0, -2.0], abs=0.01)
    assert size == pytest.approx([0.6, 0.2, 0.2], abs=0.02)
    assert abs(math.sin(fitted - yaw)) < 0.05


def test_classifier_tells_tableware_from_furniture_by_size_colour_and_footing():
    def best(size, bottom, color, invalid=0.0):
        probs = classify(np.array(size), bottom, np.array(color), invalid)
        return max(probs, key=probs.get)

    assert best([0.22, 0.22, 0.07], 0.75, [0.8, 0.15, 0.15]) == "bowl"
    assert best([0.11, 0.085, 0.095], 0.75, [0.05, 0.05, 0.06]) == "mug"
    assert best([0.08, 0.08, 0.19], 0.75, [0.8, 0.85, 0.9], invalid=0.3) == "wine_glass"
    assert best([1.2, 0.8, 0.75], 0.0, [0.8, 0.67, 0.5]) == "table"
    assert best([0.56, 0.44, 0.22], 0.55, [0.8, 0.67, 0.5]) == "drawer"
    assert best([4.5, 0.1, 1.45], 0.0, [0.9, 0.9, 0.88]) == "wall"


def test_every_phase_level_names_its_module_choices():
    assert set(PHASES) >= {"phase0", "phase1"}
    assert phase_modules("phase0")["perception"] == "ideal"
    assert phase_modules("phase1")["perception"] == "rgbd"
    with pytest.raises(ValueError):
        phase_modules("phase9")


def test_switching_the_phase_level_changes_the_sensors_and_the_robot_follows(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=4.0, seed=3, run_dir=tmp_path).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    try:
        assert "detections" in host.sim.sensors("w:robot-1")
        _pump_until(host, lambda: runtime.perception_mode == "detections" and len(runtime.estimator.tracks) > 3)
        assert console_state(host)["phase"]["level"] == "phase0"
        with pytest.raises(ValueError):
            host.set_phase("phase9")
        state = host.set_phase("phase1")
        assert state["level"] == "phase1" and host.sim.events[-1]["type"] == "phase/switch"
        assert host.sim.sensors("w:robot-1").get("imaging") == "rgbd"
        _pump_until(host, lambda: runtime.perception_mode == "rgbd")
        _pump_until(host, lambda: len(runtime.estimator.tracks) > 3)
        assert all(t.voxels is not None for t in runtime.estimator.tracks.values()), \
            "after the switch every object comes from the RGB-D pipeline"
        host.set_phase("phase0")
        _pump_until(host, lambda: runtime.perception_mode == "detections")
    finally:
        runtime.stop()
        host.stop()


def test_phase1_exploration_builds_an_accurate_ego_state_from_rgbd(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=4.0, seed=3, run_dir=tmp_path,
                       phase="phase1").start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    finished = threading.Event()

    def explore() -> None:
        job = call("POST", f"{api.url}/ego/explore")
        call("POST", f"{api.url}/skill/wait", {"job": job["id"], "timeoutS": 120}, timeout=130)
        time.sleep(1.5)
        finished.set()

    try:
        threading.Thread(target=explore, daemon=True).start()
        _pump_until(host, finished.is_set)
        record = host.observer.latest["w:robot-1"]
        assert record["phase"] == "phase1"
        assert record["recall"] >= 0.75, record
        assert record["meanPositionErrorM"] < 0.03, record
        assert record["meanSizeErrorM"] < 0.04, record
        assert record["falsePositives"] <= 1, record
        assert "leak" not in record
        assert record["categoryAccuracy"] >= 0.8, record
        objects = call("GET", f"{api.url}/ego/summary")["objects"]
        assert {"bowl", "drawer"} <= {o["category"] for o in objects}
        walls = [o for o in objects if o["category"] == "wall"]
        assert len(walls) >= 3, "near walls fill the image edge to edge and still become tracks"

        def plane(wall: dict) -> tuple[str, float]:
            yaw = math.radians(wall["yawDeg"])
            if abs(math.cos(yaw)) > abs(math.sin(yaw)):
                return "y", round(wall["position"][1], 1)
            return "x", round(wall["position"][0], 1)

        # One wall may stay in two pieces where furniture in front of it hides its middle.
        assert len({plane(w) for w in walls}) >= len(walls) - 1, f"views of one wall merge into one track: {walls}"
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def _frame(depth: np.ndarray, mask: np.ndarray) -> dict:
    return {"depth": depth, "mask": mask, "rgb": np.full((*depth.shape, 3), 180, np.uint8), "fovy": 58.0}


CAM = {"p": [0.0, 0.0, 3.0], "q": [1.0, 0.0, 0.0, 0.0]}


def test_a_region_whose_base_is_cut_by_the_image_or_hidden_by_something_nearer_has_no_seen_bottom():
    from steward.ego.perception import segments

    h, w = 60, 80
    depth = np.full((h, w), 4.0, np.float32)
    mask = np.zeros((h, w), np.uint8)
    mask[10:30, 5:25] = 1  # standing free: the floor right below is about as far
    depth[10:30, 5:25] = 3.0
    depth[30:, 5:25] = 2.95
    mask[5:25, 40:70] = 2  # a wall with a counter right below it, much nearer
    depth[5:25, 40:70] = 3.4
    mask[25:45, 40:70] = 3  # the counter: background below it is farther
    depth[25:45, 40:70] = 2.0
    mask[45:60, 30:38] = 4  # reaches the lower image edge
    depth[45:60, 30:38] = 2.5
    seen = {round(seg.range, 1): seg.bottom_seen for seg in segments(_frame(depth, mask), CAM)}
    assert seen == {3.0: True, 3.4: False, 2.0: True, 2.5: False}


def test_a_partial_view_without_its_base_is_classified_by_its_visible_top():
    tabletop = classify([1.2, 0.8, 0.30], 0.45, [0.8, 0.7, 0.5], bottom_seen=False)
    assert max(tabletop, key=tabletop.get) == "table"
    hidden_wall = classify([3.5, 0.04, 1.03], 0.47, [0.9, 0.9, 0.9], bottom_seen=False)
    assert max(hidden_wall, key=hidden_wall.get) == "wall"
    floating = classify([1.2, 0.8, 0.30], 0.45, [0.8, 0.7, 0.5])
    assert floating.get("table", 0.0) < 0.2, "a base in view that is off the floor is not a table"
