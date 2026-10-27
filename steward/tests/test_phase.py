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
        categories = {o["category"] for o in call("GET", f"{api.url}/ego/summary")["objects"]}
        assert {"bowl", "drawer"} <= categories
    finally:
        api.stop()
        runtime.stop()
        host.stop()
