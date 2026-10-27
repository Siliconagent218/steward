"""Chores task: drawers, nesting bowls, size-ordered rows, and one full round checked against World State."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService
from steward.chores import PHASES, Chores, row_offsets
from steward.ego.estimator import EgoEstimator
from steward.ego.planner import Box2D, protrusion
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.mission import _explore
from steward.world.chores_check import check_sideboard, check_swap, check_table, drawer_contents
from steward.world.console import console_state
from steward.world.server import StewardHost
from steward.world.sim import WorldSim


@pytest.fixture
def chores_scene(kitchen_root) -> dict:
    return AuthoringService(kitchen_root).load_scenario("phase0-chores")["sceneState"]


def _drawer_q(sim: WorldSim, entity: str) -> float:
    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, sim.build.names[entity].joints["drawer_slide"])
    return float(sim.data.qpos[sim.model.jnt_qposadr[jid]])


def _drive(sim: WorldSim, x: float, y: float, yaw: float, seconds: float) -> None:
    sim.command("w:robot-1", {"base": [x, y, yaw]})
    sim.step_seconds(seconds)


def test_scene_starts_with_toys_in_the_sideboard_drawer_and_tableware_on_the_table(chores_scene):
    sim = WorldSim(chores_scene)
    sim.step_seconds(1.0)
    support = {r["a"]["entity"]: r["b"] for r in sim.world_state()["relations"] if r["type"] == "supported-by"}
    assert support["w:teddy-bear"] == {"entity": "w:sideboard-1", "link": "drawer"}
    assert support["w:scissors"] == {"entity": "w:sideboard-1", "link": "drawer"}
    for eid in ("w:bowl-red", "w:bowl-green", "w:bowl-blue", "w:wine-glass", "w:mug"):
        assert support[eid]["entity"] == "w:table-1"
    parts = {cat for ent in sim.template.entities for _, _, cat in sim.perception_parts(ent)}
    assert "drawer" in parts


def test_a_grasped_handle_slides_the_drawer_and_its_contents(chores_scene):
    sim = WorldSim(chores_scene)
    _drive(sim, -1.6, 0.6, 0.0, 6)
    _drive(sim, -1.6, 0.6, math.pi, 4)
    _drive(sim, -2.15, 0.6, math.pi, 5)
    sim.command("w:robot-1", {"lift": 0.706, "reach": 0.10})
    sim.step_seconds(3)
    sim.command("w:robot-1", {"gripper": "close"})
    sim.step_seconds(0.5)
    assert sim.events[-1]["type"] == "grasp/handle"
    assert sim.sensors("w:robot-1")["gripper"]["holdingHandle"]
    bear_x = sim._entity_pose("w:teddy-bear")["p"][0]
    _drive(sim, -1.85, 0.6, math.pi, 3)
    assert _drawer_q(sim, "w:sideboard-1") == pytest.approx(0.30, abs=0.02)
    assert sim._entity_pose("w:teddy-bear")["p"][0] - bear_x == pytest.approx(0.30, abs=0.03)
    sim.command("w:robot-1", {"gripper": "open"})
    sim.step_seconds(0.5)
    assert sim.events[-1]["type"] == "grasp/handle-release"
    assert _drawer_q(sim, "w:sideboard-1") == pytest.approx(0.30, abs=0.02)
    categories = {d["category"] for d in sim.sensors("w:robot-1")["detections"]}
    assert "teddy_bear" in categories, "the open drawer shows what is inside"


def _det(category: str, p: list[float], size: list[float], color: str) -> dict:
    return {"category": category, "pose": {"p": p, "q": [1, 0, 0, 0]}, "size": size, "range": 1.0, "color": color}


def test_nested_bowls_stay_separate_tracks_and_know_what_holds_them():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    frame_odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.005, "yaw": 0.005}}
    cam = est.camera_pose()
    inv = np.linalg.inv  # noqa: F841 - poses below are expressed through pose_compose
    from steward.math3d import pose_compose, pose_inverse

    def local(p):
        return pose_compose(pose_inverse(cam), {"p": p, "q": [1, 0, 0, 0]})["p"]

    board_top = 0.85
    world = [
        _det("sideboard", [1.5, 0.0, 0.425], [0.45, 1.4, 0.85], "beige"),
        _det("bowl", [1.5, 0.0, board_top + 0.035], [0.22, 0.22, 0.07], "red"),
        _det("bowl", [1.5, 0.0, board_top + 0.008 + 0.03], [0.17, 0.17, 0.06], "green"),
        _det("bowl", [1.5, 0.0, board_top + 0.016 + 0.025], [0.13, 0.13, 0.05], "blue"),
    ]
    for k in range(5):
        dets = [{**d, "pose": {"p": local(d["pose"]["p"]), "q": [1, 0, 0, 0]}} for d in world]
        est.update({"stamp": 0.1 * (k + 1), "odometry": frame_odometry, "joints": {"lift": 0.6, "reach": 0.1},
                    "gripper": {"state": "open", "holding": False}, "detections": dets})
    bowls = {t.color: t for t in est.tracks.values() if t.category == "bowl"}
    assert set(bowls) == {"red", "green", "blue"}
    board = next(t for t in est.tracks.values() if t.category == "sideboard")
    assert est.support_of(bowls["blue"]) is bowls["green"]
    assert est.support_of(bowls["green"]) is bowls["red"]
    assert est.support_of(bowls["red"]) is board


def test_drawer_protrusion_is_measured_from_the_furniture_front():
    cabinet = Box2D("cab", 0.0, 0.0, 0.0, 0.25, 0.3)
    closed = Box2D("d", 0.02, 0.0, 0.0, 0.24, 0.25)
    opened = Box2D("d", 0.32, 0.0, 0.0, 0.24, 0.25)
    assert protrusion(closed, [cabinet]) == pytest.approx(0.01, abs=1e-6)
    assert protrusion(opened, [cabinet]) == pytest.approx(0.31, abs=1e-6)
    assert protrusion(opened, []) == 0.0


def test_row_offsets_space_items_by_width_around_the_centre():
    offsets = row_offsets([0.1, 0.2, 0.3], gap=0.05)
    assert offsets == pytest.approx([-0.30, -0.10, 0.20])


def test_one_chores_round_is_confirmed_by_world_state(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=10.0, seed=5, run_dir=tmp_path).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    try:
        def truth():
            return host.sim.world_state(), {g["id"]: g for g in console_state(host)["groundTruth"]}

        before = drawer_contents(truth()[0])
        runner = Chores(api.url)
        _explore(api.url, 120)
        runner.arrange_on_table()
        ok, detail = check_table(*truth())
        assert ok, detail
        runner.stack_on_sideboard()
        ok, detail = check_sideboard(*truth())
        assert ok, detail
        runner.swap_drawers()
        ok, detail = check_swap(before, truth()[0])
        assert ok, detail
        assert PHASES == ("table", "sideboard", "drawers")
        assert host.observer.report()["w:robot-1"]["leaks"] == 0
    finally:
        api.stop()
        runtime.stop()
        host.stop()
