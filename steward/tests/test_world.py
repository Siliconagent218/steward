from __future__ import annotations

import threading
import time

import pytest

from steward.world.sim import WorldSim


def test_running_clock_restarts_after_reset_and_does_not_jump_on_speed_change(kitchen_scene):
    sim = WorldSim(kitchen_scene, speed=4.0)
    sim.start()
    try:
        time.sleep(0.6)
        assert sim.time > 1.5
        sim.reset()
        time.sleep(0.3)
        assert sim.time < 1.6, "a reset must restart the clock instead of fast-forwarding to the old time"
        sim.speed = 1.0
        time.sleep(0.5)
        before = sim.time
        sim.speed = 8.0
        time.sleep(0.25)
        assert sim.time - before < 3.0, "raising the speed must not replay elapsed wall time"
    finally:
        sim.stop()


def test_free_camera_view_hides_walls_on_the_camera_side_only(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    south = sim._cutaway_walls({"azimuth": 90, "elevation": -50})
    southeast = sim._cutaway_walls({"azimuth": 135, "elevation": -45})
    assert len(south) == 1 and len(southeast) == 2
    assert sim._cutaway_walls({"azimuth": 90, "elevation": -89}) == []
    groups = sim.model.geom_group.copy()
    png = sim.render({"lookat": [0, 0, 0.5], "distance": 8, "azimuth": 90, "elevation": -50}, 160, 120)
    assert png.startswith(b"\x89PNG")
    assert (sim.model.geom_group == groups).all(), "rendering must restore the wall geoms"


def test_ego_view_rebuilds_do_not_corrupt_world_renders(kitchen_scene):
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator
    from steward.world.ego_view import EgoViewRenderer

    sim = WorldSim(kitchen_scene, seed=1)
    camera = {"lookat": [0, 0, 0.5], "distance": 8, "azimuth": 90, "elevation": -50}
    before = sim.render(camera, 160, 120)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    view = EgoViewRenderer()
    for _ in range(3):
        est.update(sim.sensors("w:robot-1"))
        est.tracks.pop(next(iter(est.tracks)), None)  # change the entity set to force a rebuild
        assert view.render("w:robot-1", est.to_scene().to_dict(), camera, 160, 120).startswith(b"\x89PNG")
    view.close()
    assert sim.render(camera, 160, 120) == before
    sim.stop()


def test_render_from_worker_thread_is_served_by_main_thread_pump(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    result: dict = {}

    def worker() -> None:
        result["png"] = sim.render(sim.robot_camera_name("w:robot-1"), 160, 120)

    thread = threading.Thread(target=worker)
    thread.start()
    while thread.is_alive():
        sim.pump(0.05)
    thread.join()
    assert result["png"].startswith(b"\x89PNG")


def test_render_without_main_thread_pump_times_out_without_blocking_physics(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            sim.render("overview", 160, 120, timeout=0.2)
        except Exception as exc:  # noqa: BLE001 - captured for the assertion
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    t0 = sim.time
    sim.step_seconds(0.1)
    assert isinstance(errors[0], TimeoutError)
    assert sim.time == pytest.approx(t0 + 0.1, abs=0.01)
