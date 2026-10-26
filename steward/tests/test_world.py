from __future__ import annotations

import threading

import pytest

from steward.world.sim import WorldSim


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
