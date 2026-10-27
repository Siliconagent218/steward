"""Console views: each renders on its own thread, and views stay in step through the frame clock."""

from __future__ import annotations

import json
import threading
import time

import pytest

from steward.assets.generators import steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.net import call, fetch
from steward.viewloop import FrameClock, ViewLoop, ViewSettings
from steward.world.server import StewardHost

CAMERA = {"lookat": [0.0, 0.0, 0.5], "distance": 8.0, "azimuth": 90.0, "elevation": -50.0}


def test_frame_ticks_lie_on_the_shared_wall_clock():
    clock = FrameClock(now=lambda: 100.04)
    assert clock.next_tick(10) == (1001, pytest.approx(100.1))
    assert clock.next_tick(10, after=1005) == (1006, pytest.approx(100.6))
    assert clock.next_tick(4)[0] == 401


@pytest.mark.parametrize("body", [
    {"camera": CAMERA, "fps": 0, "width": 320, "height": 240},
    {"camera": CAMERA, "fps": 60, "width": 320, "height": 240},
    {"camera": {**CAMERA, "distance": -1}, "fps": 10, "width": 320, "height": 240},
    {"camera": {**CAMERA, "lookat": [0, 0]}, "fps": 10, "width": 320, "height": 240},
    {"camera": CAMERA, "fps": 10, "width": 10, "height": 240},
    {"fps": 10, "width": 320, "height": 240},
])
def test_invalid_view_settings_are_rejected(body):
    with pytest.raises(ValueError):
        ViewSettings.parse(body)


def test_a_view_renders_on_its_own_thread_only_while_viewed_and_releases_there():
    threads: list[str] = []

    def render(settings: ViewSettings):
        threads.append(threading.current_thread().name)
        return b"img", "image/jpeg", {}

    loop = ViewLoop("probe", render, close=lambda: threads.append("close:" + threading.current_thread().name),
                    idle_s=0.5).start()
    try:
        loop.configure(ViewSettings(CAMERA, 20, 64, 64))
        time.sleep(0.3)
        assert threads == [], "nobody fetched a frame yet"
        deadline = time.monotonic() + 2.0
        while loop.latest() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        first = loop.latest()
        assert first is not None and first.index == round(first.stamp * 20)
        time.sleep(0.3)
        assert loop.latest(after=first.index).index > first.index
        time.sleep(1.0)
        rendered = len(threads)
        time.sleep(0.4)
        assert len(threads) == rendered, "rendering stops once nobody views"
    finally:
        loop.stop()
    assert set(threads) == {"view-probe", "close:view-probe"}


def test_views_in_step_through_the_clock_alone_render_the_same_frames():
    frames: dict[str, list[int]] = {"a": [], "b": []}

    def make(name: str):
        def render(settings: ViewSettings):
            time.sleep(0.01 if name == "a" else 0.03)  # different render costs, same ticks
            return b"img", "image/jpeg", {}
        return render

    loops = {name: ViewLoop(name, make(name)).start() for name in frames}
    try:
        for loop in loops.values():
            loop.configure(ViewSettings(CAMERA, 10, 64, 64))
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            for name, loop in loops.items():
                frame = loop.latest()
                if frame is not None and (not frames[name] or frames[name][-1] != frame.index):
                    frames[name].append(frame.index)
            time.sleep(0.02)
    finally:
        for loop in loops.values():
            loop.stop()
    shared = set(frames["a"]) & set(frames["b"])
    assert len(shared) >= 0.7 * max(len(frames["a"]), len(frames["b"])) >= 10, frames
    assert all(loop.status()["fps"] == pytest.approx(10, abs=1.5) for loop in loops.values())


def test_the_host_renders_the_world_off_the_main_thread_and_relays_the_robot_rendered_ego_view(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    api = ego_api(runtime).start()
    runtime.view.start()  # only the view: perception would need the host's main thread for sensor frames
    try:
        host.attach_ego_api("w:robot-1", api.url)
        wca = host.wca.url
        state = call("POST", f"{wca}/console/view", {"camera": CAMERA, "fps": 5, "width": 160, "height": 120})
        assert state["world"]["settings"]["fps"] == 5 and state["ego"]["w:robot-1"]["settings"]["fps"] == 5
        got: dict[str, list[tuple[int, bytes]]] = {"world": [], "ego": []}
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and min(len(v) for v in got.values()) < 4:
            for view in got:
                after = f"&after={got[view][-1][0]}" if got[view] else ""
                status, image, headers = fetch(f"{wca}/console/frame?view={view}{after}")
                if status == 200:
                    got[view].append((int(headers["X-Frame-Index"]), image[:2]))
                    assert json.loads(headers["X-Frame-Camera"])["azimuth"] == 90.0
            time.sleep(0.05)
        assert all(len(v) >= 4 for v in got.values()), "frames arrive with nobody pumping the main thread"
        assert all(head == b"\xff\xd8" for v in got.values() for _, head in v)
        world, ego = {i for i, _ in got["world"]}, {i for i, _ in got["ego"]}
        assert len(world & ego) >= 2, f"both views render the same ticks: {sorted(world)} / {sorted(ego)}"
    finally:
        runtime.view.stop()
        api.stop()
        host.stop()


def test_the_ego_view_draws_tracks_in_box_slots_and_recompiles_only_to_grow(kitchen_scene):
    import copy

    from steward.ego.estimator import EgoEstimator
    from steward.ego.view import SLOT_MIN, EgoViewRenderer
    from steward.world.sim import WorldSim

    sim = WorldSim(kitchen_scene, seed=1)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.update(sim.sensors("w:robot-1"))
    ego = est.to_scene().to_dict()
    settings = ViewSettings(CAMERA, 10, 160, 120)
    view = EgoViewRenderer()
    try:
        assert view.render(ego, settings).startswith(b"\xff\xd8")
        compiled = view._compiled
        fewer = copy.deepcopy(ego)
        fewer["entities"] = [e for e in fewer["entities"] if e["role"] == "robot"]
        view.render(fewer, settings)
        view.render(ego, settings)
        assert view._compiled is compiled, "tracks coming and going reuse the compiled model"
        track = next(e for e in ego["entities"] if e["role"] != "robot")
        many = copy.deepcopy(ego)
        many["entities"] += [{**copy.deepcopy(track), "id": f"e:copy-{k}"} for k in range(SLOT_MIN)]
        view.render(many, settings)
        assert view._compiled is not compiled and len(view._compiled.slots) == 2 * SLOT_MIN
    finally:
        view.close()
        sim.stop()
