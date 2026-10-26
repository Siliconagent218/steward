"""Phase 0 acceptance: the robot puts the cup on the table using only the Ego API."""

from __future__ import annotations

import time

from steward.assets.generators import steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.mission import cup_to_table
from steward.net import HttpError, call
from steward.world.server import StewardHost


def test_cup_to_table_end_to_end(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", speed=6.0, seed=11, run_dir=tmp_path).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    try:
        result = cup_to_table(api.url)
        time.sleep(0.3)
        ws = host.sim.world_state()
        assert any(r["type"] == "supported-by" and r["a"]["entity"] == "w:cup-1" and r["b"]["entity"] == "w:table-1"
                   for r in ws["relations"])
        steps = [s["step"] for s in result["log"]]
        assert steps[-1] == "place"
        report = host.observer.report()["w:robot-1"]
        assert report["leaks"] == 0
        assert report["latest"]["meanPositionErrorM"] < report["first"]["meanPositionErrorM"]
        with_receipt = call("POST", f"{api.url}/ego/predict", {"skill": "place", "params": {"on": result["table"]}})
        assert not with_receipt["feasible"], "nothing is held after placing"
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def test_robot_port_exposes_no_world_state(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", speed=1.0, run_dir=tmp_path).start()
    try:
        robot = host.robot_servers["w:robot-1"].url
        frame = call("GET", f"{robot}/sensors")
        assert "w:" not in str(frame)
        for path in ("/wca/scene", "/wca/status", "/assets"):
            try:
                call("GET", f"{robot}{path}")
                raise AssertionError(f"robot port served {path}")
            except HttpError as exc:
                assert exc.status == 404
    finally:
        host.stop()
