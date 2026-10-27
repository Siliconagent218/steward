from __future__ import annotations

import pytest

from steward.authoring.service import AuthoringService
from steward.net import HttpError, call
from steward.world.server import StewardHost
from steward.world.sim import EditError, WorldSim
from tests.test_host import _pump_while


def _supported_by(sim: WorldSim, entity: str) -> str | None:
    rel = next((r for r in sim.world_state()["relations"] if r["type"] == "supported-by" and r["a"]["entity"] == entity), None)
    return rel["b"]["entity"] if rel else None


def _settle(sim: WorldSim, seconds: float = 0.5) -> None:
    sim.step(int(seconds / sim.timestep))


def test_move_places_an_object_on_another_support(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    _settle(sim)
    result = sim.edit_move("w:cup-1", {"on": "w:table-1", "offset": [0.2, 0.1]})
    _settle(sim)
    assert result["ok"] and _supported_by(sim, "w:cup-1") == "w:table-1"
    assert sim.events[-1]["type"] == "world/edit"


def test_moving_furniture_carries_what_rests_on_it(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    _settle(sim)
    plate_before = sim._entity_pose("w:plate-1")["p"]
    result = sim.edit_move("w:table-1", {"at": {"x": 0.9, "y": -0.6}})
    _settle(sim)
    plate_after = sim._entity_pose("w:plate-1")["p"]
    assert result["carried"] == ["w:plate-1"]
    assert plate_after[0] - plate_before[0] == pytest.approx(0.3, abs=0.02)
    assert _supported_by(sim, "w:plate-1") == "w:table-1"
    assert sim.world_state()["entities"][[e["id"] for e in sim.world_state()["entities"]].index("w:table-1")]["pose"]["value"]["p"][0] == pytest.approx(0.9)


def test_interpenetrating_moves_and_protected_entities_are_rejected(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    _settle(sim)
    before = sim._entity_pose("w:table-1")["p"]
    with pytest.raises(EditError, match="interpenetrate"):
        sim.edit_move("w:table-1", {"at": {"x": 0.0, "y": 2.18}})
    assert sim._entity_pose("w:table-1")["p"] == pytest.approx(before)
    for entity in ("w:robot-1", "w:room"):
        with pytest.raises(EditError, match="cannot be edited"):
            sim.edit_move(entity, {"at": {"x": 0, "y": 0}})
    with pytest.raises(EditError, match="unknown"):
        sim.edit_remove("w:nothing")


def test_spawn_and_remove_keep_the_running_state(kitchen_root, kitchen_scene):
    sim = WorldSim(kitchen_scene)
    sim.command("w:robot-1", {"base": [-1.0, -1.2, 0.0]})
    _settle(sim, 0.4)
    t, robot = sim.time, sim._entity_pose("w:robot-1")["p"]
    ctrl = sim.data.ctrl.copy()
    record = AuthoringService(kitchen_root).library.get("bowl")
    sim.edit_spawn("w:bowl-2", record["template"], {"on": "w:table-1", "offset": [0.3, -0.1]}, {"id": "bowl"})
    assert sim.time == pytest.approx(t)
    assert sim._entity_pose("w:robot-1")["p"] == pytest.approx(robot, abs=1e-6)
    assert sim.data.ctrl == pytest.approx(ctrl), "actuator targets survive the rebuild"
    _settle(sim)
    assert _supported_by(sim, "w:bowl-2") == "w:table-1"
    sim.edit_remove("w:plate-1")
    assert sim.template.find("w:plate-1") is None
    assert not any("w:plate-1" in (r["a"]["entity"], r["b"]["entity"]) for r in sim.world_state()["relations"])
    with pytest.raises(EditError, match="already used"):
        sim.edit_spawn("w:bowl-2", record["template"], {"at": {"x": 0, "y": 0}})


def test_wca_edit_endpoint_accepts_ids_without_prefix(kitchen_root):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen").start()
    url = f"{host.wca.url}/wca/edit"
    try:
        moved = _pump_while(host, lambda: call("POST", url, {"action": "move", "entity": "cup-1", "on": "table-1"}))
        assert moved["value"]["entity"] == "w:cup-1"
        refused = _pump_while(host, lambda: call("POST", url, {"action": "remove", "entity": "robot-1"}))
        assert isinstance(refused["error"], HttpError) and refused["error"].status == 409
        assert "cannot be edited" in str(refused["error"])
    finally:
        host.stop()
