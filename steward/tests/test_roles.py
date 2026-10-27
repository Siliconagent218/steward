"""Scene-level roles: furniture and look-alikes partly seen are named by what the whole scene says."""

from __future__ import annotations

from steward.chores import _objects
from steward.ego.roles import RoleConfig, assign_roles


def _obj(oid, category, probs, pos, extent, yaw=0.0):
    return {"id": oid, "category": category, "categories": probs, "existence": 1.0, "position": pos, "yawDeg": yaw,
            "extentM": extent, "sizeM": extent}


SCENE = [
    _obj("e:35", "table", {"table": 0.85, "counter": 0.13, "sideboard": 0.02}, [1.0, -1.86, 0.39], [1.17, 0.8, 0.75]),
    _obj("e:3", "table", {"table": 0.44, "cabinet": 0.36, "sideboard": 0.14, "counter": 0.06}, [2.56, 0.59, 0.41], [0.64, 0.6, 0.8]),
    _obj("e:39", "counter", {"counter": 0.5, "sideboard": 0.42, "table": 0.08}, [-2.73, 0.58, 0.43], [1.4, 0.47, 0.84], -90),
    _obj("e:40", "counter", {"counter": 0.7, "sideboard": 0.2, "table": 0.1}, [0.0, 2.17, 0.46], [1.76, 0.62, 0.89]),
    _obj("e:7", "drawer", {"drawer": 1.0}, [2.37, 0.59, 0.6], [0.58, 0.52, 0.42], -90),
    _obj("e:8", "drawer", {"drawer": 1.0}, [-2.43, 0.59, 0.7], [0.57, 0.46, 0.21], -90),
    _obj("e:9", "cup", {"cup": 0.57, "mug": 0.43}, [0.76, -2.26, 0.8], [0.08, 0.07, 0.1]),
]


def test_one_of_a_kind_furniture_and_drawers_resolve_confused_categories():
    roles = {oid: role for oid, (role, _) in assign_roles(SCENE, RoleConfig()).items()}
    assert roles["e:35"] == "table"
    assert roles["e:3"] == "cabinet", "the table is taken and a drawer sits in it"
    assert roles["e:39"] == "sideboard", "the counter is likelier elsewhere and a drawer sits in it"
    assert roles["e:40"] == "counter"
    assert roles["e:7"] == roles["e:8"] == "drawer" and roles["e:9"] == "cup"


def test_without_one_of_a_kind_knowledge_each_object_keeps_its_best_category():
    roles = {oid: role for oid, (role, _) in assign_roles(SCENE, RoleConfig(singletons=(), drawer_evidence=1.0)).items()}
    assert roles["e:3"] == "table" and roles["e:39"] == "counter"


def test_tasks_find_objects_by_role():
    roles = assign_roles(SCENE, RoleConfig())
    summary = {"objects": [{**o, "role": roles[o["id"]][0]} for o in SCENE]}
    assert [o["id"] for o in _objects(summary, "sideboard")] == ["e:39"]
    assert [o["id"] for o in _objects(summary, "cabinet")] == ["e:3"]


def test_telemetry_carries_each_objects_role_next_to_its_perceived_category():
    import numpy as np

    from steward.assets.generators import steward_bot
    from steward.ego.estimator import Track
    from steward.ego.runtime import EgoRuntime, RobotLink

    runtime = EgoRuntime(RobotLink("http://127.0.0.1:1"), steward_bot().to_dict())
    for oid, counts, pos, size in (("e:1", {"table": 0.85, "counter": 0.15}, [1.0, -1.9, 0.38], [1.2, 0.8, 0.75]),
                                   ("e:2", {"table": 0.5, "cabinet": 0.4, "counter": 0.1}, [2.6, 0.6, 0.4], [0.6, 0.5, 0.8])):
        runtime.estimator.tracks[oid] = Track(id=oid, category_counts=dict(counts), pos=np.array(pos), var=np.full(3, 1e-4),
                                              yaw=0.0, yaw_var=1e-3, size=np.array(size), observations=5, first_seen=0.0,
                                              last_seen=0.0, existence=1.0)
    entities = {e["id"]: e for e in runtime._telemetry_scene()["entities"]}
    assert entities["e:2"]["category"]["value"] == "table"
    assert entities["e:2"]["stateVars"]["role"]["value"] == "cabinet" and entities["e:1"]["stateVars"]["role"]["value"] == "table"


def test_a_second_one_of_a_kind_object_is_not_given_the_taken_role():
    corner = _obj("e:132", "counter", {"counter": 0.56, "sideboard": 0.37, "unknown": 0.05, "table": 0.02},
                  [2.94, 2.06, 0.76], [1.21, 0.5, 1.48])
    roles = assign_roles([*SCENE, corner], RoleConfig())
    assert roles["e:40"][0] == "counter" and roles["e:39"][0] == "sideboard"
    assert roles["e:132"][0] == "unknown", "the home's one counter and one sideboard are elsewhere"


def test_roles_stay_put_between_views_unless_the_evidence_clearly_turns():
    a = _obj("e:1", "sideboard", {"sideboard": 0.48, "counter": 0.42, "table": 0.1}, [-2.7, 0.6, 0.43], [1.4, 0.47, 0.84])
    b = _obj("e:2", "sideboard", {"sideboard": 0.46, "counter": 0.44, "table": 0.1}, [0.0, 2.1, 0.46], [1.8, 0.6, 0.9])
    first = {k: v[0] for k, v in assign_roles([a, b], RoleConfig()).items()}
    assert first == {"e:1": "sideboard", "e:2": "counter"}
    a["categories"] = {"sideboard": 0.45, "counter": 0.45, "table": 0.1}
    b["categories"] = {"sideboard": 0.47, "counter": 0.43, "table": 0.1}
    assert {k: v[0] for k, v in assign_roles([a, b], RoleConfig(), first).items()} == first, "a small swing keeps the roles"
    b["categories"] = {"sideboard": 0.9, "counter": 0.05, "table": 0.05}
    assert assign_roles([a, b], RoleConfig(), first)["e:2"][0] == "sideboard", "clear evidence still wins"


def test_a_flat_object_resting_on_a_bowl_is_a_nested_bowl():
    bowl = _obj("e:5", "bowl", {"bowl": 1.0}, [-2.78, 0.41, 0.89], [0.25, 0.25, 0.08])
    rim = {**_obj("e:6", "plate", {"plate": 0.86, "bowl": 0.14}, [-2.8, 0.45, 0.89], [0.17, 0.16, 0.02]), "on": "e:5"}
    loose = _obj("e:7", "plate", {"plate": 0.86, "bowl": 0.14}, [1.0, -1.5, 0.76], [0.24, 0.24, 0.02])
    roles = assign_roles([bowl, rim, loose], RoleConfig())
    assert roles["e:6"][0] == "bowl" and roles["e:7"][0] == "plate"
