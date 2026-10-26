from __future__ import annotations

import copy

import pytest

from steward.assets.generators import cup
from steward.authoring.service import AuthoringError, AuthoringService, kitchen_spec


def test_publishing_the_same_template_is_idempotent(tmp_path):
    svc = AuthoringService(tmp_path)
    template = cup().to_dict()
    first = svc.publish_asset("cup", template)["version"]
    second = svc.publish_asset("cup", template)["version"]
    assert first == second
    assert [a["name"] for a in svc.library.search(category="cup")] == ["cup"]


def test_authored_template_without_mass_fails_validation(tmp_path):
    svc = AuthoringService(tmp_path)
    template = cup().to_dict()
    link = template["body"]["links"][0]
    link["geometry"] = {"visual": link["geometry"]["collision"]}
    del link["inertial"]
    result = svc.create_asset(template=template)
    assert not result["validation"]["ok"]
    with pytest.raises(AuthoringError):
        svc.publish_asset("ghost-cup", template)


def test_compose_rejects_interpenetrating_furniture(kitchen_root, tmp_path):
    svc = AuthoringService(kitchen_root)
    spec = copy.deepcopy(kitchen_spec())
    spec["name"] = "overlap"
    spec["entities"].append({"id": "w:table-2", "asset": "dining-table", "at": {"x": 0.7, "y": -0.9}})
    with pytest.raises(AuthoringError, match="interpenetrate"):
        svc.compose_scene(spec)


def test_object_placed_on_support_rests_on_its_top(kitchen_scene):
    cup_ent = next(e for e in kitchen_scene["entities"] if e["id"] == "w:cup-1")
    assert cup_ent["pose"]["value"]["p"][2] == pytest.approx(0.901, abs=1e-3)
