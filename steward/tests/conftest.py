from __future__ import annotations

import pytest

from steward.authoring.service import AuthoringService, build_default_kitchen


@pytest.fixture(scope="session")
def kitchen_root(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("steward")
    build_default_kitchen(AuthoringService(root))
    return str(root)


@pytest.fixture
def kitchen_scene(kitchen_root) -> dict:
    return AuthoringService(kitchen_root).load_scenario("phase0-kitchen")["sceneState"]
