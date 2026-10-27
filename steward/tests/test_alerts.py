"""Robot alerts: unexpected exceptions and agent-reported problems reach the operator console."""

from __future__ import annotations

from steward.assets.generators import steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.ego.skills import SkillExecutor
from steward.net import HttpError, call
from steward.world.chat import ChatConfig, HarnessChat


def _state() -> dict:
    return {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "joints": {"lift": 0.6, "reach": 0.1},
            "gripper": {"state": "open", "holding": False}, "entities": {}, "toolOffset": 0.26}


def test_an_unexpected_skill_exception_raises_an_alert_but_an_expected_failure_does_not():
    errors: list[str] = []

    def broken(_cmd: dict) -> None:
        raise AssertionError("encoder disagrees with the plan")

    executor = SkillExecutor(broken, _state, lambda _t: None, on_error=errors.append)
    job = executor.start("look_around", {"id": "look"})
    executor.wait(job.id, 5)
    assert job.state == "failed" and "AssertionError" in job.message
    assert len(errors) == 1 and "AssertionError: encoder disagrees" in errors[0]
    job = executor.start("dance", {"id": "nope"})
    executor.wait(job.id, 5)
    assert job.state == "failed" and len(errors) == 1, "an unknown skill is an expected failure, not an alert"


def test_ego_api_exceptions_and_reported_problems_become_alerts_until_acknowledged():
    runtime = EgoRuntime(RobotLink("http://127.0.0.1:1"), steward_bot().to_dict())

    def explode():
        raise AssertionError("track table corrupted")

    runtime.summary = explode
    api = ego_api(runtime).start()
    try:
        try:
            call("GET", f"{api.url}/ego/summary")
            raise AssertionError("expected a 500")
        except HttpError as exc:
            assert exc.status == 500
        call("POST", f"{api.url}/ego/alert", {"message": "椅子卡在桌子和墙之间，需要人工挪开"})
        call("POST", f"{api.url}/ego/alert", {"message": "椅子卡在桌子和墙之间，需要人工挪开"})
        active = call("GET", f"{api.url}/ego/alerts")["active"]
        assert [a["source"] for a in active] == ["api", "agent"]
        assert active[1]["count"] == 2, "a repeated message is counted, not duplicated"
        assert call("POST", f"{api.url}/ego/alerts/ack", {})["active"] == []
    finally:
        api.stop()


def test_robot_alerts_are_printed_in_the_chat_and_returned_for_the_ego_view(tmp_path):
    runtime = EgoRuntime(RobotLink("http://127.0.0.1:1"), steward_bot().to_dict())
    api = ego_api(runtime).start()
    home = tmp_path / "robot-1"
    (home / "profiles" / "ego").mkdir(parents=True)
    try:
        chat = HarnessChat(ChatConfig(dsh=tmp_path / "dsh", cwd=tmp_path, env_file=tmp_path / ".env", robot_home=home,
                                      ego_api=api.url), lambda: "unused")
        runtime.alert("技能 pick 出现未预料的异常：AssertionError: boom", "skill")
        state = chat.state()
        assert [a["source"] for a in state["agents"]["robot"]["alerts"]] == ["skill"]
        errors = [m for m in state["messages"] if m["role"] == "error"]
        assert len(errors) == 1 and errors[0]["agent"] == "robot" and "AssertionError: boom" in errors[0]["text"]
        assert len([m for m in chat.state()["messages"] if m["role"] == "error"]) == 1, "printed once"
        chat.acknowledge_alerts()
        assert chat.state()["agents"]["robot"]["alerts"] == []
    finally:
        api.stop()


def test_a_published_plan_is_validated_and_shown_with_the_running_skill(tmp_path):
    runtime = EgoRuntime(RobotLink("http://127.0.0.1:1"), steward_bot().to_dict())
    api = ego_api(runtime).start()
    home = tmp_path / "robot-1"
    (home / "profiles" / "ego").mkdir(parents=True)
    try:
        assert call("GET", f"{api.url}/ego/plan")["plan"] is None
        try:
            call("POST", f"{api.url}/ego/plan", {"goal": "整理", "steps": [{"name": "餐桌", "status": "maybe"}]})
            raise AssertionError("expected a 409")
        except HttpError as exc:
            assert exc.status == 409
        call("POST", f"{api.url}/ego/plan", {"goal": "家务循环", "source": "家务循环", "round": 3, "state": "escaping",
                                             "current": "自救：后撤", "difficulty": "连续 3 个阶段失败",
                                             "steps": [{"name": "餐桌", "status": "failed"}, {"name": "抽屉", "status": "pending"}]})
        chat = HarnessChat(ChatConfig(dsh=tmp_path / "dsh", cwd=tmp_path, env_file=tmp_path / ".env", robot_home=home,
                                      ego_api=api.url), lambda: "unused")
        view = chat.state()["agents"]["robot"]["plan"]
        assert view["plan"]["state"] == "escaping" and view["plan"]["round"] == 3
        assert [st["status"] for st in view["plan"]["steps"]] == ["failed", "pending"]
        assert view["skill"] is None
    finally:
        api.stop()
