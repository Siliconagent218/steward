"""Self-diagnosis: failure messages sorted into causes, with findings from the robot's own records."""

from __future__ import annotations

import pytest

from steward.diagnosis import classify_failure, diagnose


@pytest.mark.parametrize(("message", "cause"), [
    ("no table in the ego state", "perception-missing"),
    ("pick infeasible: ['no support surface is known under e:6; observe it first']", "support-unknown"),
    ("pick stayed above the risk limit: 0.77 ['target existence probability is only 0.23']", "low-confidence"),
    ("open failed: missed the handle of e:17", "grasp-miss"),
    ("navigate failed: turn did not converge (yaw error 0.14 rad)", "base-blocked"),
    ("place failed: joint lift did not reach 0.836 (at 0.945)", "arm-blocked"),
    ("e:91 (chair) is in the way and there is no free spot to move it to", "no-space"),
    ("cannot open e:10: ['dock pose in front of e:10 collides with e:5']", "path-blocked"),
    ("something nobody anticipated", "unknown"),
])
def test_failure_messages_are_sorted_into_causes(message, cause):
    assert classify_failure(message).key == cause


def _obj(oid, category, xy, extent, confidence=1.0, categories=None):
    return {"id": oid, "category": category, "existence": 1.0, "position": [xy[0], xy[1], 0.4], "extentM": extent,
            "sizeM": extent, "yawDeg": 0.0, "categoryConfidence": confidence, "categories": categories or {category: confidence}}


def test_diagnosis_names_the_dominant_cause_and_flags_duplicate_and_doubtful_tracks():
    summary = {"self": {"x": 1.0, "y": 0.0, "yawDeg": 0.0, "holding": None}, "objects": [
        _obj("e:1", "table", (1.0, -1.9), [1.2, 0.8, 0.75]),
        _obj("e:22", "table", (-2.6, 0.6), [1.39, 0.77, 0.85]),
        _obj("e:40", "sideboard", (-2.7, 0.6), [1.4, 0.47, 0.84], 0.46, {"sideboard": 0.46, "counter": 0.33, "table": 0.21}),
    ]}
    failures = [{"message": "no sideboard in the ego state", "pose": [1.0, 0.0]},
                {"message": "open failed: missed the handle of e:30", "pose": [-1.5, 0.6]},
                {"message": "pick infeasible: ['no support surface is known under e:6']", "pose": [0.0, 0.0]},
                {"message": "no sideboard in the ego state", "pose": [1.0, 0.1]}]
    diagnosis = diagnose(failures, summary, {"table": 1, "sideboard": 1})
    assert diagnosis.cause.key == "perception-missing" and (diagnosis.explained, diagnosis.total) == (2, 4)
    text = diagnosis.text
    assert "失败原因分布" in text and "基本没有移动" not in text
    assert "有 2 个 table" in text and "e:22 在 (-2.60, 0.60)" in text
    assert "e:40 被认作 sideboard 的把握只有 0.46（其次 counter 0.33、table 0.21）" in text
    assert text.splitlines()[-1].startswith("建议：")


def test_diagnosis_works_from_the_failure_log_alone():
    diagnosis = diagnose([{"message": "navigate failed: base is blocked", "pose": None}], None, {"table": 1})
    assert diagnosis.cause.key == "base-blocked" and "依据" not in diagnosis.text
