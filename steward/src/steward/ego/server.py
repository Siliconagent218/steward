"""Ego API: the robot-local HTTP interface the Ego Harness tools call.

Every response is derived from the Ego-State and the robot's own receipts and jobs. A client may
take exclusive control (``/ego/control``); skill and explore requests then carry ``owner``.
"""

from __future__ import annotations

from steward.ego.runtime import EgoRuntime
from steward.net import HttpError, JsonServer


def ego_api(runtime: EgoRuntime, port: int = 0) -> JsonServer:
    def guarded(fn):
        def wrapped(q, b):
            try:
                return fn(q, b or {})
            except KeyError as exc:
                raise HttpError(404, str(exc).strip("'")) from None
            except (PermissionError, RuntimeError, ValueError) as exc:
                raise HttpError(409, str(exc)) from None
        return wrapped

    def control(b):
        if b.get("action") == "acquire":
            return runtime.acquire_control(b.get("owner", ""))
        if b.get("action") == "release":
            return runtime.release_control(b.get("owner", ""))
        raise ValueError("action must be acquire or release")

    routes = {
        ("GET", "/ego/summary"): lambda q, b: runtime.summary(),
        ("GET", "/ego/state"): lambda q, b: runtime.ego_state().to_dict(),
        ("GET", "/ego/query"): lambda q, b: {"objects": runtime.query(
            q.get("category"), q.get("near"), float(q["radius"]) if "radius" in q else None)},
        ("POST", "/ego/observe"): lambda q, b: runtime.observe(),
        ("POST", "/ego/explore"): lambda q, b: runtime.explore(b.get("owner")),
        ("GET", "/ego/control"): lambda q, b: runtime.control(),
        ("POST", "/ego/control"): lambda q, b: control(b),
        ("POST", "/ego/predict"): lambda q, b: runtime.predict(b["skill"], b.get("params", {})),
        ("GET", "/ego/receipt"): lambda q, b: runtime.receipt(q["id"]),
        ("POST", "/skill/run"): lambda q, b: runtime.run(b["receipt"], b.get("owner")),
        ("GET", "/skill/status"): lambda q, b: runtime.job(q["job"]),
        ("POST", "/skill/wait"): lambda q, b: runtime.wait(b["job"], float(b.get("timeoutS", 120))),
        ("POST", "/skill/stop"): lambda q, b: runtime.stop_motion(),
    }
    return JsonServer({k: guarded(v) for k, v in routes.items()}, port=port, name="ego-api")
