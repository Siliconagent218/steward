"""Scripted Phase 0 mission over the Ego API: put the cup on the table.

The script uses exactly the operations the Ego Harness tools expose (explore, summary, predict,
run, wait), so it doubles as an end-to-end check of the Ego API without a language model.
"""

from __future__ import annotations

import math
import time

from steward.net import call


class MissionFailed(RuntimeError):
    pass


def _find(summary: dict, category: str, min_probability: float = 0.25) -> dict | None:
    """The most likely object of a category, using the full category distribution so that an
    object briefly mislabelled by perception is still found."""
    cands = [o for o in summary["objects"]
             if o["categories"].get(category, 0.0) >= min_probability and o["existence"] >= 0.5]
    return max(cands, key=lambda o: (o["categories"][category], -o["uncertaintyM"])) if cands else None


def _predict_until_safe(api: str, skill: str, params: dict, attempts: int = 5, settle_s: float = 1.0) -> dict:
    for _ in range(attempts):
        receipt = call("POST", f"{api}/ego/predict", {"skill": skill, "params": params})
        if receipt["withinRiskLimit"]:
            return receipt
        if not receipt["feasible"]:
            raise MissionFailed(f"{skill} infeasible: {receipt['issues']}")
        time.sleep(settle_s)
    raise MissionFailed(f"{skill} stayed above the risk limit: {receipt['risk']} {receipt['issues']}")


def _approach_to_observe(api: str, target: dict, distance: float = 1.1, timeout_s: float = 120.0) -> dict | None:
    """Drive to a floor pose about ``distance`` from the target, facing it, to observe it closely."""
    me = call("GET", f"{api}/ego/summary")["self"]
    tx, ty = target["position"][:2]
    base = math.atan2(me["y"] - ty, me["x"] - tx)
    for offset in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6):
        ang = base + offset
        x, y = tx + distance * math.cos(ang), ty + distance * math.sin(ang)
        receipt = call("POST", f"{api}/ego/predict",
                       {"skill": "navigate", "params": {"x": x, "y": y, "yaw": math.atan2(ty - y, tx - x)}})
        if receipt["withinRiskLimit"]:
            _run(api, receipt, timeout_s)
            time.sleep(1.5)
            return receipt
    return None


def _run(api: str, receipt: dict, timeout_s: float) -> dict:
    job = call("POST", f"{api}/skill/run", {"receipt": receipt["id"]})
    job = call("POST", f"{api}/skill/wait", {"job": job["id"], "timeoutS": timeout_s}, timeout=timeout_s + 10)
    if job["state"] != "succeeded":
        raise MissionFailed(f"{receipt['skill']} {job['state']}: {job['message']}")
    return job


def cup_to_table(api: str, timeout_s: float = 180.0) -> dict:
    log = []
    job = call("POST", f"{api}/ego/explore")
    job = call("POST", f"{api}/skill/wait", {"job": job["id"], "timeoutS": timeout_s}, timeout=timeout_s + 10)
    log.append({"step": "explore", "state": job["state"]})
    summary = call("GET", f"{api}/ego/summary")
    cup, table = _find(summary, "cup"), _find(summary, "table")
    if cup is None or table is None:
        raise MissionFailed(f"after exploring, cup={cup is not None} table={table is not None}")
    log.append({"step": "found", "cup": cup["id"], "table": table["id"], "cupUncertaintyM": cup["uncertaintyM"]})
    first = call("POST", f"{api}/ego/predict", {"skill": "pick", "params": {"target": cup["id"]}})
    if not first["withinRiskLimit"]:
        moved = _approach_to_observe(api, cup)
        cup = next(o for o in call("GET", f"{api}/ego/summary")["objects"] if o["id"] == cup["id"])
        log.append({"step": "observe-closer", "navigated": moved is not None, "cupUncertaintyM": cup["uncertaintyM"]})
    pick = _predict_until_safe(api, "pick", {"target": cup["id"]})
    log.append({"step": "predict-pick", "receipt": pick["id"], "risk": pick["risk"]})
    log.append({"step": "pick", "message": _run(api, pick, timeout_s)["message"]})
    place = _predict_until_safe(api, "place", {"on": table["id"]})
    log.append({"step": "predict-place", "receipt": place["id"], "risk": place["risk"], "spot": place["plan"]["spot"]})
    log.append({"step": "place", "message": _run(api, place, timeout_s)["message"]})
    return {"log": log, "cup": cup["id"], "table": table["id"], "summary": call("GET", f"{api}/ego/summary")}
