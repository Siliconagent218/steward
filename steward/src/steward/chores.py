"""Scripted chores over the Ego API, repeated round after round.

One round has three phases:

1. **table**: the tableware (bowls, wine glass, mug) stands in one row on the dining table,
   ordered from the smallest to the largest footprint;
2. **sideboard**: the tableware moves to the sideboard top; bowls nest largest to smallest in
   one stack, the other pieces stand in a row beside it (pieces already in place stay);
3. **drawers**: both drawers are opened, the contents of the sideboard drawer and the cabinet
   drawer are exchanged, and both drawers are closed.

Everything is located in the Ego-State by category, colour and estimated size; placements use
explicit offsets in the support's own frame, and every motion goes through a Prediction Receipt.
"""

from __future__ import annotations

import itertools
import math
import time

from steward.mission import (
    MissionFailed,
    _approach_to_observe,
    _explore,
    _run,
    _safe_receipt,
    _wait_for_control,
)
from steward.net import HttpError, call

TABLEWARE = ("bowl", "wine_glass", "mug")
ROW_GAP = 0.06
TABLE_ROW_Y = 0.18
VIEW_DISTANCE = 1.1


def _summary(api: str) -> dict:
    return call("GET", f"{api}/ego/summary")


def _objects(summary: dict, category: str) -> list[dict]:
    return [o for o in summary["objects"] if o["category"] == category and o["existence"] >= 0.5]


def _one(summary: dict, category: str) -> dict:
    found = _objects(summary, category)
    if not found:
        raise MissionFailed(f"no {category} in the ego state")
    return max(found, key=lambda o: o["existence"])


def _area(obj: dict) -> float:
    e = obj["extentM"] or obj["sizeM"]
    return e[0] * e[1]


def _width(obj: dict) -> float:
    e = obj["extentM"] or obj["sizeM"]
    return max(e[0], e[1])


def tableware(summary: dict) -> list[dict]:
    """Tableware tracks ordered from the smallest to the largest footprint."""
    items = [o for cat in TABLEWARE for o in _objects(summary, cat)]
    return sorted(items, key=_area)


def row_offsets(widths: list[float], gap: float = ROW_GAP) -> list[float]:
    """Centres of a row of items with the given widths, centred on 0."""
    total = sum(widths) + gap * (len(widths) - 1)
    out, x = [], -total / 2
    for w in widths:
        out.append(x + w / 2)
        x += w + gap
    return out


def _local(support: dict, xy: list[float]) -> tuple[float, float]:
    yaw = math.radians(support["yawDeg"])
    dx, dy = xy[0] - support["position"][0], xy[1] - support["position"][1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


def _refresh(api: str, obj: dict) -> dict:
    found = next((o for o in _summary(api)["objects"] if o["id"] == obj["id"]), None)
    if found is None:
        raise MissionFailed(f"{obj['id']} ({obj['category']}) is no longer in the ego state")
    return found


def _move(api: str, obj: dict, place: dict, observe: dict, log: list, timeout_s: float, attempts: int = 2) -> dict:
    """Pick ``obj`` (observing it again and retrying after a missed grasp) and place it with
    ``place`` params; return the place job."""
    for attempt in range(attempts):
        pick = _safe_receipt(api, "pick", {"target": obj["id"]}, obj, log)
        try:
            _run(api, pick, timeout_s)
            break
        except MissionFailed as exc:
            if attempt + 1 == attempts or "missed" not in str(exc) and "not closing" not in str(exc):
                raise
            log.append({"step": "pick-retry", "object": obj["id"], "error": str(exc)})
            _approach_to_observe(api, obj)
            obj = _refresh(api, obj)
    receipt = _safe_receipt(api, "place", place, observe, log)
    job = _run(api, receipt, timeout_s)
    log.append({"step": "place", "object": obj["id"], **place})
    time.sleep(0.8)
    return job


def _view(api: str, furniture: dict, timeout_s: float) -> None:
    """Stand in front of a piece of furniture, facing it, to see its drawer up close."""
    yaw = math.radians(furniture["yawDeg"])
    n = (math.cos(yaw), math.sin(yaw))
    half = (furniture["extentM"] or furniture["sizeM"])[0] / 2
    x = furniture["position"][0] + n[0] * (half + VIEW_DISTANCE)
    y = furniture["position"][1] + n[1] * (half + VIEW_DISTANCE)
    receipt = call("POST", f"{api}/ego/predict",
                   {"skill": "navigate", "params": {"x": x, "y": y, "yaw": math.atan2(-n[1], -n[0])}})
    if not receipt["withinRiskLimit"]:
        raise MissionFailed(f"cannot reach the front of {furniture['id']}: {receipt['issues']}")
    _run(api, receipt, timeout_s)
    time.sleep(1.5)


def _drawer_of(summary: dict, furniture: dict) -> dict | None:
    drawers = _objects(summary, "drawer")
    near = [d for d in drawers if math.dist(d["position"][:2], furniture["position"][:2]) < 0.6]
    return min(near, key=lambda d: math.dist(d["position"][:2], furniture["position"][:2])) if near else None


class Chores:
    """Runs the chores phases against one robot's Ego API."""

    def __init__(self, api: str, timeout_s: float = 180.0, on_step=None):
        self.api = api
        self.timeout_s = timeout_s
        self.on_step = on_step
        self.log: list[dict] = []

    def _step(self, phase: str, text: str, **extra) -> None:
        record = {"phase": phase, "text": text, **extra}
        self.log.append(record)
        if self.on_step is not None:
            self.on_step(record)

    # ------------------------------------------------------------------ table
    def arrange_on_table(self) -> list[str]:
        s = _summary(self.api)
        table = _one(s, "table")
        items = tableware(s)
        if len(items) < 2:
            raise MissionFailed(f"only {len(items)} pieces of tableware are known")
        offsets = row_offsets([_width(o) for o in items])
        order = []
        for obj, x in zip(items, offsets, strict=True):
            obj = _refresh(self.api, obj)
            order.append(obj["id"])
            lx, ly = _local(table, obj["position"])
            if obj["on"] == table["id"] and abs(lx - x) < 0.03 and abs(ly - TABLE_ROW_Y) < 0.03:
                continue
            _move(self.api, obj, {"on": table["id"], "offset": [x, TABLE_ROW_Y]}, table, self.log, self.timeout_s)
            self._step("table", f"{obj.get('color') or ''} {obj['category']} → 餐桌第 {len(order)} 位", object=obj["id"])
        return order

    # -------------------------------------------------------------- sideboard
    def stack_on_sideboard(self) -> None:
        s = _summary(self.api)
        board = _one(s, "sideboard")
        bowls = sorted(_objects(s, "bowl"), key=_area, reverse=True)
        others = sorted([o for cat in TABLEWARE if cat != "bowl" for o in _objects(s, cat)], key=_area, reverse=True)
        widths = ([_width(bowls[0])] if bowls else []) + [_width(o) for o in others]
        offsets = row_offsets(widths, gap=0.08)
        slot = iter(offsets)
        below = None
        for bowl in bowls:
            bowl = _refresh(self.api, bowl)
            if below is None:
                offset = next(slot)
                if bowl["on"] != board["id"]:
                    _move(self.api, bowl, {"on": board["id"], "offset": [0.0, offset]}, board, self.log, self.timeout_s)
                    self._step("sideboard", f"{bowl.get('color') or ''} 碗 → 餐边柜（叠放底座）", object=bowl["id"])
            else:
                below = _refresh(self.api, below)
                if bowl["on"] != below["id"]:
                    _move(self.api, bowl, {"on": below["id"]}, below, self.log, self.timeout_s)
                    self._step("sideboard", f"{bowl.get('color') or ''} 碗 → 叠进 {below.get('color') or ''} 碗",
                               object=bowl["id"])
            below = bowl
        for obj in others:
            obj = _refresh(self.api, obj)
            offset = next(slot)
            if obj["on"] == board["id"]:
                continue
            _move(self.api, obj, {"on": board["id"], "offset": [0.0, offset]}, board, self.log, self.timeout_s)
            self._step("sideboard", f"{obj['category']} → 餐边柜", object=obj["id"])

    # ----------------------------------------------------------------- drawers
    def _open(self, furniture: dict) -> dict:
        _view(self.api, furniture, self.timeout_s)
        s = _summary(self.api)
        drawer = _drawer_of(s, furniture)
        if drawer is None:
            raise MissionFailed(f"no drawer seen in front of {furniture['id']}")
        if drawer.get("openingM", 0.0) < 0.2:
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "open", "params": {"target": drawer["id"]}})
            if not receipt["withinRiskLimit"]:
                raise MissionFailed(f"cannot open {drawer['id']}: {receipt['issues']}")
            _run(self.api, receipt, self.timeout_s)
            time.sleep(1.5)
            self._step("drawers", f"打开 {furniture['category']} 的抽屉", object=drawer["id"])
        return _refresh(self.api, drawer)

    def _close(self, drawer: dict) -> None:
        drawer = _refresh(self.api, drawer)
        if drawer.get("openingM", 0.0) < 0.05:
            return
        receipt = call("POST", f"{self.api}/ego/predict", {"skill": "close", "params": {"target": drawer["id"]}})
        if not receipt["withinRiskLimit"]:
            raise MissionFailed(f"cannot close {drawer['id']}: {receipt['issues']}")
        _run(self.api, receipt, self.timeout_s)
        self._step("drawers", "关上抽屉", object=drawer["id"])

    def swap_drawers(self) -> None:
        s = _summary(self.api)
        board, cabinet = _one(s, "sideboard"), _one(s, "cabinet")
        a = self._open(board)
        a_items = [o["id"] for o in _summary(self.api)["objects"] if o["on"] == a["id"]]
        b = self._open(cabinet)
        b_items = [o["id"] for o in _summary(self.api)["objects"] if o["on"] == b["id"]]
        for items, src, dst in ((a_items, a, b), (b_items, b, a)):
            for oid in items:
                obj = _refresh(self.api, {"id": oid, "category": "item"})
                dst = _refresh(self.api, dst)
                _move(self.api, obj, {"on": dst["id"]}, dst, self.log, self.timeout_s)
                self._step("drawers", f"{obj['category']}：抽屉对调", object=oid, to=dst["id"])
        self._close(b)
        self._close(a)

    # ---------------------------------------------------------------- recovery
    def recover(self) -> None:
        """After a failed step: put down a held object on the table, then look around again."""
        try:
            s = _summary(self.api)
            if s["self"].get("holding"):
                table = _one(s, "table")
                receipt = _safe_receipt(self.api, "place", {"on": table["id"]}, table, self.log)
                _run(self.api, receipt, self.timeout_s)
            _explore(self.api, self.timeout_s)
        except (MissionFailed, HttpError) as exc:
            self.log.append({"step": "recover-failed", "error": str(exc)})
            time.sleep(2.0)


PHASES = ("table", "sideboard", "drawers")


def chores(api: str, rounds: int, on_step=None, on_phase=None, on_failure=None, on_wait=None,
           timeout_s: float = 180.0) -> dict:
    """Run ``rounds`` chores rounds (``<= 0`` repeats until the process stops).

    ``on_phase(round, phase)`` is called after each completed phase and ``on_failure(round, phase,
    message)`` after a failed one; a failed phase is followed by recovery and the next phase.
    """
    runner = Chores(api, timeout_s, on_step)
    runner.log.append({"step": "explore", "state": _explore(api, timeout_s)["state"]})
    completed = []
    for index in (range(rounds) if rounds > 0 else itertools.count()):
        for phase in PHASES:
            _wait_for_control(api, on_wait)
            try:
                {"table": runner.arrange_on_table, "sideboard": runner.stack_on_sideboard,
                 "drawers": runner.swap_drawers}[phase]()
            except (MissionFailed, HttpError) as exc:
                if on_failure is not None:
                    on_failure(index, phase, str(exc))
                if rounds > 0:
                    raise
                _wait_for_control(api, on_wait)
                runner.recover()
                continue
            completed.append((index, phase))
            if on_phase is not None:
                on_phase(index, phase)
    return {"completed": completed, "log": runner.log}
