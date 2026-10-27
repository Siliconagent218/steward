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
import re
import time

from steward.mission import (
    Blocked,
    MissionFailed,
    _approach_to_observe,
    _explore,
    _run,
    _safe_receipt,
    _wait_for_control,
    infeasible,
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


def _offset(support: dict, along: float, across: float) -> list[float]:
    """An offset in the support's frame with ``along`` on its longer horizontal side, whichever
    local axis that is (asset frames and fitted boxes differ)."""
    e = support["extentM"] or support["sizeM"]
    return [along, across] if e[0] >= e[1] else [across, along]


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
        raise infeasible(f"navigate to the front of {furniture['id']}", receipt)
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

    def _clearing(self, phase: str, action):
        """Run ``action``; when it is blocked by movable entities, move each one out of the way
        (at most two) and try again."""
        cleared: set[str] = set()
        while True:
            try:
                return action()
            except Blocked as exc:
                movable = [b for b in exc.blockers if b.get("movable") and b["id"] not in cleared]
                if not movable or len(cleared) >= 2:
                    raise
                self._clear(phase, movable[0])
                cleared.add(movable[0]["id"])

    def _clear(self, phase: str, blocker: dict) -> None:
        """Carry a blocking floor object to a parking spot out of the way."""
        obj = _refresh(self.api, {"id": blocker["id"], "category": blocker["category"]})
        spots = blocker.get("parkAt") or []
        if not spots:
            raise MissionFailed(f"{blocker['id']} ({blocker['category']}) is in the way and there is no free spot to move it to")
        self._step(phase, f"清障：{blocker['category']} {blocker['id']} 挡住了操作路径，先把它搬开", object=blocker["id"])
        pick = _safe_receipt(self.api, "pick", {"target": obj["id"]}, obj, self.log)
        _run(self.api, pick, self.timeout_s)
        for spot in spots:
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "place", "params": {"at": spot}})
            if receipt["withinRiskLimit"]:
                _run(self.api, receipt, self.timeout_s)
                self._step(phase, f"清障完成：{blocker['category']} {blocker['id']} 放到 ({spot[0]:.2f}, {spot[1]:.2f})",
                           object=blocker["id"])
                return
        raise MissionFailed(f"picked {blocker['id']} but none of the parking spots {spots} is reachable")

    def _step(self, phase: str, text: str, **extra) -> None:
        record = {"phase": phase, "text": text, **extra}
        self.log.append(record)
        if self.on_step is not None:
            self.on_step(record)

    # ------------------------------------------------------------------ table
    def arrange_on_table(self) -> list[str]:
        table = _one(_summary(self.api), "table")
        # Plan the row only after a close look: a transparent glass is not detected in every frame.
        _approach_to_observe(self.api, table, distance=1.3)
        time.sleep(2.0)
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
            want = _offset(table, x, TABLE_ROW_Y)
            lx, ly = _local(table, obj["position"])
            if obj["on"] == table["id"] and abs(lx - want[0]) < 0.03 and abs(ly - want[1]) < 0.03:
                continue
            self._clearing("table", lambda o=obj, w=want: _move(self.api, o, {"on": table["id"], "offset": w},
                                                                 table, self.log, self.timeout_s))
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
                    self._clearing("sideboard", lambda b=bowl, o=offset: _move(
                        self.api, b, {"on": board["id"], "offset": _offset(board, o, 0.0)}, board, self.log, self.timeout_s))
                    self._step("sideboard", f"{bowl.get('color') or ''} 碗 → 餐边柜（叠放底座）", object=bowl["id"])
            else:
                below = _refresh(self.api, below)
                if bowl["on"] != below["id"]:
                    self._clearing("sideboard", lambda b=bowl, u=below: _move(
                        self.api, b, {"on": u["id"]}, u, self.log, self.timeout_s))
                    self._step("sideboard", f"{bowl.get('color') or ''} 碗 → 叠进 {below.get('color') or ''} 碗",
                               object=bowl["id"])
            below = bowl
        for obj in others:
            obj = _refresh(self.api, obj)
            offset = next(slot)
            if obj["on"] == board["id"]:
                continue
            self._clearing("sideboard", lambda o=obj, off=offset: _move(
                self.api, o, {"on": board["id"], "offset": _offset(board, off, 0.0)}, board, self.log, self.timeout_s))
            self._step("sideboard", f"{obj['category']} → 餐边柜", object=obj["id"])

    # ----------------------------------------------------------------- drawers
    def _open(self, furniture: dict) -> dict:
        return self._clearing("drawers", lambda: self._open_once(furniture))

    def _open_once(self, furniture: dict) -> dict:
        _view(self.api, furniture, self.timeout_s)
        s = _summary(self.api)
        drawer = _drawer_of(s, furniture)
        if drawer is None:
            raise MissionFailed(f"no drawer seen in front of {furniture['id']}")
        if drawer.get("openingM", 0.0) < 0.2:
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "open", "params": {"target": drawer["id"]}})
            if not receipt["withinRiskLimit"]:
                raise infeasible(f"open {drawer['id']}", receipt)
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
            raise infeasible(f"close {drawer['id']}", receipt)
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
                self._clearing("drawers", lambda o=obj, d=dst: _move(self.api, o, {"on": d["id"]}, d, self.log,
                                                                       self.timeout_s))
                self._step("drawers", f"{obj['category']}：抽屉对调", object=oid, to=dst["id"])
        self._clearing("drawers", lambda: self._close(b))
        self._clearing("drawers", lambda: self._close(a))

    # ---------------------------------------------------------------- recovery
    def recover(self, message: str = "") -> None:
        """After a failed step: put down a held object (a floor-standing one where it is, anything
        else on the table), and look around again only when something was missing from view."""
        try:
            s = _summary(self.api)
            held_id = s["self"].get("holding")
            if held_id:
                held = next((o for o in s["objects"] if o["id"] == held_id), None)
                extent = (held or {}).get("extentM") or [0, 0, 0]
                if held is not None and held["position"][2] - extent[2] / 2 < 0.15:
                    receipt = call("POST", f"{self.api}/ego/predict",
                                   {"skill": "place", "params": {"at": held["position"][:2]}})
                else:
                    table = _one(s, "table")
                    receipt = _safe_receipt(self.api, "place", {"on": table["id"]}, table, self.log)
                if receipt["withinRiskLimit"]:
                    _run(self.api, receipt, self.timeout_s)
            if any(word in message for word in PERCEPTION_FAILURES):
                _explore(self.api, self.timeout_s)
        except (MissionFailed, HttpError) as exc:
            self.log.append({"step": "recover-failed", "error": str(exc)})
            time.sleep(2.0)


PHASES = ("table", "sideboard", "drawers")
PERCEPTION_FAILURES = ("ego state", "no drawer seen", "not seen", "lost")
LIVELOCK_REPEATS = 2
LIVELOCK_WAIT_S = 180.0


def _signature(phase: str, message: str) -> str:
    """A failure with its numbers and ids stripped, to recognise the same failure repeating."""
    return phase + ":" + re.sub(r"[-\d.]+|e:\d+|rcpt-\d+", "#", message)[:200]


def _report_livelock(api: str, text: str, on_wait=None) -> None:
    """Raise an operator alert and wait (without moving) until it is acknowledged or a while passes."""
    try:
        alert = call("POST", f"{api}/ego/alert", {"message": text, "source": "mission", "level": "blocked"})
    except HttpError:
        time.sleep(LIVELOCK_WAIT_S)
        return
    if on_wait is not None:
        on_wait("operator（活锁告警待处理）")
    deadline = time.monotonic() + LIVELOCK_WAIT_S
    while time.monotonic() < deadline:
        active = call("GET", f"{api}/ego/alerts")["active"]
        if all(a["id"] != alert["id"] for a in active):
            return
        time.sleep(3.0)


def chores(api: str, rounds: int, on_step=None, on_phase=None, on_failure=None, on_wait=None,
           timeout_s: float = 180.0) -> dict:
    """Run ``rounds`` chores rounds (``<= 0`` repeats until the process stops).

    ``on_phase(round, phase)`` is called after each completed phase and ``on_failure(round, phase,
    message)`` after a failed one; a failed phase is followed by recovery and the next phase.
    """
    runner = Chores(api, timeout_s, on_step)
    runner.log.append({"step": "explore", "state": _explore(api, timeout_s)["state"]})
    completed = []
    streak: dict[str, int] = {}
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
                runner.recover(str(exc))
                sig = _signature(phase, str(exc))
                streak = {sig: streak.get(sig, 0) + 1}
                if streak[sig] >= LIVELOCK_REPEATS:
                    _report_livelock(api, f"家务循环卡住：「{phase}」阶段连续 {streak[sig]} 次因同一原因失败，"
                                          f"已停止原地重试。原因：{str(exc)[:300]}。请移开障碍或调整任务后点「知道了」继续。",
                                     on_wait)
                    streak = {}
                continue
            streak = {}
            completed.append((index, phase))
            if on_phase is not None:
                on_phase(index, phase)
    return {"completed": completed, "log": runner.log}
