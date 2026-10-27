"""World State checks for the chores task (Steward plane: reads ground truth, never runs on a robot).

Each check takes a World State (``/wca/scene``) and the console's ground-truth boxes
(``/console/state`` ``groundTruth``) and returns ``(ok, detail)``.
"""

from __future__ import annotations

import math

TABLEWARE = ("bowl", "wine_glass", "mug")


def _category(scene: dict) -> dict[str, str]:
    return {e["id"]: e["category"]["value"] for e in scene["entities"]}


def _support(scene: dict) -> dict[str, dict]:
    return {r["a"]["entity"]: r["b"] for r in scene["relations"] if r["type"] == "supported-by"}


def _of(scene: dict, category: str) -> list[str]:
    return [eid for eid, cat in _category(scene).items() if cat == category]


def _area(boxes: dict[str, dict], eid: str) -> float:
    size = boxes[eid]["size"]
    return size[0] * size[1]


def _pose(scene: dict, eid: str) -> dict:
    return next(e for e in scene["entities"] if e["id"] == eid)["pose"]["value"]


def check_table(scene: dict, boxes: dict[str, dict]) -> tuple[bool, str]:
    """All tableware rests directly on the table, ordered along the table's long axis by size."""
    table = _of(scene, "table")[0]
    items = [eid for cat in TABLEWARE for eid in _of(scene, cat)]
    support = _support(scene)
    off = [eid for eid in items if support.get(eid, {}).get("entity") != table]
    t = _pose(scene, table)
    yaw = 2 * math.atan2(t["q"][3], t["q"][0])

    def along(eid: str) -> float:
        p = _pose(scene, eid)["p"]
        return (p[0] - t["p"][0]) * math.cos(yaw) + (p[1] - t["p"][1]) * math.sin(yaw)

    by_size = sorted(items, key=lambda e: _area(boxes, e))
    by_place = sorted(items, key=along)
    ok = not off and by_size == by_place
    detail = "顺序 " + " < ".join(e.removeprefix("w:") for e in by_place)
    return ok, detail + (f"；不在桌上：{off}" if off else "")


def check_sideboard(scene: dict, boxes: dict[str, dict]) -> tuple[bool, str]:
    """Bowls nest largest to smallest on the sideboard; the other tableware stands on it."""
    board = _of(scene, "sideboard")[0]
    support = _support(scene)
    bowls = sorted(_of(scene, "bowl"), key=lambda e: _area(boxes, e), reverse=True)
    others = [eid for cat in TABLEWARE if cat != "bowl" for eid in _of(scene, cat)]
    problems = []
    for i, bowl in enumerate(bowls):
        want = board if i == 0 else bowls[i - 1]
        if support.get(bowl, {}).get("entity") != want:
            problems.append(f"{bowl} 在 {support.get(bowl, {}).get('entity')} 上，应在 {want}")
    for eid in others:
        if support.get(eid, {}).get("entity") != board or support[eid].get("link") == "drawer":
            problems.append(f"{eid} 不在餐边柜台面上")
    chain = " ⊃ ".join(b.removeprefix("w:") for b in bowls)
    return not problems, f"叠放 {chain}" + (f"；{'；'.join(problems)}" if problems else "")


def drawer_contents(scene: dict) -> dict[str, set[str]]:
    """Objects resting in each drawer, keyed by the furniture that holds the drawer."""
    out: dict[str, set[str]] = {}
    for item, b in _support(scene).items():
        if b.get("link") == "drawer":
            out.setdefault(b["entity"], set()).add(item)
    return out


def drawers_closed(scene: dict, tolerance: float = 0.02) -> bool:
    for ent in scene["entities"]:
        for joint in ent["body"].get("joints", []):
            if joint["id"] == "drawer_slide" and joint["q"]["value"] and joint["q"]["value"][0] > tolerance:
                return False
    return True


def check_swap(before: dict[str, set[str]], scene: dict) -> tuple[bool, str]:
    """The sideboard and cabinet drawers exchanged their contents and both are closed."""
    after = drawer_contents(scene)
    board, cabinet = _of(scene, "sideboard")[0], _of(scene, "cabinet")[0]
    swapped = after.get(board, set()) == before.get(cabinet, set()) and after.get(cabinet, set()) == before.get(board, set())
    closed = drawers_closed(scene)

    def names(s: set[str]) -> str:
        return "、".join(sorted(e.removeprefix("w:") for e in s)) or "空"

    detail = f"餐边柜抽屉：{names(after.get(board, set()))}；柜子抽屉：{names(after.get(cabinet, set()))}"
    return swapped and closed, detail + ("" if closed else "；抽屉没有关好")
