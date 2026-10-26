"""2D navigation planning over the Ego-State.

Floor-level ES entities (bottom below ``floor_obstacle_height``) become oriented rectangles on
an occupancy grid, inflated by the robot's circumscribed radius; A* finds an 8-connected path
that is shortened by line-of-sight checks. Docking manoeuvres near furniture use exact
rectangle-overlap checks of the robot footprint instead of the inflated grid.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from steward.math3d import yaw_from_quat
from steward.scene.schema import SceneState


@dataclass
class Box2D:
    id: str
    cx: float
    cy: float
    yaw: float
    hx: float
    hy: float

    def corners(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        r = np.array([[c, -s], [s, c]])
        local = np.array([[self.hx, self.hy], [-self.hx, self.hy], [-self.hx, -self.hy], [self.hx, -self.hy]])
        return local @ r.T + np.array([self.cx, self.cy])

    def contains(self, x: float, y: float, inflate: float = 0.0) -> bool:
        dx, dy = x - self.cx, y - self.cy
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= self.hx + inflate and abs(ly) <= self.hy + inflate

    def distance(self, x: float, y: float) -> float:
        dx, dy = x - self.cx, y - self.cy
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        lx, ly = abs(c * dx - s * dy) - self.hx, abs(s * dx + c * dy) - self.hy
        return math.hypot(max(lx, 0.0), max(ly, 0.0)) - (0.0 if lx > 0 or ly > 0 else min(-lx, -ly))


def entity_box(ent) -> tuple[Box2D, float, float]:
    """Footprint box of a single-link ES entity plus its bottom and top heights."""
    geom = ent.rigid.links[0].geometry["collision"][0]
    sx, sy, sz = geom["size"]
    p = ent.pose.value["p"]
    box = Box2D(ent.id, p[0], p[1], yaw_from_quat(ent.pose.value["q"]), sx / 2, sy / 2)
    return box, p[2] - sz / 2, p[2] + sz / 2


def floor_obstacles(es: SceneState, exclude: set[str], floor_obstacle_height: float = 0.35,
                    min_existence: float = 0.3) -> list[Box2D]:
    out = []
    for ent in es.entities:
        if ent.id == es.self_id or ent.id in exclude or ent.existence < min_existence:
            continue
        box, bottom, _ = entity_box(ent)
        if bottom < floor_obstacle_height:
            out.append(box)
    return out


def rects_overlap(a: Box2D, b: Box2D) -> bool:
    """Separating-axis test for two oriented rectangles."""
    pa, pb = a.corners(), b.corners()
    for poly in (pa, pb):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            proj_a, proj_b = pa @ axis, pb @ axis
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True


@dataclass
class Grid:
    x0: float
    y0: float
    res: float
    occ: np.ndarray

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round((x - self.x0) / self.res)), int(round((y - self.y0) / self.res))

    def point(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + i * self.res, self.y0 + j * self.res

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]

    def free(self, i: int, j: int) -> bool:
        return self.inside(i, j) and not self.occ[i, j]


def build_grid(obstacles: list[Box2D], inflate: float, around: list[tuple[float, float]], res: float = 0.05,
               margin: float = 1.0) -> Grid:
    xs = [p[0] for p in around] + [c for b in obstacles for c in b.corners()[:, 0]]
    ys = [p[1] for p in around] + [c for b in obstacles for c in b.corners()[:, 1]]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    nx, ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    gx = x0 + np.arange(nx) * res
    gy = y0 + np.arange(ny) * res
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    occ = np.zeros((nx, ny), dtype=bool)
    for b in obstacles:
        c, s = math.cos(-b.yaw), math.sin(-b.yaw)
        dx, dy = xx - b.cx, yy - b.cy
        lx, ly = c * dx - s * dy, s * dx + c * dy
        occ |= (np.abs(lx) <= b.hx + inflate) & (np.abs(ly) <= b.hy + inflate)
    return Grid(x0, y0, res, occ)


def astar(grid: Grid, start: tuple[float, float], goal: tuple[float, float], start_clear: float = 0.2) -> list | None:
    si, sj = grid.cell(*start)
    gi, gj = grid.cell(*goal)
    if not grid.inside(si, sj) or not grid.inside(gi, gj) or grid.occ[gi, gj]:
        return None
    r = int(start_clear / grid.res)
    occ = grid.occ.copy()
    occ[max(0, si - r) : si + r + 1, max(0, sj - r) : sj + r + 1] = False
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
    open_heap = [(0.0, 0.0, si, sj)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {(si, sj): 0.0}
    while open_heap:
        _, g, i, j = heapq.heappop(open_heap)
        if (i, j) == (gi, gj):
            path = [(i, j)]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            pts = [grid.point(a, b) for a, b in path]
            pts[0], pts[-1] = tuple(start), tuple(goal)
            return _shortcut(Grid(grid.x0, grid.y0, grid.res, occ), pts)
        if g > cost.get((i, j), math.inf):
            continue
        for di, dj, w in moves:
            ni, nj = i + di, j + dj
            if not (0 <= ni < occ.shape[0] and 0 <= nj < occ.shape[1]) or occ[ni, nj]:
                continue
            ng = g + w
            if ng < cost.get((ni, nj), math.inf):
                cost[(ni, nj)] = ng
                came[(ni, nj)] = (i, j)
                heapq.heappush(open_heap, (ng + math.hypot(gi - ni, gj - nj), ng, ni, nj))
    return None


def _line_free(grid: Grid, a: tuple[float, float], b: tuple[float, float]) -> bool:
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (grid.res / 2)))
    for t in np.linspace(0.0, 1.0, n):
        i, j = grid.cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not grid.free(i, j):
            return False
    return True


def _shortcut(grid: Grid, pts: list) -> list:
    out = [pts[0]]
    k = 0
    while k < len(pts) - 1:
        nxt = len(pts) - 1
        while nxt > k + 1 and not _line_free(grid, pts[k], pts[nxt]):
            nxt -= 1
        out.append(pts[nxt])
        k = nxt
    return out


def path_length(path: list) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:], strict=False))


def path_clearance(path: list, obstacles: list[Box2D], step: float = 0.05) -> float:
    best = math.inf
    for a, b in zip(path, path[1:], strict=False):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for t in np.linspace(0.0, 1.0, n):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            for box in obstacles:
                best = min(best, box.distance(x, y))
    return best
