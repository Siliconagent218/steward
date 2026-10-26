"""ESim (Phase 0): consequence prediction over the Ego-State, producing Prediction Receipts.

Phase 0 ESim is geometric and probabilistic rather than a physics rollout: it plans the exact
motion the skill would execute, checks the robot footprint and path against ES obstacles, and
converts ES uncertainty into a risk score (for example the probability that the gripper misses
a target whose position is uncertain). The receipt carries that plan; running a receipt executes
the predicted plan, not a fresh one. Physics rollouts with belief sampling arrive in Phase 2.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from steward.ego.planner import (
    Box2D,
    astar,
    build_grid,
    entity_box,
    floor_obstacles,
    path_clearance,
    path_length,
    rects_overlap,
)
from steward.math3d import yaw_from_quat
from steward.scene.schema import Entity, SceneState


@dataclass
class EsimConfig:
    capture_radius: float = 0.05
    receipt_ttl: float = 60.0
    dock_gap: float = 0.07
    predock_backoff: float = 0.35
    reach_min: float = 0.08
    reach_max: float = 0.65
    lift_min: float = 0.32
    lift_max: float = 1.28
    place_edge_margin: float = 0.10
    place_clearance: float = 0.14
    nav_margin: float = 0.03


class PredictionError(ValueError):
    pass


class Esim:
    def __init__(self, robot_model: dict, config: EsimConfig | None = None):
        self.config = config or EsimConfig()
        tree = robot_model["body"]
        base = next(link for link in tree["links"] if link["id"] == "base")
        bx, by, _ = base["geometry"]["collision"][0]["size"]
        self.half_x, self.half_y = bx / 2, by / 2
        self.radius = math.hypot(self.half_x, self.half_y)
        lift = next(j for j in tree["joints"] if j["id"] == "lift")
        grip = next(j for j in tree["joints"] if j["id"] == "gripper_mount")
        self.tool_offset = lift["origin"]["value"]["p"][0] + grip["origin"]["value"]["p"][0]
        self._ids = itertools.count(1)

    # ---------------------------------------------------------------- helpers
    def _self(self, es: SceneState) -> Entity:
        return es.entity(es.self_id)

    def _robot_xy_yaw(self, es: SceneState) -> tuple[float, float, float]:
        me = self._self(es)
        p = me.pose.value
        return p["p"][0], p["p"][1], yaw_from_quat(p["q"])

    def _held(self, es: SceneState) -> Entity | None:
        for rel in es.relations:
            if rel.type == "attached" and rel.b.get("entity") == es.self_id:
                return es.find(rel.a["entity"])
        return None

    @staticmethod
    def _sigma_xy(ent: Entity) -> float:
        cov = (ent.pose.unc or {}).get("cov")
        if not cov:
            return 0.0
        return math.sqrt((cov[0] + cov[7]) / 2)

    @staticmethod
    def _affords(ent: Entity, name: str) -> bool:
        return any(a.value == name for a in ent.rigid.links[0].affordances)

    def _support_of(self, es: SceneState, ent: Entity) -> Entity | None:
        for rel in es.relations:
            if rel.type == "supported-by" and rel.a.get("entity") == ent.id:
                return es.find(rel.b["entity"])
        return None

    def _receipt(self, es: SceneState, skill: str, params: dict, risk: float, issues: list[str], plan: dict,
                 feasible: bool = True) -> dict:
        return {
            "id": f"rcpt-{next(self._ids)}",
            "skill": skill,
            "params": params,
            "esVersion": es.version,
            "createdAt": es.stamp,
            "expiresAt": es.stamp + self.config.receipt_ttl,
            "risk": round(float(min(1.0, max(0.0, risk))), 4),
            "feasible": feasible,
            "issues": issues,
            "plan": plan,
        }

    # -------------------------------------------------------------- approach
    def _approach(self, es: SceneState, support: Entity, target_xy: tuple[float, float], height: float,
                  exclude: set[str]) -> dict:
        """Choose the support side to dock at; return path, dock pose and reach, or raise."""
        sbox, _, _ = entity_box(support)
        rx, ry, _ = self._robot_xy_yaw(es)
        c, s = math.cos(sbox.yaw), math.sin(sbox.yaw)
        dx, dy = target_xy[0] - sbox.cx, target_xy[1] - sbox.cy
        tx, ty = c * dx + s * dy, -s * dx + c * dy
        candidates = [((sbox.hx, ty), (1.0, 0.0), sbox.hx - tx), ((-sbox.hx, ty), (-1.0, 0.0), sbox.hx + tx),
                      ((tx, sbox.hy), (0.0, 1.0), sbox.hy - ty), ((tx, -sbox.hy), (0.0, -1.0), sbox.hy + ty)]
        cfg = self.config
        obstacles_all = floor_obstacles(es, exclude)
        grid = build_grid(obstacles_all, self.radius + cfg.nav_margin, [(rx, ry), target_xy])
        best, reasons = None, []
        if not (cfg.lift_min <= height <= cfg.lift_max):
            raise PredictionError(f"height {height:.2f} m is outside the lift range [{cfg.lift_min}, {cfg.lift_max}]")
        for edge, normal, depth in candidates:
            standoff = self.half_x + cfg.dock_gap
            reach = standoff + depth - self.tool_offset
            if not (cfg.reach_min <= reach <= cfg.reach_max):
                reasons.append(f"side {normal}: reach {reach:.2f} m out of range")
                continue
            nx, ny = c * normal[0] - s * normal[1], s * normal[0] + c * normal[1]
            ex, ey = sbox.cx + c * edge[0] - s * edge[1], sbox.cy + s * edge[0] + c * edge[1]
            dock = (ex + nx * standoff, ey + ny * standoff)
            yaw = math.atan2(-ny, -nx)
            predock = (dock[0] + nx * cfg.predock_backoff, dock[1] + ny * cfg.predock_backoff)
            blocked = self._footprint_blocked(obstacles_all, dock, yaw)
            if blocked:
                reasons.append(f"side {normal}: dock pose collides with {blocked}")
                continue
            blocked = self._turn_blocked(obstacles_all, predock)
            if blocked:
                reasons.append(f"side {normal}: no room to turn at the pre-dock pose ({blocked})")
                continue
            path = astar(grid, (rx, ry), predock)
            if path is None:
                reasons.append(f"side {normal}: no collision-free path to the pre-dock pose")
                continue
            length = path_length(path)
            if best is None or length < best["pathLength"]:
                best = {"path": [list(p) for p in path], "predock": list(predock), "dock": list(dock), "yaw": yaw,
                        "reach": reach, "pathLength": length,
                        "clearance": path_clearance(path, obstacles_all) if len(path) > 1 else math.inf}
        if best is None:
            raise PredictionError("no feasible approach: " + "; ".join(reasons))
        return best

    def _turn_blocked(self, obstacles: list[Box2D], xy: tuple[float, float]) -> str | None:
        """The robot turns in place at the pre-dock pose, sweeping its circumscribed circle."""
        for box in obstacles:
            if box.distance(xy[0], xy[1]) < self.radius + self.config.nav_margin:
                return box.id
        return None

    def _footprint_blocked(self, obstacles: list[Box2D], xy: tuple[float, float], yaw: float) -> str | None:
        me = Box2D("robot", xy[0], xy[1], yaw, self.half_x + 0.01, self.half_y + 0.01)
        for box in obstacles:
            if rects_overlap(me, box):
                return box.id
        return None

    # ----------------------------------------------------------------- skills
    def predict(self, es: SceneState, skill: str, params: dict) -> dict:
        try:
            if skill == "navigate":
                return self._predict_navigate(es, params)
            if skill == "pick":
                return self._predict_pick(es, params)
            if skill == "place":
                return self._predict_place(es, params)
            raise PredictionError(f"unknown skill {skill!r}; known: navigate, pick, place")
        except PredictionError as exc:
            return self._receipt(es, skill, params, 1.0, [str(exc)], {}, feasible=False)

    def _predict_navigate(self, es: SceneState, params: dict) -> dict:
        rx, ry, ryaw = self._robot_xy_yaw(es)
        goal = (float(params["x"]), float(params["y"]))
        yaw = float(params.get("yaw", ryaw))
        obstacles = floor_obstacles(es, set())
        grid = build_grid(obstacles, self.radius + self.config.nav_margin, [(rx, ry), goal])
        blocked = self._footprint_blocked(obstacles, goal, yaw)
        if blocked:
            raise PredictionError(f"goal pose collides with {blocked}")
        path = astar(grid, (rx, ry), goal)
        if path is None:
            raise PredictionError("no collision-free path to the goal")
        clearance = path_clearance(path, obstacles) if len(path) > 1 else math.inf
        issues, risk = [], 0.02
        if clearance < self.radius + 0.05:
            issues.append(f"path passes within {clearance:.2f} m of an obstacle")
            risk = 0.2
        plan = {"path": [list(p) for p in path], "yaw": yaw, "pathLength": path_length(path)}
        return self._receipt(es, "navigate", params, risk, issues, plan)

    def _predict_pick(self, es: SceneState, params: dict) -> dict:
        target = es.find(params.get("target", ""))
        if target is None:
            raise PredictionError(f"unknown target {params.get('target')!r}")
        if self._held(es) is not None:
            raise PredictionError("the gripper is already holding an object")
        if not self._affords(target, "graspable"):
            raise PredictionError(f"{target.id} ({target.category.value}) is not graspable")
        support = self._support_of(es, target)
        if support is None:
            raise PredictionError(f"no support surface is known under {target.id}; observe it first")
        tp = target.pose.value["p"]
        approach = self._approach(es, support, (tp[0], tp[1]), tp[2], {target.id})
        sigma = math.hypot(self._sigma_xy(target), self._sigma_xy(self._self(es)))
        r = self.config.capture_radius
        p_miss = math.exp(-(r * r) / (2 * sigma * sigma)) if sigma > 0 else 0.0
        issues = []
        if p_miss > 0.05:
            issues.append(f"target position uncertainty {sigma * 100:.1f} cm gives {p_miss:.0%} miss probability; "
                          "observe the target more closely first")
        existence_risk = 1.0 - target.existence
        if existence_risk > 0.2:
            issues.append(f"target existence probability is only {target.existence:.2f}")
        risk = 1.0 - (1.0 - p_miss) * (1.0 - existence_risk)
        plan = {**approach, "target": target.id, "support": support.id, "lift": tp[2],
                "carryLift": min(self.config.lift_max, tp[2] + 0.10), "captureRadius": self.config.capture_radius}
        return self._receipt(es, "pick", params, risk, issues, plan)

    def _predict_place(self, es: SceneState, params: dict) -> dict:
        held = self._held(es)
        if held is None:
            raise PredictionError("the gripper is not holding anything")
        support = es.find(params.get("on", ""))
        if support is None:
            raise PredictionError(f"unknown support {params.get('on')!r}")
        if not self._affords(support, "support-surface"):
            raise PredictionError(f"{support.id} ({support.category.value}) is not a support surface")
        sbox, _, top = entity_box(support)
        held_box, hb, ht = entity_box(held)
        held_h = ht - hb
        spot, issues = self._choose_spot(es, support, sbox, held, held_box, params.get("offset"))
        height = top + held_h / 2 + 0.02
        approach = self._approach(es, support, spot, height, {held.id})
        edge = self._edge_distance(sbox, spot) - max(held_box.hx, held_box.hy)
        sigma = self._sigma_xy(support)
        risk = 0.02
        if edge < 2 * sigma + 0.03:
            issues.append(f"placement is {edge * 100:.0f} cm from the edge with {sigma * 100:.1f} cm support uncertainty")
            risk = 0.4
        fragility = (held.rigid.links[0].material.value or {}).get("fragility", 0.0) if held.rigid.links[0].material else 0.0
        risk = 1.0 - (1.0 - risk) * (1.0 - 0.05 * fragility)
        plan = {**approach, "object": held.id, "support": support.id, "spot": list(spot), "lift": height,
                "supportBox": {"cx": sbox.cx, "cy": sbox.cy, "yaw": sbox.yaw, "hx": sbox.hx, "hy": sbox.hy},
                "heldHalf": max(held_box.hx, held_box.hy)}
        return self._receipt(es, "place", params, risk, issues, plan)

    @staticmethod
    def _edge_distance(sbox: Box2D, xy: tuple[float, float]) -> float:
        dx, dy = xy[0] - sbox.cx, xy[1] - sbox.cy
        c, s = math.cos(-sbox.yaw), math.sin(-sbox.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return min(sbox.hx - abs(lx), sbox.hy - abs(ly))

    def _choose_spot(self, es: SceneState, support: Entity, sbox: Box2D, held: Entity, held_box: Box2D,
                     offset: list | None) -> tuple[tuple[float, float], list[str]]:
        cfg = self.config
        others = []
        for rel in es.relations:
            if rel.type == "supported-by" and rel.b.get("entity") == support.id and rel.a["entity"] != held.id:
                ent = es.find(rel.a["entity"])
                if ent is not None:
                    others.append(entity_box(ent)[0])
        c, s = math.cos(sbox.yaw), math.sin(sbox.yaw)

        def world(lx: float, ly: float) -> tuple[float, float]:
            return sbox.cx + c * lx - s * ly, sbox.cy + s * lx + c * ly

        if offset is not None:
            spot = world(float(offset[0]), float(offset[1]))
            if self._edge_distance(sbox, spot) < max(held_box.hx, held_box.hy):
                raise PredictionError("requested offset puts the object over the edge")
            for o in others:
                if math.hypot(spot[0] - o.cx, spot[1] - o.cy) < cfg.place_clearance:
                    raise PredictionError(f"requested spot is occupied by {o.id}")
            return spot, []
        rx, ry, _ = self._robot_xy_yaw(es)
        margin = cfg.place_edge_margin + max(held_box.hx, held_box.hy)
        best, best_cost = None, math.inf
        for lx in np.arange(-sbox.hx + margin, sbox.hx - margin + 1e-9, 0.05):
            for ly in np.arange(-sbox.hy + margin, sbox.hy - margin + 1e-9, 0.05):
                spot = world(float(lx), float(ly))
                if any(math.hypot(spot[0] - o.cx, spot[1] - o.cy) < cfg.place_clearance for o in others):
                    continue
                edge = min(sbox.hx - abs(lx), sbox.hy - abs(ly))
                cost = 0.3 * math.hypot(spot[0] - rx, spot[1] - ry) - 2.0 * edge
                if cost < best_cost:
                    best, best_cost = spot, cost
        if best is None:
            raise PredictionError(f"no free spot on {support.id}")
        return best, []
