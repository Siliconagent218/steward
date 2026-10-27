"""Ego Runtime (ER): perception loop, Ego-State store, ESim receipts, skill execution and the
operator view of the Ego-State (rendered on its own thread, see :mod:`steward.viewloop`).

The runtime talks to the world only through the robot port (sensors, actuator commands,
telemetry). It never holds World State identifiers or ground truth.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from steward.ego.classify import classify
from steward.ego.perception import box_of, decode_rgbd, segments
from steward.ego.esim import Esim, EsimConfig
from steward.ego.estimator import SELF_ID, EgoEstimator
from steward.ego.skills import MotionConfig, SkillExecutor
from steward.ego.view import EgoViewRenderer
from steward.math3d import yaw_from_quat
from steward.net import HttpError, call
from steward.scene.schema import SceneState
from steward.viewloop import ViewLoop


@dataclass
class RuntimeConfig:
    perception_hz: float = 10.0
    telemetry_hz: float = 2.0
    max_risk: float = 0.35
    esim: EsimConfig = field(default_factory=EsimConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)


class RobotLink:
    """HTTP client of one robot port: the only path between the Ego System and the world."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def sensors(self) -> dict:
        return call("GET", f"{self.url}/sensors")

    def rgbd(self) -> bytes:
        """One RGB-D frame (``npz``) when the world provides raw imaging instead of detections."""
        return call("GET", f"{self.url}/sensors/rgbd")

    def command(self, cmd: dict) -> None:
        call("POST", f"{self.url}/command", cmd)

    def telemetry(self, scene: dict) -> None:
        call("POST", f"{self.url}/telemetry", scene)

    def camera_png(self, width: int = 320, height: int = 240) -> bytes:
        return call("GET", f"{self.url}/camera/rgb?width={width}&height={height}")


ALERT_TEXT_MAX = 1500
PLAN_STEP_STATES = frozenset({"pending", "active", "done", "failed", "skipped"})

class EgoRuntime:
    def __init__(self, link: RobotLink, robot_model: dict, config: RuntimeConfig | None = None,
                 out_dir: Path | None = None):
        self.link = link
        self.config = config or RuntimeConfig()
        self.estimator = EgoEstimator(robot_model=robot_model)
        self.esim = Esim(robot_model, self.config.esim)
        self.out_dir = out_dir
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.receipts: dict[str, dict] = {}
        self.last_error: str | None = None
        self.control_owner: str | None = None
        self.perception_mode: str | None = None
        self.alerts: list[dict] = []
        self.plan: dict | None = None
        self._alert_ids = itertools.count(1)
        self.executor = SkillExecutor(self.link.command, self._motion_state, self._mark_held, self.config.motion,
                                      shift=self._shift_track, on_error=lambda msg: self.alert(msg, "skill"))
        renderer = EgoViewRenderer()
        self.view = ViewLoop("ego", lambda settings: (
            renderer.render(self.ego_state().to_dict(), settings), "image/jpeg",
            {"esVersion": self.estimator.version, "t": round(self.estimator.stamp, 3)}), renderer.close)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start perception in a worker thread. The first frame is taken there too: RGB-D frames
        are rendered on the world's main thread, which may be the caller."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ego-perception", daemon=True)
        self._thread.start()
        self.view.start()

    def stop(self) -> None:
        self._running = False
        self.view.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _poll_once(self) -> None:
        frame = self.link.sensors()
        mode = "rgbd" if frame.get("imaging") == "rgbd" else "detections"
        if mode != self.perception_mode:
            with self._lock:
                if self.perception_mode is not None:
                    # The sensors changed kind (a phase switch): objects are perceived afresh with
                    # the new pipeline; what the gripper holds and the running skill are kept.
                    self.estimator.forget_objects()
                self.perception_mode = mode
        if mode == "rgbd":
            rgbd = decode_rgbd(self.link.rgbd())
            with self._lock:
                if rgbd["header"]["stamp"] < self.estimator.stamp - 1e-6:
                    self._restart()
                self.estimator.update_rgbd(rgbd)
            return
        with self._lock:
            if frame["stamp"] < self.estimator.stamp - 1e-6:
                self._restart()
            self.estimator.update(frame)

    def _restart(self) -> None:
        """The world clock went backwards: a new episode started, which the robot experiences as a
        reboot. Forget the Ego-State, receipts and any running skill."""
        self.executor.cancel()
        self.estimator = EgoEstimator(robot_model=self.estimator.robot_model, config=self.estimator.config)
        self.receipts.clear()

    def _loop(self) -> None:
        period = 1.0 / self.config.perception_hz
        telemetry_every = max(1, int(self.config.perception_hz / self.config.telemetry_hz))
        k = 0
        while self._running:
            t0 = time.monotonic()
            try:
                self._poll_once()
                k += 1
                if k % telemetry_every == 0:
                    self.link.telemetry(self.ego_state().to_dict())
                self.last_error = None
            except (HttpError, OSError) as exc:
                self.last_error = f"robot port unavailable: {exc}"
            except Exception as exc:  # noqa: BLE001 - perception must keep running; the operator is alerted
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.alert(f"感知循环出现异常：{type(exc).__name__}: {exc}", "runtime")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    # --------------------------------------------------------------- state
    def ego_state(self) -> SceneState:
        with self._lock:
            return self.estimator.to_scene()

    def _motion_state(self) -> dict:
        with self._lock:
            est = self.estimator
            entities = {t.id: {"p": t.pos.tolist()} for t in est.tracks.values()}
            return {"stamp": est.stamp, "odometry": dict(est.odometry), "joints": dict(est.joints),
                    "gripper": dict(est.gripper), "entities": entities, "toolOffset": self.esim.tool_offset}

    def _mark_held(self, track_id: str) -> None:
        with self._lock:
            self.estimator.mark_held(track_id)

    def _shift_track(self, track_id: str, delta: list[float]) -> None:
        with self._lock:
            self.estimator.shift(track_id, delta)

    # ------------------------------------------------------------- Ego API
    def summary(self) -> dict:
        es = self.ego_state()
        me = es.entity(SELF_ID)
        p = me.pose.value
        held = next((r.a["entity"] for r in es.relations if r.type == "attached"), None)
        supports = {r.a["entity"]: r.b["entity"] for r in es.relations if r.type == "supported-by"}
        objects = []
        for ent in es.entities:
            if ent.id == SELF_ID:
                continue
            cov = ent.pose.unc["cov"]
            size = ent.rigid.links[0].geometry["collision"][0]["size"]
            objects.append({
                "id": ent.id,
                "category": ent.category.value,
                "categoryConfidence": round(ent.category.unc["p"].get(ent.category.value, 1.0), 3),
                "categories": ent.category.unc["p"],
                "position": [round(v, 3) for v in ent.pose.value["p"]],
                "yawDeg": round(math.degrees(yaw_from_quat(ent.pose.value["q"])), 1),
                "uncertaintyM": round(math.sqrt((cov[0] + cov[7]) / 2), 4),
                "sizeM": [round(v, 3) for v in size],
                "on": supports.get(ent.id),
                "existence": ent.existence,
                "lastSeenAgoS": round(es.stamp - (ent.pose.t or es.stamp), 2),
                "affordances": [a.value for a in ent.rigid.links[0].affordances],
                "color": ent.state_vars["color"].value if "color" in ent.state_vars else None,
                "extentM": self._extent(ent.id),
                **({"openingM": round(self._opening(ent.id), 3)} if ent.category.value == "drawer" else {}),
                **self._front(ent),
            })
        job = self.executor.current.to_dict() if self.executor.current else None
        return {
            "version": es.version,
            "t": round(es.stamp, 3),
            "self": {"x": round(p["p"][0], 3), "y": round(p["p"][1], 3),
                     "yawDeg": round(math.degrees(yaw_from_quat(p["q"])), 1),
                     "holding": held, "joints": {j.id: round(j.q.value[0], 3) for j in me.rigid.joints if j.q.value}},
            "objects": objects,
            "currentSkill": job,
            "perception": self.perception_mode,
            "runtimeError": self.last_error,
        }

    def _extent(self, track_id: str) -> list[float] | None:
        """Estimated physical size of a track, without the uncertainty margin added to ``sizeM``."""
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return [round(float(v), 3) for v in track.size] if track is not None else None

    def _front(self, ent) -> dict:
        """Heading of the face looking into the room, for furniture and drawers."""
        affordances = [a.value for a in ent.rigid.links[0].affordances]
        if "support-surface" not in affordances and "pullable" not in affordances:
            return {}
        with self._lock:
            track = self.estimator.tracks.get(ent.id)
            if track is None:
                return {}
            box = self.estimator.front(track)
        return {"frontDeg": round(math.degrees(box.yaw), 1), "frontHalfM": round(box.hx, 3)}

    def _opening(self, track_id: str) -> float:
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return self.estimator.drawer_opening(track) if track is not None else 0.0

    def query(self, category: str | None = None, near: str | None = None, radius: float | None = None) -> list[dict]:
        objs = self.summary()["objects"]
        if category:
            objs = [o for o in objs if o["category"] == category]
        if near:
            anchor = next((o for o in self.summary()["objects"] if o["id"] == near), None)
            if anchor is None:
                raise ValueError(f"unknown entity {near!r}")
            r = radius if radius is not None else 1.0
            objs = [o for o in objs if o["id"] != near and math.dist(o["position"][:2], anchor["position"][:2]) <= r]
        return objs

    def observe(self) -> dict:
        frame = self.link.sensors()
        if frame.get("imaging") == "rgbd":
            rgbd = decode_rgbd(self.link.rgbd())
            with self._lock:
                cam = self.estimator.camera_pose()
            visible = []
            for seg in segments(rgbd, cam):
                centre, size, _ = box_of(seg.points)
                probs = classify(size, centre[2] - size[2] / 2, seg.color, seg.invalid_frac)
                best = max(probs, key=probs.get)
                visible.append({"category": best, "confidence": round(probs[best], 2), "rangeM": round(seg.range, 2)})
        else:
            visible = [{"category": d["category"], "rangeM": round(d["range"], 2)} for d in frame["detections"]]
        out = {"t": frame["stamp"], "visible": visible}
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"observe-{frame['stamp']:.2f}.png"
            path.write_bytes(self.link.camera_png())
            out["imagePath"] = str(path)
        return out

    def predict(self, skill: str, params: dict) -> dict:
        es = self.ego_state()
        receipt = self.esim.predict(es, skill, params)
        receipt["withinRiskLimit"] = receipt["feasible"] and receipt["risk"] <= self.config.max_risk
        receipt["maxRisk"] = self.config.max_risk
        with self._lock:
            self.receipts[receipt["id"]] = receipt
        return receipt

    def receipt(self, receipt_id: str) -> dict:
        if receipt_id not in self.receipts:
            raise KeyError(f"unknown receipt {receipt_id!r}")
        return self.receipts[receipt_id]

    # ------------------------------------------------------------------ alerts
    def alert(self, message: str, source: str, level: str = "error", goal: str = "", step: str = "",
              diagnosis: str = "") -> dict:
        """Record a situation the robot cannot resolve by itself (an unexpected exception, or a
        problem its agent or mission reports) with the goal and step it was pursuing and the
        reporter's own diagnosis. A goal or step left empty is taken from the published plan. A
        repeat of an unacknowledged message only bumps its count."""
        message = message[:ALERT_TEXT_MAX]
        with self._lock:
            plan = self.plan or {}
            for existing in reversed(self.alerts):
                if not existing["acknowledged"] and existing["message"] == message:
                    existing["count"] += 1
                    existing["t"] = round(self.estimator.stamp, 2)
                    return existing
            record = {"id": next(self._alert_ids), "t": round(self.estimator.stamp, 2), "level": level,
                      "source": source, "message": message, "count": 1, "acknowledged": False,
                      "goal": (goal or plan.get("goal", ""))[:ALERT_TEXT_MAX],
                      "step": (step or plan.get("current", ""))[:ALERT_TEXT_MAX],
                      "diagnosis": diagnosis[:ALERT_TEXT_MAX]}
            self.alerts.append(record)
            del self.alerts[:-50]
            return record

    def alerts_view(self) -> dict:
        with self._lock:
            return {"alerts": [dict(a) for a in self.alerts],
                    "active": [dict(a) for a in self.alerts if not a["acknowledged"]]}

    def acknowledge(self, alert_id: int | None = None) -> dict:
        with self._lock:
            for record in self.alerts:
                if alert_id is None or record["id"] == alert_id:
                    record["acknowledged"] = True
        return self.alerts_view()

    # -------------------------------------------------------------------- plan
    def set_plan(self, plan: dict) -> dict:
        """Publish what the robot is trying to achieve: a goal, its steps with their status, the
        step in progress and the latest difficulty. A client (a mission script, the robot agent)
        owns its content; the runtime only stores the latest one and stamps it."""
        goal = plan.get("goal")
        steps = plan.get("steps", [])
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("a plan needs a goal")
        if not isinstance(steps, list) or not all(isinstance(st, dict) and isinstance(st.get("name"), str)
                                                   and st.get("status") in PLAN_STEP_STATES for st in steps):
            raise ValueError(f"plan steps need a name and a status in {sorted(PLAN_STEP_STATES)}")
        record = {"goal": goal[:300], "source": str(plan.get("source", "agent"))[:60],
                  "steps": [{"name": st["name"][:200], "status": st["status"]} for st in steps[:30]],
                  "round": plan.get("round"), "current": str(plan.get("current") or "")[:400],
                  "difficulty": str(plan.get("difficulty") or "")[:600], "state": str(plan.get("state") or "running")[:40],
                  "t": round(self.estimator.stamp, 2)}
        with self._lock:
            self.plan = record
        return record

    def plan_view(self) -> dict:
        """The latest published plan (``None`` before any) and the skill running now."""
        running = next((j for j in reversed(list(self.executor.jobs.values())) if j.state == "running"), None)
        with self._lock:
            plan = dict(self.plan) if self.plan else None
        return {"plan": plan, "skill": running.to_dict() if running else None, "t": round(self.estimator.stamp, 2)}

    # ----------------------------------------------------------------- control
    def control(self) -> dict:
        return {"owner": self.control_owner}

    def acquire_control(self, owner: str) -> dict:
        """Give ``owner`` exclusive use of skills until it releases control; the running skill stops.
        Clients that are not the owner cannot start skills meanwhile, but anyone may still stop."""
        if not owner:
            raise ValueError("control needs an owner name")
        with self._lock:
            if self.control_owner not in (None, owner):
                raise PermissionError(f"the robot is already controlled by {self.control_owner}")
            self.control_owner = owner
        self.executor.stop()
        return self.control()

    def release_control(self, owner: str) -> dict:
        with self._lock:
            if self.control_owner == owner:
                self.control_owner = None
        return self.control()

    def _check_control(self, owner: str | None) -> None:
        if self.control_owner is not None and owner != self.control_owner:
            raise PermissionError(f"the robot is controlled by {self.control_owner}; wait until it releases control")

    def run(self, receipt_id: str, owner: str | None = None) -> dict:
        """Execute a predicted plan. The runtime re-checks the receipt here, where the motion happens."""
        self._check_control(owner)
        receipt = self.receipt(receipt_id)
        now = self.estimator.stamp
        if receipt.get("used"):
            raise PermissionError(f"receipt {receipt_id} was already executed; predict again")
        if not receipt["feasible"]:
            raise PermissionError(f"receipt {receipt_id} is not feasible: {'; '.join(receipt['issues'])}")
        if now > receipt["expiresAt"]:
            raise PermissionError(f"receipt {receipt_id} expired at t={receipt['expiresAt']:.1f}s; predict again")
        if receipt["risk"] > self.config.max_risk:
            raise PermissionError(f"receipt risk {receipt['risk']:.2f} exceeds the limit {self.config.max_risk:.2f}")
        receipt["used"] = True
        return self.executor.start(receipt["skill"], receipt).to_dict()

    def explore(self, owner: str | None = None) -> dict:
        """Look around in place. Rotation without translation needs no receipt."""
        self._check_control(owner)
        return self.executor.start("look_around", {"id": "look-around", "plan": {}}).to_dict()

    def job(self, job_id: str) -> dict:
        if job_id not in self.executor.jobs:
            raise KeyError(f"unknown job {job_id!r}")
        return self.executor.jobs[job_id].to_dict()

    def wait(self, job_id: str, timeout: float = 120.0) -> dict:
        return self.executor.wait(job_id, timeout).to_dict()

    def stop_motion(self) -> dict:
        return self.executor.stop()
