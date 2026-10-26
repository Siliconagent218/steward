"""Ego Runtime (ER): perception loop, Ego-State store, ESim receipts and skill execution.

The runtime talks to the world only through the robot port (sensors, actuator commands,
telemetry). It never holds World State identifiers or ground truth.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from steward.ego.esim import Esim, EsimConfig
from steward.ego.estimator import SELF_ID, EgoEstimator
from steward.ego.skills import MotionConfig, SkillExecutor
from steward.math3d import yaw_from_quat
from steward.net import HttpError, call
from steward.scene.schema import SceneState


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

    def command(self, cmd: dict) -> None:
        call("POST", f"{self.url}/command", cmd)

    def telemetry(self, scene: dict) -> None:
        call("POST", f"{self.url}/telemetry", scene)

    def camera_png(self, width: int = 320, height: int = 240) -> bytes:
        return call("GET", f"{self.url}/camera/rgb?width={width}&height={height}")


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
        self.executor = SkillExecutor(self.link.command, self._motion_state, self._mark_held, self.config.motion)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._running = True
        self._poll_once()
        self._thread = threading.Thread(target=self._loop, name="ego-perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _poll_once(self) -> None:
        frame = self.link.sensors()
        with self._lock:
            self.estimator.update(frame)

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
            "runtimeError": self.last_error,
        }

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

    def run(self, receipt_id: str) -> dict:
        """Execute a predicted plan. The runtime re-checks the receipt here, where the motion happens."""
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

    def explore(self) -> dict:
        """Look around in place. Rotation without translation needs no receipt."""
        return self.executor.start("look_around", {"id": "look-around", "plan": {}}).to_dict()

    def job(self, job_id: str) -> dict:
        if job_id not in self.executor.jobs:
            raise KeyError(f"unknown job {job_id!r}")
        return self.executor.jobs[job_id].to_dict()

    def wait(self, job_id: str, timeout: float = 120.0) -> dict:
        return self.executor.wait(job_id, timeout).to_dict()

    def stop_motion(self) -> dict:
        return self.executor.stop()
