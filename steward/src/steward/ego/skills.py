"""Skill execution: turn a Prediction Receipt's plan into timed actuator commands.

Motion uses only the robot's own estimates (odometry, encoders, ES). Base motion ramps a
position setpoint along each path segment at a bounded speed in simulation (or wall) time taken
from sensor stamps; joints settle to targets with one sag correction. A stop request interrupts
the current skill between control ticks and holds the current pose.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from steward.math3d import wrap_angle


class SkillStopped(Exception):
    pass


class SkillFailed(Exception):
    pass


@dataclass
class MotionConfig:
    linear_speed: float = 0.35
    angular_speed: float = 0.9
    position_tol: float = 0.03
    yaw_tol: float = 0.04
    joint_tol: float = 0.01
    tick_s: float = 0.04
    retract_reach: float = 0.10
    stow_lift: float = 0.60
    dock_tol: float = 0.06
    dock_yaw_tol: float = 0.12
    align_min: float = 0.01
    align_max: float = 0.15


@dataclass
class Job:
    id: str
    skill: str
    receipt: str
    state: str = "running"
    message: str = ""
    started: float = 0.0
    finished: float | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "skill": self.skill, "receipt": self.receipt, "state": self.state,
                "message": self.message, "started": self.started, "finished": self.finished, "log": self.log[-20:]}


class SkillExecutor:
    """Runs at most one skill at a time in a worker thread."""

    def __init__(self, command: Callable[[dict], None], state: Callable[[], dict], mark_held: Callable[[str], None],
                 config: MotionConfig | None = None):
        self._command = command
        self._state = state
        self._mark_held = mark_held
        self.config = config or MotionConfig()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None

    # ------------------------------------------------------------------ public
    def start(self, skill: str, receipt: dict) -> Job:
        with self._lock:
            if self.current is not None and self.current.state == "running":
                raise RuntimeError(f"skill {self.current.skill} ({self.current.id}) is still running")
            job = Job(id=f"job-{next(self._ids)}", skill=skill, receipt=receipt["id"], started=self._state()["stamp"])
            self.jobs[job.id] = job
            self.current = job
            self._stop.clear()
        threading.Thread(target=self._run, args=(job, skill, receipt), name=f"skill-{job.id}", daemon=True).start()
        return job

    def cancel(self) -> None:
        """Interrupt the current skill without commanding the body (used when the world restarts)."""
        self._stop.set()

    def stop(self) -> dict:
        self._stop.set()
        st = self._state()
        o = st["odometry"]
        self._command({"base": [o["x"], o["y"], o["yaw"]]})
        return {"stopped": self.current.id if self.current and self.current.state == "running" else None}

    def wait(self, job_id: str, timeout: float) -> Job:
        deadline = time.monotonic() + timeout
        job = self.jobs[job_id]
        while job.state == "running" and time.monotonic() < deadline:
            time.sleep(0.05)
        return job

    # ---------------------------------------------------------------- runner
    def _run(self, job: Job, skill: str, receipt: dict) -> None:
        try:
            plan = receipt.get("plan", {})
            if skill == "look_around":
                self._look_around(job)
            elif skill == "navigate":
                self._drive(job, plan["path"], plan["yaw"])
            elif skill == "pick":
                self._pick(job, plan)
            elif skill == "place":
                self._place(job, plan)
            else:
                raise SkillFailed(f"unknown skill {skill}")
            job.state, job.message = "succeeded", job.message or f"{skill} completed"
        except SkillStopped:
            job.state, job.message = "stopped", "stopped on request"
        except SkillFailed as exc:
            job.state, job.message = "failed", str(exc)
            self._safe_retract()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller through the job record
            job.state, job.message = "failed", f"{type(exc).__name__}: {exc}"
            self._safe_retract()
        finally:
            job.finished = self._state()["stamp"]

    def _safe_retract(self) -> None:
        try:
            self._command({"reach": self.config.retract_reach})
        except Exception:  # noqa: BLE001 - best effort after a failure that is already reported
            pass

    def _tick(self) -> dict:
        if self._stop.is_set():
            raise SkillStopped()
        time.sleep(self.config.tick_s)
        return self._state()

    # ------------------------------------------------------------- primitives
    def _turn_to(self, yaw_target: float) -> None:
        st = self._state()
        o = st["odometry"]
        x, y, yaw0 = o["x"], o["y"], o["yaw"]
        delta = wrap_angle(yaw_target - yaw0)
        duration = abs(delta) / self.config.angular_speed
        t0 = st["stamp"]
        deadline = t0 + duration * 3 + 5.0
        while True:
            st = self._tick()
            frac = 1.0 if duration == 0 else min(1.0, (st["stamp"] - t0) / duration)
            self._command({"base": [x, y, yaw0 + delta * frac]})
            err = abs(wrap_angle(yaw_target - st["odometry"]["yaw"]))
            if frac >= 1.0 and err < self.config.yaw_tol:
                return
            if st["stamp"] > deadline:
                raise SkillFailed(f"turn did not converge (yaw error {err:.2f} rad)")

    def _translate_to(self, x1: float, y1: float, yaw: float) -> None:
        st = self._state()
        x0, y0 = st["odometry"]["x"], st["odometry"]["y"]
        # The base yaw joint is continuous: express the heading in the same unwrapped frame as the
        # odometry, otherwise the position controller spins the base by a full turn.
        current = st["odometry"]["yaw"]
        yaw = current + wrap_angle(yaw - current)
        length = math.hypot(x1 - x0, y1 - y0)
        duration = length / self.config.linear_speed
        t0 = st["stamp"]
        deadline = t0 + duration * 3 + 5.0
        while True:
            st = self._tick()
            frac = 1.0 if duration == 0 else min(1.0, (st["stamp"] - t0) / duration)
            self._command({"base": [x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, yaw]})
            err = math.hypot(x1 - st["odometry"]["x"], y1 - st["odometry"]["y"])
            if frac >= 1.0 and err < self.config.position_tol:
                return
            if st["stamp"] > deadline:
                raise SkillFailed(f"base did not reach the waypoint (error {err:.2f} m)")

    def _drive(self, job: Job, path: list, final_yaw: float) -> None:
        for wx, wy in path[1:]:
            o = self._state()["odometry"]
            if math.hypot(wx - o["x"], wy - o["y"]) < self.config.position_tol:
                continue
            heading = math.atan2(wy - o["y"], wx - o["x"])
            self._turn_to(heading)
            self._translate_to(wx, wy, heading)
            job.log.append(f"reached waypoint ({wx:.2f}, {wy:.2f})")
        self._turn_to(final_yaw)

    def _set_joint(self, name: str, target: float, timeout: float = 8.0) -> None:
        self._command({name: target})
        t0 = self._state()["stamp"]
        corrected = False
        while True:
            st = self._tick()
            q = st["joints"][name]
            if abs(q - target) < self.config.joint_tol:
                return
            settled = st["stamp"] - t0 > 1.5
            if settled and not corrected:
                self._command({name: target + (target - q)})
                corrected = True
            if st["stamp"] - t0 > timeout:
                raise SkillFailed(f"joint {name} did not reach {target:.3f} (at {q:.3f})")

    def _wait(self, seconds: float) -> None:
        t0 = self._state()["stamp"]
        while self._tick()["stamp"] - t0 < seconds:
            pass

    def _dock(self, job: Job, plan: dict) -> None:
        self._set_joint("reach", self.config.retract_reach)
        self._drive(job, plan["path"], plan["yaw"])
        self._translate_to(plan["dock"][0], plan["dock"][1], plan["yaw"])
        o = self._state()["odometry"]
        dist = math.hypot(o["x"] - plan["dock"][0], o["y"] - plan["dock"][1])
        dyaw = abs(wrap_angle(o["yaw"] - plan["yaw"]))
        if dist > self.config.dock_tol or dyaw > self.config.dock_yaw_tol:
            self._backoff(plan)
            raise SkillFailed(f"docking missed the planned pose by {dist:.2f} m / {math.degrees(dyaw):.0f} deg")
        job.log.append("docked")

    def _gripper_xy(self) -> np.ndarray:
        """Gripper position from the robot's own odometry and reach encoder."""
        st = self._state()
        o = st["odometry"]
        ahead = st["toolOffset"] + st["joints"]["reach"]
        return np.array([o["x"] + ahead * math.cos(o["yaw"]), o["y"] + ahead * math.sin(o["yaw"])])

    def _align_lateral(self, target_xy: list[float]) -> None:
        """Shift the base sideways so the target lies on the arm's line of reach."""
        o = self._state()["odometry"]
        left = np.array([-math.sin(o["yaw"]), math.cos(o["yaw"])])
        lateral = float((np.array(target_xy) - np.array([o["x"], o["y"]])) @ left)
        if self.config.align_min < abs(lateral) <= self.config.align_max:
            self._translate_to(o["x"] + lateral * left[0], o["y"] + lateral * left[1], o["yaw"])

    def _backoff(self, plan: dict) -> None:
        self._translate_to(plan["predock"][0], plan["predock"][1], plan["yaw"])

    def _reach_to(self, target_xy: list[float]) -> float:
        st = self._state()
        o = st["odometry"]
        heading = np.array([math.cos(o["yaw"]), math.sin(o["yaw"])])
        rel = np.array(target_xy) - np.array([o["x"], o["y"]])
        return float(rel @ heading) - st["toolOffset"]

    def _over_support(self, plan: dict) -> bool:
        """True when the held object's footprint at the gripper lies within the support top."""
        box = plan["supportBox"]
        g = self._gripper_xy()
        dx, dy = g[0] - box["cx"], g[1] - box["cy"]
        c, s = math.cos(-box["yaw"]), math.sin(-box["yaw"])
        lx, ly = c * dx - s * dy, s * dx + c * dy
        margin = plan.get("heldHalf", 0.05) * 0.6
        return abs(lx) <= box["hx"] - margin and abs(ly) <= box["hy"] - margin

    # ---------------------------------------------------------------- skills
    def _look_around(self, job: Job) -> None:
        yaw0 = self._state()["odometry"]["yaw"]
        for k in range(1, 5):
            self._turn_to(wrap_angle(yaw0 + k * math.pi / 2))
            self._wait(0.6)
            job.log.append(f"looked toward {math.degrees(wrap_angle(yaw0 + k * math.pi / 2)):.0f} deg")

    def _pick(self, job: Job, plan: dict) -> None:
        self._dock(job, plan)
        target = self._state()["entities"].get(plan["target"])
        if target is None:
            raise SkillFailed(f"target {plan['target']} is no longer in the ego state")
        self._align_lateral(target["p"][:2])
        self._set_joint("lift", target["p"][2])
        self._set_joint("reach", max(0.05, self._reach_to(target["p"][:2])))
        miss = float(np.linalg.norm(self._gripper_xy() - np.array(target["p"][:2])))
        if miss > plan.get("captureRadius", 0.05):
            self._set_joint("reach", self.config.retract_reach)
            self._backoff(plan)
            raise SkillFailed(f"gripper is {miss * 100:.0f} cm from {plan['target']}; not closing")
        self._command({"gripper": "close"})
        self._wait(0.4)
        if not self._state()["gripper"]["holding"]:
            self._command({"gripper": "open"})
            self._set_joint("reach", self.config.retract_reach)
            self._backoff(plan)
            raise SkillFailed(f"grasp missed {plan['target']}")
        self._mark_held(plan["target"])
        job.log.append(f"grasped {plan['target']}")
        self._set_joint("lift", plan["carryLift"])
        self._set_joint("reach", self.config.retract_reach)
        self._backoff(plan)
        job.message = f"holding {plan['target']}"

    def _place(self, job: Job, plan: dict) -> None:
        self._dock(job, plan)
        self._align_lateral(plan["spot"])
        self._set_joint("lift", plan["lift"])
        self._set_joint("reach", max(0.05, self._reach_to(plan["spot"])))
        if not self._over_support(plan):
            self._set_joint("reach", self.config.retract_reach)
            self._backoff(plan)
            raise SkillFailed(f"gripper is not over {plan['support']}; keeping {plan['object']}")
        self._command({"gripper": "open"})
        self._wait(0.6)
        job.log.append(f"released {plan['object']} on {plan['support']}")
        self._set_joint("reach", self.config.retract_reach)
        self._set_joint("lift", max(plan["lift"], self.config.stow_lift))
        self._backoff(plan)
        job.message = f"placed {plan['object']} on {plan['support']}"
