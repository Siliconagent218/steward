"""Operator chat with the World Harness and the robots' Ego Harnesses.

Each console message goes to one harness (``world`` or ``robot``) and runs one
``dsh --profile <profile> --json`` process in that harness's own DSH home. The first turn opens a
Session; later turns pass ``--session-id`` so each harness keeps its own conversation. A
``--patch`` overlay points the harness's provider at this host's World Control API or at the
robot's Ego API. Progress is read from the NDJSON run events and kept as one transcript that the
console polls.

A robot turn takes exclusive control of the robot through the Ego API first (stopping the running
skill and pausing other clients such as the scripted shuttle) and releases it when the turn ends.
The Ego Harness keeps its robot-only tool set and empty workspace; the operator's words are its
only extra input.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from steward.net import HttpError, call

WORLD_BRIEF = (
    "你是 Steward 的 World Harness 操作员助手。一个世界仿真正在运行（机器人正在执行任务），"
    "操作员会用自然语言要求对 World State 做运行中的增量修改：移动、添加或删除物体与家具。\n"
    "规则：先用 world_scene 查看当前实体 id 与位置，再用 world_edit 执行修改；"
    "需要新物体时用 asset_search 选资产；不要启动、停止或重载 episode，不要读写文件或运行 shell 命令。"
    "world_edit 被拒绝时（例如穿模、物体正被机器人拿着），说明原因并给出可行的替代方案。"
    "完成后用一两句中文说明做了什么。\n\n操作员请求："
)

ROBOT_BRIEF = (
    "你是家用服务机器人 robot-1。操作员通过控制台给你下达新任务；在这一轮对话期间你独占控制机器人，"
    "原来的后台任务会暂停，等你这一轮结束后再继续。\n"
    "做法：先用 ego_state 看看你已经知道哪些物体；找不到或不确定时用 ego_explore 环顾，或导航靠近观察。"
    "每个动作先用 ego_predict 得到收据，风险超限就先靠近观察再预测，然后用 skill_run 执行；"
    "执行后用 ego_state 自查结果。完成或无法完成时，用一两句中文向操作员汇报。\n\n操作员任务："
)

ROBOT_OWNER = "operator"


def load_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines of a dotenv file; missing files yield no variables."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return out


@dataclass
class ChatConfig:
    """Where the harnesses live. ``robot_home`` / ``ego_api`` may be unset to disable robot chat."""

    dsh: Path
    cwd: Path
    env_file: Path
    world_home: Path | None = None
    robot_home: Path | None = None
    ego_api: str | None = None
    timeout_s: float = 900.0


@dataclass
class _Agent:
    key: str
    label: str
    profile: str
    home: Path | None
    cwd: Path
    brief: str
    follow_up: str
    patch: Callable[[], str]
    before_turn: Callable[[], None] | None = None
    after_turn: Callable[[], None] | None = None
    extra_reason: Callable[[], str | None] | None = None
    session_id: str | None = None
    busy: bool = False
    proc: subprocess.Popen | None = None


@dataclass
class _Entry:
    id: int
    agent: str
    role: str  # user | assistant | tool | error | system
    text: str
    t: float = field(default_factory=time.time)
    tool: str | None = None
    input: object = None
    status: str | None = None
    call_id: str | None = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "agent": self.agent, "role": self.role, "text": self.text, "t": self.t}
        if self.tool is not None:
            d |= {"tool": self.tool, "input": self.input, "status": self.status}
        return d


class HarnessChat:
    def __init__(self, config: ChatConfig | None, wca_url: Callable[[], str]):
        self.config = config
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []
        self._next_id = 1
        self.agents: dict[str, _Agent] = {}
        if config is None:
            return
        self.agents["world"] = _Agent(
            key="world", label="World Harness", profile="world-task", home=config.world_home, cwd=config.cwd,
            brief=WORLD_BRIEF, follow_up="操作员请求：",
            patch=lambda: f"- id: steward-world-api\n  config:\n    endpoint: '{wca_url()}'\n")
        robot_home = config.robot_home
        self.agents["robot"] = _Agent(
            key="robot", label="robot-1", profile="ego", home=robot_home,
            cwd=(robot_home / "workspace") if robot_home else config.cwd,
            brief=ROBOT_BRIEF, follow_up="操作员任务：",
            patch=lambda: (f"- id: steward-ego-runtime\n  config:\n    endpoint: '{config.ego_api}'\n"
                           f"    owner: {ROBOT_OWNER}\n"),
            before_turn=lambda: self._robot_control("acquire"),
            after_turn=lambda: self._robot_control("release"),
            extra_reason=lambda: None if config.ego_api else "the robot's Ego API is not known to this host")

    # ------------------------------------------------------------ public API
    def unavailable_reason(self, key: str) -> str | None:
        agent = self.agents.get(key)
        if agent is None or self.config is None:
            return "operator chat is disabled"
        if not self.config.dsh.exists():
            return f"DSH launcher not found at {self.config.dsh}; run steward/scripts/setup-dsh-profiles.sh"
        if agent.home is None or not (agent.home / "profiles" / agent.profile).exists():
            return f"profile {agent.profile!r} missing in {agent.home}; run steward/scripts/setup-dsh-profiles.sh"
        return agent.extra_reason() if agent.extra_reason else None

    def state(self, since: int = 0) -> dict:
        with self._lock:
            agents = {key: {"label": a.label, "available": self.unavailable_reason(key) is None,
                            "reason": self.unavailable_reason(key), "busy": a.busy, "sessionId": a.session_id}
                      for key, a in self.agents.items()}
            messages = [e.to_dict() for e in self._entries if e.id > since]
        if "robot" in agents and self.config and self.config.ego_api:
            agents["robot"]["control"] = self._robot_owner()
        return {"agents": agents, "messages": messages}

    def send(self, key: str, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise HttpError(400, "message is empty")
        if key not in self.agents:
            raise HttpError(400, f"unknown chat target {key!r}; use one of {sorted(self.agents)}")
        reason = self.unavailable_reason(key)
        if reason:
            raise HttpError(503, reason)
        agent = self.agents[key]
        with self._lock:
            if agent.busy:
                raise HttpError(409, f"{agent.label} is still working on the previous message")
            agent.busy = True
            entry = self._add(key, "user", text)
            first = agent.session_id is None
        threading.Thread(target=self._run, args=(agent, text, first), name=f"chat-{key}", daemon=True).start()
        return {"accepted": True, "id": entry.id}

    def reset(self, key: str) -> dict:
        agent = self.agents.get(key)
        if agent is None:
            raise HttpError(400, f"unknown chat target {key!r}")
        with self._lock:
            if agent.busy:
                raise HttpError(409, "cannot start a new conversation while a message is running")
            agent.session_id = None
            self._add(key, "system", f"已开始新的 {agent.label} 会话")
        return {"reset": True}

    def stop(self) -> None:
        for agent in self.agents.values():
            proc = agent.proc
            if proc is not None and proc.poll() is None:
                proc.terminate()

    # -------------------------------------------------------------- internals
    def _robot_control(self, action: str) -> None:
        call("POST", f"{self.config.ego_api}/ego/control", {"action": action, "owner": ROBOT_OWNER})

    def _robot_owner(self) -> str | None:
        try:
            return call("GET", f"{self.config.ego_api}/ego/control", timeout=0.5)["owner"]
        except (HttpError, OSError):  # the robot's Ego API is down; the console shows no owner
            return None

    def _add(self, agent: str, role: str, text: str, **extra) -> _Entry:
        entry = _Entry(self._next_id, agent, role, text, **extra)
        self._next_id += 1
        self._entries.append(entry)
        del self._entries[:-400]
        return entry

    def _report(self, agent: _Agent, text: str) -> None:
        with self._lock:
            self._add(agent.key, "error", text)

    def _run(self, agent: _Agent, text: str, first: bool) -> None:
        cfg = self.config
        fd, name = tempfile.mkstemp(prefix=f"steward-{agent.key}-", suffix=".yml")
        with os.fdopen(fd, "w") as fh:
            fh.write(agent.patch())
        patch = Path(name)
        args = [str(cfg.dsh), "--profile", agent.profile, "--patch", str(patch), "--json"]
        if agent.session_id:
            args += ["--session-id", agent.session_id]
        args.append("-")
        env = os.environ | load_env_file(cfg.env_file) | {"DSH_HOME": str(agent.home)}
        stderr_tail: list[str] = []
        controlled = False
        try:
            if agent.before_turn is not None:
                agent.before_turn()
                controlled = True
            agent.cwd.mkdir(parents=True, exist_ok=True)
            proc = agent.proc = subprocess.Popen(args, cwd=agent.cwd, env=env, stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            proc.stdin.write((agent.brief if first else agent.follow_up) + text)
            proc.stdin.close()
            err_reader = threading.Thread(target=lambda: stderr_tail.extend(proc.stderr), daemon=True)
            err_reader.start()
            timer = threading.Timer(cfg.timeout_s, proc.kill)
            timer.start()
            last_text: str | None = None
            for line in proc.stdout:
                last_text = self._ingest(agent, line, last_text)
            code = proc.wait()
            timer.cancel()
            err_reader.join(2)
            if code != 0:
                self._report(agent, f"{agent.label} 运行失败：{''.join(stderr_tail[-6:]).strip() or f'exit code {code}'}")
        except HttpError as exc:
            self._report(agent, f"无法接管机器人：{exc}")
        except OSError as exc:
            self._report(agent, f"无法启动 {agent.label}：{exc}")
        finally:
            patch.unlink(missing_ok=True)
            if controlled and agent.after_turn is not None:
                try:
                    agent.after_turn()
                except (HttpError, OSError) as exc:
                    self._report(agent, f"交还机器人控制失败：{exc}")
            with self._lock:
                agent.busy = False
                agent.proc = None

    def _ingest(self, agent: _Agent, line: str, last_text: str | None) -> str | None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return last_text
        kind = event.get("type")
        with self._lock:
            if kind == "session":
                agent.session_id = event.get("sessionId") or agent.session_id
            elif kind == "text" and event.get("text", "").strip():
                last_text = event["text"].strip()
                self._add(agent.key, "assistant", last_text)
            elif kind == "tool_call":
                self._add(agent.key, "tool", "", tool=event.get("tool"), input=event.get("input"), status="running",
                          call_id=event.get("callId"))
            elif kind == "tool_result":
                for entry in reversed(self._entries):
                    if entry.role == "tool" and entry.call_id == event.get("callId"):
                        entry.status = event.get("status")
                        entry.text = str(event.get("result", ""))[:1200]
                        break
            elif kind == "final":
                agent.session_id = event.get("sessionId") or agent.session_id
                final = (event.get("text") or "").strip()
                if final and final != last_text:
                    self._add(agent.key, "assistant", final)
                    last_text = final
        return last_text
