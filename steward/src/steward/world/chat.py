"""Operator chat with the World Harness: console messages become World Harness (DSH) turns.

Each message runs one ``dsh --profile world-task --json`` process in the World Harness home. The
first turn opens a Session; later turns pass ``--session-id`` so the World Harness keeps the
conversation. A ``--patch`` overlay points the World Harness tools at this host's World Control
API, so edits always reach the running world. Progress is read from the NDJSON run events and kept
as a transcript that the console polls.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from steward.net import HttpError

OPERATOR_BRIEF = (
    "你是 Steward 的 World Harness 操作员助手。一个世界仿真正在运行（机器人正在执行任务），"
    "操作员会用自然语言要求对 World State 做运行中的增量修改：移动、添加或删除物体与家具。\n"
    "规则：先用 world_scene 查看当前实体 id 与位置，再用 world_edit 执行修改；"
    "需要新物体时用 asset_search 选资产；不要启动、停止或重载 episode，不要读写文件或运行 shell 命令。"
    "world_edit 被拒绝时（例如穿模、物体正被机器人拿着），说明原因并给出可行的替代方案。"
    "完成后用一两句中文说明做了什么。\n\n操作员请求："
)


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
    dsh: Path
    dsh_home: Path
    cwd: Path
    env_file: Path
    profile: str = "world-task"
    timeout_s: float = 600.0


@dataclass
class _Entry:
    id: int
    role: str  # user | assistant | tool | error | system
    text: str
    t: float = field(default_factory=time.time)
    tool: str | None = None
    input: object = None
    status: str | None = None
    call_id: str | None = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "role": self.role, "text": self.text, "t": self.t}
        if self.tool is not None:
            d |= {"tool": self.tool, "input": self.input, "status": self.status}
        return d


class WorldChat:
    def __init__(self, config: ChatConfig | None, wca_url):
        self.config = config
        self._wca_url = wca_url
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []
        self._next_id = 1
        self.session_id: str | None = None
        self.busy = False
        self._proc: subprocess.Popen | None = None

    # ------------------------------------------------------------ public API
    def unavailable_reason(self) -> str | None:
        if self.config is None:
            return "World Harness chat is disabled"
        if not self.config.dsh.exists():
            return f"DSH launcher not found at {self.config.dsh}; run steward/scripts/setup-dsh-profiles.sh"
        if not (self.config.dsh_home / "profiles" / self.config.profile).exists():
            return f"profile {self.config.profile!r} missing in {self.config.dsh_home}; run steward/scripts/setup-dsh-profiles.sh"
        return None

    def state(self, since: int = 0) -> dict:
        with self._lock:
            return {"available": self.unavailable_reason() is None, "reason": self.unavailable_reason(),
                    "busy": self.busy, "sessionId": self.session_id,
                    "messages": [e.to_dict() for e in self._entries if e.id > since]}

    def send(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise HttpError(400, "message is empty")
        reason = self.unavailable_reason()
        if reason:
            raise HttpError(503, reason)
        with self._lock:
            if self.busy:
                raise HttpError(409, "the World Harness is still working on the previous message")
            self.busy = True
            entry = self._add("user", text)
            first = self.session_id is None
        threading.Thread(target=self._run, args=(text, first), name="world-chat", daemon=True).start()
        return {"accepted": True, "id": entry.id}

    def reset(self) -> dict:
        with self._lock:
            if self.busy:
                raise HttpError(409, "cannot start a new conversation while a message is running")
            self.session_id = None
            self._add("system", "已开始新的 World Harness 会话")
        return {"reset": True}

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    # -------------------------------------------------------------- internals
    def _add(self, role: str, text: str, **extra) -> _Entry:
        entry = _Entry(self._next_id, role, text, **extra)
        self._next_id += 1
        self._entries.append(entry)
        del self._entries[:-400]
        return entry

    def _patch_file(self) -> Path:
        patch = f"- id: steward-world-api\n  config:\n    endpoint: '{self._wca_url()}'\n"
        fd, name = tempfile.mkstemp(prefix="steward-wh-", suffix=".yml")
        with os.fdopen(fd, "w") as fh:
            fh.write(patch)
        return Path(name)

    def _run(self, text: str, first: bool) -> None:
        cfg = self.config
        patch = self._patch_file()
        args = [str(cfg.dsh), "--profile", cfg.profile, "--patch", str(patch), "--json"]
        if self.session_id:
            args += ["--session-id", self.session_id]
        args.append("-")
        env = os.environ | load_env_file(cfg.env_file) | {"DSH_HOME": str(cfg.dsh_home)}
        last_text: str | None = None
        stderr_tail: list[str] = []
        try:
            self._proc = subprocess.Popen(args, cwd=cfg.cwd, env=env, stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._proc.stdin.write((OPERATOR_BRIEF if first else "操作员请求：") + text)
            self._proc.stdin.close()
            err_reader = threading.Thread(target=lambda: stderr_tail.extend(self._proc.stderr), daemon=True)
            err_reader.start()
            timer = threading.Timer(cfg.timeout_s, self._proc.kill)
            timer.start()
            for line in self._proc.stdout:
                last_text = self._ingest(line, last_text)
            code = self._proc.wait()
            timer.cancel()
            err_reader.join(2)
            if code != 0:
                detail = "".join(stderr_tail[-6:]).strip() or f"exit code {code}"
                with self._lock:
                    self._add("error", f"World Harness 运行失败：{detail}")
        except OSError as exc:
            with self._lock:
                self._add("error", f"无法启动 World Harness：{exc}")
        finally:
            patch.unlink(missing_ok=True)
            with self._lock:
                self.busy = False
                self._proc = None

    def _ingest(self, line: str, last_text: str | None) -> str | None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return last_text
        kind = event.get("type")
        with self._lock:
            if kind == "session":
                self.session_id = event.get("sessionId") or self.session_id
            elif kind == "text" and event.get("text", "").strip():
                last_text = event["text"].strip()
                self._add("assistant", last_text)
            elif kind == "tool_call":
                self._add("tool", "", tool=event.get("tool"), input=event.get("input"), status="running",
                          call_id=event.get("callId"))
            elif kind == "tool_result":
                for entry in reversed(self._entries):
                    if entry.role == "tool" and entry.call_id == event.get("callId"):
                        entry.status = event.get("status")
                        entry.text = str(event.get("result", ""))[:1200]
                        break
            elif kind == "final":
                self.session_id = event.get("sessionId") or self.session_id
                final = (event.get("text") or "").strip()
                if final and final != last_text:
                    self._add("assistant", final)
                    last_text = final
        return last_text
