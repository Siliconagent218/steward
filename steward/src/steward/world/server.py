"""Steward host: WSim, Observer and authoring behind the World Control API, plus robot ports.

Two kinds of listeners keep the information boundary explicit:

* the **World Control API** (WCA) port serves World State, clock and episode control, rendering,
  Observer reports and asset/scene/scenario authoring. Only the Steward plane (the World
  Harness, operators, tests) may reach it;
* one **robot port** per robot serves only that robot's sensor frames, camera images, actuator
  commands and one-way telemetry. Nothing on a robot port returns World State identifiers.
"""

from __future__ import annotations

from pathlib import Path

from steward.authoring.service import AuthoringError, AuthoringService
from steward.net import Binary, HttpError, JsonServer
from steward.observer.observer import Observer
from steward.world.sim import WorldSim


class StewardHost:
    def __init__(self, root: Path | str, *, scenario: str | None = None, speed: float = 1.0, seed: int = 0,
                 run_dir: Path | None = None, wca_port: int = 0, robot_ports: dict[str, int] | None = None):
        self.authoring = AuthoringService(root)
        self.speed = speed
        self.seed = seed
        self.run_dir = run_dir
        self.sim: WorldSim | None = None
        self.scenario: dict | None = None
        self.observer = Observer(self._require_sim, run_dir)
        self.wca = JsonServer(self._wca_routes(), port=wca_port, name="wca")
        self._robot_port_config = robot_ports or {}
        self.robot_servers: dict[str, JsonServer] = {}
        if scenario is not None:
            self.load_episode(scenario)

    # --------------------------------------------------------------- episodes
    def load_episode(self, name: str) -> dict:
        scenario = self.authoring.load_scenario(name)
        if self.sim is not None:
            self.sim.stop()
        self.sim = WorldSim(scenario["sceneState"], speed=self.speed, seed=self.seed)
        self.scenario = scenario
        for rid in scenario["robots"]:
            if rid not in self.robot_servers:
                server = JsonServer(self._robot_routes(rid), port=self._robot_port_config.get(rid, 0), name=f"robot-{rid}")
                self.robot_servers[rid] = server.start()
        return {"scenario": name, "robots": {rid: self.robot_servers[rid].url for rid in scenario["robots"]},
                "task": scenario["task"]}

    def _require_sim(self) -> WorldSim:
        if self.sim is None:
            raise HttpError(409, "no episode is loaded")
        return self.sim

    def start(self) -> StewardHost:
        self.wca.start()
        if self.sim is not None:
            self.sim.start()
        return self

    def stop(self) -> None:
        for server in self.robot_servers.values():
            server.stop()
        self.wca.stop()
        if self.sim is not None:
            self.sim.stop()

    # ------------------------------------------------------------------- WCA
    def _wca_routes(self) -> dict:
        a = self.authoring

        def guard(fn):
            def wrapped(q, b):
                try:
                    return fn(q, b or {})
                except AuthoringError as exc:
                    raise HttpError(400, str(exc)) from None
                except KeyError as exc:
                    raise HttpError(404, str(exc)) from None
            return wrapped

        def episode(q, b):
            action = b.get("action")
            if action == "load":
                result = self.load_episode(b["scenario"])
                self.sim.start()
                return result
            if action == "reset":
                self._require_sim().reset()
                return {"reset": True}
            raise HttpError(400, "action must be load or reset")

        def clock(q, b):
            sim = self._require_sim()
            action = b.get("action")
            if action == "pause":
                sim.paused = True
            elif action == "resume":
                sim.paused = False
            elif action == "speed":
                sim.speed = float(b["value"])
            else:
                raise HttpError(400, "action must be pause, resume or speed")
            return {"paused": sim.paused, "speed": sim.speed, "t": sim.time}

        def render(q, b):
            sim = self._require_sim()
            camera = q.get("camera", "overview")
            if camera.startswith("robot:"):
                camera = sim.robot_camera_name(camera.removeprefix("robot:"))
            return Binary(sim.render(camera, int(q.get("width", 800)), int(q.get("height", 600))), "image/png")

        return {
            ("GET", "/wca/status"): guard(lambda q, b: {"scenario": (self.scenario or {}).get("name"),
                                                        **(self.sim.status() if self.sim else {"loaded": False})}),
            ("GET", "/wca/scene"): guard(lambda q, b: self._require_sim().world_state()),
            ("POST", "/wca/clock"): guard(clock),
            ("POST", "/wca/episode"): guard(episode),
            ("GET", "/wca/render"): guard(render),
            ("GET", "/observer/latest"): guard(lambda q, b: self.observer.latest),
            ("GET", "/observer/report"): guard(lambda q, b: self.observer.report()),
            ("GET", "/assets/generators"): guard(lambda q, b: {"generators": a.generators()}),
            ("GET", "/assets"): guard(lambda q, b: {"assets": a.library.search(q.get("query"), q.get("category"))}),
            ("GET", "/assets/get"): guard(lambda q, b: a.library.get(q["name"], q.get("version"))),
            ("POST", "/assets/create"): guard(lambda q, b: a.create_asset(b.get("generator"), b.get("params"),
                                                                          b.get("template"))),
            ("POST", "/assets/preview"): guard(lambda q, b: {"images": a.preview_asset(b["template"],
                                                                                       b.get("stem", "draft"))}),
            ("POST", "/assets/publish"): guard(lambda q, b: a.publish_asset(b["name"], b["template"], b.get("meta"))),
            ("GET", "/scenes"): guard(lambda q, b: {"scenes": a.list_scenes()}),
            ("POST", "/scenes/compose"): guard(lambda q, b: a.compose_scene(b)),
            ("POST", "/scenes/render"): guard(lambda q, b: {"images": a.render_scene_views(b["name"])}),
            ("GET", "/scenarios"): guard(lambda q, b: {"scenarios": a.list_scenarios()}),
            ("POST", "/scenarios/save"): guard(lambda q, b: a.save_scenario(b)),
        }

    # ------------------------------------------------------------ robot port
    def _robot_routes(self, rid: str) -> dict:
        def camera(q, b):
            sim = self._require_sim()
            png = sim.render(sim.robot_camera_name(rid), int(q.get("width", 320)), int(q.get("height", 240)))
            return Binary(png, "image/png")

        def telemetry(q, b):
            if not isinstance(b, dict):
                raise HttpError(400, "telemetry body must be an ego SceneState")
            self.observer.ingest(rid, b)
            return {"accepted": True}

        return {
            ("GET", "/sensors"): lambda q, b: self._require_sim().sensors(rid),
            ("GET", "/camera/rgb"): camera,
            ("POST", "/command"): lambda q, b: self._require_sim().command(rid, b or {}),
            ("POST", "/telemetry"): telemetry,
        }
