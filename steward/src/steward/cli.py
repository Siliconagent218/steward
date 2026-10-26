"""Steward command line.

    steward build-kitchen            generate, validate and publish the Phase 0 kitchen
    steward world                    run WSim + Observer + WCA + robot ports
    steward ego                      run one robot's Ego Runtime + Ego API
    steward up                       run world and ego for every scenario robot in one process
    steward demo                     scripted "cup to table" mission, checked against World State
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "var" / "steward"


def _robot_model() -> dict:
    from steward.assets.generators import steward_bot

    return steward_bot().to_dict()


def _wait_forever(stoppers: list, pump=None) -> None:
    """Block until SIGINT/SIGTERM; the main thread serves render requests through ``pump``."""
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    while not done.is_set():
        if pump is not None:
            pump(0.05)
        else:
            done.wait(0.5)
    for stop in stoppers:
        stop()


def cmd_build_kitchen(args) -> int:
    from steward.authoring.service import AuthoringService, build_default_kitchen

    out = build_default_kitchen(AuthoringService(args.root))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _host(args, run_dir: Path):
    from steward.world.server import StewardHost

    ports = {"w:robot-1": args.robot_port} if getattr(args, "robot_port", 0) else {}
    return StewardHost(args.root, scenario=args.scenario, speed=args.speed, seed=args.seed, run_dir=run_dir,
                       wca_port=args.wca_port, robot_ports=ports).start()


def cmd_world(args) -> int:
    run_dir = Path(args.root) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    host = _host(args, run_dir)
    print(json.dumps({"wca": host.wca.url, "robots": {k: v.url for k, v in host.robot_servers.items()},
                      "runDir": str(run_dir)}), flush=True)
    _wait_forever([host.stop], host.pump)
    return 0


def _start_ego(robot_url: str, api_port: int, out_dir: Path):
    from steward.ego.runtime import EgoRuntime, RobotLink
    from steward.ego.server import ego_api

    runtime = EgoRuntime(RobotLink(robot_url), _robot_model(), out_dir=out_dir)
    runtime.start()
    api = ego_api(runtime, port=api_port).start()
    return runtime, api


def cmd_ego(args) -> int:
    runtime, api = _start_ego(args.robot_url, args.api_port, Path(args.root) / "ego" / "observations")
    print(json.dumps({"egoApi": api.url, "robot": args.robot_url}), flush=True)
    _wait_forever([api.stop, runtime.stop])
    return 0


def cmd_up(args) -> int:
    run_dir = Path(args.root) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    host = _host(args, run_dir)
    robot_url = host.robot_servers["w:robot-1"].url
    runtime, api = _start_ego(robot_url, args.api_port, run_dir / "ego")
    print(json.dumps({"wca": host.wca.url, "robot": robot_url, "egoApi": api.url, "runDir": str(run_dir),
                      "task": host.scenario["task"]}, ensure_ascii=False), flush=True)
    _wait_forever([api.stop, runtime.stop, host.stop], host.pump)
    return 0


def cmd_demo(args) -> int:
    from steward.authoring.service import AuthoringService, build_default_kitchen
    from steward.mission import MissionFailed, cup_to_table

    if "phase0-kitchen" not in AuthoringService(args.root).list_scenarios():
        build_default_kitchen(AuthoringService(args.root))
    run_dir = Path(args.root) / "runs" / ("demo-" + time.strftime("%Y%m%d-%H%M%S"))
    host = _host(args, run_dir)
    runtime, api = _start_ego(host.robot_servers["w:robot-1"].url, 0, run_dir / "ego")
    t0 = time.monotonic()
    outcome: dict = {}

    def mission() -> None:
        try:
            outcome["result"] = cup_to_table(api.url)
        except MissionFailed as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=mission, name="mission", daemon=True)
    worker.start()
    while worker.is_alive():
        host.pump(0.05)
    ok_mission = "result" in outcome
    result = outcome.get("result") or {"error": outcome.get("error")}
    time.sleep(1.0 / max(args.speed, 0.1))
    ws = host.sim.world_state()
    on_table = any(r["type"] == "supported-by" and r["a"]["entity"] == "w:cup-1" and r["b"]["entity"] == "w:table-1"
                   for r in ws["relations"])
    (run_dir / "overview-final.png").write_bytes(host.sim.render("overview", 800, 600))
    report = {
        "missionCompleted": ok_mission,
        "groundTruthCupOnTable": on_table,
        "wallSeconds": round(time.monotonic() - t0, 1),
        "simSeconds": round(host.sim.time, 1),
        "mission": result.get("log", result),
        "observer": host.observer.report(),
        "egoObjects": [(o["id"], o["category"], o["position"], o["uncertaintyM"], o["existence"])
                       for o in runtime.summary()["objects"]],
        "observerMatches": host.observer.latest.get("w:robot-1", {}).get("matches"),
        "events": [e for e in host.sim.events if e["type"].startswith("grasp/")],
        "runDir": str(run_dir),
    }
    api.stop()
    runtime.stop()
    host.stop()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok_mission and on_table else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steward")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Steward data root (assets, scenes, runs)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-kitchen").set_defaults(fn=cmd_build_kitchen)
    for name, fn in (("world", cmd_world), ("up", cmd_up), ("demo", cmd_demo)):
        p = sub.add_parser(name)
        p.add_argument("--scenario", default="phase0-kitchen")
        p.add_argument("--speed", type=float, default=1.0 if name != "demo" else 4.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--wca-port", type=int, default=7400 if name != "demo" else 0)
        p.add_argument("--robot-port", type=int, default=7410 if name != "demo" else 0)
        if name == "up":
            p.add_argument("--api-port", type=int, default=7420)
        p.set_defaults(fn=fn)
    p = sub.add_parser("ego")
    p.add_argument("--robot-url", default="http://127.0.0.1:7410")
    p.add_argument("--api-port", type=int, default=7420)
    p.set_defaults(fn=cmd_ego)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
