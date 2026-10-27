"""Steward command line.

    steward build-kitchen            generate, validate and publish the Phase 0 kitchen
    steward world                    run WSim + Observer + WCA + robot ports
    steward ego                      run one robot's Ego Runtime + Ego API
    steward up                       run world and ego for every scenario robot in one process
    steward demo                     scripted "cup to table" mission, checked against World State
    steward shuttle                  carry objects around table → cabinet → sideboard, step by step
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

STEWARD_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = STEWARD_DIR / "var" / "steward"


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
                       wca_port=args.wca_port, robot_ports=ports, chat=_chat_config(args)).start()


def _chat_config(args):
    """World Harness chat for the console: the steward-local DSH launcher and world home."""
    from steward.world.chat import ChatConfig

    if not getattr(args, "world_harness", None):
        return None
    home = Path(args.world_harness)
    return ChatConfig(dsh=STEWARD_DIR / "dsh" / "node_modules" / ".bin" / "dsh", dsh_home=home,
                      cwd=STEWARD_DIR, env_file=STEWARD_DIR / ".env")


def cmd_world(args) -> int:
    run_dir = Path(args.root) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    host = _host(args, run_dir)
    print(json.dumps({"console": f"{host.wca.url}/console", "wca": host.wca.url, "robots": {k: v.url for k, v in host.robot_servers.items()},
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
    print(json.dumps({"console": f"{host.wca.url}/console", "wca": host.wca.url, "robot": robot_url, "egoApi": api.url, "runDir": str(run_dir),
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


def _ground_truth_locations(scene: dict, supports: list[str], objects: list[str]) -> dict[str, str | None]:
    """Support category under each carried object category, from World State relations."""
    category = {e["id"]: e["category"]["value"] for e in scene["entities"]}
    out: dict[str, str | None] = {cat: None for cat in objects}
    for rel in scene["relations"]:
        a, b = category.get(rel["a"]["entity"]), category.get(rel["b"]["entity"])
        if rel["type"] == "supported-by" and a in out and b in supports:
            out[a] = b
    return out


SUPPORT_NAMES = {"table": "餐桌", "cabinet": "柜子", "sideboard": "餐边柜", "counter": "操作台"}
OBJECT_NAMES = {"cup": "杯子", "bowl": "碗", "plate": "盘子"}


def cmd_shuttle(args) -> int:
    """Shuttle objects between table, cabinet and sideboard, checking each move against World State."""
    from steward.mission import MissionFailed, shuttle
    from steward.net import call

    supports, objects = ["table", "cabinet", "sideboard"], ["cup", "bowl"]
    host = runtime = api = None
    if args.api:
        api_url = args.api.rstrip("/")
        def world_scene():
            return call("GET", f"{args.wca.rstrip('/')}/wca/scene")
    else:
        from steward.authoring.service import AuthoringService, build_default_kitchen

        if args.scenario not in AuthoringService(args.root).list_scenarios():
            build_default_kitchen(AuthoringService(args.root))
        run_dir = Path(args.root) / "runs" / ("shuttle-" + time.strftime("%Y%m%d-%H%M%S"))
        host = _host(args, run_dir)
        runtime, api = _start_ego(host.robot_servers["w:robot-1"].url, 0, run_dir / "ego")
        api_url = api.url
        def world_scene():
            return host.sim.world_state()
        print(json.dumps({"console": f"{host.wca.url}/console"}), flush=True)
    t0 = time.monotonic()
    results: list[bool] = []

    def on_step(index: int, move: dict) -> None:
        time.sleep(0.5)
        truth = _ground_truth_locations(world_scene(), supports, objects)
        ok = truth.get(move["object"]) == move["to"]
        results.append(ok)
        if args.steps > 0:
            filled = int(30 * (index + 1) / args.steps)
            progress = f"[{'#' * filled}{'-' * (30 - filled)}] {index + 1}/{args.steps}"
        else:
            progress = f"[第 {index + 1} 步，第 {index // 3 + 1} 轮]"
        print(f"{progress}  "
              f"{OBJECT_NAMES.get(move['object'], move['object'])}：{SUPPORT_NAMES.get(move['from'], '其他位置')} → "
              f"{SUPPORT_NAMES[move['to']]}  World State {'确认' if ok else '不一致'} "
              f"{ {OBJECT_NAMES.get(k, k): SUPPORT_NAMES.get(v, v) for k, v in truth.items()} }  "
              f"({time.monotonic() - t0:.0f}s)", flush=True)

    def on_failure(index: int, message: str) -> None:
        print(f"[第 {index + 1} 步失败，重新观察后继续]  {message}  ({time.monotonic() - t0:.0f}s)", flush=True)

    outcome: dict = {}

    def mission() -> None:
        try:
            outcome["result"] = shuttle(api_url, args.steps, tuple(supports), tuple(objects), on_step=on_step,
                                        on_failure=on_failure)
        except MissionFailed as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=mission, name="shuttle", daemon=True)
    worker.start()
    interrupted = False
    try:
        while worker.is_alive():
            if host is not None:
                host.pump(0.05)
            else:
                worker.join(0.2)
    except KeyboardInterrupt:
        interrupted = True
        if host is None:
            call("POST", f"{api_url}/skill/stop", {})
    finished = "result" in outcome or (interrupted and "error" not in outcome)
    ok = finished and (args.steps <= 0 or len(results) == args.steps) and all(results)
    print(json.dumps({"completedSteps": len(results), "allConfirmedByWorldState": ok,
                      "error": outcome.get("error")}, ensure_ascii=False), flush=True)
    if host is not None:
        api.stop()
        runtime.stop()
        host.stop()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steward")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Steward data root (assets, scenes, runs)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-kitchen").set_defaults(fn=cmd_build_kitchen)
    for name, fn in (("world", cmd_world), ("up", cmd_up), ("demo", cmd_demo), ("shuttle", cmd_shuttle)):
        p = sub.add_parser(name)
        p.add_argument("--scenario", default="phase0-kitchen" if name == "demo" else "phase0-shuttle")
        in_process = name in ("demo", "shuttle")
        p.add_argument("--speed", type=float, default=4.0 if in_process else 1.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--wca-port", type=int, default=0 if in_process else 7400)
        p.add_argument("--robot-port", type=int, default=0 if in_process else 7410)
        if not in_process:
            p.add_argument("--world-harness", default=str(STEWARD_DIR / "var" / "dsh" / "world"),
                           help="DSH home of the World Harness used by the console chat; '' disables the chat")
        if name == "up":
            p.add_argument("--api-port", type=int, default=7420)
        if name == "shuttle":
            p.add_argument("--steps", type=int, default=6, help="number of moves; 0 repeats until Ctrl-C")
            p.add_argument("--api", help="Ego API of a running `steward up`; omit to run in-process")
            p.add_argument("--wca", default="http://127.0.0.1:7400", help="WCA of the running world (with --api)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("ego")
    p.add_argument("--robot-url", default="http://127.0.0.1:7410")
    p.add_argument("--api-port", type=int, default=7420)
    p.set_defaults(fn=cmd_ego)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
