"""Run the Phase 0 cup-to-table mission across seeds and print a progress bar per seed.

    .venv/bin/python scripts/seed_sweep.py --seeds 20 --speed 6
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService, build_default_kitchen
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.mission import MissionFailed, cup_to_table
from steward.world.server import StewardHost


def bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total}"


def run_seed(root: str, seed: int, speed: float) -> tuple[bool, str]:
    host = StewardHost(root, scenario="phase0-kitchen", speed=speed, seed=seed).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    try:
        cup_to_table(api.url)
        time.sleep(0.3)
        ws = host.sim.world_state()
        on_table = any(r["type"] == "supported-by" and r["a"]["entity"] == "w:cup-1" and r["b"]["entity"] == "w:table-1"
                       for r in ws["relations"])
        return on_table, "ok" if on_table else "mission reported success but the cup is not on the table"
    except MissionFailed as exc:
        lines = [str(exc)[:240]]
        for o in runtime.summary()["objects"]:
            if o["category"] != "wall":
                lines.append(f"    {o['id']} {o['category']} pos={o['position']} size={o['sizeM']} "
                             f"sigma={o['uncertaintyM']} on={o['on']} p={o['categories']}")
        return False, "\n".join(lines)
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--speed", type=float, default=6.0)
    args = parser.parse_args()
    root = tempfile.mkdtemp(prefix="steward-sweep-")
    build_default_kitchen(AuthoringService(root))
    failures = []
    t0 = time.monotonic()
    for i in range(args.seeds):
        seed = args.first + i
        ok, message = run_seed(root, seed, args.speed)
        if not ok:
            failures.append((seed, message))
        print(f"{bar(i + 1, args.seeds)}  seed {seed}: {'ok' if ok else 'FAIL ' + message}  "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
    print(f"done: {args.seeds - len(failures)}/{args.seeds} passed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
