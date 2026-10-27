"""Multi-view preview rendering of asset templates and scenes."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from steward.glrender import close_renderer
from steward.scene.extent import entity_obb
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, SceneState, exact, pose

VIEWS = {"front": (1.0, -1.2, 0.8), "top": (0.05, -0.05, 1.6)}


def render_template(template: dict, out_dir: Path, stem: str, size: tuple[int, int] = (480, 360)) -> list[Path]:
    """Render the asset from each view in :data:`VIEWS`; return the PNG paths."""
    ent = Entity.from_dict(template)
    ent.id = "w:asset"
    ent.pose = exact(pose())
    probe = build_mjcf(SceneState(mode="world", entities=[ent]))
    model = probe.compile()
    data = mujoco.MjData(model)
    probe.apply_initial(model, data)
    obb = entity_obb(model, data, probe.names["w:asset"])
    center = np.asarray(obb.center)
    radius = max(float(np.linalg.norm(obb.size)), 0.2)
    cameras = []
    for view, direction in VIEWS.items():
        d = np.asarray(direction) / np.linalg.norm(direction)
        cameras.append({"name": f"preview_{view}", "pos": (center + d * radius * 1.6).tolist(), "target": center.tolist(),
                        "fovy": 45})
    return render_scene(SceneState(mode="world", entities=[ent]), cameras, out_dir, stem, size)


def render_scene(scene: SceneState, cameras: list[dict], out_dir: Path, stem: str,
                 size: tuple[int, int] = (480, 360)) -> list[Path]:
    build = build_mjcf(scene, extra_cameras=cameras)
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, size[1], size[0])
    paths = []
    try:
        for cam in cameras:
            renderer.update_scene(data, camera=cam["name"])
            img = renderer.render()
            path = out_dir / f"{stem}-{cam['name'].removeprefix('preview_')}.png"
            Image.fromarray(img).save(path)
            paths.append(path)
    finally:
        close_renderer(renderer)
    return paths
