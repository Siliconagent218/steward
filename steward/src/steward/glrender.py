"""MuJoCo rendering helpers shared by WSim, asset previews and the Ego-State view.

All rendering runs on the process main thread (macOS OpenGL rule).
"""

from __future__ import annotations

import mujoco


def close_renderer(renderer: mujoco.Renderer) -> None:
    """Close a renderer with its own OpenGL context current.

    ``mujoco.Renderer.close`` frees its MuJoCo render context without making its OpenGL context
    current, which would release GPU objects of whichever renderer happens to be current.
    """
    context = getattr(renderer, "_gl_context", None)
    if context is not None:
        context.make_current()
    renderer.close()


def free_camera(spec: dict) -> mujoco.MjvCamera:
    """MuJoCo free camera: it looks along (cos el cos az, cos el sin az, sin el) at ``lookat``
    from ``distance`` away; a negative elevation looks down."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(v) for v in spec["lookat"]]
    cam.distance = float(spec["distance"])
    cam.azimuth = float(spec["azimuth"])
    cam.elevation = float(spec["elevation"])
    return cam
