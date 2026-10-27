"""MuJoCo rendering helpers shared by WSim, asset previews and the Ego-State view.

Each renderer is created, used and closed on one thread, which owns its OpenGL context: WSim
sensor images and asset previews on the process main thread, console views on their view loop
threads (:mod:`steward.viewloop`).
"""

from __future__ import annotations

import io

import mujoco
import numpy as np
from PIL import Image

IMAGE_FORMATS = {"png": "image/png", "jpeg": "image/jpeg"}


def encode_image(pixels: np.ndarray, fmt: str = "png") -> bytes:
    """Encode RGB pixels as PNG (lossless) or JPEG (quality 85, several times faster to encode)."""
    buf = io.BytesIO()
    if fmt == "jpeg":
        Image.fromarray(pixels).save(buf, format="JPEG", quality=85)
    elif fmt == "png":
        Image.fromarray(pixels).save(buf, format="PNG")
    else:
        raise ValueError(f"image format must be one of {sorted(IMAGE_FORMATS)}, got {fmt!r}")
    return buf.getvalue()


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
