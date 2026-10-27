"""Robot-side perception for RGB-D frames: segments → 3D points → geometry and category.

The world only provides raw sensor data: an RGB image, a depth image and anonymous instance masks
(as a class-agnostic segmenter would produce). This module back-projects every masked region into
the robot's ``map`` frame using the robot's own camera pose estimate, measures the visible geometry,
and classifies it against the robot's shape priors. It never sees World State identifiers.

Camera convention: the sensor mount frame looks along +x with +z up; pixel (u, v) at depth d maps
to ``[d, -(u - cx) d / f, -(v - cy) d / f]`` in that frame, with ``f = (H / 2) / tan(fovy / 2)``.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass

import numpy as np

from steward.math3d import matrix_from_quat


@dataclass
class PerceptionConfig:
    max_range: float = 6.0
    min_points: int = 20
    max_points: int = 400
    floor_cut: float = 0.012
    occluder_gap: float = 0.15  # something this much nearer right below a region hides its base
    occluded_share: float = 0.3  # share of the lower edge hidden that leaves the base unseen


@dataclass
class Segment:
    """One masked region of a frame, in the robot's map frame."""

    points: np.ndarray  # (N, 3)
    color: np.ndarray  # mean RGB in [0, 1]
    pixels: int
    invalid_frac: float  # share of masked pixels without a depth reading (glass, far)
    border: bool  # the region touches the image border (truncated view)
    range: float  # median distance from the camera
    bottom_seen: bool = True  # the lowest points are the object's base, not an image edge or an occluder


def decode_rgbd(blob: bytes) -> dict:
    """Unpack the robot port's ``/sensors/rgbd`` payload."""
    z = np.load(io.BytesIO(blob))
    return {"rgb": z["rgb"], "depth": z["depth"].astype(np.float32) / 1000.0, "mask": z["mask"],
            "fovy": float(z["fovy"]), "header": json.loads(str(z["header"]))}


def segments(frame: dict, cam_pose: dict, config: PerceptionConfig | None = None,
             rng: np.random.Generator | None = None) -> list[Segment]:
    """Back-project every masked region of an RGB-D frame into map-frame points."""
    cfg = config or PerceptionConfig()
    rng = rng or np.random.default_rng(0)
    depth, mask, rgb = frame["depth"], frame["mask"], frame["rgb"]
    h, w = depth.shape
    f = (h / 2.0) / math.tan(math.radians(frame["fovy"]) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rot = matrix_from_quat(cam_pose["q"])
    origin = np.asarray(cam_pose["p"], float)
    out = []
    for instance in np.unique(mask):
        if instance == 0:
            continue
        ys, xs = np.nonzero(mask == instance)
        d = depth[ys, xs]
        valid = (d > 0.05) & (d < cfg.max_range)
        if int(valid.sum()) < cfg.min_points:
            continue
        u, v, dv = xs[valid].astype(np.float64), ys[valid].astype(np.float64), d[valid].astype(np.float64)
        if len(dv) > cfg.max_points:
            keep = rng.choice(len(dv), cfg.max_points, replace=False)
            u, v, dv = u[keep], v[keep], dv[keep]
        local = np.stack([dv, -(u - cx) * dv / f, -(v - cy) * dv / f], axis=1)
        points = local @ rot.T + origin
        points = points[points[:, 2] > cfg.floor_cut]
        if len(points) < cfg.min_points:
            continue
        out.append(Segment(
            points=points,
            color=rgb[ys, xs].reshape(-1, 3).mean(axis=0) / 255.0,
            pixels=len(ys),
            invalid_frac=float(1.0 - valid.mean()),
            border=bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1),
            range=float(np.median(dv)),
            bottom_seen=_bottom_seen(mask, depth, instance, ys, xs, cfg),
        ))
    return out


def _bottom_seen(mask: np.ndarray, depth: np.ndarray, instance: int, ys: np.ndarray, xs: np.ndarray,
                 cfg: PerceptionConfig) -> bool:
    """Whether a region's lowest pixels show the object's base: not when the region reaches the
    lower image edge (the base is out of view), nor when much of its lower edge has something
    clearly nearer right below it (the base is hidden, for example a wall behind a counter). A
    base on the floor has floor right below it, at nearly the same distance."""
    h = mask.shape[0]
    if ys.max() >= h - 1:
        return False
    sel = np.zeros(mask.shape, dtype=bool)
    sel[ys, xs] = True
    edge_y, edge_x = np.nonzero(sel[:-2] & ~sel[1:-1])
    if len(edge_y) == 0:
        return True
    here = depth[edge_y, edge_x]
    below = depth[edge_y + 2, edge_x]
    other = mask[edge_y + 2, edge_x] != instance
    measured = (here > 0.05) & other
    if not measured.any():
        return True
    hidden = measured & (below > 0.05) & (below < here - cfg.occluder_gap)
    return float(hidden.sum()) / float(measured.sum()) < cfg.occluded_share


def _hull(xy: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (monotone chain), counter-clockwise."""
    pts = np.unique(np.round(xy, 4), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def box_of(points: np.ndarray, trim: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Oriented box of a point set: centre, full extents (x along the longer horizontal side) and
    yaw, from the minimum-area rectangle around the footprint. With ``trim`` the box is fitted
    again without the given share of points at either end of each horizontal axis (scattered depth
    noise)."""
    if trim > 0 and len(points) > 50:
        centre, _, yaw = box_of(points)
        c, s = math.cos(yaw), math.sin(yaw)
        rel = points - centre
        coords = np.stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c, rel[:, 2]], axis=1)
        lo, hi = np.quantile(coords, trim, axis=0), np.quantile(coords, 1 - trim, axis=0)
        # Trim horizontally only: thin vertical parts such as table legs hold few points but set the base.
        keep = np.all((coords[:, :2] >= lo[:2]) & (coords[:, :2] <= hi[:2]), axis=1)
        if keep.sum() >= 20:
            points = points[keep]
    xy = points[:, :2]
    hull = _hull(xy)
    best = None
    if len(hull) >= 3:
        edges = np.roll(hull, -1, axis=0) - hull
        for angle in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (math.pi / 2), 4)):
            c, s = math.cos(angle), math.sin(angle)
            lx, ly = hull @ np.array([c, s]), hull @ np.array([-s, c])
            area = (lx.max() - lx.min()) * (ly.max() - ly.min())
            if best is None or area < best[0]:
                best = (area, angle)
    yaw = best[1] if best else 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = xy @ np.array([c, s]), xy @ np.array([-s, c])
    ex, ey = lx.max() - lx.min(), ly.max() - ly.min()
    mx, my = (lx.max() + lx.min()) / 2, (ly.max() + ly.min()) / 2
    if ey > ex:
        ex, ey, yaw = ey, ex, yaw + math.pi / 2
    centre_xy = mx * np.array([c, s]) + my * np.array([-s, c])
    z_lo, z_hi = points[:, 2].min(), points[:, 2].max()
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return np.array([centre_xy[0], centre_xy[1], (z_lo + z_hi) / 2]), np.array([ex, ey, z_hi - z_lo]), yaw


def box_at_yaw(points: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Box of a point set in a given heading (for parts aligned with a known frame); the longer
    horizontal side is reported as x, turning the heading by a quarter when needed."""
    c, s = math.cos(yaw), math.sin(yaw)
    xy = points[:, :2]
    lx, ly = xy @ np.array([c, s]), xy @ np.array([-s, c])
    ex, ey = lx.max() - lx.min(), ly.max() - ly.min()
    mx, my = (lx.max() + lx.min()) / 2, (ly.max() + ly.min()) / 2
    centre_xy = mx * np.array([c, s]) + my * np.array([-s, c])
    if ey > ex:
        ex, ey, yaw = ey, ex, yaw + math.pi / 2
    z_lo, z_hi = points[:, 2].min(), points[:, 2].max()
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return np.array([centre_xy[0], centre_xy[1], (z_lo + z_hi) / 2]), np.array([ex, ey, z_hi - z_lo]), yaw
