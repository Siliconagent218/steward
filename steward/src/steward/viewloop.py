"""Frame-clocked view rendering on a dedicated thread.

A console view (the World State view in the Steward host, the Ego-State view on the robot) renders
on its own thread with its own OpenGL resources, never on a simulation, estimation or request
thread. Views in different processes stay in step through :class:`FrameClock` alone: frame ``k``
falls at ``k / fps`` seconds of Unix time, so renderers told the same rate render at the same
instants without exchanging messages (separate hosts keep their clocks in step with NTP or PTP).
A renderer that cannot finish a frame within one tick skips to the next tick instead of drifting.
"""

from __future__ import annotations

import collections
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from steward.net import Binary

MAX_FPS = 30.0
SIZE_RANGE = (64, 2048)
IDLE_S = 3.0
RATE_WINDOW_S = 2.0


@dataclass(frozen=True)
class ViewSettings:
    """What a view renders: a free camera (``lookat``, ``distance``, ``azimuth``, ``elevation``),
    the frame rate and the image size."""

    camera: dict
    fps: float
    width: int
    height: int

    @staticmethod
    def parse(body: dict) -> ViewSettings:
        """Validate view settings received over HTTP.

        :param body: ``{"camera": {...}, "fps": n, "width": w, "height": h}``.
        :returns: the settings.
        :raises ValueError: when a field is missing or out of range.
        """
        try:
            cam = body["camera"]
            lookat = [float(v) for v in cam["lookat"]]
            camera = {"lookat": lookat, "distance": float(cam["distance"]), "azimuth": float(cam["azimuth"]),
                      "elevation": float(cam["elevation"])}
            fps, width, height = float(body["fps"]), int(body["width"]), int(body["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"view settings need camera {{lookat, distance, azimuth, elevation}}, fps, width, height: {exc}") from None
        numbers = [*lookat, camera["distance"], camera["azimuth"], camera["elevation"]]
        if len(lookat) != 3 or not camera["distance"] > 0 or not all(map(math.isfinite, numbers)):
            raise ValueError("camera needs three finite lookat values and a positive distance")
        if not 0 < fps <= MAX_FPS:
            raise ValueError(f"fps must be in (0, {MAX_FPS:g}]")
        if not (SIZE_RANGE[0] <= width <= SIZE_RANGE[1] and SIZE_RANGE[0] <= height <= SIZE_RANGE[1]):
            raise ValueError(f"width and height must be in {list(SIZE_RANGE)}")
        return ViewSettings(camera, fps, width, height)

    def to_dict(self) -> dict:
        return {"camera": self.camera, "fps": self.fps, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class Frame:
    """One rendered frame: its tick number and Unix time, the encoded image, the camera it was
    rendered with, how long rendering took, and renderer-specific facts (sim time, ES version)."""

    index: int
    stamp: float
    image: bytes
    content_type: str
    camera: dict
    render_ms: float
    meta: dict = field(default_factory=dict)


class FrameClock:
    """Frame ticks on the shared wall clock: tick ``k`` at ``k / fps`` seconds of Unix time."""

    def __init__(self, now: Callable[[], float] = time.time):
        self.now = now

    def next_tick(self, fps: float, after: int | None = None) -> tuple[int, float]:
        """The first tick strictly in the future and after tick ``after``.

        :param fps: frames per second.
        :param after: the last tick already rendered, if any.
        :returns: the tick number and its Unix time.
        """
        index = math.floor(self.now() * fps) + 1
        if after is not None and index <= after:
            index = after + 1
        return index, index / fps


Render = Callable[[ViewSettings], tuple[bytes, str, dict]]


class ViewLoop:
    """Renders one view on its own thread at the ticks of a :class:`FrameClock`.

    ``render(settings)`` returns ``(image, content_type, meta)`` and ``close()`` releases its
    resources; both run only on the loop thread, which therefore owns every OpenGL object the view
    creates. The loop renders only while someone fetched a frame within the last ``idle_s``
    seconds, and keeps the latest frame for :meth:`latest`.
    """

    def __init__(self, name: str, render: Render, close: Callable[[], None] | None = None,
                 clock: FrameClock | None = None, idle_s: float = IDLE_S):
        self.name = name
        self._render = render
        self._close = close
        self.clock = clock or FrameClock()
        self.idle_s = idle_s
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._settings: ViewSettings | None = None
        self._frame: Frame | None = None
        self._stamps: collections.deque[float] = collections.deque()
        self._last_view = -math.inf
        self.error: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> ViewLoop:
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"view-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def configure(self, settings: ViewSettings) -> dict:
        """Render with ``settings`` from the next tick on.

        :param settings: camera, rate and size.
        :returns: the loop status.
        """
        with self._lock:
            self._settings = settings
        self._wake.set()
        return self.status()

    def latest(self, after: int | None = None) -> Frame | None:
        """The newest frame if its tick is after ``after`` (``None`` otherwise); counts as viewing.

        :param after: the tick of the frame the viewer already has.
        :returns: a newer frame or ``None``.
        """
        with self._lock:
            idle = time.monotonic() - self._last_view > self.idle_s
            self._last_view = time.monotonic()
            frame = self._frame
        if idle:
            self._wake.set()
        return frame if frame is not None and (after is None or frame.index > after) else None

    def status(self) -> dict:
        """Settings, measured frame rate over the last two seconds, last tick and render time."""
        with self._lock:
            stamps = list(self._stamps)
            frame, settings = self._frame, self._settings
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0
        return {"settings": settings.to_dict() if settings else None, "fps": round(rate, 2),
                "frame": frame.index if frame else None, "renderMs": frame.render_ms if frame else None,
                "error": self.error}

    def _run(self) -> None:
        try:
            last: int | None = None
            while self._running:
                with self._lock:
                    settings = self._settings
                    active = time.monotonic() - self._last_view <= self.idle_s
                if settings is None or not active:
                    self._wake.wait(0.2)
                    self._wake.clear()
                    continue
                index, stamp = self.clock.next_tick(settings.fps, last)
                delay = stamp - self.clock.now()
                if delay > 0 and self._wake.wait(delay):
                    self._wake.clear()
                    continue  # new settings or a stop request: take the next tick with them
                started = time.perf_counter()
                try:
                    image, content_type, meta = self._render(settings)
                except Exception as exc:  # noqa: BLE001 - kept in status and frame responses; the loop retries
                    self.error = f"{type(exc).__name__}: {exc}"
                    self._wake.wait(0.5)
                    continue
                self.error = None
                last = index
                frame = Frame(index, stamp, image, content_type, settings.camera,
                              round((time.perf_counter() - started) * 1000.0, 1), meta)
                with self._lock:
                    self._frame = frame
                    self._stamps.append(stamp)
                    while self._stamps and self._stamps[0] < stamp - RATE_WINDOW_S:
                        self._stamps.popleft()
        finally:
            if self._close is not None:
                self._close()


def frame_response(frame: Frame, status: dict) -> Binary:
    """An HTTP response carrying a frame image, with its tick, camera and the loop status in headers.

    :param frame: the frame.
    :param status: :meth:`ViewLoop.status` of the loop that rendered it.
    :returns: the response body.
    """
    return Binary(frame.image, frame.content_type, {
        "X-Frame-Index": str(frame.index), "X-Frame-Stamp": f"{frame.stamp:.3f}",
        "X-Frame-Camera": json.dumps(frame.camera), "X-Frame-Meta": json.dumps(frame.meta),
        "X-Render-Fps": str(status["fps"]), "X-Render-Ms": str(frame.render_ms),
        "Cache-Control": "no-store"})
