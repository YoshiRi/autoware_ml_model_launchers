from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import subprocess
from typing import Any


BACKENDS = ("mss", "ffmpeg")

# "2560/344x1600/215+0+0" as printed by `xrandr --listmonitors`
_XRANDR_GEOMETRY = re.compile(r"(\d+)(?:/\d+)?x(\d+)(?:/\d+)?\+(-?\d+)\+(-?\d+)")
# "1920x1080+2560+0", the geometry string users type
_REGION = re.compile(r"^\s*(\d+)x(\d+)(?:\+(-?\d+)\+(-?\d+))?\s*$")


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Region:
    """A screen rectangle in X11 display coordinates."""

    width: int
    height: int
    x: int = 0
    y: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CaptureError(f"region must have a positive size: {self}")

    @property
    def geometry(self) -> str:
        return f"{self.width}x{self.height}+{self.x}+{self.y}"

    def to_mss_monitor(self) -> dict[str, int]:
        return {"left": self.x, "top": self.y, "width": self.width, "height": self.height}

    def to_json(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height, "x": self.x, "y": self.y}


def parse_region(text: str) -> Region:
    """Parse an X11 style geometry, "1920x1080+2560+0" or "1920x1080"."""
    match = _REGION.match(text or "")
    if match is None:
        raise CaptureError(f"region must look like WxH+X+Y: {text!r}")
    width, height, x, y = match.groups()
    return Region(width=int(width), height=int(height), x=int(x or 0), y=int(y or 0))


def parse_xrandr_monitors(text: str) -> list[Region]:
    """Read `xrandr --listmonitors` output into one region per monitor."""
    regions = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Monitors:"):
            continue
        match = _XRANDR_GEOMETRY.search(stripped)
        if match is None:
            continue
        width, height, x, y = (int(value) for value in match.groups())
        regions.append(Region(width=width, height=height, x=x, y=y))
    return regions


def list_monitors(display: str | None = None) -> list[Region]:
    """List monitors, preferring xrandr so both backends see the same layout."""
    if shutil.which("xrandr"):
        environment = dict(os.environ)
        if display:
            environment["DISPLAY"] = display
        try:
            result = subprocess.run(
                ["xrandr", "--listmonitors"],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CaptureError(f"xrandr failed: {exc}") from exc
        if result.returncode == 0:
            regions = parse_xrandr_monitors(result.stdout)
            if regions:
                return regions

    monitors = _mss_monitors()
    if monitors:
        return monitors
    raise CaptureError("could not list monitors; pass an explicit region instead")


def resolve_region(
    region: str = "",
    monitor: int = 0,
    display: str | None = None,
) -> Region:
    """Turn the region/monitor parameters into one rectangle to capture."""
    # An explicit region wins; monitor 0 means the whole virtual screen.
    if region:
        return parse_region(region)

    monitors = list_monitors(display)
    if monitor <= 0:
        left = min(item.x for item in monitors)
        top = min(item.y for item in monitors)
        right = max(item.x + item.width for item in monitors)
        bottom = max(item.y + item.height for item in monitors)
        return Region(width=right - left, height=bottom - top, x=left, y=top)
    if monitor > len(monitors):
        raise CaptureError(f"monitor {monitor} does not exist; found {len(monitors)}")
    return monitors[monitor - 1]


def select_backend(backend: str = "auto") -> str:
    """Resolve the backend name, falling back to ffmpeg when mss is missing."""
    if backend not in ("auto", *BACKENDS):
        raise CaptureError(f"unknown backend: {backend}")
    if backend != "auto":
        return backend
    return "mss" if _mss_module() is not None else "ffmpeg"


def normalize_display(display: str | None = None) -> str:
    """Return a DISPLAY string that always names a screen, as x11grab wants."""
    value = display or os.environ.get("DISPLAY") or ":0"
    return value if "." in value.rsplit(":", 1)[-1] else f"{value}.0"


def build_x11grab_command(
    region: Region,
    fps: float,
    display: str | None = None,
    show_cursor: bool = False,
) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "x11grab",
        "-draw_mouse",
        "1" if show_cursor else "0",
        "-framerate",
        f"{fps:g}",
        "-video_size",
        f"{region.width}x{region.height}",
        "-i",
        f"{normalize_display(display)}+{region.x},{region.y}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]


class FfmpegScreenCapture:
    """Reads BGR frames from an ffmpeg x11grab pipe, which needs no python deps."""

    def __init__(
        self,
        region: Region,
        fps: float,
        display: str | None = None,
        show_cursor: bool = False,
    ) -> None:
        import numpy

        self.region = region
        self.paced = True  # ffmpeg emits frames at -framerate, so do not sleep
        self._numpy = numpy
        self._frame_bytes = region.width * region.height * 3
        self.command = build_x11grab_command(region, fps, display, show_cursor)
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise CaptureError(f"could not start ffmpeg: {exc}") from exc

    def read(self) -> Any:
        buffer = bytearray(self._frame_bytes)
        view = memoryview(buffer)
        read = 0
        while read < self._frame_bytes:
            if self.process.stdout is None:
                raise CaptureError("ffmpeg stdout is closed")
            chunk = self.process.stdout.readinto(view[read:])
            if not chunk:
                raise CaptureError(self._exit_reason())
            read += chunk
        frame = self._numpy.frombuffer(bytes(buffer), self._numpy.uint8)
        return frame.reshape(self.region.height, self.region.width, 3)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()

    def _exit_reason(self) -> str:
        self.process.poll()
        stderr = b""
        if self.process.stderr is not None:
            try:
                stderr = self.process.stderr.read() or b""
            except OSError:
                stderr = b""
        detail = stderr.decode(errors="replace").strip()
        return f"x11grab stopped ({self.process.returncode}): {detail or 'no output'}"


class MssScreenCapture:
    """Grabs frames through mss, which is faster but is an extra dependency."""

    def __init__(self, region: Region, display: str | None = None) -> None:
        import cv2
        import numpy

        module = _mss_module()
        if module is None:
            raise CaptureError("mss is not installed; use the ffmpeg backend")

        self.region = region
        self.paced = False  # grabs return immediately, so the caller must pace
        self._cv2 = cv2
        self._numpy = numpy
        self._monitor = region.to_mss_monitor()
        try:
            self._sct = module.mss(display=display) if display else module.mss()
        except Exception as exc:  # mss raises its own error types per platform
            raise CaptureError(f"could not open the screen: {exc}") from exc

    def read(self) -> Any:
        shot = self._sct.grab(self._monitor)
        frame = self._numpy.frombuffer(shot.rgb, self._numpy.uint8)
        rgb = frame.reshape(shot.height, shot.width, 3)
        return self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self._sct.close()


def create_capture(
    region: Region,
    fps: float,
    backend: str = "auto",
    display: str | None = None,
    show_cursor: bool = False,
) -> FfmpegScreenCapture | MssScreenCapture:
    resolved = select_backend(backend)
    if resolved == "mss":
        return MssScreenCapture(region, display)
    return FfmpegScreenCapture(region, fps, display, show_cursor)


def _mss_module() -> Any:
    try:
        import mss
    except ImportError:
        return None
    return mss


def _mss_monitors() -> list[Region]:
    module = _mss_module()
    if module is None:
        return []
    try:
        with module.mss() as sct:
            # monitors[0] is the union of all screens; the rest are physical.
            return [
                Region(
                    width=int(item["width"]),
                    height=int(item["height"]),
                    x=int(item["left"]),
                    y=int(item["top"]),
                )
                for item in sct.monitors[1:]
            ]
    except Exception:
        return []
