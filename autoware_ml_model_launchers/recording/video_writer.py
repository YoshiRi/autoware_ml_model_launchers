from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .spec import progress_path, stamp_csv_path, utc_now


# Frames buffered before the writer is opened, used to estimate fps from stamps.
FPS_PROBE_FRAMES = 20
DEFAULT_FPS = 10.0
PROGRESS_FLUSH_SEC = 5.0


def select_encoder(encoder: str = "auto") -> str:
    """Resolve the encoder name, falling back to OpenCV when ffmpeg is missing."""
    if encoder not in {"auto", "ffmpeg", "opencv"}:
        raise ValueError(f"unknown encoder: {encoder}")
    if encoder == "auto":
        return "ffmpeg" if shutil.which("ffmpeg") else "opencv"
    return encoder


def build_ffmpeg_command(path: Path, width: int, height: int, fps: float, crf: int) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.3f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]


class FfmpegFrameSink:
    """Pipes raw BGR frames into ffmpeg, so no intermediate images hit the disk."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, crf: int) -> None:
        width, height = size
        self.path = path
        self.command = build_ffmpeg_command(path, width, height, fps, crf)
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # Own session: a SIGINT aimed at the recorder's process group must not
            # reach ffmpeg, which would abort before writing the moov atom. ffmpeg
            # finalizes on stdin EOF instead, even if the recorder is killed.
            start_new_session=True,
        )

    def write(self, image: Any) -> None:
        if self.process.stdin is None:
            return
        self.process.stdin.write(image.tobytes())

    def close(self) -> str | None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
            return "ffmpeg did not exit in time"
        if self.process.returncode != 0:
            stderr = b""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read() or b""
            return f"ffmpeg exited with {self.process.returncode}: {stderr.decode(errors='replace')}"
        return None


class OpenCvFrameSink:
    """Fallback writer; mp4v plays in desktop players but not in most browsers."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, codec: str = "mp4v") -> None:
        import cv2

        self.path = path
        self.command = None
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {path}")

    def write(self, image: Any) -> None:
        self.writer.write(image)

    def close(self) -> str | None:
        self.writer.release()
        return None


def create_frame_sink(
    path: Path,
    size: tuple[int, int],
    fps: float,
    encoder: str = "auto",
    crf: int = 23,
) -> FfmpegFrameSink | OpenCvFrameSink:
    resolved = select_encoder(encoder)
    if resolved == "ffmpeg":
        return FfmpegFrameSink(path, size, fps, crf)
    return OpenCvFrameSink(path, size, fps)


class TopicVideoRecorder:
    """Turns the frames of one image topic into a video file plus its sidecars."""

    def __init__(
        self,
        topic: str,
        path: Path,
        encoder: str = "auto",
        crf: int = 23,
        fps: float | None = None,
        stamp_csv: bool = True,
        logger: Any = None,
    ) -> None:
        self.topic = topic
        self.path = Path(path)
        self.encoder = encoder
        self.crf = crf
        self.fixed_fps = fps
        self.logger = logger

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sink: FfmpegFrameSink | OpenCvFrameSink | None = None
        self.size: tuple[int, int] | None = None
        self.output_fps: float | None = None
        self.frames = 0
        self.dropped = 0
        self.first_stamp: float | None = None
        self.last_stamp: float | None = None
        self.error: str | None = None
        self.closed = False
        self.pending: list[tuple[float | None, Any]] = []

        self.stamp_file = None
        if stamp_csv:
            self.stamp_file = stamp_csv_path(self.path).open("w", encoding="utf-8")
            self.stamp_file.write("frame_idx,stamp\n")
        self.write_progress()

    def add_frame(self, image: Any, stamp: float | None) -> None:
        if image is None:
            self.dropped += 1
            return
        if stamp is not None:
            if self.first_stamp is None:
                self.first_stamp = stamp
            self.last_stamp = stamp

        if self.sink is None:
            self.pending.append((stamp, image))
            if self.fixed_fps is not None or len(self.pending) >= FPS_PROBE_FRAMES:
                self._open_sink()
            return
        self._write(image, stamp)

    def close(self) -> None:
        if self.closed:
            return
        if self.sink is None and self.pending:
            self._open_sink()  # a very short recording should still produce a file
        if self.sink is not None:
            error = self.sink.close()
            if error and not self.error:
                self.error = error
            self.sink = None
        if self.stamp_file is not None:
            self.stamp_file.close()
            self.stamp_file = None
        self.closed = True
        self.write_progress()
        self._log(
            f"{self.topic}: {self.frames} frames"
            + (f", {self.dropped} undecodable" if self.dropped else "")
            + f" -> {self.path}"
        )

    def measured_fps(self) -> float:
        stamps = [stamp for stamp, _ in self.pending if stamp is not None]
        if len(stamps) >= 2 and stamps[-1] > stamps[0]:
            return (len(stamps) - 1) / (stamps[-1] - stamps[0])
        self._log(f"{self.topic}: could not estimate fps, using {DEFAULT_FPS}", level="warn")
        return DEFAULT_FPS

    def progress(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "sink": "video",
            "file": str(self.path),
            "frames": self.frames,
            "dropped": self.dropped,
            "fps": self.output_fps,
            "width": self.size[0] if self.size else None,
            "height": self.size[1] if self.size else None,
            "first_stamp": self.first_stamp,
            "last_stamp": self.last_stamp,
            "state": "closed" if self.closed else "recording",
            "error": self.error,
            "updated_at": utc_now(),
        }

    def write_progress(self) -> None:
        path = progress_path(self.path)
        try:
            path.write_text(json.dumps(self.progress(), indent=2) + "\n", encoding="utf-8")
        except OSError as exc:  # progress is diagnostics only, never fail recording for it
            self._log(f"{self.topic}: could not write progress: {exc}", level="warn")

    def _open_sink(self) -> None:
        fps = self.fixed_fps if self.fixed_fps is not None else self.measured_fps()
        height, width = self.pending[0][1].shape[:2]
        self.size = (width, height)
        self.output_fps = fps
        try:
            self.sink = create_frame_sink(self.path, self.size, fps, self.encoder, self.crf)
        except (RuntimeError, OSError) as exc:
            self.error = str(exc)
            self.pending.clear()
            self._log(f"{self.topic}: {exc}", level="error")
            return

        self._log(f"{self.topic} -> {self.path} ({width}x{height} @ {fps:.2f} fps)")
        for stamp, image in self.pending:
            self._write(image, stamp)
        self.pending.clear()

    def _write(self, image: Any, stamp: float | None) -> None:
        if self.sink is None:
            return
        if self.size is not None and (image.shape[1], image.shape[0]) != self.size:
            import cv2

            image = cv2.resize(image, self.size)
        if not image.flags["C_CONTIGUOUS"]:
            import numpy

            image = numpy.ascontiguousarray(image)
        try:
            self.sink.write(image)
        except (BrokenPipeError, OSError) as exc:
            self.error = f"writer failed: {exc}"
            self._log(f"{self.topic}: {self.error}", level="error")
            self.sink = None
            return
        if self.stamp_file is not None:
            self.stamp_file.write(f"{self.frames},{'' if stamp is None else f'{stamp:.9f}'}\n")
        self.frames += 1

    def _log(self, message: str, level: str = "info") -> None:
        if self.logger is None:
            print(message)
            return
        getattr(self.logger, level)(message)
