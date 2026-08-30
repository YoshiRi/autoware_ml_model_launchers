from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import uuid

from ..launcher_dashboard.process_manager import _tail_file
from ..launcher_dashboard.registry import LauncherRegistry
from .bag_recorder import bag_output_state, build_bag_record_command
from .spec import (
    BagOptions,
    RecordingConfig,
    RecordingError,
    TopicRecording,
    VideoOptions,
    build_manifest,
    default_session_id,
    progress_path,
    read_progress,
    resolve_recordings,
    utc_now,
    write_manifest,
)


VIDEO_RECORDER_MODULE = "autoware_ml_model_launchers.recording.video_recorder"
DEFAULT_STOP_SETTLE_SEC = 3.0
BAG_WATCH_POLL_SEC = 1.0


@dataclass
class ManagedRecording:
    recording_id: str
    sink: str
    session_id: str
    session_dir: Path
    command: list[str]
    started_at: str
    log_path: Path
    process: subprocess.Popen
    outputs: list[dict[str, Any]] = field(default_factory=list)
    stopped_at: str | None = None

    def refresh(self) -> None:
        self.process.poll()

    def running(self) -> bool:
        self.refresh()
        return self.process.returncode is None

    def to_json(self) -> dict[str, Any]:
        self.refresh()
        return {
            "id": self.recording_id,
            "sink": self.sink,
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "pid": self.process.pid,
            "running": self.process.returncode is None,
            "returncode": self.process.returncode,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "command": self.command,
            "log_path": str(self.log_path),
            "outputs": [self._output_json(output) for output in self.outputs],
        }

    def _output_json(self, output: dict[str, Any]) -> dict[str, Any]:
        merged = dict(output)
        path = Path(output["file"])
        if self.sink == "video":
            progress = read_progress(progress_path(path)) or {}
            merged.update(
                {
                    "frames": progress.get("frames"),
                    "dropped": progress.get("dropped"),
                    "fps": progress.get("fps"),
                    "first_stamp": progress.get("first_stamp"),
                    "last_stamp": progress.get("last_stamp"),
                    "state": progress.get("state"),
                    "error": progress.get("error"),
                    "exists": path.is_file(),
                    "bytes": path.stat().st_size if path.is_file() else 0,
                }
            )
        else:
            merged.update(bag_output_state(path))
        return merged


class RecordingManager:
    """Owns recorder child processes, their outputs, and the session manifest."""

    def __init__(
        self,
        config: RecordingConfig | None = None,
        registry: LauncherRegistry | None = None,
        bag_running: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config or RecordingConfig()
        self.registry = registry
        self.bag_running = bag_running
        self._recordings: dict[str, ManagedRecording] = {}
        self._session_id: str | None = None
        self._lock = threading.Lock()
        self._watcher: threading.Thread | None = None

    # -- session ----------------------------------------------------------
    def session_id(self, requested: str | None = None) -> str:
        with self._lock:
            if requested:
                self._session_id = str(requested)
            elif self._session_id is None:
                self._session_id = default_session_id()
            return self._session_id

    def session_dir(self, session_id: str | None = None) -> Path:
        return self.config.session_dir(session_id or self.session_id())

    # -- planning ---------------------------------------------------------
    def preview(
        self,
        payload: dict[str, Any],
        processes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        plan = self._build_plan(payload, processes)
        return {
            "session_id": plan["session_id"],
            "session_dir": str(plan["session_dir"]),
            "skipped": plan["skipped"],
            "recorders": [
                {
                    "sink": item["sink"],
                    "command": item["command"],
                    "outputs": item["outputs"],
                }
                for item in plan["recorders"]
            ],
        }

    def start(
        self,
        payload: dict[str, Any],
        processes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        plan = self._build_plan(payload, processes)
        session_dir: Path = plan["session_dir"]
        log_dir = session_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        started: list[dict[str, Any]] = []
        for item in plan["recorders"]:
            recording_id = uuid.uuid4().hex[:10]
            log_path = log_dir / f"{recording_id}_{item['sink']}.log"
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"$ {' '.join(item['command'])}\n")
                stream.flush()
                process = subprocess.Popen(
                    item["command"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            managed = ManagedRecording(
                recording_id=recording_id,
                sink=item["sink"],
                session_id=plan["session_id"],
                session_dir=session_dir,
                command=item["command"],
                started_at=utc_now(),
                log_path=log_path,
                process=process,
                outputs=item["outputs"],
            )
            with self._lock:
                self._recordings[recording_id] = managed
            started.append(managed.to_json())

        if not started:
            raise RecordingError("no topics resolved for this record request")

        if payload.get("stop_with_bag"):
            self._start_bag_watcher(float(payload.get("settle_sec", DEFAULT_STOP_SETTLE_SEC)))

        return {
            "session_id": plan["session_id"],
            "session_dir": str(session_dir),
            "skipped": plan["skipped"],
            "recordings": started,
        }

    # -- control ----------------------------------------------------------
    def stop(self, recording_id: str, timeout_sec: float = 30.0) -> dict[str, Any]:
        with self._lock:
            managed = self._recordings.get(recording_id)
        if managed is None:
            raise RecordingError(f"unknown recording: {recording_id}")
        self._stop_managed(managed, timeout_sec)
        return managed.to_json()

    def stop_all(self, timeout_sec: float = 30.0) -> list[dict[str, Any]]:
        with self._lock:
            recordings = list(self._recordings.values())
        for managed in recordings:
            self._stop_managed(managed, timeout_sec)
        return [managed.to_json() for managed in recordings]

    def list_recordings(self) -> list[dict[str, Any]]:
        with self._lock:
            recordings = list(self._recordings.values())
        return [managed.to_json() for managed in recordings]

    def status(self) -> dict[str, Any]:
        recordings = self.list_recordings()
        with self._lock:
            session_id = self._session_id
        return {
            "session_id": session_id,
            "session_dir": str(self.session_dir(session_id)) if session_id else None,
            "recording": any(item["running"] for item in recordings),
            "recordings": recordings,
            "config": self.config.to_json(),
        }

    def tail_log(self, recording_id: str, lines: int = 200) -> str:
        with self._lock:
            managed = self._recordings.get(recording_id)
        if managed is None or not managed.log_path.is_file():
            return ""
        return _tail_file(managed.log_path, lines)

    def finalize(
        self,
        processes: list[dict[str, Any]] | None = None,
        bag: dict[str, Any] | None = None,
        clear: bool = True,
        clips: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Write manifest.json for the current session and optionally end it."""
        with self._lock:
            session_id = self._session_id
            recordings = list(self._recordings.values())
        if session_id is None:
            raise RecordingError("no recording session to finalize")

        session_dir = self.session_dir(session_id)
        entries: list[dict[str, Any]] = []
        for managed in recordings:
            if managed.session_id != session_id:
                continue
            state = managed.to_json()
            for output in state["outputs"]:
                entry = dict(output)
                entry.update(
                    {
                        "id": managed.recording_id,
                        "sink": managed.sink,
                        "started_at": managed.started_at,
                        "stopped_at": managed.stopped_at,
                    }
                )
                entry["file"] = _relative_to(Path(output["file"]), session_dir)
                entries.append(entry)

        manifest = build_manifest(session_id, session_dir, entries, processes, bag, clips)
        path = write_manifest(session_dir, manifest)
        if clear:
            with self._lock:
                self._session_id = None
        return {"manifest_path": str(path), "manifest": manifest}

    # -- internals --------------------------------------------------------
    def _build_plan(
        self,
        payload: dict[str, Any],
        processes: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        session_id = self.session_id(payload.get("session_id"))
        session_dir = self.session_dir(session_id)
        recordings, skipped = resolve_recordings(
            payload,
            self.config,
            session_id,
            registry=self.registry,
            processes=processes,
        )
        if not recordings:
            raise RecordingError("no topics resolved for this record request")

        video = self.config.video.merge(payload.get("video"))
        bag = self.config.bag.merge(payload.get("bag"))
        recorders: list[dict[str, Any]] = []

        video_topics = [item for item in recordings if item.sink == "video"]
        if video_topics:
            _check_stem_collisions(video_topics)
            recorders.append(self._video_recorder(video_topics, session_dir, video))

        bag_topics = [item for item in recordings if item.sink == "bag"]
        if bag_topics:
            recorders.append(self._bag_recorder(bag_topics, session_dir, bag, payload))

        return {
            "session_id": session_id,
            "session_dir": session_dir,
            "skipped": skipped,
            "recorders": recorders,
        }

    def _video_recorder(
        self,
        topics: list[TopicRecording],
        session_dir: Path,
        options: VideoOptions,
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            "-m",
            VIDEO_RECORDER_MODULE,
            "-o",
            str(session_dir),
            "--encoder",
            options.encoder,
            "--crf",
            str(options.crf),
            "--wait",
            f"{options.wait:g}",
        ]
        if options.fps is not None:
            command += ["--fps", f"{options.fps:g}"]
        if not options.stamp_csv:
            command.append("--no-stamp-csv")

        outputs = []
        for item in topics:
            command += ["-t", item.topic, "-n", item.stem]
            output = item.to_json()
            output["file"] = str(session_dir / f"{item.stem}.mp4")
            outputs.append(output)
        return {"sink": "video", "command": command, "outputs": outputs}

    def _bag_recorder(
        self,
        topics: list[TopicRecording],
        session_dir: Path,
        options: BagOptions,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        output_dir = _unique_bag_dir(session_dir, _bag_dir_name(topics, payload))
        command = build_bag_record_command(
            output_dir,
            [item.topic for item in topics],
            storage=options.storage,
        )
        outputs = []
        for item in topics:
            output = item.to_json()
            output["file"] = str(output_dir)
            outputs.append(output)
        return {"sink": "bag", "command": command, "outputs": outputs}

    def _stop_managed(self, managed: ManagedRecording, timeout_sec: float) -> None:
        managed.refresh()
        if managed.process.returncode is None:
            # SIGINT first: recorders finalize their files on interrupt, and a
            # SIGTERM-first stop would truncate the video and the bag metadata.
            for sig, wait in (
                (signal.SIGINT, timeout_sec),
                (signal.SIGTERM, 10.0),
                (signal.SIGKILL, 5.0),
            ):
                _signal_process_group(managed.process, sig)
                try:
                    managed.process.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if managed.stopped_at is None:
            managed.stopped_at = utc_now()

    def _start_bag_watcher(self, settle_sec: float) -> None:
        if self.bag_running is None:
            return
        with self._lock:
            if self._watcher is not None and self._watcher.is_alive():
                return
            self._watcher = threading.Thread(
                target=self._watch_bag,
                args=(settle_sec,),
                name="recording_bag_watcher",
                daemon=True,
            )
            self._watcher.start()

    def _watch_bag(self, settle_sec: float) -> None:
        seen_running = False
        while True:
            time.sleep(BAG_WATCH_POLL_SEC)
            if not any(managed.running() for managed in self._snapshot()):
                return
            try:
                running = bool(self.bag_running())
            except Exception:  # never let a status probe kill the watcher
                continue
            if running:
                seen_running = True
                continue
            if seen_running:
                # Let the perception pipeline emit the last few frames.
                time.sleep(settle_sec)
                self.stop_all()
                return

    def _snapshot(self) -> list[ManagedRecording]:
        with self._lock:
            return list(self._recordings.values())


def _check_stem_collisions(topics: list[TopicRecording]) -> None:
    """Two topics writing the same file would silently lose one of them."""
    by_stem: dict[str, list[str]] = {}
    for item in topics:
        by_stem.setdefault(item.stem, []).append(item.topic)
    collisions = {stem: names for stem, names in by_stem.items() if len(names) > 1}
    if collisions:
        stem, names = next(iter(collisions.items()))
        raise RecordingError(
            f"{len(names)} topics resolve to the same output file '{stem}': "
            f"{', '.join(sorted(names))}. Add {{arg_name}} or {{topic_slug}} to "
            "the recording file_layout, or give each topic its own name."
        )


def _bag_dir_name(topics: list[TopicRecording], payload: dict[str, Any]) -> str:
    explicit = payload.get("bag_name")
    if explicit:
        return str(explicit)
    clips = {item.context.get("clip", "") for item in topics}
    if len(clips) == 1:
        only = clips.pop()
        if only:
            return f"{only}_bag"
    run_ids = {item.context.get("run_id", "") for item in topics}
    if len(run_ids) == 1:
        only = run_ids.pop()
        if only:
            return f"{only}_bag"
    return "topics_bag"


def _unique_bag_dir(session_dir: Path, name: str) -> Path:
    # ros2 bag record refuses to write into an existing directory.
    candidate = session_dir / name
    index = 2
    while candidate.exists():
        candidate = session_dir / f"{name}_{index}"
        index += 1
    return candidate


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError):
        return
