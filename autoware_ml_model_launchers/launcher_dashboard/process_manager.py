from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import threading
import uuid
from typing import Any

from .registry import LauncherRegistry, build_launch_command


MAX_LOG_TAIL_LINES = 500
MAX_LOG_TAIL_BYTES = 2 * 1024 * 1024
LOG_TAIL_CHUNK_BYTES = 64 * 1024


@dataclass
class ManagedProcess:
    process_id: str
    launcher_id: str
    label: str
    command: list[str]
    started_at: str
    log_path: Path
    process: subprocess.Popen

    def refresh(self) -> None:
        self.process.poll()

    def to_json(self) -> dict[str, Any]:
        self.refresh()
        return {
            "id": self.process_id,
            "launcher_id": self.launcher_id,
            "label": self.label,
            "pid": self.process.pid,
            "running": self.process.returncode is None,
            "returncode": self.process.returncode,
            "started_at": self.started_at,
            "command": self.command,
            "log_path": str(self.log_path),
        }


class ProcessManager:
    def __init__(self, registry: LauncherRegistry, log_dir: Path | None = None) -> None:
        self.registry = registry
        self.log_dir = log_dir or Path("/tmp/autoware_ml_model_launchers/launcher_dashboard")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()

    def start(
        self,
        launcher_id: str,
        args: dict[str, Any] | None = None,
        label_suffix: str | None = None,
    ) -> dict[str, Any]:
        spec = self.registry.get(launcher_id)
        command = build_launch_command(spec, args)
        process_id = uuid.uuid4().hex[:10]
        log_path = self.log_dir / f"{process_id}_{launcher_id}.log"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"$ {' '.join(command)}\n")
            stream.flush()
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

        label = spec.label if not label_suffix else f"{spec.label} {label_suffix}"
        managed = ManagedProcess(
            process_id=process_id,
            launcher_id=launcher_id,
            label=label,
            command=command,
            started_at=datetime.now(timezone.utc).isoformat(),
            log_path=log_path,
            process=process,
        )
        with self._lock:
            self._processes[process_id] = managed
        return managed.to_json()

    def start_multi_yolox(
        self, cameras: list[str], args: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if self.registry.multi_yolox is None:
            raise ValueError("multi_yolox is not configured")
        if not cameras:
            raise ValueError("at least one camera is required")

        spec = self.registry.multi_yolox
        starts = []
        shared_args = spec.defaults()
        shared_args.update(args or {})
        for camera in cameras:
            launch_args = dict(shared_args)
            launch_args["camera_namespace"] = camera
            starts.append(self.start(spec.launcher_id, launch_args, label_suffix=camera))
        return starts

    def stop(self, process_id: str, timeout_sec: float = 5.0) -> dict[str, Any]:
        with self._lock:
            managed = self._processes[process_id]
        if managed.process.returncode is None:
            self._signal_process_group(managed.process, signal.SIGTERM)
            try:
                managed.process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                self._signal_process_group(managed.process, signal.SIGKILL)
                managed.process.wait(timeout=timeout_sec)
        return managed.to_json()

    def stop_all(self) -> list[dict[str, Any]]:
        with self._lock:
            process_ids = list(self._processes)
        return [self.stop(process_id) for process_id in process_ids]

    def close(self, process_id: str) -> dict[str, Any]:
        with self._lock:
            managed = self._processes[process_id]
        managed.refresh()
        if managed.process.returncode is None:
            raise ValueError("running processes must be stopped before closing")
        process = managed.to_json()
        with self._lock:
            self._processes.pop(process_id, None)
        return process

    def close_all(self) -> list[dict[str, Any]]:
        with self._lock:
            process_ids = list(self._processes)
        closed = []
        for process_id in process_ids:
            with self._lock:
                managed = self._processes.get(process_id)
            if managed is None:
                continue
            managed.refresh()
            if managed.process.returncode is not None:
                closed.append(managed.to_json())
                with self._lock:
                    self._processes.pop(process_id, None)
        return closed

    def list_processes(self) -> list[dict[str, Any]]:
        with self._lock:
            processes = list(self._processes.values())
        return [process.to_json() for process in processes]

    def tail_log(self, process_id: str, lines: int = 200) -> str:
        with self._lock:
            managed = self._processes[process_id]
        if not managed.log_path.is_file():
            return ""
        return _tail_file(managed.log_path, lines)

    @staticmethod
    def _signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return


def _tail_file(path: Path, lines: int) -> str:
    line_limit = min(max(int(lines), 1), MAX_LOG_TAIL_LINES)
    chunks: deque[bytes] = deque()
    newline_count = 0

    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        remaining = min(position, MAX_LOG_TAIL_BYTES)

        while remaining > 0 and newline_count <= line_limit:
            read_size = min(LOG_TAIL_CHUNK_BYTES, remaining)
            position -= read_size
            stream.seek(position)
            chunk = stream.read(read_size)
            chunks.appendleft(chunk)
            newline_count += chunk.count(b"\n")
            remaining -= read_size

    log_lines = b"".join(chunks).splitlines(keepends=True)
    return b"".join(log_lines[-line_limit:]).decode("utf-8", errors="replace")
