from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
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
    group_id: str | None = None
    run_id: str | None = None
    variant_id: str | None = None
    outputs: dict[str, str] | None = None
    model_path: str | None = None

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
            "group_id": self.group_id,
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "outputs": dict(self.outputs or {}),
            "model_path": self.model_path,
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
        metadata: dict[str, Any] | None = None,
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
        metadata = metadata or {}
        managed = ManagedProcess(
            process_id=process_id,
            launcher_id=launcher_id,
            label=label,
            command=command,
            started_at=datetime.now(timezone.utc).isoformat(),
            log_path=log_path,
            process=process,
            group_id=metadata.get("group_id"),
            run_id=metadata.get("run_id"),
            variant_id=metadata.get("variant_id"),
            outputs=metadata.get("outputs"),
            model_path=spec.resolve_model_path(args),
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

    def preview_comparison(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = self._build_comparison_plan(payload)
        planned_processes = []
        for item in plan["items"]:
            spec = self.registry.get(item["launcher_id"])
            planned_processes.append(
                {
                    "label": item["label"],
                    "launcher_id": item["launcher_id"],
                    "variant_id": item["variant_id"],
                    "run_id": plan["run_id"],
                    "group_id": plan["group_id"],
                    "args": item["args"],
                    "outputs": item["outputs"],
                    "command": build_launch_command(spec, item["args"]),
                    "model_path": spec.resolve_model_path(item["args"]),
                }
            )
        return {
            "run_id": plan["run_id"],
            "group_id": plan["group_id"],
            "processes": planned_processes,
        }

    def start_comparison(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = self._build_comparison_plan(payload)
        processes = []
        for item in plan["items"]:
            processes.append(
                self.start(
                    item["launcher_id"],
                    item["args"],
                    label_suffix=f"{plan['run_id']} {item['label']}",
                    metadata={
                        "group_id": plan["group_id"],
                        "run_id": plan["run_id"],
                        "variant_id": item["variant_id"],
                        "outputs": item["outputs"],
                    },
                )
            )
        return {
            "run_id": plan["run_id"],
            "group_id": plan["group_id"],
            "processes": processes,
        }

    def stop_group(self, group_id: str) -> list[dict[str, Any]]:
        with self._lock:
            process_ids = [
                process_id
                for process_id, process in self._processes.items()
                if process.group_id == group_id
            ]
        return [self.stop(process_id) for process_id in process_ids]

    def close_group(self, group_id: str) -> list[dict[str, Any]]:
        with self._lock:
            process_ids = [
                process_id
                for process_id, process in self._processes.items()
                if process.group_id == group_id
            ]
        closed = []
        for process_id in process_ids:
            try:
                closed.append(self.close(process_id))
            except ValueError:
                continue
        return closed

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

    def _build_comparison_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = _sanitize_token(str(payload.get("run_id") or _default_run_id()))
        camera = str(payload.get("camera_namespace", "camera5"))
        common_args = _dict_payload(payload.get("common_args", {}), "common_args")
        variants = payload.get("variants", [])
        if not isinstance(variants, list) or not variants:
            raise ValueError("at least one comparison variant is required")

        group_id = f"comparison:{run_id}"
        items = []
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("comparison variants must be objects")
            if variant.get("enabled", True) is False:
                continue

            launcher_id = str(variant["launcher_id"])
            spec = self.registry.get(launcher_id)
            variant_id = _sanitize_token(str(variant["id"]))
            label = str(variant.get("label") or variant_id)
            args = dict(common_args)
            args.update(_dict_payload(variant.get("args", {}), f"{variant_id} args"))
            args["camera_namespace"] = camera

            outputs: dict[str, str] = {}
            if spec.isolation is not None and payload.get("auto_isolate", True):
                context = {
                    "run_id": run_id,
                    "variant_id": variant_id,
                    "camera_namespace": camera,
                }
                isolation_args = spec.isolation.build_args(context)
                args.update(isolation_args)
                outputs = {
                    name: isolation_args[name]
                    for name in spec.isolation.output_templates
                    if name in isolation_args
                }

            items.append(
                {
                    "launcher_id": launcher_id,
                    "variant_id": variant_id,
                    "label": label,
                    "args": args,
                    "outputs": outputs,
                }
            )

        if not items:
            raise ValueError("at least one enabled comparison variant is required")
        return {"run_id": run_id, "group_id": group_id, "items": items}

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


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _sanitize_token(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "_-" else "_" for character in value)
    return cleaned.strip("_") or "run"


def _dict_payload(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)
