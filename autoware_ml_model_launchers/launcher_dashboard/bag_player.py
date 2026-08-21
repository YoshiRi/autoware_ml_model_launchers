from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
from typing import Any

from .process_manager import _tail_file


CONTROL_SERVICE_TYPES = {
    "pause": "rosbag2_interfaces/srv/Pause",
    "resume": "rosbag2_interfaces/srv/Resume",
    "get_rate": "rosbag2_interfaces/srv/GetRate",
    "set_rate": "rosbag2_interfaces/srv/SetRate",
    "is_paused": "rosbag2_interfaces/srv/IsPaused",
    "play_next": "rosbag2_interfaces/srv/PlayNext",
    "burst": "rosbag2_interfaces/srv/Burst",
    "seek": "rosbag2_interfaces/srv/Seek",
}

PLAYER_IDENTIFYING_CONTROLS = {"get_rate", "set_rate", "play_next", "burst", "seek"}


@dataclass
class ManagedBagPlayer:
    player_id: str
    bag_path: Path
    command: list[str]
    started_at: str
    log_path: Path
    process: subprocess.Popen
    requested_rate: float
    service_names: dict[str, str] = field(default_factory=dict)

    def refresh(self) -> None:
        self.process.poll()

    def to_json(self, service_names: dict[str, str] | None = None) -> dict[str, Any]:
        self.refresh()
        services = service_names if service_names is not None else self.service_names
        return {
            "id": self.player_id,
            "bag_path": str(self.bag_path),
            "pid": self.process.pid,
            "running": self.process.returncode is None,
            "returncode": self.process.returncode,
            "started_at": self.started_at,
            "command": self.command,
            "log_path": str(self.log_path),
            "requested_rate": self.requested_rate,
            "controls": {name: name in services for name in CONTROL_SERVICE_TYPES},
            "service_names": services,
        }


class BagPlayerManager:
    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or Path("/tmp/autoware_ml_model_launchers/launcher_dashboard")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._player: ManagedBagPlayer | None = None
        self._lock = threading.Lock()
        self._service_cache: tuple[float, dict[str, str]] = (0.0, {})
        self._playlist: list[Path] = []
        self._playlist_index = 0
        self._playlist_options: dict[str, Any] = {}
        self._stopping = False
        self._playlist_thread: threading.Thread | None = None

    def start(
        self,
        bag_path: str | None = None,
        rate: Any = 1.0,
        loop: bool = False,
        clock: bool = False,
        start_paused: bool = False,
        bag_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._player is not None:
                self._player.refresh()
                if self._player.process.returncode is None:
                    raise ValueError("bag player is already running")

        paths = resolve_bag_paths(bag_path, bag_paths)
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise ValueError(f"bag path does not exist: {missing[0]}")

        playback_rate = _coerce_positive_float(rate, "rate")
        player_id = uuid.uuid4().hex[:10]
        log_path = self.log_dir / f"{player_id}_bag_play.log"
        options = {
            "rate": playback_rate,
            "loop": loop,
            "clock": clock,
            "start_paused": start_paused,
        }
        with self._lock:
            self._playlist = paths
            self._playlist_index = 0
            self._playlist_options = options
            self._stopping = False
            self._service_cache = (0.0, {})

        player = self._launch(paths[0], player_id, log_path, options)
        if len(paths) > 1:
            self._start_playlist_thread()
        return player.to_json({})

    def stop(self, timeout_sec: float = 5.0) -> dict[str, Any]:
        with self._lock:
            player = self._player
            self._stopping = True
        if player is None:
            return self.status()
        player.refresh()
        if player.process.returncode is None:
            self._signal_process_group(player.process, signal.SIGTERM)
            try:
                player.process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                self._signal_process_group(player.process, signal.SIGKILL)
                player.process.wait(timeout=timeout_sec)
        return player.to_json({})

    def _launch(
        self,
        bag_path: Path,
        player_id: str,
        log_path: Path,
        options: dict[str, Any],
    ) -> ManagedBagPlayer:
        command = build_bag_play_command(
            bag_path,
            rate=options["rate"],
            loop=options["loop"],
            clock=options["clock"],
            start_paused=options["start_paused"],
        )
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

        player = ManagedBagPlayer(
            player_id=player_id,
            bag_path=bag_path,
            command=command,
            started_at=datetime.now(timezone.utc).isoformat(),
            log_path=log_path,
            process=process,
            requested_rate=options["rate"],
        )
        with self._lock:
            self._player = player
        return player

    def _start_playlist_thread(self) -> None:
        thread = threading.Thread(target=self._run_playlist, name="bag_playlist", daemon=True)
        with self._lock:
            self._playlist_thread = thread
        thread.start()

    def _run_playlist(self) -> None:
        """Play the remaining bags one after another, as a shell loop would."""
        while True:
            with self._lock:
                player = self._player
                stopping = self._stopping
            if player is None or stopping:
                return
            returncode = player.process.wait()
            with self._lock:
                if self._stopping:
                    return
                if returncode != 0:
                    self._playlist = self._playlist[: self._playlist_index + 1]
                    return
                next_index = self._playlist_index + 1
                if next_index >= len(self._playlist):
                    # Keep the index on the last bag played, so status stays valid.
                    return
                self._playlist_index = next_index
                next_path = self._playlist[next_index]
                player_id = player.player_id
                log_path = player.log_path
                options = dict(self._playlist_options)
            self._launch(next_path, player_id, log_path, options)

    def pause(self) -> dict[str, Any]:
        self._call_service("pause", "{}")
        return self.status(force_service_refresh=True)

    def resume(self) -> dict[str, Any]:
        self._call_service("resume", "{}")
        return self.status(force_service_refresh=True)

    def set_rate(self, rate: Any) -> dict[str, Any]:
        playback_rate = _coerce_positive_float(rate, "rate")
        self._call_service("set_rate", f"{{rate: {playback_rate}}}")
        with self._lock:
            if self._player is not None:
                self._player.requested_rate = playback_rate
        return self.status(force_service_refresh=True)

    def status(self, force_service_refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            player = self._player
        service_names = self.discover_control_services(force=force_service_refresh)
        if player is None:
            return {
                "running": False,
                "external_player": bool(service_names),
                "controls": {name: name in service_names for name in CONTROL_SERVICE_TYPES},
                "service_names": service_names,
                "playlist": None,
            }

        player.refresh()
        player.service_names = service_names
        state = player.to_json(service_names)
        state["playlist"] = self.playlist_status()
        # A playlist is still "running" between two bags, while no child exists.
        state["running"] = self.is_running()
        return state

    def is_running(self) -> bool:
        """Cheap liveness probe for watchers; does not touch the ROS graph."""
        with self._lock:
            player = self._player
            pending = (
                not self._stopping and self._playlist_index + 1 < len(self._playlist)
            )
        if pending:
            return True
        if player is None:
            return False
        player.refresh()
        return player.process.returncode is None

    def playlist_status(self) -> dict[str, Any] | None:
        with self._lock:
            total = len(self._playlist)
            if total <= 1:
                return None
            index = min(self._playlist_index, total - 1)
            return {
                "index": index,
                "total": total,
                "current": str(self._playlist[index]),
            }

    def tail_log(self, lines: int = 200) -> str:
        with self._lock:
            player = self._player
        if player is None or not player.log_path.is_file():
            return ""
        return _tail_file(player.log_path, lines)

    def discover_control_services(self, force: bool = False) -> dict[str, str]:
        now = time.monotonic()
        cached_at, service_names = self._service_cache
        if not force and now - cached_at < 2.0:
            return dict(service_names)

        services = _list_ros_services()
        selected = _select_player_services(services)
        self._service_cache = (now, selected)
        return dict(selected)

    def _call_service(self, control: str, request: str) -> None:
        with self._lock:
            player = self._player
        if player is not None:
            player.refresh()

        services = self.discover_control_services(force=True)
        service_name = services.get(control)
        if service_name is None:
            raise ValueError(f"rosbag player service is not available: {control}")

        command = [
            "ros2",
            "service",
            "call",
            service_name,
            CONTROL_SERVICE_TYPES[control],
            request,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if player is not None:
            _append_command_result(player.log_path, command, result)
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(details or f"{control} failed with code {result.returncode}")

    @staticmethod
    def _signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return


def build_bag_play_command(
    bag_path: Path,
    rate: float = 1.0,
    loop: bool = False,
    clock: bool = False,
    start_paused: bool = False,
) -> list[str]:
    command = [
        "ros2",
        "bag",
        "play",
        str(bag_path),
        "--disable-keyboard-controls",
        "--rate",
        _format_rate(rate),
    ]
    if loop:
        command.append("--loop")
    if clock:
        command.append("--clock")
    if start_paused:
        command.append("--start-paused")
    return command


def resolve_bag_paths(
    bag_path: str | None = None,
    bag_paths: list[str] | None = None,
) -> list[Path]:
    """Resolve one bag, an explicit list, or an @file holding one path per line."""
    if bag_paths:
        paths = [Path(str(item)).expanduser() for item in bag_paths]
    elif bag_path and str(bag_path).startswith("@"):
        paths = read_bag_list(Path(str(bag_path)[1:]).expanduser())
    elif bag_path:
        paths = [Path(str(bag_path)).expanduser()]
    else:
        paths = []
    if not paths:
        raise ValueError("no bag path was given")
    return paths


def read_bag_list(list_path: Path) -> list[Path]:
    """Read a playlist file: one bag path per line, # comments and blanks ignored."""
    if not list_path.is_file():
        raise ValueError(f"bag list does not exist: {list_path}")
    paths = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paths.append(Path(stripped).expanduser())
    if not paths:
        raise ValueError(f"bag list is empty: {list_path}")
    return paths


def browse_bag_path(path: str | None = None) -> dict[str, Any]:
    current_path = Path(path).expanduser() if path else Path.home()
    if current_path.is_file():
        current_path = current_path.parent
    current_path = current_path.resolve()
    if not current_path.is_dir():
        raise ValueError(f"bag browser path is not a directory: {current_path}")

    entries = []
    try:
        children = list(current_path.iterdir())
    except PermissionError as exc:
        raise ValueError(f"permission denied: {current_path}") from exc

    for child in children:
        try:
            is_dir = child.is_dir()
            is_file = child.is_file()
        except OSError:
            continue
        if not is_dir and not is_file:
            continue
        playable = _looks_like_rosbag(child, is_dir=is_dir)
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "kind": "directory" if is_dir else "file",
                "playable": playable,
            }
        )

    entries.sort(key=lambda entry: (entry["kind"] != "directory", not entry["playable"], entry["name"]))
    return {
        "path": str(current_path),
        "parent": str(current_path.parent) if current_path.parent != current_path else None,
        "entries": entries,
    }


def _coerce_positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if number <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return number


def _format_rate(rate: float) -> str:
    return f"{rate:g}"


def _looks_like_rosbag(path: Path, is_dir: bool) -> bool:
    if is_dir:
        return (path / "metadata.yaml").is_file()
    return path.suffix.lower() in {".mcap", ".db3", ".sqlite3"}


def _list_ros_services() -> list[tuple[str, str]]:
    try:
        result = subprocess.run(
            ["ros2", "service", "list", "-t"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    services = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "[" not in stripped or "]" not in stripped:
            continue
        name, service_type = stripped.split("[", 1)
        services.append((name.strip(), service_type.split("]", 1)[0].strip()))
    return services


def _select_player_services(services: list[tuple[str, str]]) -> dict[str, str]:
    control_by_service_type = {service_type: name for name, service_type in CONTROL_SERVICE_TYPES.items()}
    by_prefix: dict[str, dict[str, str]] = {}
    for service_name, service_type in services:
        control = control_by_service_type.get(service_type)
        if control is None or "/" not in service_name:
            continue
        prefix, suffix = service_name.rsplit("/", 1)
        if suffix != control:
            continue
        by_prefix.setdefault(prefix, {})[control] = service_name

    if not by_prefix:
        return {}

    best_controls = max(
        by_prefix.values(),
        key=lambda controls: (
            len(PLAYER_IDENTIFYING_CONTROLS.intersection(controls)),
            len(controls),
        ),
    )
    if not PLAYER_IDENTIFYING_CONTROLS.intersection(best_controls):
        return {}
    return dict(best_controls)


def _append_command_result(
    log_path: Path,
    command: list[str],
    result: subprocess.CompletedProcess,
) -> None:
    with log_path.open("a", encoding="utf-8", errors="replace") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        if result.stdout:
            stream.write(result.stdout)
        if result.stderr:
            stream.write(result.stderr)
