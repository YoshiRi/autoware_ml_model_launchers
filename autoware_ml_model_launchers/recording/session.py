from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml

from ..launcher_dashboard.bag_player import BagPlayerManager, resolve_bag_paths
from ..launcher_dashboard.process_manager import ProcessManager
from ..launcher_dashboard.registry import (
    LauncherRegistry,
    build_launch_command,
    default_registry_path,
    load_registry,
)
from .manager import RecordingManager
from .spec import RecordingConfig, RecordingError, utc_now


DEFAULTS: dict[str, Any] = {
    "camera_namespace": "camera5",
    "rate": 1.0,
    "clock": True,
    "loop": False,
    # A model without a cached TensorRT engine takes minutes to come up.
    "startup_timeout": 900.0,
    "settle_sec": 3.0,
    "topic_poll_sec": 2.0,
}


class SessionError(ValueError):
    pass


def load_sessions(path: Path) -> list[dict[str, Any]]:
    """Merge the defaults block into every session entry of a session config."""
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise SessionError(f"{path}: config must be a mapping")

    defaults = dict(DEFAULTS)
    defaults.update(data.get("defaults") or {})
    sessions = data.get("sessions") or []
    if not isinstance(sessions, list) or not sessions:
        raise SessionError(f"{path}: no 'sessions' entries")

    merged = []
    for index, session in enumerate(sessions):
        if not isinstance(session, dict):
            raise SessionError(f"{path}: session #{index + 1} must be a mapping")
        entry = dict(defaults)
        entry.update(session)
        entry.setdefault("id", f"session_{index + 1}")
        if not entry.get("bags") and not entry.get("clips"):
            raise SessionError(f"{path}: session '{entry['id']}' is missing 'bags' or 'clips'")
        if not entry.get("comparison") and not entry.get("launcher"):
            raise SessionError(
                f"{path}: session '{entry['id']}' needs either 'comparison' or 'launcher'"
            )
        session_clips(entry)  # validate the clip list while the file is in hand
        merged.append(entry)
    return merged


def session_clips(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a session into the bags it plays, one clip per output name."""
    # A session without `clips` is a single unnamed clip, which keeps the plain
    # one-bag-per-session config working unchanged.
    clips = session.get("clips")
    if not clips:
        return [_clip_entry(session, {"bags": session.get("bags")}, "")]
    if not isinstance(clips, list):
        raise SessionError(f"session '{session.get('id')}': clips must be a list")

    expanded = []
    seen: set[str] = set()
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise SessionError(f"session '{session.get('id')}': each clip must be a mapping")
        name = str(clip.get("name") or f"clip_{index + 1}")
        if name in seen:
            raise SessionError(f"session '{session.get('id')}': duplicate clip name '{name}'")
        seen.add(name)
        expanded.append(_clip_entry(session, clip, name))
    return expanded


def _clip_entry(session: dict[str, Any], clip: dict[str, Any], name: str) -> dict[str, Any]:
    bags = clip.get("bags") or clip.get("bag")
    if not bags:
        raise SessionError(f"session '{session.get('id')}': clip '{name or session['id']}' has no bag")
    return {
        "name": name,
        "bags": bags,
        "rate": clip.get("rate", session["rate"]),
        "clock": bool(clip.get("clock", session["clock"])),
        "loop": bool(clip.get("loop", session["loop"])),
    }


def run_sessions(
    config_path: Path,
    only: list[str] | None = None,
    dry_run: bool = False,
    keep_going: bool = False,
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    sessions = load_sessions(config_path)
    if only:
        wanted = set(only)
        sessions = [session for session in sessions if session["id"] in wanted]
        if not sessions:
            raise SessionError("no sessions left after filtering")

    registry = load_registry(registry_path or default_registry_path())
    results = []
    for session in sessions:
        result = (
            _plan_session(session, registry)
            if dry_run
            else _run_session(session, registry)
        )
        results.append(result)
        if not result["ok"] and not keep_going:
            print("aborting (use --keep-going to continue past failures)")
            break
    return results


def _recording_config(session: dict[str, Any], registry: LauncherRegistry) -> RecordingConfig:
    config = RecordingConfig.from_registry(registry)
    if session.get("output_root"):
        config = replace(config, output_root=Path(str(session["output_root"])).expanduser())
    if session.get("file_layout"):
        config = replace(config, file_layout=str(session["file_layout"]))
    return config


def _launch_payloads(session: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    comparison = session.get("comparison")
    launcher = session.get("launcher")
    if comparison is not None and not isinstance(comparison, dict):
        raise SessionError(f"session '{session['id']}': comparison must be a mapping")
    if launcher is not None and not isinstance(launcher, dict):
        raise SessionError(f"session '{session['id']}': launcher must be a mapping")
    if comparison is not None:
        comparison = dict(comparison)
        comparison.setdefault("camera_namespace", session["camera_namespace"])
    return comparison, launcher


def _record_payload(
    session: dict[str, Any],
    group_id: str | None,
    clip: str = "",
) -> dict[str, Any]:
    record = dict(session.get("record") or {})
    payload: dict[str, Any] = {
        "topics": record.get("topics") or [],
        "session_id": record.get("session_id") or session["id"],
        "clip": clip,
    }
    if record.get("include_sinks"):
        payload["include_sinks"] = record["include_sinks"]
    if record.get("video"):
        payload["video"] = record["video"]
    if record.get("bag"):
        payload["bag"] = record["bag"]
    if record.get("from_group", True) and group_id:
        payload["from_group_id"] = group_id
    return payload


def _plan_session(session: dict[str, Any], registry: LauncherRegistry) -> dict[str, Any]:
    """Resolve everything a run would do, without starting a single process."""
    comparison, launcher = _launch_payloads(session)
    config = _recording_config(session, registry)
    manager = ProcessManager(registry, Path("/tmp/autoware_ml_model_launchers/session_dry_run"))
    recording_manager = RecordingManager(config=config, registry=registry)

    print(f"\n=== {session['id']} (dry run)")
    clips = session_clips(session)
    for clip in clips:
        paths = _clip_bag_paths(clip)
        label = clip["name"] or "single"
        shown = paths[0] if len(paths) == 1 else f"{paths[0]} (+{len(paths) - 1} more)"
        print(f"  clip  : {label} <- {shown} @ rate {clip['rate']}")

    planned: list[dict[str, Any]] = []
    group_id = None
    if comparison is not None:
        plan = manager.preview_comparison(comparison)
        group_id = plan["group_id"]
        for process in plan["processes"]:
            print(f"  launch: {' '.join(process['command'])}")
            planned.append(process)
    if launcher is not None:
        spec = registry.get(str(launcher["launcher_id"]))
        command = build_launch_command(spec, launcher.get("args", {}))
        print(f"  launch: {' '.join(command)}")
        planned.append(
            {
                "id": "planned",
                "launcher_id": spec.launcher_id,
                "command": command,
                "outputs": {},
                "group_id": None,
                "run_id": None,
                "variant_id": None,
            }
        )

    session_dir = ""
    for clip in clips:
        payload = _record_payload(session, group_id, clip["name"])
        if not payload.get("from_group_id"):
            payload["from_process_ids"] = [
                process["id"] for process in planned if "id" in process
            ]
        try:
            preview = recording_manager.preview(payload, planned)
        except RecordingError as exc:
            print(f"  record: {exc}")
            return {"id": session["id"], "ok": False, "detail": str(exc)}

        session_dir = preview["session_dir"]
        for recorder in preview["recorders"]:
            for output in recorder["outputs"]:
                print(f"  output: [{clip['name'] or 'single'}] {output['topic']}")
                print(f"          -> {output['file']}")
        for skipped in preview["skipped"]:
            print(f"  skip  : {skipped.get('arg_name', '')} {skipped['reason']}")

    return {
        "id": session["id"],
        "ok": True,
        "detail": f"{len(clips)} clip(s) planned",
        "session_dir": session_dir,
    }


def _run_session(session: dict[str, Any], registry: LauncherRegistry) -> dict[str, Any]:
    comparison, launcher = _launch_payloads(session)
    config = _recording_config(session, registry)
    log_dir = config.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    process_manager = ProcessManager(registry, log_dir)
    bag_manager = BagPlayerManager(log_dir)
    recording_manager = RecordingManager(
        config=config,
        registry=registry,
        bag_running=bag_manager.is_running,
    )

    print(f"\n=== {session['id']}")
    detail = ""
    ok = False
    manifest_path = None
    clips = session_clips(session)
    clip_results: list[dict[str, Any]] = []
    try:
        # Fail before a multi-minute model start-up if a bag path is wrong.
        _check_bags_exist(clips)
        group_id = None
        if comparison is not None:
            started = process_manager.start_comparison(comparison)
            group_id = started["group_id"]
            print(f"  launched {len(started['processes'])} comparison process(es)")
        if launcher is not None:
            process = process_manager.start(
                str(launcher["launcher_id"]), launcher.get("args", {})
            )
            print(f"  launched {process['label']}")

        settle_sec = float(session["settle_sec"])
        payload = _record_payload(session, group_id, clips[0]["name"])
        if not payload.get("from_group_id"):
            payload["from_process_ids"] = [
                process["id"] for process in process_manager.list_processes()
            ]
        preview = recording_manager.preview(payload, process_manager.list_processes())
        topics = [
            output["topic"]
            for recorder in preview["recorders"]
            for output in recorder["outputs"]
        ]
        print(f"  waiting for {len(topics)} topic(s)")
        timeout = float(session["startup_timeout"])
        if not wait_for_topics(topics, timeout, float(session["topic_poll_sec"])):
            raise SessionError("topics never appeared")
        _check_alive(process_manager)
        time.sleep(settle_sec)

        # The models stay up for the whole batch; only the recorders and the bag
        # player are restarted per clip, so a cold TensorRT build is paid once.
        for index, clip in enumerate(clips):
            label = clip["name"] or "single"
            print(f"  --- clip {index + 1}/{len(clips)}: {label}")
            _check_alive(process_manager)
            clip_payload = _record_payload(session, group_id, clip["name"])
            clip_payload.setdefault("from_process_ids", payload.get("from_process_ids", []))
            started_recording = recording_manager.start(
                clip_payload, process_manager.list_processes()
            )
            time.sleep(settle_sec)

            bag_path, bag_paths = _bag_arguments(clip)
            bag_manager.start(
                bag_path,
                rate=clip["rate"],
                loop=clip["loop"],
                clock=clip["clock"],
                bag_paths=bag_paths,
            )
            started_at = utc_now()
            while bag_manager.is_running():
                time.sleep(1.0)
            # Let the pipeline emit the images of the last few frames.
            time.sleep(settle_sec)
            recording_manager.stop_all()
            clip_results.append(
                {
                    "name": clip["name"] or None,
                    "bags": [str(item) for item in _clip_bag_paths(clip)],
                    "rate": clip["rate"],
                    "started_at": started_at,
                    "stopped_at": utc_now(),
                }
            )
            files = [
                output["file"]
                for recording in started_recording["recordings"]
                for output in recording["outputs"]
            ]
            print(f"      wrote {len(files)} output(s)")
        ok = True
        detail = started_recording["session_dir"]
    except (SessionError, RecordingError, ValueError, OSError) as exc:
        detail = str(exc)
        print(f"  FAILED: {detail}")
    finally:
        # Recorders first, so the files are finalized while the publishers live.
        recording_manager.stop_all()
        bag_manager.stop()
        try:
            finalized = recording_manager.finalize(
                processes=process_manager.list_processes(),
                bag=bag_manager.status(),
                clips=clip_results,
            )
            manifest_path = finalized["manifest_path"]
            print(f"  manifest: {manifest_path}")
        except RecordingError:
            pass
        process_manager.stop_all()

    return {
        "id": session["id"],
        "ok": ok,
        "detail": detail,
        "manifest_path": manifest_path,
    }


def wait_for_topics(topics: list[str], timeout_sec: float, poll_sec: float = 2.0) -> bool:
    """Wait until every topic is advertised, as auto_record.py did before a run."""
    if not topics:
        return True
    deadline = time.monotonic() + timeout_sec
    last_report = 0.0
    while time.monotonic() < deadline:
        available = _list_topics()
        missing = [topic for topic in topics if topic not in available]
        if not missing:
            return True
        if time.monotonic() - last_report > 30.0:
            print(f"    still waiting for {missing}")
            last_report = time.monotonic()
        time.sleep(poll_sec)
    print(f"    ERROR: topics never appeared: {missing}")
    return False


def _list_topics() -> set[str]:
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"], capture_output=True, text=True, timeout=20.0, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    return set(result.stdout.split())


def _check_alive(process_manager: ProcessManager) -> None:
    for process in process_manager.list_processes():
        if not process["running"]:
            raise SessionError(
                f"{process['label']} exited with {process['returncode']}; see {process['log_path']}"
            )


def _bag_arguments(clip: dict[str, Any]) -> tuple[str | None, list[str] | None]:
    """Accept a single bag, an explicit list, or an @playlist file."""
    bags = clip["bags"]
    if isinstance(bags, list):
        return None, [str(item) for item in bags]
    return str(bags), None


def _clip_bag_paths(clip: dict[str, Any]) -> list[Path]:
    bag_path, bag_paths = _bag_arguments(clip)
    return resolve_bag_paths(bag_path, bag_paths)


def _check_bags_exist(clips: list[dict[str, Any]]) -> None:
    for clip in clips:
        for path in _clip_bag_paths(clip):
            if not path.exists():
                raise SessionError(f"bag does not exist: {path}")
