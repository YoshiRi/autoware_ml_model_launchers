from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any

from ..launcher_dashboard.registry import LauncherRegistry, RegistryError


MANIFEST_SCHEMA = "launcher_recording/1"
MANIFEST_FILE_NAME = "manifest.json"

SINKS = ("video", "bag")

DEFAULT_OUTPUT_ROOT = "~/Documents/launcher_recordings"
DEFAULT_SESSION_LAYOUT = "{session_id}"
DEFAULT_FILE_LAYOUT = "{run_id}/{variant_id}_{camera_namespace}_{arg_name}"

CONTEXT_KEYS = (
    "session_id",
    "clip",
    "run_id",
    "variant_id",
    "camera_namespace",
    "arg_name",
    "launcher_id",
    "topic_slug",
)


class RecordingError(ValueError):
    pass


@dataclass(frozen=True)
class VideoOptions:
    encoder: str = "auto"
    crf: int = 23
    wait: float = 60.0
    stamp_csv: bool = True
    fps: float | None = None

    @classmethod
    def from_mapping(cls, data: Any) -> "VideoOptions":
        if not isinstance(data, dict):
            return cls()
        fps = data.get("fps")
        return cls(
            encoder=str(data.get("encoder", cls.encoder)),
            crf=int(data.get("crf", cls.crf)),
            wait=float(data.get("wait", cls.wait)),
            stamp_csv=bool(data.get("stamp_csv", cls.stamp_csv)),
            fps=None if fps is None else float(fps),
        )

    def merge(self, data: Any) -> "VideoOptions":
        if not isinstance(data, dict) or not data:
            return self
        merged = {**self.to_json(), **data}
        return VideoOptions.from_mapping(merged)

    def to_json(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder,
            "crf": self.crf,
            "wait": self.wait,
            "stamp_csv": self.stamp_csv,
            "fps": self.fps,
        }


@dataclass(frozen=True)
class BagOptions:
    storage: str | None = "mcap"

    @classmethod
    def from_mapping(cls, data: Any) -> "BagOptions":
        if not isinstance(data, dict):
            return cls()
        storage = data.get("storage", cls.storage)
        return cls(storage=None if storage in (None, "") else str(storage))

    def merge(self, data: Any) -> "BagOptions":
        if not isinstance(data, dict) or not data:
            return self
        return BagOptions.from_mapping({**self.to_json(), **data})

    def to_json(self) -> dict[str, Any]:
        return {"storage": self.storage}


@dataclass(frozen=True)
class RecordingConfig:
    output_root: Path = Path(DEFAULT_OUTPUT_ROOT).expanduser()
    session_layout: str = DEFAULT_SESSION_LAYOUT
    file_layout: str = DEFAULT_FILE_LAYOUT
    video: VideoOptions = field(default_factory=VideoOptions)
    bag: BagOptions = field(default_factory=BagOptions)

    @classmethod
    def from_mapping(cls, data: Any) -> "RecordingConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            output_root=Path(str(data.get("output_root", DEFAULT_OUTPUT_ROOT))).expanduser(),
            session_layout=str(data.get("session_layout", DEFAULT_SESSION_LAYOUT)),
            file_layout=str(data.get("file_layout", DEFAULT_FILE_LAYOUT)),
            video=VideoOptions.from_mapping(data.get("video", {})),
            bag=BagOptions.from_mapping(data.get("bag", {})),
        )

    @classmethod
    def from_registry(cls, registry: LauncherRegistry | None) -> "RecordingConfig":
        if registry is None:
            return cls()
        return cls.from_mapping(registry.recording or {})

    def session_dir(self, session_id: str) -> Path:
        relative = _format_layout(self.session_layout, {"session_id": session_id})
        return (self.output_root / relative).expanduser()

    def file_stem(self, context: dict[str, str]) -> str:
        return _format_layout(self.file_layout, context)

    def to_json(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "session_layout": self.session_layout,
            "file_layout": self.file_layout,
            "video": self.video.to_json(),
            "bag": self.bag.to_json(),
        }


@dataclass(frozen=True)
class TopicRecording:
    """One topic to record, with the sink and the output stem already resolved."""

    topic: str
    sink: str
    stem: str
    context: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "sink": self.sink,
            "stem": self.stem,
            "clip": self.context.get("clip") or None,
            "run_id": self.context.get("run_id") or None,
            "variant_id": self.context.get("variant_id") or None,
            "camera_namespace": self.context.get("camera_namespace") or None,
            "arg_name": self.context.get("arg_name") or None,
            "launcher_id": self.context.get("launcher_id") or None,
        }


def default_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_sink(topic: str) -> str:
    """Guess a sink from a topic name when the caller did not say which one."""
    segments = [segment for segment in topic.split("/") if segment]
    if any(segment in {"image", "image_raw", "compressed"} for segment in segments):
        return "video"
    return "bag"


def resolve_recordings(
    payload: dict[str, Any],
    config: RecordingConfig,
    session_id: str,
    registry: LauncherRegistry | None = None,
    processes: list[dict[str, Any]] | None = None,
) -> tuple[list[TopicRecording], list[dict[str, str]]]:
    """Resolve a record request into topics plus a list of skipped entries."""
    recordings: list[TopicRecording] = []
    skipped: list[dict[str, str]] = []
    clip = sanitize_token(str(payload.get("clip") or ""))

    for entry in payload.get("topics") or []:
        recording = _explicit_recording(entry, config, session_id, clip)
        recordings.append(recording)

    selected = _select_processes(payload, processes or [])
    for process in selected:
        found, process_skipped = _process_recordings(
            process, config, session_id, registry, clip
        )
        recordings.extend(found)
        skipped.extend(process_skipped)

    include = payload.get("include_sinks")
    if include:
        allowed = {str(sink) for sink in include}
        recordings = [item for item in recordings if item.sink in allowed]

    return _deduplicate(recordings), skipped


def build_manifest(
    session_id: str,
    session_dir: Path,
    recordings: list[dict[str, Any]],
    processes: list[dict[str, Any]] | None = None,
    bag: dict[str, Any] | None = None,
    clips: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "session_id": session_id,
        "session_dir": str(session_dir),
        "created_at": utc_now(),
        "bag": bag,
        # One entry per played bag in a batch run; recordings join on "clip".
        "clips": clips or [],
        "processes": [_manifest_process(process) for process in processes or []],
        "recordings": recordings,
    }


def write_manifest(session_dir: Path, manifest: dict[str, Any]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / MANIFEST_FILE_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_progress(path: Path) -> dict[str, Any] | None:
    """Read a recorder progress sidecar, tolerating a partially written file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def progress_path(video_path: Path) -> Path:
    return video_path.with_suffix(video_path.suffix + ".progress.json")


def stamp_csv_path(video_path: Path) -> Path:
    return video_path.with_suffix(video_path.suffix + ".stamps.csv")


def sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    return re.sub(r"_+", "_", cleaned).strip("_.")


def topic_slug(topic: str) -> str:
    return sanitize_token(topic.strip("/").replace("/", "_"))


def command_arg_value(command: list[str], name: str) -> str | None:
    prefix = f"{name}:="
    for item in reversed(command):
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _explicit_recording(
    entry: Any,
    config: RecordingConfig,
    session_id: str,
    clip: str = "",
) -> TopicRecording:
    if isinstance(entry, str):
        entry = {"topic": entry}
    if not isinstance(entry, dict):
        raise RecordingError("each topics entry must be an object or a topic string")

    topic = str(entry.get("topic", "")).strip()
    if not topic:
        raise RecordingError("each topics entry needs a topic")

    sink = str(entry.get("sink") or infer_sink(topic))
    if sink not in SINKS:
        raise RecordingError(f"unknown sink for {topic}: {sink}")

    name = entry.get("name")
    context = _base_context(session_id, clip)
    context["topic_slug"] = topic_slug(topic)
    stem = _sanitize_stem(str(name)) if name else context["topic_slug"]
    if not stem:
        raise RecordingError(f"empty output name for {topic}")
    return TopicRecording(
        topic=topic, sink=sink, stem=_with_clip(stem, clip, config), context=context
    )


def _select_processes(
    payload: dict[str, Any],
    processes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_id = payload.get("from_group_id")
    process_ids = payload.get("from_process_ids")
    if not group_id and not process_ids:
        return []

    wanted_ids = {str(item) for item in process_ids or []}
    selected = [
        process
        for process in processes
        if (group_id is not None and process.get("group_id") == group_id)
        or process.get("id") in wanted_ids
    ]
    if not selected:
        raise RecordingError("no matching processes for the record request")
    return selected


def _process_recordings(
    process: dict[str, Any],
    config: RecordingConfig,
    session_id: str,
    registry: LauncherRegistry | None,
    clip: str = "",
) -> tuple[list[TopicRecording], list[dict[str, str]]]:
    launcher_id = str(process.get("launcher_id", ""))
    if registry is None:
        return [], [{"launcher_id": launcher_id, "reason": "registry is not available"}]
    try:
        spec = registry.get(launcher_id)
    except RegistryError as exc:
        return [], [{"launcher_id": launcher_id, "reason": str(exc)}]
    if spec.record is None:
        return [], [{"launcher_id": launcher_id, "reason": "launcher declares no record args"}]

    command = [str(item) for item in process.get("command", [])]
    outputs = process.get("outputs") or {}
    camera = command_arg_value(command, "camera_namespace") or ""
    variant_id = process.get("variant_id") or launcher_id

    recordings: list[TopicRecording] = []
    skipped: list[dict[str, str]] = []
    for arg_name in spec.record.arg_names():
        context = _base_context(session_id, clip)
        context.update(
            {
                "run_id": sanitize_token(str(process.get("run_id") or "")),
                "variant_id": sanitize_token(str(variant_id)),
                "camera_namespace": sanitize_token(camera),
                "arg_name": sanitize_token(arg_name.replace("/", "_")),
                "launcher_id": sanitize_token(launcher_id),
            }
        )
        topic = _resolve_topic(arg_name, outputs, command, spec, context, camera)
        if not topic:
            skipped.append(
                {
                    "launcher_id": launcher_id,
                    "arg_name": arg_name,
                    "reason": "topic could not be resolved; pass it explicitly",
                }
            )
            continue
        context["topic_slug"] = topic_slug(topic)
        stem = _sanitize_stem(config.file_stem(context))
        if not stem:
            stem = context["topic_slug"]
        recordings.append(
            TopicRecording(
                topic=topic,
                sink=spec.record.sink_for(arg_name) or infer_sink(topic),
                stem=_with_clip(stem, clip, config),
                context=context,
            )
        )
    return recordings, skipped


def _resolve_topic(
    arg_name: str,
    outputs: dict[str, Any],
    command: list[str],
    spec: Any,
    context: dict[str, str],
    camera: str,
) -> str | None:
    isolated = outputs.get(arg_name)
    if isolated:
        return str(isolated)
    explicit = command_arg_value(command, arg_name)
    if explicit:
        return explicit
    if not camera:
        return None
    return spec.record.default_topic(arg_name, {"camera_namespace": camera})


def _deduplicate(recordings: list[TopicRecording]) -> list[TopicRecording]:
    seen: dict[tuple[str, str], TopicRecording] = {}
    for recording in recordings:
        seen.setdefault((recording.sink, recording.topic), recording)
    return list(seen.values())


def _base_context(session_id: str, clip: str = "") -> dict[str, str]:
    context = {key: "" for key in CONTEXT_KEYS}
    context["session_id"] = sanitize_token(session_id)
    context["clip"] = clip
    return context


def _with_clip(stem: str, clip: str, config: RecordingConfig) -> str:
    """Give every clip of a batch its own subdirectory."""
    # Without this, two clips of the same session resolve to the same file name
    # and the second overwrites the first. A layout placing {clip} itself is
    # left alone.
    if not clip or "{clip}" in config.file_layout:
        return stem
    return f"{clip}/{stem}"


def _manifest_process(process: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": process.get("id"),
        "launcher_id": process.get("launcher_id"),
        "label": process.get("label"),
        "run_id": process.get("run_id"),
        "variant_id": process.get("variant_id"),
        "group_id": process.get("group_id"),
        "command": list(process.get("command") or []),
        "outputs": dict(process.get("outputs") or {}),
    }


def _format_layout(layout: str, context: dict[str, str]) -> str:
    try:
        formatted = layout.format(**context)
    except KeyError as exc:
        raise RecordingError(f"unknown recording layout key: {exc.args[0]}") from exc
    except IndexError as exc:
        raise RecordingError("recording layout must use named keys") from exc
    return _sanitize_stem(formatted)


def _sanitize_stem(value: str) -> str:
    """Drop empty path segments left by absent metadata, and sanitize the rest."""
    segments = [sanitize_token(segment) for segment in value.split("/")]
    return "/".join(segment for segment in segments if segment)
