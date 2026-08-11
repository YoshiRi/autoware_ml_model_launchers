#!/usr/bin/env python3
"""Command line entry points for recording launcher debug topics."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

from ..launcher_dashboard.registry import default_registry_path, load_registry
from .manager import RecordingManager
from .spec import RecordingConfig, RecordingError


def _load_config(registry_path: Path | None) -> tuple[RecordingConfig, Any]:
    path = registry_path or default_registry_path()
    registry = load_registry(path)
    return RecordingConfig.from_registry(registry), registry


def build_record_topics_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record_topics",
        description=(
            "Record image topics to mp4 and other topics to a rosbag, then write "
            "a manifest.json describing the session."
        ),
    )
    parser.add_argument(
        "-t", "--topic", action="append", dest="video_topics", metavar="TOPIC", default=[],
        help="image topic recorded as video (repeatable)",
    )
    parser.add_argument(
        "-b", "--bag-topic", action="append", dest="bag_topics", metavar="TOPIC", default=[],
        help="topic recorded into a rosbag (repeatable)",
    )
    parser.add_argument(
        "-n", "--name", action="append", dest="names", metavar="NAME", default=[],
        help="output stem for the video topic at the same position as -t (repeatable)",
    )
    parser.add_argument("-o", "--output-root", default=None, help="override the output root")
    parser.add_argument("--session-id", default=None, help="session directory name")
    parser.add_argument("--fps", type=float, default=None, help="fixed output fps")
    parser.add_argument("--encoder", default=None, choices=["auto", "ffmpeg", "opencv"])
    parser.add_argument("--crf", type=int, default=None, help="libx264 crf, lower is better")
    parser.add_argument("--wait", type=float, default=None, help="seconds to wait for publishers")
    parser.add_argument("--storage", default=None, help="rosbag storage id, e.g. mcap or sqlite3")
    parser.add_argument("--registry", type=Path, default=None, help="launcher registry YAML")
    parser.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    return parser


def record_topics_main(argv: list[str] | None = None) -> int:
    args = build_record_topics_parser().parse_args(argv)
    if not args.video_topics and not args.bag_topics:
        raise SystemExit("at least one -t/--topic or -b/--bag-topic is required")
    if len(args.names) > len(args.video_topics):
        raise SystemExit(f"got {len(args.names)} -n names for {len(args.video_topics)} topics")

    config, _registry = _load_config(args.registry)
    if args.output_root:
        config = replace(config, output_root=Path(args.output_root).expanduser())

    topics: list[dict[str, Any]] = []
    for index, topic in enumerate(args.video_topics):
        entry: dict[str, Any] = {"topic": topic, "sink": "video"}
        if index < len(args.names):
            entry["name"] = args.names[index]
        topics.append(entry)
    topics += [{"topic": topic, "sink": "bag"} for topic in args.bag_topics]

    payload: dict[str, Any] = {"topics": topics, "session_id": args.session_id}
    payload["video"] = _present(
        {"fps": args.fps, "encoder": args.encoder, "crf": args.crf, "wait": args.wait}
    )
    payload["bag"] = _present({"storage": args.storage})

    manager = RecordingManager(config=config)
    if args.dry_run:
        plan = manager.preview(payload)
        for recorder in plan["recorders"]:
            print(f"[{recorder['sink']}] $ {' '.join(recorder['command'])}")
        print(f"session dir: {plan['session_dir']}")
        return 0

    return _run_foreground(manager, payload, duration=args.duration)


def _run_foreground(
    manager: RecordingManager,
    payload: dict[str, Any],
    duration: float | None = None,
    processes: list[dict[str, Any]] | None = None,
) -> int:
    try:
        started = manager.start(payload, processes)
    except RecordingError as exc:
        raise SystemExit(str(exc))

    print(f"session dir: {started['session_dir']}")
    for recording in started["recordings"]:
        print(f"[{recording['sink']}] $ {' '.join(recording['command'])}")
    for skipped in started["skipped"]:
        print(f"skipped {skipped.get('arg_name', '')}: {skipped['reason']}")
    print("recording... press Ctrl-C to stop")

    deadline = None if duration is None else time.monotonic() + duration
    try:
        while True:
            time.sleep(1.0)
            if deadline is not None and time.monotonic() >= deadline:
                break
            if not any(item["running"] for item in manager.list_recordings()):
                print("all recorders exited")
                break
    except KeyboardInterrupt:
        print("\nstopping recorders")

    manager.stop_all()
    result = manager.finalize(processes=processes)
    print(f"manifest: {result['manifest_path']}")
    for entry in result["manifest"]["recordings"]:
        frames = entry.get("frames")
        detail = f"{frames} frames" if frames is not None else f"{entry.get('bytes', 0)} bytes"
        print(f"  {entry['sink']:5s} {entry['file']} ({detail})")
    return 0


def _present(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def build_record_session_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record_session",
        description=(
            "Run launch, playback, and recording for every session in a config "
            "file, unattended."
        ),
    )
    parser.add_argument("config", help="session YAML config (see docs/topic_recording_design.md)")
    parser.add_argument("--only", action="append", metavar="ID", help="run only these session ids")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failure")
    parser.add_argument("--registry", type=Path, default=None, help="launcher registry YAML")
    return parser


def record_session_main(argv: list[str] | None = None) -> int:
    from .session import run_sessions

    args = build_record_session_parser().parse_args(argv)
    results = run_sessions(
        Path(args.config).expanduser(),
        only=args.only,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        registry_path=args.registry,
    )
    print("\n=== summary")
    for result in results:
        print(f"  {'ok  ' if result['ok'] else 'FAIL'} {result['id']}: {result.get('detail', '')}")
    if not args.dry_run:
        print(json.dumps({"sessions": results}, indent=2))
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(record_topics_main())
