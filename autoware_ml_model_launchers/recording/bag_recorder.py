from __future__ import annotations

from pathlib import Path


def build_bag_record_command(
    output_dir: Path,
    topics: list[str],
    storage: str | None = "mcap",
) -> list[str]:
    """Build a ros2 bag record command for an explicit topic list."""
    if not topics:
        raise ValueError("at least one topic is required to record a bag")
    command = ["ros2", "bag", "record", "-o", str(output_dir)]
    if storage:
        command += ["--storage", storage]
    return command + list(topics)


def bag_output_state(output_dir: Path) -> dict[str, object]:
    """Report whether a recorded bag directory exists yet and how large it is."""
    metadata = output_dir / "metadata.yaml"
    size = 0
    if output_dir.is_dir():
        for child in output_dir.rglob("*"):
            try:
                if child.is_file():
                    size += child.stat().st_size
            except OSError:
                continue
    return {
        "exists": output_dir.is_dir(),
        "finalized": metadata.is_file(),
        "bytes": size,
    }
