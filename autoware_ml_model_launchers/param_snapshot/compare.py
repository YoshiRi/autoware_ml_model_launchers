from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SnapshotLoader(yaml.SafeLoader):
    """YAML loader that keeps RTUI's Python-specific tags as plain YAML values."""


def _construct_unknown(loader: SnapshotLoader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return None


SnapshotLoader.add_multi_constructor("", _construct_unknown)


@dataclass(frozen=True)
class ParameterRecord:
    key: str
    value: Any
    raw_value: Any


def load_yaml_text(text: str) -> Any:
    return yaml.load(text, Loader=SnapshotLoader)


def load_yaml_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return load_yaml_text(stream.read())


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            parsed = yaml.safe_load(stripped)
        except yaml.YAMLError:
            return value
        if parsed is None and stripped.lower() not in {"null", "~"}:
            return value
        if isinstance(parsed, str) and parsed == stripped:
            return value
        return normalize_value(parsed)
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    return value


def flatten_parameters(data: dict[str, Any], prefix: str = "") -> dict[str, ParameterRecord]:
    records: dict[str, ParameterRecord] = {}
    for key, value in data.items():
        joined = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            records.update(flatten_parameters(value, joined))
            continue
        records[joined] = ParameterRecord(joined, normalize_value(value), value)
    return records


def extract_parameter_map(document: Any) -> dict[str, ParameterRecord]:
    if not isinstance(document, dict):
        return {}

    if isinstance(document.get("parameters"), dict):
        return flatten_parameters(document["parameters"])

    merged: dict[str, Any] = {}
    for node_name, node_data in document.items():
        if not isinstance(node_data, dict):
            continue
        params = node_data.get("ros__parameters")
        if isinstance(params, dict):
            for key, value in flatten_parameters(params).items():
                merged[f"{node_name}:{key}"] = value.raw_value

    if merged:
        return flatten_parameters(merged)

    ros_params = document.get("ros__parameters")
    if isinstance(ros_params, dict):
        return flatten_parameters(ros_params)

    return flatten_parameters(document)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def compare_documents(left: Any, right: Any) -> dict[str, Any]:
    left_params = extract_parameter_map(left)
    right_params = extract_parameter_map(right)
    left_keys = set(left_params)
    right_keys = set(right_params)

    added = sorted(right_keys - left_keys)
    removed = sorted(left_keys - right_keys)
    common = sorted(left_keys & right_keys)
    changed = [key for key in common if left_params[key].value != right_params[key].value]
    same = [key for key in common if left_params[key].value == right_params[key].value]

    return {
        "summary": {
            "left_params": len(left_params),
            "right_params": len(right_params),
            "same": len(same),
            "changed": len(changed),
            "added": len(added),
            "removed": len(removed),
        },
        "changed": [
            {
                "key": key,
                "left": _jsonable(left_params[key].value),
                "right": _jsonable(right_params[key].value),
                "left_raw": _jsonable(left_params[key].raw_value),
                "right_raw": _jsonable(right_params[key].raw_value),
            }
            for key in changed
        ],
        "added": [
            {"key": key, "right": _jsonable(right_params[key].value)}
            for key in added
        ],
        "removed": [
            {"key": key, "left": _jsonable(left_params[key].value)}
            for key in removed
        ],
    }


def compare_texts(left_text: str, right_text: str) -> dict[str, Any]:
    return compare_documents(load_yaml_text(left_text), load_yaml_text(right_text))


def compare_files(left_path: Path, right_path: Path) -> dict[str, Any]:
    return compare_documents(load_yaml_file(left_path), load_yaml_file(right_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Autoware/ROS parameter snapshots.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    result = compare_files(args.left, args.right)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    summary = result["summary"]
    print(
        "left={left_params} right={right_params} same={same} changed={changed} "
        "added={added} removed={removed}".format(**summary)
    )
    for item in result["changed"]:
        print(f"~ {item['key']}: {item['left']} -> {item['right']}")
    for item in result["added"]:
        print(f"+ {item['key']}: {item['right']}")
    for item in result["removed"]:
        print(f"- {item['key']}: {item['left']}")


if __name__ == "__main__":
    main()
