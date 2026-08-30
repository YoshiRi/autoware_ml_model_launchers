from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import yaml


PACKAGE_NAME = "autoware_ml_model_launchers"
REGISTRY_FILE_NAME = "launcher_dashboard.yaml"


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ArgSpec:
    name: str
    value_type: str
    default: Any = None
    choices: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    group: str = "basic"

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "ArgSpec":
        value_type = str(data.get("type", "string"))
        if value_type not in {"string", "bool", "int", "float"}:
            raise RegistryError(f"unsupported type for {name}: {value_type}")
        choices = tuple(str(item) for item in data.get("choices", ()))
        suggestions = tuple(str(item) for item in data.get("suggestions", ()))
        return cls(
            name=name,
            value_type=value_type,
            default=data.get("default"),
            choices=choices,
            suggestions=suggestions,
            group=str(data.get("group", "basic")),
        )

    def coerce(self, value: Any) -> str:
        if self.value_type == "bool":
            return "true" if _coerce_bool(value) else "false"
        if self.value_type == "int":
            return str(int(value))
        if self.value_type == "float":
            return str(float(value))

        text = "" if value is None else str(value)
        if self.choices and text not in self.choices:
            raise RegistryError(f"{self.name} must be one of: {', '.join(self.choices)}")
        return text

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.value_type,
            "default": self.default,
            "choices": list(self.choices),
            "suggestions": list(self.suggestions),
            "group": self.group,
        }


@dataclass(frozen=True)
class LauncherSpec:
    launcher_id: str
    label: str
    package: str
    file: str
    args: dict[str, ArgSpec]
    isolation: "IsolationSpec | None" = None
    model_path_template: tuple[str, ...] = ()
    record: "RecordSpec | None" = None

    @classmethod
    def from_mapping(cls, launcher_id: str, data: dict[str, Any]) -> "LauncherSpec":
        args_data = data.get("args", {})
        if not isinstance(args_data, dict):
            raise RegistryError(f"launcher {launcher_id} args must be a mapping")
        args = {
            name: ArgSpec.from_mapping(name, spec if isinstance(spec, dict) else {})
            for name, spec in args_data.items()
        }
        return cls(
            launcher_id=launcher_id,
            label=str(data.get("label", launcher_id)),
            package=str(data["package"]),
            file=str(data["file"]),
            args=args,
            isolation=IsolationSpec.from_mapping(data.get("isolation", {})),
            model_path_template=_model_path_templates(
                data.get("model_path_template"),
                args,
                launcher_id,
            ),
            record=RecordSpec.from_mapping(data.get("record", {})),
        )

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.args.items()}

    def resolve_model_path(self, args: dict[str, Any] | None = None) -> str | None:
        """
        Return the model file this launcher loads for ``args``.

        The launch files derive model paths from several args, so the registry
        cannot know the model from any single arg. Each candidate template is
        tried in order and the first one whose referenced args all carry a
        value wins, which lets a launcher declare an explicit override arg
        first and the derived default afterwards. Returns ``None`` when the
        launcher declares no template or no candidate is fully populated.
        """
        if not self.model_path_template:
            return None
        values = self.defaults()
        values.update(args or {})
        context = {
            name: "" if value is None else str(value) for name, value in values.items()
        }
        for template in self.model_path_template:
            if all(context.get(field) for field in _template_fields(template)):
                return _format_template(template, context)
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.launcher_id,
            "label": self.label,
            "package": self.package,
            "file": self.file,
            "args": {name: spec.to_json() for name, spec in self.args.items()},
            "isolation": self.isolation.to_json() if self.isolation else None,
            "model_path_template": list(self.model_path_template),
            "model_path": self.resolve_model_path(),
            "record": self.record.to_json() if self.record else None,
        }


@dataclass(frozen=True)
class RecordSpec:
    """Declares which launch args carry topics worth recording, and to which sink."""

    video_args: tuple[str, ...]
    bag_args: tuple[str, ...]
    topic_templates: dict[str, str]

    @classmethod
    def from_mapping(cls, data: Any) -> "RecordSpec | None":
        if not isinstance(data, dict):
            return None
        video_args = _string_sequence(data.get("video_args", ()), "record video_args")
        bag_args = _string_sequence(data.get("bag_args", ()), "record bag_args")
        overlap = sorted(set(video_args) & set(bag_args))
        if overlap:
            raise RegistryError(f"record args listed twice: {', '.join(overlap)}")
        if not video_args and not bag_args:
            return None
        return cls(
            video_args=video_args,
            bag_args=bag_args,
            topic_templates=_string_mapping(
                data.get("topic_templates", {}), "record topic_templates"
            ),
        )

    def sink_for(self, arg_name: str) -> str | None:
        if arg_name in self.video_args:
            return "video"
        if arg_name in self.bag_args:
            return "bag"
        return None

    def arg_names(self) -> tuple[str, ...]:
        return self.video_args + self.bag_args

    def default_topic(self, arg_name: str, context: dict[str, str]) -> str | None:
        """Topic the launch file would use when the arg is left at its default."""
        template = self.topic_templates.get(arg_name)
        if template is None:
            return None
        return _format_template(template, context)

    def to_json(self) -> dict[str, Any]:
        return {
            "video_args": list(self.video_args),
            "bag_args": list(self.bag_args),
            "topic_templates": dict(self.topic_templates),
        }


@dataclass(frozen=True)
class IsolationSpec:
    arg_templates: dict[str, str]
    output_templates: dict[str, str]

    @classmethod
    def from_mapping(cls, data: Any) -> "IsolationSpec | None":
        if not isinstance(data, dict):
            return None
        arg_templates = _string_mapping(data.get("arg_templates", {}), "isolation arg_templates")
        output_templates = _string_mapping(data.get("output_args", {}), "isolation output_args")
        if not arg_templates and not output_templates:
            return None
        return cls(arg_templates=arg_templates, output_templates=output_templates)

    def build_args(self, context: dict[str, str]) -> dict[str, str]:
        args = {
            name: _format_template(template, context)
            for name, template in self.arg_templates.items()
        }
        args.update(
            {
                name: _format_template(template, context)
                for name, template in self.output_templates.items()
            }
        )
        return args

    def to_json(self) -> dict[str, Any]:
        return {
            "arg_templates": dict(self.arg_templates),
            "output_args": dict(self.output_templates),
        }


@dataclass(frozen=True)
class CameraLauncherPresetSpec:
    launcher_id: str
    cameras: tuple[str, ...]
    args: dict[str, ArgSpec]

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        default_launcher_id: str,
        error_prefix: str,
    ) -> "CameraLauncherPresetSpec":
        args_data = data.get("args", {})
        if not isinstance(args_data, dict):
            raise RegistryError(f"{error_prefix} args must be a mapping")
        return cls(
            launcher_id=str(data.get("launcher_id", default_launcher_id)),
            cameras=tuple(str(camera) for camera in data.get("cameras", ())),
            args={
                name: ArgSpec.from_mapping(name, spec if isinstance(spec, dict) else {})
                for name, spec in args_data.items()
            },
        )

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.args.items()}

    def to_json(self) -> dict[str, Any]:
        return {
            "launcher_id": self.launcher_id,
            "cameras": list(self.cameras),
            "args": {name: spec.to_json() for name, spec in self.args.items()},
        }


@dataclass(frozen=True)
class ComparisonVariantSpec:
    variant_id: str
    label: str
    launcher_id: str
    args: dict[str, Any]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ComparisonVariantSpec":
        args = data.get("args", {})
        if not isinstance(args, dict):
            raise RegistryError("model_comparison variant args must be a mapping")
        variant_id = str(data["id"])
        return cls(
            variant_id=variant_id,
            label=str(data.get("label", variant_id)),
            launcher_id=str(data["launcher_id"]),
            args=dict(args),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.variant_id,
            "label": self.label,
            "launcher_id": self.launcher_id,
            "args": dict(self.args),
        }


@dataclass(frozen=True)
class ModelComparisonSpec:
    cameras: tuple[str, ...]
    args: dict[str, ArgSpec]
    variants: tuple[ComparisonVariantSpec, ...]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ModelComparisonSpec":
        args_data = data.get("args", {})
        variants_data = data.get("variants", ())
        if not isinstance(args_data, dict):
            raise RegistryError("model_comparison args must be a mapping")
        if not isinstance(variants_data, list):
            raise RegistryError("model_comparison variants must be a list")
        return cls(
            cameras=tuple(str(camera) for camera in data.get("cameras", ())),
            args={
                name: ArgSpec.from_mapping(name, spec if isinstance(spec, dict) else {})
                for name, spec in args_data.items()
            },
            variants=tuple(
                ComparisonVariantSpec.from_mapping(item)
                for item in variants_data
                if isinstance(item, dict)
            ),
        )

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.args.items()}

    def to_json(self) -> dict[str, Any]:
        return {
            "cameras": list(self.cameras),
            "args": {name: spec.to_json() for name, spec in self.args.items()},
            "variants": [variant.to_json() for variant in self.variants],
        }


@dataclass(frozen=True)
class LauncherRegistry:
    launchers: dict[str, LauncherSpec]
    multi_yolox: CameraLauncherPresetSpec | None = None
    tlr_validation: CameraLauncherPresetSpec | None = None
    model_comparison: ModelComparisonSpec | None = None
    recording: dict[str, Any] | None = None

    def get(self, launcher_id: str) -> LauncherSpec:
        try:
            return self.launchers[launcher_id]
        except KeyError as exc:
            raise RegistryError(f"unknown launcher: {launcher_id}") from exc

    def to_json(self) -> dict[str, Any]:
        return {
            "launchers": [spec.to_json() for spec in self.launchers.values()],
            "multi_yolox": self.multi_yolox.to_json() if self.multi_yolox else None,
            "tlr_validation": self.tlr_validation.to_json() if self.tlr_validation else None,
            "model_comparison": (
                self.model_comparison.to_json() if self.model_comparison else None
            ),
            "recording": dict(self.recording or {}),
        }


def build_launch_command(spec: LauncherSpec, args: dict[str, Any] | None = None) -> list[str]:
    command = ["ros2", "launch", spec.package, spec.file]
    provided = args or {}
    unknown = sorted(set(provided) - set(spec.args))
    if unknown:
        raise RegistryError(f"unknown args for {spec.launcher_id}: {', '.join(unknown)}")
    merged = spec.defaults()
    merged.update(provided)
    for name, value in merged.items():
        if value is None:
            continue
        command.append(f"{name}:={spec.args[name].coerce(value)}")
    return command


def load_registry(path: Path) -> LauncherRegistry:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    launchers_data = data.get("launchers", {})
    if not isinstance(launchers_data, dict):
        raise RegistryError("launchers must be a mapping")
    launchers = {
        launcher_id: LauncherSpec.from_mapping(launcher_id, spec)
        for launcher_id, spec in launchers_data.items()
    }
    multi_data = data.get("multi_yolox")
    multi_yolox = (
        CameraLauncherPresetSpec.from_mapping(multi_data, "yolox_camera", "multi_yolox")
        if isinstance(multi_data, dict)
        else None
    )
    tlr_data = data.get("tlr_validation")
    tlr_validation = (
        CameraLauncherPresetSpec.from_mapping(
            tlr_data,
            "tlr_detect_and_classifier",
            "tlr_validation",
        )
        if isinstance(tlr_data, dict)
        else None
    )
    comparison_data = data.get("model_comparison")
    model_comparison = (
        ModelComparisonSpec.from_mapping(comparison_data)
        if isinstance(comparison_data, dict)
        else None
    )
    recording_data = data.get("recording")
    return LauncherRegistry(
        launchers=launchers,
        multi_yolox=multi_yolox,
        tlr_validation=tlr_validation,
        model_comparison=model_comparison,
        recording=dict(recording_data) if isinstance(recording_data, dict) else None,
    )


def default_registry_path() -> Path:
    source_tree_path = Path(__file__).parents[2] / "config" / REGISTRY_FILE_NAME
    if source_tree_path.is_file():
        return source_tree_path

    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError as exc:
        raise RegistryError("ament_index_python is required to locate installed config") from exc

    return Path(get_package_share_directory(PACKAGE_NAME)) / "config" / REGISTRY_FILE_NAME


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise RegistryError(f"invalid bool value: {value}")


def _string_mapping(data: Any, name: str) -> dict[str, str]:
    if not isinstance(data, dict):
        raise RegistryError(f"{name} must be a mapping")
    return {str(key): str(value) for key, value in data.items()}


def _template_fields(template: str) -> list[str]:
    return [field for _, field, _, _ in Formatter().parse(template) if field]


def _model_path_templates(
    data: Any,
    args: dict[str, ArgSpec],
    launcher_id: str,
) -> tuple[str, ...]:
    if data is None:
        return ()
    if isinstance(data, str):
        templates = (data,)
    elif isinstance(data, list) and all(isinstance(item, str) for item in data):
        templates = tuple(data)
    else:
        raise RegistryError(
            f"launcher {launcher_id} model_path_template must be a string or a list of strings"
        )
    for template in templates:
        for field in _template_fields(template):
            if field not in args:
                raise RegistryError(
                    f"launcher {launcher_id} model_path_template references unknown arg: {field}"
                )
    return templates


def _string_sequence(data: Any, name: str) -> tuple[str, ...]:
    if isinstance(data, str) or not isinstance(data, (list, tuple)):
        raise RegistryError(f"{name} must be a list")
    return tuple(str(item) for item in data)


def _format_template(template: str, context: dict[str, str]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        raise RegistryError(f"unknown isolation template key: {exc.args[0]}") from exc
