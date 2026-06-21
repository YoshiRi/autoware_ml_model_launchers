from __future__ import annotations

from .types import BackendConfig


BACKEND_ALIASES = {
    "yolo": "ultralytics",
    "ultralytics_yolo": "ultralytics",
    "ultralytics": "ultralytics",
    "world": "yolo_world",
    "yolo-world": "yolo_world",
    "yolo_world": "yolo_world",
    "yoloworld": "yolo_world",
    "dfine": "dfine",
    "d-fine": "dfine",
    "rfdetr": "rfdetr",
    "rf-detr": "rfdetr",
    "fake": "fake",
}


def canonical_backend_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key not in BACKEND_ALIASES:
        supported = ", ".join(sorted(BACKEND_ALIASES))
        raise ValueError(f"Unsupported backend {name!r}. Supported: {supported}")
    return BACKEND_ALIASES[key]


def create_backend(config: BackendConfig, *, load: bool = True):
    """
    Create a detector backend and load its model during backend construction by default.

    Only the selected backend module is imported. When `load` is true, the returned detector is
    ready for inference; dependency and model errors are raised during construction.
    """
    name = canonical_backend_name(config.backend)
    if name == "ultralytics":
        from .backends.ultralytics_backend import UltralyticsYoloBackend

        backend = UltralyticsYoloBackend(config, autoload=load)
    elif name == "yolo_world":
        from .backends.yolo_world_backend import YoloWorldBackend

        backend = YoloWorldBackend(config, autoload=load)
    elif name == "dfine":
        from .backends.dfine_backend import DFineBackend

        backend = DFineBackend(config, autoload=load)
    elif name == "rfdetr":
        from .backends.rfdetr_backend import RFDetrBackend

        backend = RFDetrBackend(config, autoload=load)
    elif name == "fake":
        from .backends.fake_backend import FakeBackend

        backend = FakeBackend(config, autoload=load)
    else:
        raise AssertionError(f"Unhandled backend {name}")

    return backend
