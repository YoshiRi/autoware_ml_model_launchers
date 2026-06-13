from __future__ import annotations

from .types import BackendConfig


BACKEND_ALIASES = {
    "yolo": "ultralytics",
    "ultralytics_yolo": "ultralytics",
    "ultralytics": "ultralytics",
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


def create_backend(config: BackendConfig):
    """
    Create a backend with lazy imports.

    Only the selected backend module is imported. Heavy ML libraries are imported inside
    backend.load(), so missing dependencies do not break unrelated backends or unit tests.
    """
    name = canonical_backend_name(config.backend)
    if name == "ultralytics":
        from .backends.ultralytics_backend import UltralyticsYoloBackend

        return UltralyticsYoloBackend(config)
    if name == "dfine":
        from .backends.dfine_backend import DFineBackend

        return DFineBackend(config)
    if name == "rfdetr":
        from .backends.rfdetr_backend import RFDetrBackend

        return RFDetrBackend(config)
    if name == "fake":
        from .backends.fake_backend import FakeBackend

        return FakeBackend(config)
    raise AssertionError(f"Unhandled backend {name}")
