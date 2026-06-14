from __future__ import annotations

from typing import List

import numpy as np

from ..types import BackendConfig, Detection


class BaseDetectorBackend:
    """Small interface shared by all detector backends."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.loaded = False

    @property
    def name(self) -> str:
        return self.config.backend

    def load(self) -> None:
        self.loaded = True

    def require_loaded(self) -> None:
        """Raise when inference is attempted before model initialization."""
        if not self.loaded:
            raise RuntimeError(f"{self.name} detector model is not loaded")

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        self.require_loaded()
        raise NotImplementedError
