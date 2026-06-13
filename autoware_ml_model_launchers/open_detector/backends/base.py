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

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        if not self.loaded:
            self.load()
        raise NotImplementedError
