from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_image_bgr(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def write_image_bgr(path: str, image_bgr: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), image_bgr)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def decode_compressed_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg(image_bgr: np.ndarray, quality: int = 85) -> bytes:
    quality = int(np.clip(quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError("Failed to encode JPEG")
    return encoded.tobytes()
