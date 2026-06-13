#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List

import cv2
import numpy as np

from .image_io import write_image_bgr


def make_image(width: int = 960, height: int = 540) -> np.ndarray:
    img = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.rectangle(img, (0, int(height * 0.62)), (width, height), (110, 110, 110), -1)
    cv2.line(img, (0, int(height * 0.80)), (width, int(height * 0.72)), (255, 255, 255), 3)
    cv2.rectangle(img, (int(width * 0.18), int(height * 0.45)), (int(width * 0.42), int(height * 0.72)), (40, 80, 180), -1)
    cv2.circle(img, (int(width * 0.23), int(height * 0.72)), 28, (20, 20, 20), -1)
    cv2.circle(img, (int(width * 0.37), int(height * 0.72)), 28, (20, 20, 20), -1)
    cv2.rectangle(img, (int(width * 0.62), int(height * 0.38)), (int(width * 0.66), int(height * 0.70)), (40, 40, 40), -1)
    cv2.circle(img, (int(width * 0.64), int(height * 0.34)), 22, (40, 40, 40), -1)
    cv2.putText(img, "synthetic test image", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Create a synthetic image for plumbing tests.")
    p.add_argument("--output", default="/tmp/open_detector_test.jpg")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args(argv)
    write_image_bgr(args.output, make_image(args.width, args.height))
    print(args.output)


if __name__ == "__main__":
    main()
