#!/usr/bin/env python3
"""Publish a screen region as sensor_msgs/CompressedImage, like a camera would."""

from __future__ import annotations

import sys
import threading
import time

import cv2
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from .capture import CaptureError, create_capture, resolve_region


REPORT_INTERVAL_SEC = 5.0

PARAMETER_DEFAULTS: dict[str, object] = {
    "output_topic": "/sensing/camera/screen/image_raw/compressed",
    "frame_id": "screen",
    "fps": 10.0,
    "region": "",
    "monitor": 0,
    "display": "",
    "backend": "auto",
    "jpeg_quality": 80,
    "resize_width": 0,
    "resize_height": 0,
    "show_cursor": False,
}

# Declared with dynamic typing so that fps:=4 works as well as fps:=4.0. A
# statically typed DOUBLE rejects the integer a user naturally types on the
# command line, and the node would die before capturing anything.
NUMERIC_PARAMETERS = ("fps", "monitor", "jpeg_quality", "resize_width", "resize_height")


def scaled_size(width: int, height: int, target_width: int, target_height: int) -> tuple[int, int]:
    """Resolve the output size, keeping the aspect ratio when one side is given."""
    if target_width <= 0 and target_height <= 0:
        return width, height
    if target_width > 0 and target_height > 0:
        return target_width, target_height
    if target_width > 0:
        return target_width, max(1, round(height * target_width / width))
    return max(1, round(width * target_height / height)), target_height


class ScreenCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("screen_camera")

        for name, default in PARAMETER_DEFAULTS.items():
            if name in NUMERIC_PARAMETERS:
                self.declare_parameter(
                    name, default, ParameterDescriptor(dynamic_typing=True)
                )
            else:
                self.declare_parameter(name, default)

        topic = self.get_parameter("output_topic").value
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.fps = max(0.1, float(self.get_parameter("fps").value))
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        display = str(self.get_parameter("display").value) or None

        self.region = resolve_region(
            region=str(self.get_parameter("region").value),
            monitor=int(self.get_parameter("monitor").value),
            display=display,
        )
        self.size = scaled_size(
            self.region.width,
            self.region.height,
            int(self.get_parameter("resize_width").value),
            int(self.get_parameter("resize_height").value),
        )
        self.capture = create_capture(
            region=self.region,
            fps=self.fps,
            backend=str(self.get_parameter("backend").value),
            display=display,
            show_cursor=bool(self.get_parameter("show_cursor").value),
        )

        # Sensor data QoS, so the ML nodes subscribe with their usual profile.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(CompressedImage, str(topic), qos)

        self.frames = 0
        self.dropped = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="screen_capture", daemon=True)

        self.get_logger().info(
            f"capturing {self.region.geometry} at {self.fps:g} fps "
            f"-> {topic} ({self.size[0]}x{self.size[1]} jpeg q{self.jpeg_quality})"
        )
        self.create_timer(REPORT_INTERVAL_SEC, self._report)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.capture.close()
        self.get_logger().info(f"published {self.frames} frames, {self.dropped} dropped")

    def _run(self) -> None:
        period = 1.0 / self.fps
        next_frame = time.monotonic()
        while not self._stop.is_set():
            try:
                frame = self.capture.read()
            except CaptureError as exc:
                self.get_logger().error(str(exc))
                return
            if frame is None:
                self.dropped += 1
                continue

            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)

            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                self.dropped += 1
                continue

            message = CompressedImage()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.frame_id
            message.format = "jpeg"
            message.data = encoded.tobytes()
            self.publisher.publish(message)
            self.frames += 1

            if not getattr(self.capture, "paced", False):
                # ffmpeg paces itself; mss returns immediately, so keep the rate here.
                next_frame += period
                delay = next_frame - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_frame = time.monotonic()

    def _report(self) -> None:
        self.get_logger().info(f"frames published: {self.frames}")


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=argv)
    try:
        node = ScreenCameraNode()
    except CaptureError as exc:
        print(f"screen_camera: {exc}", file=sys.stderr)
        rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
