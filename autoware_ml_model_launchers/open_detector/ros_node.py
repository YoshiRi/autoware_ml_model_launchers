#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .backend_loader import create_backend
from .drawing import draw_detections
from .filtering import (
    DEFAULT_DRIVING_CLASS_FILTER,
    DEFAULT_DRIVING_LABEL_MAP,
    parse_class_filter,
    parse_label_map,
    parse_string_list,
)
from .image_io import decode_compressed_image, encode_jpeg
from .runtime import OpenDetectorRuntime
from .types import BackendConfig, Detection


class OpenDetectorNode(Node):
    """
    Backend-switchable detector node.

    The ROS interface is fixed. The selected detector model is loaded during node initialization.
    """

    def __init__(self) -> None:
        super().__init__("open_detector_node")

        # Topics and image input type.
        self.declare_parameter("input_topic", "/image_raw/compressed")
        self.declare_parameter("input_image_type", "compressed")  # compressed or image
        self.declare_parameter("detections_topic", "~/detections")
        self.declare_parameter("debug_image_topic", "~/debug/image")
        self.declare_parameter("debug_compressed_topic", "~/debug/image/compressed")
        self.declare_parameter("latency_topic", "~/latency/infer_ms")

        # Backend/runtime.
        dynamic_parameter = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("backend", "ultralytics")
        self.declare_parameter("model", "")
        self.declare_parameter("device", "", dynamic_parameter)
        self.declare_parameter("imgsz", 960)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.70)
        self.declare_parameter("max_det", 100)
        self.declare_parameter("half", False)
        self.declare_parameter("prompt_classes", [], dynamic_parameter)
        self.declare_parameter("backend_config_json", "{}")

        # Output behavior.
        self.declare_parameter(
            "class_filter", DEFAULT_DRIVING_CLASS_FILTER, dynamic_parameter
        )
        self.declare_parameter("label_map", [], dynamic_parameter)
        self.declare_parameter("use_default_driving_label_map", False)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_debug_compressed", False)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("drop_if_busy", True)
        self.declare_parameter("log_interval_sec", 2.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.input_image_type = str(self.get_parameter("input_image_type").value).strip().lower()
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.debug_compressed_topic = str(self.get_parameter("debug_compressed_topic").value)
        self.latency_topic = str(self.get_parameter("latency_topic").value)

        backend = str(self.get_parameter("backend").value)
        model = str(self.get_parameter("model").value)
        device = str(self.get_parameter("device").value)
        imgsz = int(self.get_parameter("imgsz").value)
        conf_thres = float(self.get_parameter("conf_thres").value)
        iou_thres = float(self.get_parameter("iou_thres").value)
        max_det = int(self.get_parameter("max_det").value)
        half = bool(self.get_parameter("half").value)
        extra = self._parse_json_object(str(self.get_parameter("backend_config_json").value), "backend_config_json")
        prompt_classes = parse_string_list(self.get_parameter("prompt_classes").value)
        if prompt_classes:
            extra["classes"] = prompt_classes

        self.class_filter = parse_class_filter(self.get_parameter("class_filter").value)
        self.label_map = parse_label_map(self.get_parameter("label_map").value)
        if bool(self.get_parameter("use_default_driving_label_map").value):
            merged = dict(DEFAULT_DRIVING_LABEL_MAP)
            merged.update(self.label_map)
            self.label_map = merged

        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.publish_debug_compressed = bool(self.get_parameter("publish_debug_compressed").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.drop_if_busy = bool(self.get_parameter("drop_if_busy").value)
        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.det_pub = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        self.latency_pub = self.create_publisher(Float64, self.latency_topic, 10)
        self.debug_image_pub = (
            self.create_publisher(Image, self.debug_image_topic, sensor_qos)
            if self.publish_debug_image
            else None
        )
        self.debug_compressed_pub = (
            self.create_publisher(CompressedImage, self.debug_compressed_topic, sensor_qos)
            if self.publish_debug_compressed
            else None
        )

        if self.input_image_type == "compressed":
            self.sub = self.create_subscription(CompressedImage, self.input_topic, self.compressed_callback, sensor_qos)
        elif self.input_image_type == "image":
            self.sub = self.create_subscription(Image, self.input_topic, self.image_callback, sensor_qos)
        else:
            raise ValueError("input_image_type must be 'compressed' or 'image'")

        self.backend_config = BackendConfig(
            backend=backend,
            model=model,
            device=device,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            half=half,
            extra=extra,
        )
        self.get_logger().info(
            f"Loading detector: backend={backend} model={model or '<backend-default>'}"
        )
        self.backend = create_backend(self.backend_config)
        self.runtime = OpenDetectorRuntime(
            self.backend,
            class_filter=self.class_filter,
            label_map=self.label_map,
            max_det=max_det,
        )

        self.infer_lock = threading.Lock()
        self.last_log_time = 0.0
        self.frame_count = 0
        self.drop_count = 0

        self.get_logger().info(
            "Ready. input=%s type=%s backend=%s model=%s detections=%s debug_image=%s debug_compressed=%s classes=%s"
            % (
                self.input_topic,
                self.input_image_type,
                backend,
                model or "<backend-default>",
                self.detections_topic,
                self.debug_image_topic if self.publish_debug_image else "disabled",
                self.debug_compressed_topic if self.publish_debug_compressed else "disabled",
                sorted(self.class_filter) if self.class_filter else "all",
            )
        )

    @staticmethod
    def _parse_json_object(value: str, param_name: str) -> dict:
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"{param_name} must be a JSON object string: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{param_name} must be a JSON object string")
        return parsed

    def compressed_callback(self, msg: CompressedImage) -> None:
        image = decode_compressed_image(msg.data)
        if image is None:
            self.get_logger().warn("Failed to decode CompressedImage")
            return
        self._process_image(image, msg.header)

    def image_callback(self, msg: Image) -> None:
        try:
            image = self._image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode Image: {type(exc).__name__}: {exc}")
            return
        self._process_image(image, msg.header)

    @staticmethod
    def _image_msg_to_bgr(msg: Image) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        enc = str(msg.encoding).lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if enc == "bgr8":
            return data.reshape((height, width, 3)).copy()
        if enc == "rgb8":
            rgb = data.reshape((height, width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if enc == "bgra8":
            bgra = data.reshape((height, width, 4))
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        if enc == "rgba8":
            rgba = data.reshape((height, width, 4))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if enc == "mono8":
            mono = data.reshape((height, width))
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    def _process_image(self, image_bgr: np.ndarray, header) -> None:
        if self.drop_if_busy and not self.infer_lock.acquire(blocking=False):
            self.drop_count += 1
            return
        if not self.drop_if_busy:
            self.infer_lock.acquire()

        try:
            result = self.runtime.update(image_bgr)
            self.latency_pub.publish(Float64(data=result.infer_ms))

            self.det_pub.publish(self._to_detection_array(result.detections, header))

            if self.publish_debug_image or self.publish_debug_compressed:
                drawn = draw_detections(image_bgr, result.detections)
                if self.debug_image_pub is not None:
                    self.debug_image_pub.publish(self._cv2_to_image_msg(drawn, header))
                if self.debug_compressed_pub is not None:
                    self.debug_compressed_pub.publish(self._cv2_to_compressed_msg(drawn, header))

            self.frame_count += 1
            now = time.monotonic()
            if now - self.last_log_time >= self.log_interval_sec:
                self.last_log_time = now
                self.get_logger().info(
                    "frame=%d det=%d raw=%d infer=%.1f ms drops=%d backend_loaded=%s"
                    % (
                        self.frame_count,
                        len(result.detections),
                        len(result.raw_detections),
                        result.infer_ms,
                        self.drop_count,
                        self.backend.loaded,
                    )
                )
        except Exception as exc:
            self.get_logger().error(f"Detector callback failed: {type(exc).__name__}: {exc}")
        finally:
            self.infer_lock.release()

    @staticmethod
    def _to_detection_array(detections: List[Detection], header) -> Detection2DArray:
        out = Detection2DArray()
        out.header = header
        for i, d in enumerate(detections):
            det = Detection2D()
            det.header = header
            det.id = str(i)
            det.bbox.center.position.x = float(d.cx)
            det.bbox.center.position.y = float(d.cy)
            det.bbox.center.theta = 0.0
            det.bbox.size_x = float(d.width)
            det.bbox.size_y = float(d.height)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(d.label)
            hyp.hypothesis.score = float(d.score)
            det.results.append(hyp)
            out.detections.append(det)
        return out

    @staticmethod
    def _cv2_to_image_msg(image_bgr: np.ndarray, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(image_bgr.shape[0])
        msg.width = int(image_bgr.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(image_bgr.shape[1] * 3)
        msg.data = image_bgr.tobytes()
        return msg

    def _cv2_to_compressed_msg(self, image_bgr: np.ndarray, header) -> CompressedImage:
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data = encode_jpeg(image_bgr, self.jpeg_quality)
        return msg


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = OpenDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except (KeyboardInterrupt, ExternalShutdownException):
                pass


if __name__ == "__main__":
    main()
