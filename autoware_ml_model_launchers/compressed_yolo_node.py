#!/usr/bin/env python3
"""Run an Ultralytics detector on ROS 2 compressed images."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Sequence
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Failed to import ultralytics. Activate the Open YOLO virtual environment "
        "or install requirements-open-yolo.txt."
    ) from exc


DrawItem = Tuple[int, int, int, int, str, float, int]


class CompressedYoloNode(Node):
    """Publish generic 2D detections from a compressed camera stream."""

    def __init__(self) -> None:
        super().__init__("compressed_yolo_node")

        self.declare_parameter("input_topic", "/image_raw/compressed")
        self.declare_parameter("detections_topic", "~/detections")
        self.declare_parameter("debug_image_topic", "~/debug/image")
        self.declare_parameter("debug_compressed_topic", "~/debug/image/compressed")
        self.declare_parameter("latency_topic", "~/latency/predict_ms")

        self.declare_parameter("model", "yolo26s.pt")
        dynamic_parameter = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("device", "", dynamic_parameter)
        self.declare_parameter("imgsz", 960)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.70)
        self.declare_parameter("max_det", 100)
        self.declare_parameter("half", False)

        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_debug_compressed", False)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("drop_if_busy", True)
        self.declare_parameter("log_interval_sec", 2.0)
        self.declare_parameter(
            "class_filter",
            [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "bus",
                "truck",
                "traffic light",
                "stop sign",
            ],
            dynamic_parameter,
        )

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.debug_compressed_topic = str(
            self.get_parameter("debug_compressed_topic").value
        )
        self.latency_topic = str(self.get_parameter("latency_topic").value)

        self.model_name = str(self.get_parameter("model").value)
        self.device = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.iou_thres = float(self.get_parameter("iou_thres").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.half = bool(self.get_parameter("half").value)

        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        self.publish_debug_compressed = bool(
            self.get_parameter("publish_debug_compressed").value
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.drop_if_busy = bool(self.get_parameter("drop_if_busy").value)
        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)
        self.class_filter = self._normalize_class_filter(
            self.get_parameter("class_filter").value
        )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.det_pub = self.create_publisher(
            Detection2DArray, self.detections_topic, 10
        )
        self.latency_pub = self.create_publisher(Float64, self.latency_topic, 10)
        self.debug_image_pub = (
            self.create_publisher(Image, self.debug_image_topic, sensor_qos)
            if self.publish_debug_image
            else None
        )
        self.debug_compressed_pub = (
            self.create_publisher(
                CompressedImage, self.debug_compressed_topic, sensor_qos
            )
            if self.publish_debug_compressed
            else None
        )
        self.sub = self.create_subscription(
            CompressedImage, self.input_topic, self.image_callback, sensor_qos
        )

        self.infer_lock = threading.Lock()
        self.last_log_time = 0.0
        self.frame_count = 0

        self.get_logger().info(f"Loading Ultralytics model: {self.model_name}")
        self.model = YOLO(self.model_name)
        self.names = self._extract_names(self.model.names)
        self.get_logger().info(
            "Ready. input=%s, detections=%s, model=%s, device=%s, imgsz=%d"
            % (
                self.input_topic,
                self.detections_topic,
                self.model_name,
                self.device if self.device else "default",
                self.imgsz,
            )
        )

    @staticmethod
    def _normalize_class_filter(value: object) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, Iterable):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            items = []
        return {item.lower() for item in items}

    @staticmethod
    def _extract_names(names: object) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, Sequence):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    def image_callback(self, msg: CompressedImage) -> None:
        if self.drop_if_busy and not self.infer_lock.acquire(blocking=False):
            return
        if not self.drop_if_busy:
            self.infer_lock.acquire()

        try:
            image = self._compressed_to_cv2(msg)
            if image is None:
                self.get_logger().warning("Failed to decode CompressedImage")
                return

            start = time.perf_counter()
            results = self.model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                device=self.device if self.device else None,
                half=self.half,
                verbose=False,
            )
            predict_ms = (time.perf_counter() - start) * 1000.0
            self.latency_pub.publish(Float64(data=predict_ms))

            detections, draw_items = self._result_to_detections(results[0], msg)
            self.det_pub.publish(detections)
            if self.publish_debug_image or self.publish_debug_compressed:
                drawn = self._draw_detections(image, draw_items)
                if self.debug_image_pub is not None:
                    self.debug_image_pub.publish(self._cv2_to_image_msg(drawn, msg))
                if self.debug_compressed_pub is not None:
                    compressed = self._cv2_to_compressed_msg(drawn, msg)
                    if compressed is not None:
                        self.debug_compressed_pub.publish(compressed)

            self.frame_count += 1
            now = time.monotonic()
            if now - self.last_log_time >= self.log_interval_sec:
                self.last_log_time = now
                self.get_logger().info(
                    f"frame={self.frame_count} "
                    f"detections={len(detections.detections)} "
                    f"predict={predict_ms:.1f} ms"
                )
        except Exception as exc:
            self.get_logger().error(
                f"YOLO callback failed: {type(exc).__name__}: {exc}"
            )
        finally:
            self.infer_lock.release()

    @staticmethod
    def _compressed_to_cv2(msg: CompressedImage) -> Optional[np.ndarray]:
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _result_to_detections(
        self, result: object, source: CompressedImage
    ) -> Tuple[Detection2DArray, List[DrawItem]]:
        output = Detection2DArray()
        output.header = source.header
        draw_items: List[DrawItem] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return output, draw_items

        names = self._extract_names(getattr(result, "names", self.names)) or self.names
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        for index, (box, score, class_id) in enumerate(
            zip(xyxy, confidences, class_ids)
        ):
            class_name = names.get(int(class_id), str(int(class_id)))
            if self.class_filter and class_name.lower() not in self.class_filter:
                continue

            x1, y1, x2, y2 = [float(value) for value in box]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 0.0 or height <= 0.0:
                continue

            detection = Detection2D()
            detection.header = source.header
            detection.id = str(index)
            detection.bbox.center.position.x = x1 + width * 0.5
            detection.bbox.center.position.y = y1 + height * 0.5
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = width
            detection.bbox.size_y = height

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = float(score)
            detection.results.append(hypothesis)
            output.detections.append(detection)
            draw_items.append(
                (
                    round(x1),
                    round(y1),
                    round(x2),
                    round(y2),
                    class_name,
                    float(score),
                    int(class_id),
                )
            )
        return output, draw_items

    @staticmethod
    def _color_for_class(class_id: int) -> Tuple[int, int, int]:
        base = class_id * 37
        return (
            (base + 53) % 255,
            (base * 3 + 97) % 255,
            (base * 7 + 193) % 255,
        )

    def _draw_detections(
        self, image: np.ndarray, items: List[DrawItem]
    ) -> np.ndarray:
        drawn = image.copy()
        height, width = drawn.shape[:2]
        for x1, y1, x2, y2, class_name, score, class_id in items:
            start = (
                int(np.clip(x1, 0, width - 1)),
                int(np.clip(y1, 0, height - 1)),
            )
            end = (
                int(np.clip(x2, 0, width - 1)),
                int(np.clip(y2, 0, height - 1)),
            )
            color = self._color_for_class(class_id)
            cv2.rectangle(drawn, start, end, color, 2)
            cv2.putText(
                drawn,
                f"{class_name} {score:.2f}",
                (start[0], max(0, start[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return drawn

    @staticmethod
    def _cv2_to_image_msg(
        image: np.ndarray, source: CompressedImage
    ) -> Image:
        output = Image()
        output.header = source.header
        output.height = int(image.shape[0])
        output.width = int(image.shape[1])
        output.encoding = "bgr8"
        output.is_bigendian = False
        output.step = int(image.shape[1] * 3)
        output.data = image.tobytes()
        return output

    def _cv2_to_compressed_msg(
        self, image: np.ndarray, source: CompressedImage
    ) -> Optional[CompressedImage]:
        parameters = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            int(np.clip(self.jpeg_quality, 1, 100)),
        ]
        success, encoded = cv2.imencode(".jpg", image, parameters)
        if not success:
            self.get_logger().warning("Failed to encode debug CompressedImage")
            return None
        output = CompressedImage()
        output.header = source.header
        output.format = "jpeg"
        output.data = encoded.tobytes()
        return output


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = CompressedYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
