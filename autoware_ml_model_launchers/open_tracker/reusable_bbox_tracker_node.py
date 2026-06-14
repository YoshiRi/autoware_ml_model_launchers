#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32

from autoware_ml_model_launchers.open_tracker.backend_factory import (
    create_bbox_tracker_backend,
)
from autoware_ml_model_launchers.open_tracker.jsonl_logger import TrackJsonlLogger
from autoware_ml_model_launchers.open_tracker.ros_conversion import (
    detections_from_autoware,
    detections_from_detection2d_array,
    stamp_to_sec,
    tracks_to_autoware,
    tracks_to_detection2d_array,
)
from autoware_ml_model_launchers.open_tracker.runtime import ReusableBBoxTrackerRuntime


class ReusableBBoxTrackerNode(Node):
    """ROS I/O wrapper for message-agnostic 2D bounding-box tracking."""

    def __init__(self) -> None:
        super().__init__("reusable_bbox_tracker")
        self._declare_parameters()
        self.backend_name = str(self.get_parameter("tracker_backend").value)
        self.message_type = str(self.get_parameter("message_type").value).strip().lower()
        backend = create_bbox_tracker_backend(
            self.backend_name,
            iou_threshold=self.get_parameter("iou_threshold").value,
            max_missed=self.get_parameter("max_missed").value,
            min_hits=self.get_parameter("min_hits").value,
            score_threshold=self.get_parameter("score_threshold").value,
            class_agnostic=self.get_parameter("class_agnostic").value,
            emit_unmatched_tracks=self.get_parameter("emit_unmatched_tracks").value,
        )
        self.runtime = ReusableBBoxTrackerRuntime(backend)
        self.track_logger = TrackJsonlLogger(str(self.get_parameter("jsonl_path").value))

        qos_depth = int(self.get_parameter("qos_depth").value)
        self.input_topic = str(self.get_parameter("input_detections").value)
        self.output_topic = str(self.get_parameter("output_detections").value)
        self.track_ids_topic = str(self.get_parameter("output_track_ids").value)
        latency_topic = str(self.get_parameter("latency_topic").value)

        self.pub_latency = self.create_publisher(Float32, latency_topic, qos_depth)
        self.pub_track_ids = None
        if self.message_type in {"vision", "vision_msgs", "detection2d"}:
            self._configure_vision_io(qos_depth)
        elif self.message_type in {"autoware", "detected_objects_with_feature"}:
            self._configure_autoware_io(qos_depth)
        else:
            raise ValueError(
                "message_type must be one of: vision_msgs, autoware"
            )

        self.get_logger().info(
            f"reusable_bbox_tracker ready: backend={self.backend_name}, "
            f"message_type={self.message_type}, input={self.input_topic}, "
            f"output={self.output_topic}"
        )

    def _configure_vision_io(self, qos_depth: int) -> None:
        from vision_msgs.msg import Detection2DArray

        self.pub_tracks = self.create_publisher(
            Detection2DArray, self.output_topic, qos_depth
        )
        self.sub_detections = self.create_subscription(
            Detection2DArray,
            self.input_topic,
            self._on_vision_detections,
            qos_depth,
        )

    def _configure_autoware_io(self, qos_depth: int) -> None:
        from tier4_perception_msgs.msg import (
            DetectedObjectsWithFeature,
            DynamicObjectArray,
        )

        self.pub_tracks = self.create_publisher(
            DetectedObjectsWithFeature, self.output_topic, qos_depth
        )
        self.pub_track_ids = self.create_publisher(
            DynamicObjectArray, self.track_ids_topic, qos_depth
        )
        self.sub_detections = self.create_subscription(
            DetectedObjectsWithFeature,
            self.input_topic,
            self._on_autoware_detections,
            qos_depth,
        )

    def destroy_node(self) -> bool:
        self.track_logger.close()
        return super().destroy_node()

    def _declare_parameters(self) -> None:
        self.declare_parameter("message_type", "vision_msgs")
        self.declare_parameter(
            "input_detections",
            "/perception/object_recognition/detection/open_dfine/camera5/detections",
        )
        self.declare_parameter(
            "output_detections",
            "/perception/object_recognition/tracking/simple_iou/open_dfine/camera5/tracked_detections",
        )
        self.declare_parameter(
            "output_track_ids",
            "/perception/object_recognition/tracking/simple_iou/open_dfine/camera5/track_ids",
        )
        self.declare_parameter(
            "latency_topic",
            "/perception/object_recognition/tracking/simple_iou/open_dfine/camera5/latency/update_ms",
        )
        self.declare_parameter("tracker_backend", "simple_iou")
        self.declare_parameter("qos_depth", 5)
        self.declare_parameter("iou_threshold", 0.30)
        self.declare_parameter("max_missed", 5)
        self.declare_parameter("min_hits", 1)
        self.declare_parameter("score_threshold", 0.0)
        self.declare_parameter("class_agnostic", False)
        self.declare_parameter("emit_unmatched_tracks", False)
        self.declare_parameter("jsonl_path", "")

    def _on_vision_detections(self, msg) -> None:
        detections = detections_from_detection2d_array(msg)
        tracks = self._update(detections, msg.header)
        self.pub_tracks.publish(tracks_to_detection2d_array(tracks, msg.header))

    def _on_autoware_detections(self, msg) -> None:
        detections = detections_from_autoware(msg)
        tracks = self._update(detections, msg.header)
        output, track_ids = tracks_to_autoware(tracks, msg)
        self.pub_tracks.publish(output)
        self.pub_track_ids.publish(track_ids)

    def _update(self, detections, header):
        started = time.perf_counter()
        stamp_sec = stamp_to_sec(header.stamp)
        tracks = self.runtime.update(detections, stamp_sec=stamp_sec)
        latency_msg = Float32()
        latency_msg.data = float((time.perf_counter() - started) * 1000.0)
        self.pub_latency.publish(latency_msg)
        self.track_logger.write_frame(
            stamp_sec=stamp_sec,
            frame_id=header.frame_id,
            tracker_backend=self.backend_name,
            tracks=tracks,
        )
        return tracks


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ReusableBBoxTrackerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
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
