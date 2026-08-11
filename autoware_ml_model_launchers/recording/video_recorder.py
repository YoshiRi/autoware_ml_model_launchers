#!/usr/bin/env python3
"""Subscribe to image topics and write one video file per topic."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from .spec import topic_slug
from .video_writer import PROGRESS_FLUSH_SEC, TopicVideoRecorder


EPILOG = """
The message type is taken from the ROS graph, the output frame rate is
estimated from the header stamps so that a bag played at -r 0.2 still yields a
real-time video, and every file is finalized on SIGINT or SIGTERM.

example:
  ros2 run autoware_ml_model_launchers record_video
    -t /perception/traffic_light_recognition/camera5/detection/yolox/debug/image
    -n tlr_debug -o /tmp/videos
"""

# encoding -> (channels, cv2 conversion to BGR or None)
ENCODING_INFO = {
    "bgr8": (3, None),
    "rgb8": (3, cv2.COLOR_RGB2BGR),
    "bgra8": (4, cv2.COLOR_BGRA2BGR),
    "rgba8": (4, cv2.COLOR_RGBA2BGR),
    "mono8": (1, cv2.COLOR_GRAY2BGR),
    "8uc1": (1, cv2.COLOR_GRAY2BGR),
    "8uc3": (3, None),
}


def decode_compressed(msg: CompressedImage) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)


def decode_raw(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to BGR, honoring the row padding in step."""
    info = ENCODING_INFO.get(msg.encoding.lower())
    if info is None:
        raise ValueError(f"unsupported encoding: {msg.encoding}")
    channels, conversion = info
    buffer = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    image = buffer[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if conversion is not None:
        return cv2.cvtColor(image, conversion)
    return image


class ImageTopicRecorderNode(Node):
    def __init__(
        self,
        topics: list[str],
        names: list[str],
        output_dir: Path,
        extension: str,
        encoder: str,
        crf: int,
        fps: float | None,
        wait_sec: float,
        stamp_csv: bool,
    ) -> None:
        super().__init__("image_topic_recorder")

        # Best effort matches both reliable and best effort publishers.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.recorders: list[TopicVideoRecorder] = []
        for index, topic in enumerate(topics):
            name = names[index] if index < len(names) else topic_slug(topic)
            path = output_dir / f"{name}{extension}"
            recorder = TopicVideoRecorder(
                topic=topic,
                path=path,
                encoder=encoder,
                crf=crf,
                fps=fps,
                stamp_csv=stamp_csv,
                logger=self.get_logger(),
            )
            self.recorders.append(recorder)

            compressed = self._is_compressed(topic, wait_sec)
            message_type = CompressedImage if compressed else Image
            decode = decode_compressed if compressed else decode_raw
            self.create_subscription(
                message_type,
                topic,
                self._make_callback(recorder, decode),
                qos,
                # one thread per topic, so a slow encode cannot block another topic
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
            self.get_logger().info(f"subscribed to {topic} as {message_type.__name__}")

        self.create_timer(PROGRESS_FLUSH_SEC, self._report)

    def close(self) -> None:
        for recorder in self.recorders:
            recorder.close()

    def _is_compressed(self, topic: str, wait_sec: float) -> bool:
        """Ask the graph for the publisher type; fall back to the topic name."""
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            for info in self.get_publishers_info_by_topic(topic):
                if info.topic_type.endswith("CompressedImage"):
                    return True
                if info.topic_type.endswith("Image"):
                    return False
            time.sleep(0.2)
        guessed = topic.rstrip("/").endswith("compressed")
        self.get_logger().warn(
            f"no publisher found for {topic}; assuming "
            f"{'CompressedImage' if guessed else 'Image'} from the topic name"
        )
        return guessed

    def _make_callback(self, recorder: TopicVideoRecorder, decode):
        def callback(msg):
            try:
                image = decode(msg)
            except Exception as exc:  # keep the other frames and topics going
                self.get_logger().warn(f"{recorder.topic}: decode failed: {exc}")
                image = None
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            recorder.add_frame(image, stamp)

        return callback

    def _report(self) -> None:
        for recorder in self.recorders:
            recorder.write_progress()
        status = ", ".join(f"{item.topic}: {item.frames}" for item in self.recorders)
        self.get_logger().info(f"frames recorded -> {status}")


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-t", "--topic", action="append", dest="topics", metavar="TOPIC", default=[],
        help="image topic to record (repeatable)",
    )
    parser.add_argument(
        "-n", "--name", action="append", dest="names", metavar="NAME", default=[],
        help="output file stem for the topic at the same position as -t (repeatable)",
    )
    parser.add_argument("-o", "--output-dir", default="videos", help="output directory")
    parser.add_argument("--fps", type=float, default=None, help="fixed output fps")
    parser.add_argument("--encoder", default="auto", choices=["auto", "ffmpeg", "opencv"])
    parser.add_argument("--crf", type=int, default=23, help="libx264 crf, lower is better")
    parser.add_argument("--ext", default=".mp4", help="output file extension")
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for publishers")
    parser.add_argument("--no-stamp-csv", dest="stamp_csv", action="store_false")
    parser.set_defaults(stamp_csv=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.topics:
        raise SystemExit("at least one -t/--topic is required")
    if len(args.names) > len(args.topics):
        raise SystemExit(f"got {len(args.names)} -n names for {len(args.topics)} topics")

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    rclpy.init()
    node = ImageTopicRecorderNode(
        topics=args.topics,
        names=args.names,
        output_dir=Path(args.output_dir).expanduser(),
        extension=args.ext,
        encoder=args.encoder,
        crf=args.crf,
        fps=args.fps,
        wait_sec=args.wait,
        stamp_csv=args.stamp_csv,
    )
    executor = MultiThreadedExecutor(num_threads=max(2, len(args.topics) + 1))
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()  # finalize the video files first, whatever happened
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
