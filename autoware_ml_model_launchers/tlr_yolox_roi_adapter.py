#!/usr/bin/env python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile

from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from tier4_perception_msgs.msg import TrafficLightRoi
from tier4_perception_msgs.msg import TrafficLightRoiArray


class TlrYoloxRoiAdapter(Node):

    def __init__(self):
        super().__init__("tlr_yolox_roi_adapter")

        self.car_label_id = self.declare_parameter("car_label_id", 1).value
        self.pedestrian_label_id = self.declare_parameter("pedestrian_label_id", 2).value
        self.traffic_light_id_offset = self.declare_parameter(
            "traffic_light_id_offset", 100000
        ).value
        self.min_roi_width = self.declare_parameter("min_roi_width", 1).value
        self.min_roi_height = self.declare_parameter("min_roi_height", 1).value

        qos = QoSProfile(depth=1)
        self.pub = self.create_publisher(TrafficLightRoiArray, "~/output/rois", qos)
        self.sub = self.create_subscription(
            DetectedObjectsWithFeature, "~/input/objects", self.on_objects, qos
        )

    def on_objects(self, msg: DetectedObjectsWithFeature):
        out = TrafficLightRoiArray()
        out.header = msg.header

        mapped_index = 0
        for obj in msg.feature_objects:
            roi = obj.feature.roi
            if roi.width < self.min_roi_width or roi.height < self.min_roi_height:
                continue
            if not obj.object.classification:
                continue

            label = int(obj.object.classification[0].label)
            if label == self.car_label_id:
                traffic_light_type = TrafficLightRoi.CAR_TRAFFIC_LIGHT
            elif label == self.pedestrian_label_id:
                traffic_light_type = TrafficLightRoi.PEDESTRIAN_TRAFFIC_LIGHT
            else:
                continue

            tl_roi = TrafficLightRoi()
            tl_roi.roi = roi
            tl_roi.traffic_light_type = traffic_light_type
            tl_roi.traffic_light_id = int(self.traffic_light_id_offset + mapped_index)
            out.rois.append(tl_roi)
            mapped_index += 1

        self.pub.publish(out)


def main():
    rclpy.init()
    node = TlrYoloxRoiAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
