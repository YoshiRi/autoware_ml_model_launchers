import pytest

from autoware_ml_model_launchers.open_tracker.ros_conversion import (
    detections_from_autoware,
    detections_from_detection2d_array,
    tracks_to_autoware,
    tracks_to_detection2d_array,
)
from autoware_ml_model_launchers.open_tracker.types import Track


def test_vision_msgs_round_trip_preserves_track_id():
    from vision_msgs.msg import (
        Detection2D,
        Detection2DArray,
        ObjectHypothesisWithPose,
    )

    source = Detection2DArray()
    source.header.frame_id = "camera"
    detection = Detection2D()
    detection.bbox.center.position.x = 60.0
    detection.bbox.center.position.y = 45.0
    detection.bbox.size_x = 100.0
    detection.bbox.size_y = 50.0
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = "car"
    hypothesis.hypothesis.score = 0.9
    detection.results.append(hypothesis)
    source.detections.append(detection)

    detections = detections_from_detection2d_array(source)
    assert len(detections) == 1
    assert detections[0].bbox_xyxy() == (10.0, 20.0, 110.0, 70.0)
    assert detections[0].label == "car"

    output = tracks_to_detection2d_array(
        [
            Track(
                track_id=42,
                x1=12.0,
                y1=22.0,
                x2=112.0,
                y2=72.0,
                score=0.88,
                label="car",
                class_id=1,
                source_index=0,
            )
        ],
        source.header,
    )
    assert output.header.frame_id == "camera"
    assert output.detections[0].id == "42"
    assert output.detections[0].bbox.center.position.x == 62.0


def test_autoware_round_trip_preserves_object_and_emits_uuid():
    autoware_msgs = pytest.importorskip("autoware_perception_msgs.msg")
    tier4_msgs = pytest.importorskip("tier4_perception_msgs.msg")
    ObjectClassification = autoware_msgs.ObjectClassification
    DetectedObjectWithFeature = tier4_msgs.DetectedObjectWithFeature
    DetectedObjectsWithFeature = tier4_msgs.DetectedObjectsWithFeature

    source = DetectedObjectsWithFeature()
    source.header.frame_id = "camera"
    feature_object = DetectedObjectWithFeature()
    feature_object.feature.roi.x_offset = 10
    feature_object.feature.roi.y_offset = 20
    feature_object.feature.roi.width = 100
    feature_object.feature.roi.height = 50
    feature_object.object.existence_probability = 0.9
    feature_object.object.shape.dimensions.x = 4.2
    classification = ObjectClassification()
    classification.label = ObjectClassification.CAR
    classification.probability = 0.95
    feature_object.object.classification.append(classification)
    source.feature_objects.append(feature_object)

    detections = detections_from_autoware(source)
    assert len(detections) == 1
    assert detections[0].bbox_xyxy() == (10.0, 20.0, 110.0, 70.0)
    assert detections[0].class_id == ObjectClassification.CAR

    tracks = [
        Track(
            track_id=7,
            x1=12.0,
            y1=22.0,
            x2=112.0,
            y2=72.0,
            score=0.88,
            label=str(ObjectClassification.CAR),
            class_id=ObjectClassification.CAR,
            source_index=0,
        )
    ]
    output, track_ids = tracks_to_autoware(tracks, source)

    assert output.header.frame_id == "camera"
    assert output.feature_objects[0].feature.roi.x_offset == 12
    assert output.feature_objects[0].object.shape.dimensions.x == 4.2
    assert output.feature_objects[0].object.classification[0].label == ObjectClassification.CAR
    assert len(track_ids.objects) == 1
    assert any(track_ids.objects[0].id.uuid)
