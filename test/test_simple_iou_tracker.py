from autoware_ml_model_launchers.open_tracker.backends.simple_iou import SimpleIouTrackerBackend
from autoware_ml_model_launchers.open_tracker.types import Detection


def det(x1, y1, x2, y2, label="car", score=0.9, class_id=2):
    return Detection(x1, y1, x2, y2, score=score, label=label, class_id=class_id)


def test_stable_id_for_overlapping_boxes():
    tracker = SimpleIouTrackerBackend(iou_threshold=0.2)
    tracks0 = tracker.update([det(0, 0, 100, 100)])
    tracks1 = tracker.update([det(5, 5, 105, 105)])
    assert len(tracks0) == 1
    assert len(tracks1) == 1
    assert tracks0[0].track_id == tracks1[0].track_id


def test_new_id_for_non_overlapping_boxes():
    tracker = SimpleIouTrackerBackend(iou_threshold=0.2)
    tracks0 = tracker.update([det(0, 0, 100, 100)])
    tracks1 = tracker.update([det(300, 300, 400, 400)])
    assert tracks0[0].track_id != tracks1[0].track_id


def test_classwise_matching_prevents_cross_class_reuse():
    tracker = SimpleIouTrackerBackend(iou_threshold=0.2, class_agnostic=False)
    tracks0 = tracker.update([det(0, 0, 100, 100, label="car", class_id=2)])
    tracks1 = tracker.update([det(5, 5, 105, 105, label="person", class_id=0)])
    assert tracks0[0].track_id != tracks1[0].track_id


def test_class_agnostic_can_reuse_id():
    tracker = SimpleIouTrackerBackend(iou_threshold=0.2, class_agnostic=True)
    tracks0 = tracker.update([det(0, 0, 100, 100, label="car", class_id=2)])
    tracks1 = tracker.update([det(5, 5, 105, 105, label="person", class_id=0)])
    assert tracks0[0].track_id == tracks1[0].track_id
