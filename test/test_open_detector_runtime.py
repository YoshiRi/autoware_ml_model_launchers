import numpy as np

from autoware_ml_model_launchers.open_detector.runtime import OpenDetectorRuntime, TimingStats
from autoware_ml_model_launchers.open_detector.types import Detection


class Backend:
    loaded = True

    def infer(self, image_bgr):
        self.shape = image_bgr.shape
        return [
            Detection(
                x1=-10.0,
                y1=2.0,
                x2=20.0,
                y2=90.0,
                score=0.9,
                label="car",
                class_id=2,
            ),
            Detection(
                x1=1.0,
                y1=1.0,
                x2=2.0,
                y2=2.0,
                score=0.8,
                label="sports ball",
                class_id=32,
            ),
        ]


def test_runtime_clips_filters_maps_and_times_detections():
    runtime = OpenDetectorRuntime(
        Backend(),
        class_filter={"car"},
        label_map={"car": "CAR"},
        max_det=10,
    )

    result = runtime.update(np.zeros((40, 80, 3), dtype=np.uint8))

    assert result.image_size_hw == (40, 80)
    assert result.infer_ms >= 0.0
    assert len(result.raw_detections) == 2
    assert len(result.detections) == 1
    assert result.detections[0].label == "CAR"
    assert result.detections[0].x1 == 0.0
    assert result.detections[0].y2 == 39.0


def test_timing_stats_handles_empty_and_non_empty_values():
    assert TimingStats.from_values([]).to_dict() == {
        "runs": 0,
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }
    assert TimingStats.from_values([3.0, 1.0, 2.0]).to_dict() == {
        "runs": 3,
        "mean": 2.0,
        "median": 2.0,
        "min": 1.0,
        "max": 3.0,
    }
