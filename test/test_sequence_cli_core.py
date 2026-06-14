from pathlib import Path

from autoware_ml_model_launchers.open_tracker.tools_run_tracker_sequence import _frame_to_detections, _load_frames


def test_sample_sequence_loads():
    sample = Path(__file__).resolve().parents[1] / "samples" / "simple_sequence.json"
    frames = _load_frames(sample)
    assert len(frames) == 3
    detections = _frame_to_detections(frames[0])
    assert detections[0].label == "car"
    assert detections[0].class_id == 2
