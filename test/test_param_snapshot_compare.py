from autoware_ml_model_launchers.param_snapshot.compare import compare_texts


def test_rtui_snapshot_values_are_normalized_before_compare():
    left = """
entity_type: Node
name: /perception/stream_petr
info:
  publishers:
  - !!python/tuple
    - /parameter_events
    - rcl_interfaces/msg/ParameterEvent
parameters:
  use_sim_time: 'False'
  max_camera_time_diff: '0.15'
  camera_mask.camera_ids: '[8, 6, 10, 7, 9]'
"""
    right = """
entity_type: Node
name: /perception/secondary/stream_petr
parameters:
  use_sim_time: 'True'
  max_camera_time_diff: '0.2'
  camera_mask.camera_ids: '[0, 1, 2, 3, 4]'
"""

    result = compare_texts(left, right)

    assert result["summary"] == {
        "left_params": 3,
        "right_params": 3,
        "same": 0,
        "changed": 3,
        "added": 0,
        "removed": 0,
    }
    changed = {item["key"]: item for item in result["changed"]}
    assert changed["use_sim_time"]["left"] is False
    assert changed["use_sim_time"]["right"] is True
    assert changed["max_camera_time_diff"]["left"] == 0.15
    assert changed["camera_mask.camera_ids"]["right"] == [0, 1, 2, 3, 4]


def test_ros_parameter_yaml_is_flattened():
    left = """
/**:
  ros__parameters:
    debug_mode: true
    model_params:
      trt_precision: fp16
      workspace_size: 32
"""
    right = """
/**:
  ros__parameters:
    debug_mode: true
    model_params:
      trt_precision: int8
      workspace_size: 32
"""

    result = compare_texts(left, right)

    assert result["summary"]["changed"] == 1
    assert result["changed"][0]["key"] == "/**:model_params.trt_precision"
