# Autoware Parameter Snapshot UI Design

## Problem

Autoware launch files often combine XML launch arguments, ROS parameter YAML files,
model-package YAML files, runtime remaps, and node defaults. The value that matters
during a field test is the value actually loaded by the running node. A startup
snapshot gives that ground truth, but today comparing it with another run or with
the launch/config source is manual.

## Existing tools

- `ros2 param dump` can export parameters from a running node, but it is a CLI
  export primitive rather than a review UI.
- `rqt_reconfigure` and similar ROS GUI tools focus on inspecting or changing
  live parameters.
- Foxglove Studio focuses on telemetry visualization and ROS data workflows, not
  launch/config provenance diffing.
- General YAML diff tools such as `dyff` are useful for structured YAML, but they
  do not understand ROS node snapshots, flattened ROS parameters, or launch-time
  substitution context.

The useful gap is an Autoware-oriented review surface: normalize snapshots, keep
run history, compare actual startup values, and explain where each expected value
came from.

## MVP Scope

The first implementation in this package adds:

- a reusable Python comparator for RTUI-style node snapshots and ROS parameter
  YAML files;
- a browser UI served locally with no npm dependencies;
- normalization of quoted snapshot values such as `"False"`, `"0.2"`, and
  `"[8, 6, 10, 7, 9]"`;
- changed, added, and removed parameter tables with filtering.

Run it from the workspace:

```bash
python3 -m autoware_ml_model_launchers.param_snapshot.ui_server --port 8765
```

Then open `http://127.0.0.1:8765/` and select two YAML files.

The same core comparison is available from CLI:

```bash
python3 -m autoware_ml_model_launchers.param_snapshot.compare left.yaml right.yaml
```

## Target Design

### Capture

Capture should happen once the launch tree is up and lifecycle nodes have reached
the intended state. The capture job should record:

- node name and namespace;
- parameters and raw values;
- publishers, subscribers, services, and QoS when available;
- launch command, selected launch arguments, git revision, machine, ROS distro,
  Autoware branch, and timestamp.

The capture source can start with RTUI exports or `ros2 param dump`, then move to
an optional ROS 2 node that subscribes to parameter events and snapshots selected
nodes after startup stabilization.

### Normalize

Every input becomes a canonical document:

```yaml
schema_version: 1
run:
  id: 20260629_171413
  command: ros2 launch ...
nodes:
  /perception/secondary/stream_petr:
    parameters:
      is_distorted_image:
        value: true
        source: launch:streampetr_x2.launch.xml
      camera_mask.camera_ids:
        value: [0, 1, 2, 3, 4]
        source: model_package:param.yaml
```

### Compare

The UI should support:

- snapshot vs snapshot;
- snapshot vs resolved expected config;
- node A vs node B within the same run;
- allow-list and deny-list rules for volatile parameters such as `use_sim_time`;
- semantic grouping by prefix, for example `model_params`, `camera_mask`, and
  `qos_overrides`;
- export of a review report for PRs or field-test records.

### Resolve Launch/Config

Launch resolution is the hard part. A robust resolver should use ROS 2 launch APIs
instead of regex parsing:

1. parse the selected launch file with provided launch arguments;
2. collect `Node` actions and their parameter sources;
3. resolve substitutions against a `LaunchContext`;
4. load every referenced YAML file;
5. apply inline `<param name="..." value="...">` overrides in launch order;
6. emit expected parameters per node with source provenance.

The MVP intentionally avoids pretending this is complete. It compares snapshots
and parameter YAML accurately, and leaves launch provenance as the next feature.

## StreamPETR Example Findings

Comparing the two local StreamPETR snapshots showed:

- `/perception/stream_petr` had 80 parameters;
- `/perception/secondary/stream_petr` had 76 parameters;
- 19 common parameters changed;
- the important differences were `camera_mask.camera_ids`, camera 9/10 masks,
  `is_distorted_image`, `max_camera_time_diff`, and `use_sim_time`.

For `streampetr_x2.launch.xml`, logical camera inputs `camera0..4` are remapped
to physical cameras `camera8,6,10,7,9`, so `camera_mask.camera_ids: [0,1,2,3,4]`
can be correct even when subscribers are physical camera topics.

