# Launcher Dashboard Design

## Goal

Provide a local web UI that starts and stops selected ROS launch files without
requiring users to memorize long `ros2 launch` commands. The first useful flow is
multi-camera YOLOX startup, but the implementation should remain a generic
launcher dashboard.

The dashboard runs locally:

```bash
ros2 run autoware_ml_model_launchers launcher_dashboard_ui
```

Then the operator opens:

```text
http://127.0.0.1:8766/
```

## Initial Scope

- Load a launcher registry from YAML.
- Render registered launch files and their editable arguments.
- Start a registered launcher as a child `ros2 launch` process.
- Stop one process or all processes.
- Show process state, command preview, and log tail.
- Provide a YOLOX multi-camera form that starts one `yolox_camera.launch.xml`
  process per selected camera.
- Provide a TLR validation form that starts only
  `tlr_detect_and_classifier.launch.xml` for a selected camera.
- Provide a model comparison form that starts multiple launcher variants under a
  shared run ID with isolated namespaces and output topics.
- Provide a separate rosbag player panel for bag playback, stop, log visibility,
  and service-backed pause/resume/rate controls when available.
- Provide a separate recorder panel that captures the debug topics of running
  launchers to mp4 and rosbag files. See
  [topic_recording_design.md](topic_recording_design.md).

Out of scope for the first implementation:

- Authentication and remote access.
- Editing arbitrary launch files from the UI.
- ROS graph visualization beyond process/log status.
- Persistent process supervision after the UI server exits.

## Architecture

```text
Browser
  |
  | HTTP JSON API
  v
launcher_dashboard.ui_server
  |
  +-- registry.py          Loads config/launcher_dashboard.yaml
  +-- process_manager.py   Builds ros2 launch commands and manages subprocesses
  +-- bag_player.py        Starts ros2 bag play and detects control services
  +-- recording/manager.py Starts recorders for the topics of running launchers
  +-- static/index.html    Single-page UI
```

The UI server is intentionally small and uses Python stdlib HTTP primitives, the
same approach as the parameter snapshot UI. This keeps the tool easy to run in a
ROS shell without adding a web framework dependency.

## Registry

Launchers are allowlisted in `config/launcher_dashboard.yaml`. The dashboard
does not parse raw `ros2 launch --show-args` output. Argument forms are generated
from this registry so the UI can group and label controls intentionally.

Each launcher declares:

- `label`: display name
- `package`: ROS package passed to `ros2 launch`
- `file`: launch file passed to `ros2 launch`
- `args`: editable launch arguments and their type/default value
- `isolation`: optional templates for rewriting namespace and output topic args
  during comparison runs

Argument metadata may include `group`. The first implementation uses it mainly
for YOLOX multi-camera controls:

- `basic`: runtime basics such as `use_decompress` and `use_sim_time`
- `addon`: optional nodes such as ByteTrack and visualizers

Example:

```yaml
launchers:
  yolox_camera:
    label: YOLOX Camera
    package: autoware_ml_model_launchers
    file: yolox_camera.launch.xml
    args:
      camera_namespace:
        type: string
        default: camera5
      use_decompress:
        type: bool
        default: true
```

The backend rejects launcher IDs and argument names not present in this registry.
It also builds subprocess commands as argument lists, never through a shell.

Comparison presets are also declared in the registry. They contain common args
and a list of variants. A variant names a launcher and supplies launcher-specific
model args. At runtime the dashboard merges common args, variant args, and
isolation args in that order.

Registry args with `default: null` are treated as optional overrides and are not
emitted in generated `ros2 launch` commands unless the UI or preset provides a
value. This is useful for derived launch args such as `*_model_path`: comparison
variants can override only `model_name` while the launch file still resolves the
full path from the shared `data_path`, model folder, and ONNX file name.

## API

- `GET /api/launchers`
  - Returns the registry.
- `GET /api/processes`
  - Returns current child process state.
- `GET /api/logs?id=<process_id>&lines=200`
  - Returns a log tail.
- `POST /api/start`
  - Body: `{"launcher_id": "...", "args": {...}}`
  - Starts one registered launcher.
- `POST /api/start_multi_yolox`
  - Body: `{"cameras": ["camera5"], "args": {...}}`
  - Starts `yolox_camera` once per camera.
- `POST /api/preview_comparison`
  - Body: `{"run_id": "...", "camera_namespace": "camera5", "auto_isolate": true,
    "common_args": {...}, "variants": [...]}`
  - Returns planned commands and isolated output topics without starting
    processes.
- `POST /api/start_comparison`
  - Starts the planned comparison variants as one process group.
- `POST /api/stop`
  - Body: `{"id": "..."}`
  - Stops one process.
- `POST /api/stop_all`
  - Stops all managed processes.
- `POST /api/stop_group`
  - Body: `{"group_id": "..."}`
  - Stops only processes in a comparison group.
- `POST /api/close`
  - Body: `{"id": "..."}`
  - Removes one exited process from the dashboard process list.
- `POST /api/close_all`
  - Removes all exited processes from the dashboard process list.
- `POST /api/close_group`
  - Removes exited processes in a comparison group.
- `GET /api/bag/status`
  - Returns current bag player state and detected control capabilities.
- `GET /api/bag/logs?lines=200`
  - Returns a bag playback log tail.
- `GET /api/bag/browse?path=<directory>`
  - Lists server-side directories and rosbag-looking entries for path selection.
- `POST /api/bag/start`
  - Body: `{"bag_path": "...", "rate": 1.0, "loop": false, "clock": false}`
  - Starts one managed `ros2 bag play` process. `bag_path` also accepts
    `@<file>` holding one bag path per line, and `bag_paths` accepts an explicit
    list; both are played in order as a playlist.
- `POST /api/bag/stop`
  - Stops the managed bag process.
- `POST /api/bag/pause`
  - Calls the detected rosbag pause service.
- `POST /api/bag/resume`
  - Calls the detected rosbag resume service.
- `POST /api/bag/set_rate`
  - Body: `{"rate": 2.0}`
  - Calls the detected rosbag set-rate service.
- `GET /api/record/status`, `GET /api/record/logs`
- `POST /api/record/preview`, `/api/record/start`, `/api/record/stop`,
  `/api/record/stop_all`, `/api/record/finalize`
  - Recorder endpoints, documented in
    [topic_recording_design.md](topic_recording_design.md).

## YOLOX Multi-Camera Behavior

The multi-camera form is a convenience layer over the generic launcher start API.
It has separate UI sections for camera selection, basic settings, and add-on node
startup settings. For each selected camera it starts:

```bash
ros2 launch autoware_ml_model_launchers yolox_camera.launch.xml \
  camera_namespace:=<camera> \
  use_decompress:=<value> \
  enable_bytetrack:=<value> \
  enable_bytetrack_visualizer:=<value>
```

The dashboard defaults `use_decompress=true` and `use_sim_time=true` to match the
launch files' rosbag-oriented defaults.

## TLR Validation Behavior

The TLR validation tab is a focused launcher preset for traffic light recognition
checks. It starts `tlr_detect_and_classifier.launch.xml` directly instead of
using `all_single_camera_detection.launch.xml`, so YOLOX object detection and
ByteTrack are not launched.

For the selected camera it starts:

```bash
ros2 launch autoware_ml_model_launchers tlr_detect_and_classifier.launch.xml \
  camera_namespace:=<camera> \
  use_decompress:=<value> \
  use_sim_time:=<value> \
  enable_classification:=<value> \
  data_path:=<value>
```

The tab exposes common validation controls plus the detector ML package name.
It also renders the launch wiring derived from the selected camera:
decompression path, detector package path, runtime param file, ML package param
file, and the output topics. More specialized topic arguments can still be
edited through the generic launcher form after selecting
`TLR Detection/Classification` in the launcher list.

## Model Comparison Behavior

The comparison tab is a generic layer over registered launchers. It does not
know YOLOX, TLR, or open detector semantics directly. Instead, each launcher
declares an `isolation` contract:

```yaml
isolation:
  arg_templates:
    node_namespace: evaluation/{run_id}/{variant_id}/{camera_namespace}
  output_args:
    output/objects: /evaluation/{run_id}/{variant_id}/{camera_namespace}/object_recognition/rois
```

The comparison tab sends a run ID, a shared camera, common args, and a JSON list
of variants. Each variant includes `id`, `label`, `launcher_id`, and
launcher-specific `args`. When `auto_isolate` is enabled, the backend rewrites
only args declared by that launcher's isolation contract. The launch files
therefore remain the source of truth for which namespaces and topics can be
isolated.

The process manager stores `group_id`, `run_id`, `variant_id`, and generated
output topics on each managed process so the UI can stop or close a whole
comparison group.

The first comparison preset is intentionally editable JSON. This keeps the
backend common across Autoware YOLOX, TLR YOLOX, and Python open detector
launchers even though their model argument names and output message types differ.
More specialized evaluator or metric panels can consume the same run metadata
later.

The default preset demonstrates the common case where models live under the same
ML model root and folder, but different variants choose different ONNX file
names. Full path args remain available as optional overrides for launchers that
need them.

## Process Model

Each started launcher receives an in-memory process ID. The process manager keeps:

- launcher ID
- command arguments
- start time
- PID
- return code
- log file path
- optional comparison group ID, run ID, variant ID, and isolated outputs

Logs are written under:

```text
/tmp/autoware_ml_model_launchers/launcher_dashboard/
```

Stopping a process first sends `terminate()`, waits briefly, then uses `kill()`
if the process does not exit.

Closed processes are removed only from the dashboard's in-memory process list.
Their log files remain on disk under the log directory. Closing a running process
is rejected; it must be stopped first. `Close All` removes only exited processes.

Log API responses are capped to a bounded tail. Even if a launch or bag process
writes a very large log file, the backend reads at most a fixed byte window from
the end of the file and returns at most 500 lines. The browser replaces the log
panel text on refresh instead of appending, so the page does not grow with every
poll.

## Rosbag Player Boundary

Rosbag playback is useful for the same operator workflow, but it should not be
implemented as another launcher entry. It has a different lifecycle and control
surface from `ros2 launch`: playback can be paused, resumed, rate-controlled,
seeked, and stopped while publishing data to the ROS graph.

This is implemented as a separate `Bag Player` panel backed by a separate
manager class. The launcher registry remains responsible only for allowlisted
launch files. The bag player manager owns:

- starting `ros2 bag play <bag_path>` as a managed child process
- stopping the bag process group
- capturing stdout and stderr into the same log area
- browsing server-side paths without uploading or reading bag contents
- exposing service-backed controls only when the running ROS 2 rosbag player
  provides the required services

The first practical increment is intentionally small:

- play a user-provided bag path
- select a bag path through a server-side path browser
- stop playback
- set the initial playback rate before start
- show the exact command and log tail

Playlist playback was added later for scenes that are stored as many short bags.
`ros2 bag play` on Humble takes a single bag, so the manager plays the list
sequentially and reports itself as running in the gap between two bags. Without
that, a recorder set to stop with playback would stop at the first boundary.

The path browser intentionally does not use an HTML file input. Browser file
dialogs do not expose a reliable full filesystem path to JavaScript, and
uploading a bag through the browser is unsuitable for 10GB+ rosbag files. The UI
therefore lists directories on the machine running the dashboard and only checks
for lightweight rosbag candidates such as directories containing `metadata.yaml`
or single-file `.mcap`/`.db3` bags. Playback still passes only the selected path
to `ros2 bag play`.

Runtime controls such as pause, resume, speed up, and speed down are available
only behind capability detection. The backend scans `ros2 service list -t` and
uses the `rosbag2_interfaces` services that are actually present in the running
ROS graph. It does not import ROS 2 rosbag service classes directly, and it does
not hardcode the player node name; this keeps the UI more tolerant of Humble and
Jazzy differences. If the required services are not available, the UI disables
those buttons and shows an explicit unsupported state instead of failing
silently.

Controls intentionally not included in the first increment:

- seek
- single-step/play-next
- burst by message count
- topic filtering
- persistent bag path presets

## Error Visibility

The dashboard redirects each child `ros2 launch` process' stdout and stderr to a
per-process log file and exposes that log in the UI. This is important when a
launch file changes, an argument is removed, or a package cannot be found: the
error normally printed by `ros2 launch` should be visible in the dashboard log
panel, not only in the terminal that started the UI server.

The UI selects the newly started process automatically and shows its log tail.
If any managed process exits with a non-zero return code, the process list marks
it as exited and the UI automatically switches the log panel to that failed
process. Errors that occur before `ros2 launch` is spawned, such as an unknown
registry argument, are returned by the API and shown in the dashboard status bar.

## Packaging

The dashboard is installed as:

```text
launcher_dashboard_ui = autoware_ml_model_launchers.launcher_dashboard.ui_server:main
```

The static HTML is packaged as package data, and the registry YAML is installed
through the existing `config/*.yaml` data file rule.

## Follow-Up Ideas

- Detect topic health with `ros2 topic info` or native `rclpy`.
- Add named presets saved to YAML.
- Add readiness checks for model files and label files.
- Add a compact camera grid for common vehicle camera sets.
- Add graceful cleanup on UI server shutdown.
