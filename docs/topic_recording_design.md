# Topic Recording Design

## Goal

Make "play a rosbag, launch TLR/YOLOX, record the debug topics" a supported
feature of this package instead of a set of ad-hoc scripts that live next to
each evaluation dataset.

The dashboard already covers the first two steps: it starts launch files with
isolated output topics (`config/launcher_dashboard.yaml` `isolation:`) and it
plays a rosbag (`Bag Player` panel). The missing third step is capturing what
those launches publish, with file names and metadata that downstream tooling can
consume without a human renaming files.

Two entry points are in scope:

- a `Recorder` panel in the launcher dashboard, for interactive runs
- console scripts, for unattended matrix runs and for terminals without the UI

Both drive the same manager objects, so the UI is never the only place where a
recording can be described.

## What the Existing Scripts Do

These scripts are the requirements source. They are absorbed, not wrapped.

| Script | Behavior worth keeping | Behavior to drop |
| --- | --- | --- |
| `record_image_topics.py` | multiple image topics at once, `Image`/`CompressedImage` detected from the ROS graph, fps estimated from `header.stamp`, per-topic output names, files finalized on shutdown, one callback group per topic | `mp4v` as the only codec |
| `record_tlr_debug_video.py` | `frame_timestamps.csv` mapping frame index to stamp, h264 output through ffmpeg | jpg frame dump plus `rmtree` as the encoding path |
| `auto_record.py` | wait until topics are advertised before recording, settle delays, per-process log files, stop in reverse start order, run a matrix unattended, dry run | rebuilding `ros2 launch` argument strings by hand |

Recording playback-rate-independent video matters: at `-r 0.2` the wall-clock
frame rate is not the sensor frame rate, so fps must come from `header.stamp`,
as both scripts already do.

## Recorder Boundary

Recording is a third lifecycle, next to launchers and bag playback. It is not
added to the launcher registry, for the same reason the bag player is not:

- it is not started with `ros2 launch`
- it must be stopped with `SIGINT` first so that video files and mcap files are
  finalized, while `ProcessManager` deliberately sends `SIGTERM` first
- it produces artifacts on disk that need to be tracked after the process exits
- its useful status is "frames captured so far", not "process alive"

So the module gets its own manager, mirroring `bag_player.py`:

```text
launcher_dashboard/ui_server.py
  +-- process_manager.py    ros2 launch children
  +-- bag_player.py         ros2 bag play child
  +-- recording/manager.py  recorder children      <-- new
```

## Architecture

```text
autoware_ml_model_launchers/recording/
  __init__.py
  spec.py             RecordingConfig, topic resolution, template expansion, manifest
  video_writer.py     encoder selection and per-topic writer state (no ROS imports)
  video_recorder.py   rclpy node: image topics -> mp4 (+ stamp csv, + progress json)
  bag_recorder.py     builds the ros2 bag record command
  manager.py          RecordingManager: start/stop/status/tail/manifest
  session.py          unattended orchestration (launch -> wait -> record -> play -> stop)
  cli.py              console script entry points
```

`video_writer.py` is split out from the node deliberately: the encoder choice,
the fps estimation, and the sidecar bookkeeping are the parts worth unit
testing, and importing `rclpy` to test them would tie the tests to a live ROS
installation.

Recorders run as child processes, not inside the UI server. The UI server is a
stdlib `http.server` process with no ROS node, and a crashed or slow encoder must
not take the dashboard down. This matches the existing bag player decision.

## Sinks

A recording request contains one or more sinks.

- `video`: one mp4 per image topic, written by `video_recorder.py`.
  Encoder order: ffmpeg pipe (`libx264`, `yuv420p`, `+faststart`) when `ffmpeg`
  is on `PATH`, otherwise OpenCV `VideoWriter` with `mp4v`. h264 is the default
  because `mp4v` files do not play in browsers, and the dashboard is the place
  these clips get reviewed. Frames are piped to ffmpeg's stdin; no intermediate
  jpg directory is created. ffmpeg is started in its own session so that the
  `SIGINT` that stops the recorder's process group does not reach it: it must
  finalize on stdin EOF, or the mp4 loses its moov atom and will not play.
- `bag`: `ros2 bag record -o <dir> <topics...>` for everything that is not an
  image, such as `traffic_signals`, `rois`, `objects`, and latency topics. This
  keeps non-image debug output analyzable offline instead of only viewable.

Both sinks share output directory, naming, and manifest handling.

## Topic Selection

Three ways to name topics, in increasing order of automation.

1. Explicit list in the request: `[{"topic": "...", "name": "...", "sink": "video"}]`.
2. From running processes: `{"from_group_id": "comparison:<run_id>"}`. Each
   `ManagedProcess` already carries `run_id`, `variant_id`, and `outputs`, the
   isolated topic names produced by the launcher's `isolation` contract. The
   recorder resolves the group into a topic list with the run metadata attached.
3. From a registry preset, for launchers started outside a comparison run.

For case 2 and 3 the registry says which output belongs to which sink, next to
the existing isolation block, so the launch file stays the source of truth:

```yaml
  tlr_detect_and_classifier:
    isolation:
      output_args:
        output/debug/image: /evaluation/{run_id}/{variant_id}/{camera_namespace}/tlr/debug/image
        output/traffic_signals: /evaluation/{run_id}/{variant_id}/{camera_namespace}/tlr/traffic_signals
    record:
      video_args: [output/debug/image]
      bag_args: [output/traffic_signals, output/rois, output/detected_objects]
      topic_templates:
        output/debug/image: /perception/traffic_light_recognition/{camera_namespace}/detection/yolox/debug/image
```

A topic is resolved in three steps: the process' isolated `outputs`, then an
explicit `arg:=value` in the launch command, then `topic_templates`. The third
step is what makes recording work for plain runs, such as the TLR validation
tab, where nothing is isolated and the launch file computes the topic from the
camera namespace. Anything still unresolved is reported in `skipped` rather
than recorded under a guessed name.

`topic_templates` duplicates the launch file defaults, which would rot quietly.
`test_recording.py` therefore parses `launch/*.launch.xml` and asserts that each
template equals the corresponding `<arg default=...>` with `$(var x)` rewritten
to `{x}`, so a changed launch default fails the tests instead of producing an
empty recording.

The video recorder still confirms the message type from the ROS graph at
subscribe time, as `record_image_topics.py` does today, and only falls back to
the `/compressed` name heuristic when no publisher has appeared yet.

## Output Layout and Naming

A recording belongs to a session. The session directory is templated with the
same `{}` context used by isolation, so recorded file names line up with the
topics they came from:

```yaml
recording:
  output_root: ~/Documents/launcher_recordings
  session_layout: "{session_id}"
  file_layout: "{run_id}/{variant_id}_{camera_namespace}_{arg_name}"
```

```text
~/Documents/launcher_recordings/20260810_183000/
  manifest.json
  20260810_182959/                       # run_id
    tlr_960_20260629_camera5_output_debug_image.mp4
    tlr_960_20260629_camera5_output_debug_image.mp4.stamps.csv
    tlr_960_20260629_camera5_output_debug_image.mp4.progress.json
    tlr_1280_20260703_camera5_output_debug_image.mp4
  20260810_182959_bag/                   # ros2 bag record -o
    metadata.yaml
  logs/
    <recording_id>_video.log
```

Empty path segments are dropped rather than rendered, so a plain run with no
run or variant produces `tlr_detect_and_classifier_camera5_output_debug_image.mp4`
instead of a stem full of separators. The bag directory is named after the run
and gets a numeric suffix if it already exists, because `ros2 bag record`
refuses to write into an existing directory.

`spec.py` reuses `registry._format_template`, so an unknown template key fails
loudly at request time rather than producing a file named `{run_id}`.

## Manifest

The reason to build this into the package rather than keep scripts is the
manifest. Each session directory gets `manifest.json`:

```json
{
  "schema": "launcher_recording/1",
  "session_id": "20260810_183000",
  "created_at": "2026-08-10T18:30:00+00:00",
  "bag": {"path": "/data/odaiba/....mcap", "rate": 0.2, "clock": true},
  "processes": [
    {"run_id": "20260810_182959", "variant_id": "tlr_960_20260629",
     "launcher_id": "tlr_detect_and_classifier", "command": ["ros2", "launch", "..."],
     "outputs": {"output/debug/image": "/evaluation/..."}}
  ],
  "recordings": [
    {"id": "a1b2c3", "sink": "video", "topic": "/evaluation/...",
     "file": "20260810_182959/tlr_960_20260629_camera5_output_debug_image.mp4",
     "run_id": "20260810_182959", "variant_id": "tlr_960_20260629",
     "camera_namespace": "camera5",
     "frames": 1180, "dropped": 0, "fps": 9.87,
     "first_stamp": 1778.42, "last_stamp": 1897.96,
     "started_at": "...", "stopped_at": "..."}
  ]
}
```

The video recorder writes its own numbers into `<file>.progress.json` every five
seconds, reusing the reporting timer that already exists, and finalizes it on
close. The manager merges those sidecars into the manifest when the session is
finalized, so counts survive a recorder that was killed. `make_comparison_video.sh`
and similar downstream tools then read one file instead of guessing from names.

## API

New endpoints on the existing dashboard server, in the style of `/api/bag/*`:

- `GET /api/record/status` returns active recordings and the current session.
- `GET /api/record/logs?id=<recording_id>&lines=200` returns a log tail.
- `POST /api/record/preview` resolves a request into topics, files, and exact
  commands without starting anything, mirroring `/api/preview_comparison`.
- `POST /api/record/start` starts the resolved recorders.
  Body: `{"session_id": null, "from_group_id": "comparison:<run_id>",
  "topics": [...], "sinks": {"video": {...}, "bag": {...}},
  "stop_with_bag": true, "wait_sec": 60}`
- `POST /api/record/stop` with `{"id": "..."}`, and `POST /api/record/stop_all`.
- `POST /api/record/finalize` writes `manifest.json` and returns the session
  directory.

`stop_with_bag` is the interactive version of `auto_record.py`'s main loop: a
manager thread watches the managed bag player, and when it exits, waits
`settle_sec` (default 3) so the pipeline can emit the last frames, then stops the
recorders. `RecordingManager` receives only `bag_manager.is_running` as a
callable, not the manager object, so the dependency stays one-way and the
watcher cannot accidentally drive playback. `is_running` deliberately does not
touch the ROS graph: it polls the child process, because a watcher running every
second must not shell out to `ros2 service list`.

## Shutdown Semantics

`RecordingManager.stop` escalates `SIGINT` to `SIGTERM` to `SIGKILL` on the
process group, unlike `ProcessManager.stop`, which starts at `SIGTERM`. The
video recorder installs handlers for both `SIGINT` and `SIGTERM` that shut the
executor down and run `node.close()`, so the mp4 trailer is written no matter
which signal the operator's path produced. `ros2 bag record` finalizes on
`SIGINT` the same way.

A recording that is stopped before any frame arrives still writes a manifest
entry with `frames: 0`, because a silently missing file is the failure mode that
wasted the most time in the script-based workflow.

## UI

A `Recorder` panel is added to the right column, under `Bag Player`, because its
lifecycle is tied to playback rather than to a launcher form:

- topic table with sink, source topic, and resolved output file, filled by
  `POST /api/record/preview`
- `Record from group` button on each comparison group in the `Processes` panel,
  which prefills the panel from that group's `outputs`
- start, stop, stop all, and a live `frames / dropped` column polled from
  `/api/record/status`
- session directory with a copy button, matching the copy-command buttons added
  for model comparison
- log tail, reusing the existing log panel styling

## Console Scripts

```text
record_topics  = autoware_ml_model_launchers.recording.cli:record_topics_main
record_session = autoware_ml_model_launchers.recording.cli:record_session_main
record_video   = autoware_ml_model_launchers.recording.video_recorder:main
```

`record_topics` is the direct successor of `record_image_topics.py`: foreground,
Ctrl-C to finish, same `-t`/`-n`/`-o`/`--fps` flags so existing habits and shell
history keep working, plus `-b` for bag topics and `--encoder`. It drives the
same `RecordingManager` as the dashboard, so a terminal capture also produces a
session directory and a manifest.

`record_video` is the recorder node itself. The manager spawns it as
`python -m ...recording.video_recorder`, which works the same in a source tree
and in an installed workspace; the console script exists for debugging a single
topic by hand.

`record_session` is the successor of `auto_record.py` and drives the managers
directly, with no HTTP server involved:

```yaml
defaults:
  output_root: ~/Documents/launcher_recordings
  camera_namespace: camera5
  rate: 0.2
  clock: true
  startup_timeout: 900        # TensorRT engine build on a cold model
  settle_sec: 3.0
sessions:
  - id: odaiba_tlr_phase2
    bags: "@play_order_phase2.txt"    # one bag path per line, played in order
    comparison:                        # same payload shape as /api/start_comparison
      run_id: tlr_20260810
      variants: [{id: tlr_960_20260629, launcher_id: tlr_detect_and_classifier, args: {...}}]
    record: {from_group: true}
```

The full example lives in `samples/record_session.example.yaml`. `--dry-run`,
`--only <id>`, and `--keep-going` are kept from `auto_record.py`. A dry run
resolves the whole plan, including the recorder commands and their output paths,
by feeding `preview_comparison` output into the recorder planner without
starting a process.

Playback of a bag list is implemented in `BagPlayerManager`, not in the session
runner, because the dashboard benefits from it too: typing `@/path/list.txt` in
the bag path field plays the list in order, and the panel shows `running 3/42`.
A playlist reports itself as running in the gap between two bags, which is what
keeps `stop_with_bag` from cutting the recording short at the first boundary.
One logical scene stored as many one-minute bags therefore comes out as a single
continuous video.

## Batch Runs: One Launch, Many Clips

A playlist answers "many bags, one video". The opposite need is "many bags, one
video each", and re-launching the models per bag is the expensive way to get it:
a cold TensorRT engine build costs minutes, and the models are identical across
the bags being compared.

A session therefore takes a list of `clips`, each a (bag, output name) pair:

```yaml
  - id: tlr_regression_batch
    launcher:
      launcher_id: tlr_detect_and_classifier
      args: {camera_namespace: camera5, enable_classification: false}
    clips:
      - name: scene_a_night
        bag: ~/bags/scene_a.mcap
      - name: scene_b_rain
        bag: ~/bags/scene_b.mcap
        rate: 0.5                      # per-clip playback override
      - name: scene_c_phase2
        bag: "@~/bags/scene_c_order.txt"   # a clip may itself be a playlist
```

The launch happens once, the topic wait happens once, and only the recorders and
the bag player are cycled per clip: start recorders, play, wait for playback,
settle, stop recorders, next. `_check_alive` runs before every clip, so a model
that died mid-batch fails the run instead of producing empty files for the
remaining clips. All bag paths are resolved and checked for existence before the
launch, because discovering a typo after a ten-minute engine build is the worst
possible time to discover it.

`clip` is part of the naming context. When the layout does not place `{clip}`
itself, the clip name becomes a subdirectory, so two clips can never resolve to
the same file and silently overwrite each other. Setting
`file_layout: "{clip}_{arg_name}"` puts them side by side instead. Any layout
that still maps two topics onto one file is rejected at plan time with a message
naming the key to add, rather than producing one truncated video.

The whole batch shares one session directory and one `manifest.json`: `clips`
lists what was played, `recordings` carries a `clip` field, and the two join on
the clip name.

```text
~/Documents/launcher_recordings/tlr_regression_batch/
  manifest.json
  scene_a_night/tlr_detect_and_classifier_camera5_output_debug_image.mp4
  scene_a_night_bag/
  scene_b_rain/tlr_detect_and_classifier_camera5_output_debug_image.mp4
  scene_b_rain_bag/
```

A session without `clips` is treated as a single unnamed clip, so existing
one-bag-per-session configs keep working and produce no extra directory level.

## Phasing

All three phases are implemented.

1. `recording/` core: `spec.py`, `video_writer.py`, `video_recorder.py`,
   `bag_recorder.py`, `manager.py`, and the `record_topics` console script.
   Usable from a terminal on its own, and it replaces the two recorder scripts.
2. Dashboard integration: `/api/record/*`, the `Recorder` panel, and the
   `Record Group` button on a comparison group.
3. `record_session` plus bag playlist support in `BagPlayerManager`, replacing
   `auto_record.py` for unattended matrix runs.

## Ordering on Shutdown

Three managers now own child processes, and the stop order matters: recorders
first, then launchers, then the bag player. Stopping a launcher before its
recorder would end the topic mid-file, and the recorder's own `SIGINT`-first
escalation only helps if it is the one being asked to stop. `ui_server.main` and
`session.py` both follow that order.

## Out of Scope

- Recording on a remote machine, or authentication of the dashboard.
- Re-encoding, side-by-side grid composition, or annotation. `make_comparison_video.sh`
  stays a separate downstream step, fed by `manifest.json`.
- Recording non-image topics as video overlays.
- Automatic evaluation or metric computation over recorded runs.

## Follow-Up Ideas

- Serve recorded mp4 files from the dashboard for in-browser review, which the
  h264 default already allows.
- Disk space guard: refuse to start when the output filesystem is below a
  configured threshold.
- Reuse `manifest.json` as the input contract for a later comparison viewer.
