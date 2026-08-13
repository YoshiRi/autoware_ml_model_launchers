# Screen Camera Design

## Goal

Feed a desktop screen region into the ML launchers as if it were a camera, so a
model can be pointed at anything that can be displayed: a video file, a web
player, a simulator window, or a Lichtblick panel replaying someone else's bag.

The node publishes `sensor_msgs/CompressedImage` on the same topic a real camera
or a rosbag would use, so nothing downstream needs to know the images came from
a screen:

```bash
ros2 launch autoware_ml_model_launchers screen_camera.launch.xml \
  camera_namespace:=screen monitor:=1 fps:=10

ros2 launch autoware_ml_model_launchers tlr_detect_and_classifier.launch.xml \
  camera_namespace:=screen use_decompress:=true use_sim_time:=false
```

## Why Compressed Output

The camera launchers already expect a compressed feed: their decompressor
subscribes to `<input/image>/compressed` and republishes `<input/image>`, which
is how rosbag playback drives them today. Publishing JPEG therefore drops into
the existing `use_decompress:=true` path with no new wiring, and it keeps the
message shape identical to the recorded bags the same models are validated
against.

Raw `sensor_msgs/Image` output is deliberately not offered. A second output
would be a second thing to keep consistent, and the decompressor already exists.

## Capture Backends

Both backends produce BGR frames of the same region, so the node body does not
know which one it has.

- **ffmpeg x11grab**, the default when nothing else is installed. Frames are read
  as raw `bgr24` from an ffmpeg pipe. This needs no Python package, and ffmpeg is
  already an optional dependency of the topic recorder.
- **mss**, used automatically when it is importable (`requirements-screen-camera.txt`).
  Grabs are roughly an order of magnitude cheaper than starting an encoder
  pipeline, which matters above ~15 fps or on large regions.

The two differ in pacing, which the node has to respect: ffmpeg emits frames at
`-framerate` and so paces itself, while `mss.grab` returns immediately. Each
capture object exposes `paced`, and the publish loop sleeps only when it is
false. Pacing in both places would halve the effective rate.

## Region Selection

`monitor` and `region` resolve to one rectangle before capture starts:

- `region: "1920x1080+2560+0"` wins when set, in X11 geometry order.
- `monitor: N` selects the Nth monitor, 1-based.
- `monitor: 0` captures the union of all monitors.

Monitors are enumerated with `xrandr --listmonitors` rather than through mss, so
that both backends see the same layout and the same indices. mss enumeration is
only a fallback for systems without xrandr.

This matters on multi-head setups, where the X display is one wide virtual
screen: on the development machine `monitor: 0` is `6400x2160`, while
`monitor: 1` is the built-in `2560x1600` panel. Capturing the union and feeding
it to a 960x960 detector wastes most of the pixels.

## Frame Rate, Size, and Time

`fps` is the publish rate, not a screen refresh rate; a slow encode or a busy
compositor lowers it. `resize_width`/`resize_height` scale the frame before JPEG
encoding, and giving only one of them keeps the aspect ratio. Downscaling before
encoding is the cheapest way to cut bandwidth, because the JPEG is then smaller
as well.

`use_sim_time` defaults to **false** here, unlike every other launch file in this
package. Those default to true because they are driven by rosbag playback; a
screen is a live source with no `/clock`, and a node waiting for simulated time
would publish nothing. The registry entry carries the same default, and the ML
launcher started alongside it must be given `use_sim_time:=false` too.

Timestamps come from the node clock at publish time. There is no capture
timestamp to recover: X11 hands over the current framebuffer contents, not the
moment they were drawn.

## Limitations

- **X11 only.** Both backends talk to an X display. Under a Wayland session
  neither works, and capture would have to go through the PipeWire portal, which
  requires an interactive permission dialog. `XDG_SESSION_TYPE` tells which one
  is in use.
- **No `camera_info`.** The camera launchers in this package do not subscribe to
  it; only `streampetr_x2` does, and a screen has no meaningful intrinsics.
  Publishing a fabricated pinhole model would invite it to be trusted.
- **No cursor by default.** `show_cursor:=true` draws it, but only the ffmpeg
  backend can; mss never captures the pointer.
- The captured region is whatever is on screen, including overlapping windows.
  There is no per-window capture.

## Follow-Up Ideas

- A PipeWire backend for Wayland sessions.
- Publish `camera_info` when a calibration file is supplied explicitly.
- Reuse the topic recorder to capture the pseudo camera feed alongside the model
  outputs, which already works today by naming the topic explicitly.
