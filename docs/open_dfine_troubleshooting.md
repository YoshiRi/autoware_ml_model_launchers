# Open D-FINE GPU troubleshooting

## Tested result

The following fixes were required to run `open_detector_node` with the D-FINE backend on the
tested CUDA 13.0 environment:

1. Install `torchvision` from the same CUDA 13.0 nightly index as PyTorch.
2. Upgrade Pillow from 9.0.1 to 9.1.0 or newer.

After these changes, D-FINE GPU inference and debug-image visualization worked.

## Install the CUDA 13.0 PyTorch packages

The first failure was:

```text
AutoImageProcessor requires the Torchvision library but it was not found in your environment.
```

Install matching PyTorch packages from the CUDA 13.0 nightly index:

```bash
python3 -m pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu130
```

The versions used for the successful run were:

```text
torch 2.14.0.dev20260612+cu130
torchvision 0.29.0.dev20260613+cu130
```

These are nightly builds and their exact version numbers will change. Install `torch` and
`torchvision` together from the same index instead of mixing CUDA variants.

## Upgrade Pillow

After installing `torchvision`, importing the D-FINE classes still failed. The top-level message
suggested that general D-FINE dependencies were missing, but the underlying error was:

```text
AttributeError: module 'PIL.Image' has no attribute 'Resampling'
```

The interpreter was loading Pillow 9.0.1 from:

```text
/usr/lib/python3/dist-packages/PIL/Image.py
```

`PIL.Image.Resampling` is available in Pillow 9.1.0 and newer. Upgrade Pillow in the active
environment:

```bash
python3 -m pip install --upgrade "pillow>=9.1.0"
```

When using the system interpreter instead of a virtual environment, the tested command was:

```bash
python3 -m pip install --user --upgrade "pillow>=9.1.0"
```

Installing with `--user` can affect other Python tools that use the system interpreter. An isolated
virtual environment with `--system-site-packages` is preferable when one is available.

## Verify the imports

Run this check in the same shell and Python environment used to launch ROS:

```bash
python3 - <<'PY'
import PIL
import torch
import torchvision
from PIL import Image
from transformers import AutoImageProcessor, DFineForObjectDetection

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PIL:", PIL.__version__, PIL.__file__)
print("has Resampling:", hasattr(Image, "Resampling"))
print("AutoImageProcessor OK")
print("DFineForObjectDetection OK")
PY
```

The important result is:

```text
has Resampling: True
AutoImageProcessor OK
DFineForObjectDetection OK
```

The repository environment checker performs the same backend-specific imports:

```bash
ros2 run autoware_ml_model_launchers check_environment \
  --pipeline open_detector \
  --backend dfine \
  --camera camera5
```

## Launch D-FINE

Build from the Autoware workspace root and launch from the environment verified above:

```bash
cd "${AUTOWARE_WS}"
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --symlink-install --packages-select autoware_ml_model_launchers
source install/setup.bash

ros2 launch autoware_ml_model_launchers open_dfine.launch.xml \
  camera_namespace:=camera5
```

The first run may download `ustc-community/dfine-small-obj2coco`.

## Successfully tested package versions

```text
torch 2.14.0.dev20260612+cu130
torchvision 0.29.0.dev20260613+cu130
transformers 5.12.0
pillow >= 9.1.0
```

`requirements-open-detector-dfine.txt` records the Python-level D-FINE dependencies. The CUDA
specific `torch` and `torchvision` wheels still need to be installed from the appropriate PyTorch
index before installing the remaining requirements.
