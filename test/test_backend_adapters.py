import numpy as np

from autoware_ml_model_launchers.open_detector.backends.dfine_backend import DFineBackend
from autoware_ml_model_launchers.open_detector.backends.rfdetr_backend import RFDetrBackend
from autoware_ml_model_launchers.open_detector.backends.ultralytics_backend import UltralyticsYoloBackend
from autoware_ml_model_launchers.open_detector.backend_loader import create_backend
from autoware_ml_model_launchers.open_detector.types import BackendConfig


class TensorLike:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def item(self):
        return self.value.item()

    def to(self, _device):
        return self

    def tolist(self):
        return self.value.tolist()


def test_ultralytics_adapter_converts_model_result_to_detections():
    class Boxes:
        xyxy = TensorLike([[10.0, 20.0, 30.0, 50.0]])
        conf = TensorLike([0.92])
        cls = TensorLike([2])

        def __len__(self):
            return 1

    class Result:
        boxes = Boxes()
        names = {2: "car"}

    class Model:
        names = {2: "car"}

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return [Result()]

    backend = UltralyticsYoloBackend(
        BackendConfig(backend="ultralytics", device="0", imgsz=960, max_det=10),
        autoload=False,
    )
    backend.model = Model()
    backend.names = backend.model.names
    backend.loaded = True

    detections = backend.infer(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "car"
    assert detections[0].source == "ultralytics"
    assert detections[0].class_id == 2
    assert detections[0].score == 0.92
    assert detections[0].x1 == 10.0
    assert backend.model.kwargs["device"] == "0"
    assert backend.model.kwargs["imgsz"] == 960


def test_dfine_adapter_converts_postprocessed_result_to_detections():
    class ImageFactory:
        @staticmethod
        def fromarray(array):
            return type("Image", (), {"width": array.shape[1], "height": array.shape[0]})()

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class Torch:
        @staticmethod
        def no_grad():
            return NoGrad()

        @staticmethod
        def tensor(value, device=None):
            return {"value": value, "device": device}

    class Processor:
        def __call__(self, images, return_tensors):
            self.image = images
            self.return_tensors = return_tensors
            return {"pixel_values": TensorLike([1.0])}

        def post_process_object_detection(self, outputs, target_sizes, threshold):
            self.outputs = outputs
            self.target_sizes = target_sizes
            self.threshold = threshold
            return [
                {
                    "scores": [TensorLike(0.81)],
                    "labels": [TensorLike(5)],
                    "boxes": [TensorLike([1.0, 2.0, 11.0, 22.0])],
                }
            ]

    class Model:
        def __call__(self, **inputs):
            self.inputs = inputs
            return {"ok": True}

    backend = DFineBackend(
        BackendConfig(backend="dfine", conf_thres=0.4, max_det=10), autoload=False
    )
    backend.torch = Torch()
    backend.Image = ImageFactory()
    backend.processor = Processor()
    backend.model = Model()
    backend.device = "cpu"
    backend.id2label = {5: "bus"}
    backend.loaded = True

    detections = backend.infer(np.zeros((40, 80, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "bus"
    assert detections[0].source == "dfine"
    assert detections[0].class_id == 5
    assert detections[0].score == 0.81
    assert detections[0].x2 == 11.0
    assert backend.processor.threshold == 0.4


def test_rfdetr_adapter_converts_prediction_result_to_detections():
    class ImageFactory:
        @staticmethod
        def fromarray(array):
            return {"shape": array.shape}

    class Prediction:
        xyxy = np.asarray([[5.0, 6.0, 25.0, 36.0]])
        confidence = np.asarray([0.7])
        class_id = np.asarray([3])

    class Model:
        def predict(self, image, threshold):
            self.image = image
            self.threshold = threshold
            return Prediction()

    backend = RFDetrBackend(
        BackendConfig(backend="rfdetr", conf_thres=0.33, max_det=10), autoload=False
    )
    backend.Image = ImageFactory()
    backend.model = Model()
    backend.coco_classes = {2: "truck"}
    backend.loaded = True

    detections = backend.infer(np.zeros((50, 60, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "truck"
    assert detections[0].source == "rfdetr"
    assert detections[0].class_id == 3
    assert detections[0].score == 0.7
    assert detections[0].height == 30.0
    assert backend.model.threshold == 0.33


def test_create_backend_loads_selected_backend_during_construction():
    backend = create_backend(BackendConfig(backend="fake"))

    assert backend.loaded is True


def test_create_backend_can_skip_loading_for_adapter_tests():
    backend = create_backend(BackendConfig(backend="fake"), load=False)

    assert backend.loaded is False
