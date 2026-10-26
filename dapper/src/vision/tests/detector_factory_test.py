import numpy as np
import pytest

from vision import detector_factory
from vision import model_registry

PROMPTS = {"box": ("white box",), "chair": ("stool", "chair")}


@pytest.mark.parametrize(
  "label,expected",
  [
    ("white box", "white box"),
    ("a stool chair", "stool"),
    ("dog", None),
    (" Stool ", "stool"),
  ],
)
def test_match_prompt(label, expected):
  prompt_classes = {"white box": "box", "stool": "chair", "chair": "chair"}
  assert detector_factory.match_prompt(label, prompt_classes) == expected


def test_missing_model_fails_without_downloading(tmp_path):
  with pytest.raises(FileNotFoundError, match="download_models.py"):
    detector_factory.make_omdet_turbo(PROMPTS, models_root=tmp_path, device="cpu")


@pytest.mark.skipif(
  not (model_registry.DEFAULT_MODELS_ROOT / "omdet_turbo" / "config.json").is_file(),
  reason="omdet_turbo not downloaded",
)
def test_omdet_returns_one_list_per_camera():
  detector = detector_factory.make_omdet_turbo(PROMPTS, device="cpu")
  per_camera = detector.detect(np.zeros((5, 256, 256, 3), np.uint8))
  assert len(per_camera) == 5


def test_make_detections_applies_class_thresholds():
  prompt_classes = {"white box": "box", "stool": "chair"}
  detections = detector_factory._make_detections(
    camera=1,
    boxes=np.zeros((2, 4)),
    scores=np.array([0.2, 0.2]),
    labels=["white box", "stool"],
    prompt_classes=prompt_classes,
    class_thresholds={"box": 0.15, "chair": 0.3},
  )
  assert [detection.class_name for detection in detections] == ["box"]
