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
