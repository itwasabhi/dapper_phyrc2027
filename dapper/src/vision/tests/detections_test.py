import pytest

from vision import detections


def test_default_prompts_have_no_duplicates():
  mapping = detections.prompt_to_class(detections.DEFAULT_PROMPTS)
  assert mapping["white box"] == "box"
  assert mapping["mannequin"] == "manikin"


def test_prompt_to_class_rejects_shared_prompt():
  with pytest.raises(ValueError):
    detections.prompt_to_class({"a": ("x",), "b": ("x",)})


def test_to_jsonable_names_cameras():
  detection = detections.Detection(3, "box", "white box", 0.5, (0.0, 1.0, 2.0, 3.0))
  [record] = detections.to_jsonable([[], [], [], [detection], []])
  assert record["camera_name"] == "robot_0_head"
  assert record["class_name"] == "box"
