"""Detection results and the public camera / prompt conventions."""

import dataclasses
from typing import Mapping, Sequence

# Order of the stacked runtime camera observations (docs/competition/POLICY.md).
CAMERA_NAMES = (
  "overview",
  "robot_0_wrist",
  "robot_1_wrist",
  "robot_0_head",
  "robot_1_head",
)

DEFAULT_PROMPTS: Mapping[str, Sequence[str]] = {
  "box": ("white box", "cardboard box"),
  "shirt": ("t-shirt",),
  "manikin": ("mannequin", "person"),
  "chair": ("stool", "chair"),
  "robot": ("robot",),
}


@dataclasses.dataclass(frozen=True)
class Detection:
  camera: int
  class_name: str
  prompt: str
  score: float
  box_xyxy: tuple[float, float, float, float]


def prompt_to_class(prompts: Mapping[str, Sequence[str]]) -> dict[str, str]:
  """Inverts a class -> prompts map; each prompt must belong to one class."""
  mapping = {}
  for class_name, class_prompts in prompts.items():
    for prompt in class_prompts:
      if prompt in mapping:
        raise ValueError(f"Prompt {prompt!r} assigned to multiple classes")
      mapping[prompt] = class_name
  return mapping


def to_jsonable(
  detections_per_camera: Sequence[Sequence[Detection]],
) -> list[dict]:
  return [
    {**dataclasses.asdict(detection), "camera_name": CAMERA_NAMES[detection.camera]}
    for camera_detections in detections_per_camera
    for detection in camera_detections
  ]
