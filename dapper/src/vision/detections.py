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
  "box": ("white box",),
  # Best recall in eval_detector; its false positives (white boxes, robot bodies) carry no shirt
  # color, so the scene integrator's color gate drops them.
  "shirt": ("blue cloth",),
  "manikin": ("mannequin", "person"),
  "chair": ("stool", "chair"),
  "robot": ("robot",),
}

# Per-class score thresholds; classes not listed use the detector's score_threshold.
DEFAULT_CLASS_THRESHOLDS: Mapping[str, float] = {}


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
    if isinstance(class_prompts, str):
      raise TypeError(
        f"Prompts for {class_name!r} must be a sequence of strings, not a str"
      )
    for prompt in class_prompts:
      if prompt in mapping:
        raise ValueError(f"Prompt {prompt!r} assigned to multiple classes")
      mapping[prompt] = class_name
  return mapping


def parse_prompt_specs(specs: Sequence[str]) -> dict[str, tuple[str, ...]]:
  """Parses "class=prompt1|prompt2" strings; classes not given keep their DEFAULT_PROMPTS."""
  prompts = dict(DEFAULT_PROMPTS)
  for spec in specs:
    class_name, separator, joined = spec.partition("=")
    if not separator or not joined:
      raise ValueError(f"Expected class=prompt1|prompt2, got {spec!r}")
    prompts[class_name] = tuple(joined.split("|"))
  return prompts


def to_jsonable(
  detections_per_camera: Sequence[Sequence[Detection]],
) -> list[dict]:
  return [
    {**dataclasses.asdict(detection), "camera_name": CAMERA_NAMES[detection.camera]}
    for camera_detections in detections_per_camera
    for detection in camera_detections
  ]
