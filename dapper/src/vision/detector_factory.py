"""Builds open-vocabulary 2D detectors from locally downloaded Hugging Face models."""

import dataclasses
import pathlib
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from timm.models import swin_transformer
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from vision import detections as detections_lib
from vision import model_registry

DetectFn = Callable[[np.ndarray], list[list[detections_lib.Detection]]]


@dataclasses.dataclass(frozen=True)
class Detector:
  name: str
  detect: DetectFn


def _rebuild_swin_masks(model: torch.nn.Module) -> None:
  """Non-persistent timm Swin masks are left on the meta device by transformers' loader."""
  for module in model.modules():
    if isinstance(module, swin_transformer.SwinTransformerBlock) and (
      module.attn_mask is not None and module.attn_mask.is_meta
    ):
      module.register_buffer(
        "attn_mask",
        module.get_attn_mask(device=torch.device("cpu"), dtype=torch.float32),
        persistent=False,
      )


def _load(name: str, models_root: pathlib.Path, device: str):
  path = model_registry.local_path(name, models_root)
  processor = AutoProcessor.from_pretrained(path, local_files_only=True)
  model = AutoModelForZeroShotObjectDetection.from_pretrained(
    path, local_files_only=True
  )
  _rebuild_swin_masks(model)
  return processor, model.to(device).eval()


def _to_pil(rgb_batch: np.ndarray) -> list[Image.Image]:
  return [Image.fromarray(rgb) for rgb in rgb_batch]


def _make_detections(
  camera: int,
  boxes,
  scores,
  labels: Sequence[str],
  prompt_classes: Mapping[str, str],
) -> list[detections_lib.Detection]:
  detections = []
  for box, score, label in zip(boxes.tolist(), scores.tolist(), labels):
    prompt = match_prompt(label, prompt_classes)
    if prompt is None:
      continue
    detections.append(
      detections_lib.Detection(
        camera=camera,
        class_name=prompt_classes[prompt],
        prompt=prompt,
        score=score,
        box_xyxy=tuple(box),
      )
    )
  return detections


def match_prompt(label: str, prompt_classes: Mapping[str, str]) -> str | None:
  """Grounding DINO can return a sub/merged phrase; take the longest prompt it contains."""
  label = label.strip().lower()
  if label in prompt_classes:
    return label
  contained = [prompt for prompt in prompt_classes if prompt in label]
  return max(contained, key=len) if contained else None


def make_omdet_turbo(
  prompts: Mapping[str, Sequence[str]] = detections_lib.DEFAULT_PROMPTS,
  models_root: pathlib.Path = model_registry.DEFAULT_MODELS_ROOT,
  score_threshold: float = 0.3,
  nms_threshold: float = 0.3,
  device: str = "cuda",
) -> Detector:
  prompt_classes = detections_lib.prompt_to_class(prompts)
  prompt_list = list(prompt_classes)
  processor, model = _load("omdet_turbo", models_root, device)

  @torch.inference_mode()
  def detect(rgb_batch: np.ndarray):
    images = _to_pil(rgb_batch)
    inputs = processor(
      images=images, text=[prompt_list] * len(images), return_tensors="pt"
    ).to(device)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
      outputs,
      text_labels=[prompt_list] * len(images),
      target_sizes=[image.size[::-1] for image in images],
      threshold=score_threshold,
      nms_threshold=nms_threshold,
    )
    return [
      _make_detections(
        camera, result["boxes"], result["scores"], result["text_labels"], prompt_classes
      )
      for camera, result in enumerate(results)
    ]

  return Detector(name="omdet_turbo", detect=detect)


def make_grounding_dino(
  prompts: Mapping[str, Sequence[str]] = detections_lib.DEFAULT_PROMPTS,
  models_root: pathlib.Path = model_registry.DEFAULT_MODELS_ROOT,
  score_threshold: float = 0.3,
  text_threshold: float = 0.25,
  device: str = "cuda",
) -> Detector:
  prompt_classes = detections_lib.prompt_to_class(prompts)
  text = ". ".join(prompt_classes) + "."
  processor, model = _load("grounding_dino_tiny", models_root, device)

  @torch.inference_mode()
  def detect(rgb_batch: np.ndarray):
    images = _to_pil(rgb_batch)
    inputs = processor(
      images=images, text=[text] * len(images), return_tensors="pt", padding=True
    ).to(device)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
      outputs,
      inputs.input_ids,
      threshold=score_threshold,
      text_threshold=text_threshold,
      target_sizes=[image.size[::-1] for image in images],
    )
    return [
      _make_detections(
        camera, result["boxes"], result["scores"], result["text_labels"], prompt_classes
      )
      for camera, result in enumerate(results)
    ]

  return Detector(name="grounding_dino", detect=detect)


BACKENDS = {
  "omdet": make_omdet_turbo,
  "grounding_dino": make_grounding_dino,
}
