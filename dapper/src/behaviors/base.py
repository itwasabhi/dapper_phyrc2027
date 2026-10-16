"""Shared state and global settings passed between dapper/src/behaviors/ implementations."""

import dataclasses

from vision import scene as scene_lib


@dataclasses.dataclass(frozen=True)
class BehaviorCfg:
  """Global settings shared by every behavior in a sequence."""

  policy_hz: int = 20


@dataclasses.dataclass
class BehaviorState:
  """Mutable state handed to every behavior's reset(); grows as behaviors need to share more."""

  scene: scene_lib.SceneEstimate = dataclasses.field(
    default_factory=scene_lib.SceneEstimate
  )
