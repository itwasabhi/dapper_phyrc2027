"""translate_star: drives both robots out and back along evenly spaced directions (an asterisk).

Open loop in each robot's body frame; for checking that base translation works in the sim.
"""

import dataclasses
import math

import numpy as np

from behaviors import base
from behaviors import motion


@dataclasses.dataclass(frozen=True)
class Config:
  command: float = 1.0  # Fraction of BASE_LINEAR_RATE, in [0, 1].
  leg_s: float = 2.0
  pause_s: float = 0.5
  num_directions: int = 8  # Spokes, evenly spaced over 360 degrees.


class TranslateStar:
  """Each spoke is an outbound leg, a pause, the reverse leg, then a pause."""

  def __init__(
    self, cfg: base.BehaviorCfg = base.BehaviorCfg(), config: Config = Config()
  ):
    self._config = config
    self._leg_steps = round(config.leg_s * cfg.policy_hz)
    self._pause_steps = round(config.pause_s * cfg.policy_hz)
    self._spoke_steps = 2 * (self._leg_steps + self._pause_steps)
    self._total_steps = config.num_directions * self._spoke_steps
    self._step = 0

  def reset(self, state: base.BehaviorState) -> None:
    del state
    self._step = 0

  def step(self, observation) -> np.ndarray:
    del observation
    action = np.zeros((2, 9), dtype=np.float32)
    if self._step < self._total_steps:
      spoke, spoke_step = divmod(self._step, self._spoke_steps)
      phase, phase_step = divmod(spoke_step, self._leg_steps + self._pause_steps)
      if phase_step < self._leg_steps:
        angle = 2 * math.pi * spoke / self._config.num_directions
        sign = 1.0 if phase == 0 else -1.0
        action[:, motion.FORWARD_COLUMN] = sign * self._config.command * math.cos(angle)
        action[:, motion.LEFT_COLUMN] = sign * self._config.command * math.sin(angle)
    self._step += 1
    return action

  def done(self) -> bool:
    return self._step >= self._total_steps

  def close(self) -> None:
    pass
