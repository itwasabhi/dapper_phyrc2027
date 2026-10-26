"""scan_room: raises the wrist cameras, spins once and integrates a scene estimate.

Set DAPPER_DEBUG_DIR (run_solution.py --debug does) so a solution can pass a debug_dir here to
write annotated frames and scene_estimate.json.
"""

import json
import math
import pathlib
from typing import Callable

import numpy as np

from behaviors import base
from behaviors import motion
from vision import debug_writer
from vision import scene as scene_lib
from vision import scene_integrator

BASE_YAW_COLUMN = 2
LIFT_COLUMN = 3
WRIST_PITCH_COLUMN = 6
YAW_COMMAND = 0.2
LIFT_COMMAND = 1.0
# Sign needed to tilt the wrist camera toward the table tops.
WRIST_PITCH_COMMAND = -1.0
UPDATE_EVERY_N_STEPS = 4
RAISE_STEPS = 40
SPIN_MARGIN = 1.1


def get_spin_steps(cfg: base.BehaviorCfg) -> int:
  return math.ceil(
    SPIN_MARGIN * 2 * math.pi / (YAW_COMMAND * motion.BASE_ANGULAR_RATE) * cfg.policy_hz
  )


def get_total_steps(cfg: base.BehaviorCfg) -> int:
  return RAISE_STEPS + get_spin_steps(cfg)


class ScanRoom:
  """Raises the wrist cameras, spins once, and writes the resulting SceneEstimate into state."""

  def __init__(
    self,
    integrator: scene_integrator.BasicSceneIntegrator,
    cfg: base.BehaviorCfg = base.BehaviorCfg(),
    debug_dir: pathlib.Path | None = None,
    debug_writer_factory: Callable[[pathlib.Path, float], debug_writer.DebugWriter] = (
      debug_writer.DebugWriter
    ),
  ):
    self._integrator = integrator
    self._debug_dir = debug_dir
    self._spin_steps = get_spin_steps(cfg)
    self._writer = (
      debug_writer_factory(debug_dir, cfg.policy_hz / UPDATE_EVERY_N_STEPS)
      if debug_dir
      else None
    )
    self._state: base.BehaviorState | None = None
    self._step = 0

  def reset(self, state: base.BehaviorState) -> None:
    self._state = state
    self._step = 0
    state.scene = scene_lib.SceneEstimate()
    self._integrator.reset()

  def step(self, observation) -> np.ndarray:
    scene = self._integrator.update(observation)
    self._state.scene = scene
    if self._writer and self._step % UPDATE_EVERY_N_STEPS == 0:
      self._writer.write(
        self._step, observation["rgb"], scene.debug_detections, scene.debug_image
      )
    action = np.zeros((2, 9), dtype=np.float32)
    if self._step < RAISE_STEPS:
      action[:, LIFT_COLUMN] = LIFT_COMMAND
      action[:, WRIST_PITCH_COLUMN] = WRIST_PITCH_COMMAND
    elif self._step < RAISE_STEPS + self._spin_steps:
      action[:, BASE_YAW_COLUMN] = YAW_COMMAND
    self._step += 1
    return action

  def done(self) -> bool:
    return self._step >= RAISE_STEPS + self._spin_steps

  def close(self) -> None:
    if self._writer and self._state is not None:
      self._writer.close()
      out = self._debug_dir / "scene_estimate.json"
      out.write_text(json.dumps(scene_lib.to_jsonable(self._state.scene), indent=2))
