import numpy as np

from behaviors import base
from behaviors import scan_room
from vision import scene as scene_lib


class _FakeIntegrator:
  def __init__(self):
    self.reset_calls = 0

  def reset(self):
    self.reset_calls += 1

  def update(self, observation):
    return scene_lib.SceneEstimate(
      manikin=scene_lib.Pose2D(1.0, 2.0, 0.0, 0.5, 0.5, 1.0)
    )


def test_raises_then_spins_then_holds_and_updates_state():
  behavior = scan_room.ScanRoom(_FakeIntegrator())
  state = base.BehaviorState()
  behavior.reset(state)

  action = behavior.step({})
  assert action[0, scan_room.LIFT_COLUMN] == scan_room.LIFT_COMMAND
  assert action[0, scan_room.WRIST_PITCH_COLUMN] == scan_room.WRIST_PITCH_COMMAND
  assert state.scene.manikin is not None

  for _ in range(scan_room.RAISE_STEPS - 1):
    behavior.step({})
  spin_action = behavior.step({})
  assert spin_action[0, scan_room.BASE_YAW_COLUMN] == scan_room.YAW_COMMAND

  while not behavior.done():
    behavior.step({})
  assert np.array_equal(behavior.step({}), np.zeros((2, 9), np.float32))


def test_scene_estimate_is_written_when_the_scan_finishes(tmp_path):
  class _FakeWriter:
    def __init__(self, debug_dir, fps):
      pass

    def write(self, *args):
      pass

    def close(self):
      pass

  behavior = scan_room.ScanRoom(
    _FakeIntegrator(), debug_dir=tmp_path, debug_writer_factory=_FakeWriter
  )
  behavior.reset(base.BehaviorState())
  while not behavior.done():
    behavior.step({"rgb": None})
  assert (tmp_path / "scene_estimate.json").is_file()
