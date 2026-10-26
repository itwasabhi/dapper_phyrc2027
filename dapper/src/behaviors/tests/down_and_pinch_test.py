import numpy as np

from behaviors import arm
from behaviors import base
from behaviors import down_and_pinch
from behaviors import motion
from vision import scene as scene_lib

POLICY_HZ = 20
SURFACE_Z = 0.6
# Physical fingertips reach this far below the gripper center when tilted.
FINGER_DROP_M = 0.04
FINGER_FREE_M = 0.008


class _FakeRobots:
  """Joints track controller targets at once, except at the table.

  At the table the lift stops and the wrist bends up as the lift target is pushed lower, and the
  fingers stay jammed until the gripper is FINGER_FREE_M above the contact height.
  """

  def __init__(self, base_xy):
    self.base_xy = np.array(base_xy, dtype=float)
    self.targets = np.zeros((2, 9))
    self.lift = np.zeros(2)
    self.gripper_closed = np.zeros(2, bool)
    self.finger_distance = np.full(2, 0.189)
    # Distance the fingers stop at when closed: cloth between them, or nothing.
    self.closed_distance = np.array([0.06, 0.044])
    self.close_steps = np.zeros(2, int)

  def observe(self):
    joints = np.zeros((2, 13))
    grasp = np.zeros((2, 7))
    for robot in range(2):
      joints[robot, arm.LIFT_JOINT] = self.lift[robot]
      joints[robot, 4] = self.targets[robot, arm.ARM_TARGET]
      pressing = max(
        self._lowest_lift(robot) - self.targets[robot, arm.LIFT_TARGET], 0.0
      )
      joints[robot, arm.WRIST_PITCH_JOINT] = (
        self.targets[robot, arm.WRIST_PITCH_TARGET] - 2.0 * pressing
      )
      x, z = arm.get_gripper_center(
        self.lift[robot],
        self.targets[robot, arm.ARM_TARGET],
        self.targets[robot, arm.WRIST_PITCH_TARGET],
      )
      grasp[robot, :3] = (x, arm.GRIPPER_ORIGIN_Y, z)
    base_pose = np.zeros((2, 7))
    base_pose[:, :2] = self.base_xy
    base_pose[:, 3] = 1.0
    return {
      "joint_position": joints.reshape(-1),
      "controller_target": self.targets.reshape(-1),
      "grasp_pose_base": grasp.reshape(-1),
      "base_pose_world": base_pose,
      "fingertip_origin_distance": self.finger_distance.copy(),
    }

  def apply(self, action):
    rates = {
      arm.LIFT_COLUMN: (arm.LIFT_TARGET, arm.LIFT_RATE),
      arm.ARM_COLUMN: (arm.ARM_TARGET, arm.ARM_RATE),
      arm.WRIST_PITCH_COLUMN: (arm.WRIST_PITCH_TARGET, arm.WRIST_RATE),
    }
    for robot in range(2):
      for column, (target, rate) in rates.items():
        self.targets[robot, target] += action[robot, column] * rate / POLICY_HZ
      # Base yaw is 0, so base-left is world +y.
      self.base_xy[robot, 1] += action[robot, motion.LEFT_COLUMN] * 0.616 / POLICY_HZ
      self.lift[robot] = max(
        self.targets[robot, arm.LIFT_TARGET], self._lowest_lift(robot)
      )
      if action[robot, arm.GRIPPER_COLUMN] > 0.5:
        self.gripper_closed[robot] = True
      if self.gripper_closed[robot]:
        # Like the sim, the fingers only start moving after a delay.
        self.close_steps[robot] += 1
        free = self.lift[robot] >= self._lowest_lift(robot) + FINGER_FREE_M
        if self.close_steps[robot] > 10 and free:
          self.finger_distance[robot] = max(
            self.finger_distance[robot] - 0.01, self.closed_distance[robot]
          )

  def _lowest_lift(self, robot):
    _, center_z = arm.get_gripper_center(
      0.0, 0.0, self.targets[robot, arm.WRIST_PITCH_TARGET]
    )
    return SURFACE_Z + FINGER_DROP_M - center_z


def _run(targets, robots, max_steps=800):
  behavior = down_and_pinch.DownAndPinch(lambda scene, base_pose: targets)
  behavior.reset(
    base.BehaviorState(scene=scene_lib.SceneEstimate(shirt_height_m=SURFACE_Z))
  )
  closed_at = [None, None]
  for _ in range(max_steps):
    observation = robots.observe()
    action = behavior.step(observation)
    for robot in range(2):
      if action[robot, arm.GRIPPER_COLUMN] > 0.5 and closed_at[robot] is None:
        grasp = observation["grasp_pose_base"].reshape(2, 7)[robot]
        closed_at[robot] = (robots.base_xy[robot] + grasp[:2], grasp[2])
    robots.apply(action)
    if behavior.done():
      return behavior, closed_at
  raise AssertionError(f"not done; phases {behavior.phases()}")


def test_pinches_at_the_target_after_touching_the_surface_then_raises():
  robots = _FakeRobots([[0.0, 0.0], [0.0, 3.0]])
  targets = (np.array([0.62, 0.05]), np.array([0.55, 2.9]))
  behavior, closed_at = _run(targets, robots)
  assert behavior.phases() == (down_and_pinch.Phase.DONE,) * 2
  assert robots.gripper_closed.all()
  for robot in range(2):
    xy, z = closed_at[robot]
    np.testing.assert_allclose(xy, targets[robot], atol=0.015)
    # Stopped by contact, before reaching the commanded depth.
    assert z > SURFACE_Z - 0.03
  assert (robots.lift > SURFACE_Z - 0.2).all()
  # Robot 0's fingers stopped on cloth; robot 1's closed on nothing.
  assert behavior.grasped() == (True, False)


def test_skips_a_robot_whose_target_is_out_of_reach():
  robots = _FakeRobots([[0.0, 0.0], [0.0, 3.0]])
  _run((np.array([0.6, 0.0]), np.array([3.0, 3.0])), robots)
  assert robots.gripper_closed.tolist() == [True, False]


def test_done_immediately_without_targets():
  behavior = down_and_pinch.DownAndPinch(lambda scene, base_pose: (None, None))
  behavior.reset(base.BehaviorState())
  action = behavior.step(_FakeRobots([[0.0, 0.0], [0.0, 3.0]]).observe())
  assert np.array_equal(action, np.zeros((2, 9), np.float32))
  assert behavior.done()


def test_fingers_that_never_close_do_not_count_as_a_grasp():
  robots = _FakeRobots([[0.0, 0.0], [0.0, 3.0]])
  robots.closed_distance[:] = 0.189
  behavior, _ = _run((np.array([0.62, 0.05]), np.array([0.55, 2.9])), robots)
  assert behavior.grasped() == (False, False)
