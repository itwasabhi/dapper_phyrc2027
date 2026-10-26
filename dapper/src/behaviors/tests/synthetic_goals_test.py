import json
import math

import numpy as np
import pytest

from behaviors import base
from behaviors import collision
from behaviors import synthetic_goals
from vision import scene as scene_lib

MARGIN = collision.CollisionConfig().obstacle_clearance_m
TABLE = scene_lib.Pose2D(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
SCENE = scene_lib.SceneEstimate(tables=(TABLE,))


def _base_pose(xy_0=(-2.0, 0.5), xy_1=(-2.0, -0.5)):
  return np.array(
    [[*xy_0, 0.0, 1.0, 0.0, 0.0, 0.0], [*xy_1, 0.0, 1.0, 0.0, 0.0, 0.0]], np.float32
  )


def _sample_independent(seed):
  sampler = synthetic_goals.SyntheticGoalSampler(
    synthetic_goals.SyntheticGoalConfig(seed=seed)
  )
  leg = sampler.create_independent_leg()
  return leg, leg.get_goals(SCENE, _base_pose())


def test_independent_goals_are_free_and_apart():
  _, goals = _sample_independent(seed=0)
  assert all(goal is not None for goal in goals)
  assert all(not collision.contains(TABLE, [g.x, g.y], MARGIN) for g in goals)
  assert math.hypot(goals[0].x - goals[1].x, goals[0].y - goals[1].y) >= 0.6


def test_goals_are_held_across_calls_and_deterministic_per_seed():
  leg, goals = _sample_independent(seed=1)
  assert leg.get_goals(SCENE, _base_pose((3.0, 3.0), (3.0, 2.0))) == goals
  assert _sample_independent(seed=1)[1] == goals
  assert _sample_independent(seed=2)[1] != goals


def test_formation_goal_keeps_both_slots_free():
  sampler = synthetic_goals.SyntheticGoalSampler()
  leg = sampler.create_formation_leg()
  goal = leg.get_goal(SCENE, _base_pose())
  assert goal is not None
  assert leg.get_goal(SCENE, _base_pose()) == goal


def test_recorder_logs_one_line_when_the_leg_finishes(tmp_path):
  class _Behavior:
    def __init__(self):
      self.steps = 0

    def reset(self, state):
      pass

    def step(self, observation):
      self.steps += 1
      return np.zeros((2, 9), np.float32)

    def done(self):
      return self.steps >= 2

    def blocked(self):
      return (False, False)

    def close(self):
      pass

  leg = synthetic_goals.SyntheticGoalSampler().create_independent_leg()
  leg.get_goals(SCENE, _base_pose())
  log_path = tmp_path / "legs.jsonl"
  recorder = synthetic_goals.LegRecorder(_Behavior(), leg, "leg0", log_path)
  recorder.reset(base.BehaviorState())
  observation = {"base_pose_world": _base_pose()}
  for _ in range(4):
    recorder.step(observation)
  lines = log_path.read_text().splitlines()
  assert len(lines) == 1
  record = json.loads(lines[0])
  assert record["name"] == "leg0" and record["steps"] == 2
  assert record["reached"] is False


def test_goals_prefer_paths_that_need_a_detour():
  rects = collision.obstacles(SCENE)
  sampler = synthetic_goals.SyntheticGoalSampler(
    synthetic_goals.SyntheticGoalConfig(seed=3)
  )
  for _ in range(5):
    goals = sampler.create_independent_leg().get_goals(SCENE, _base_pose())
    for robot, goal in enumerate(goals):
      assert not collision.path_is_clear(
        _base_pose()[robot, :2], np.array([goal.x, goal.y]), rects, MARGIN
      )


def test_recorder_logs_the_trajectory(tmp_path):
  class _Behavior:
    def reset(self, state):
      pass

    def step(self, observation):
      return np.zeros((2, 9), np.float32)

    def done(self):
      return True

    def blocked(self):
      return (False, False)

    def close(self):
      pass

  leg = synthetic_goals.SyntheticGoalSampler().create_independent_leg()
  leg.get_goals(SCENE, _base_pose())
  log_path = tmp_path / "legs.jsonl"
  recorder = synthetic_goals.LegRecorder(_Behavior(), leg, "leg0", log_path)
  recorder.reset(base.BehaviorState())
  recorder.step({"base_pose_world": _base_pose()})
  record = json.loads(log_path.read_text())
  assert record["trajectory"] == [[[-2.0, 0.5, 0.0], [-2.0, -0.5, 0.0]]]


def test_gather_goals_are_side_by_side_and_free():
  sampler = synthetic_goals.SyntheticGoalSampler(
    synthetic_goals.SyntheticGoalConfig(pair_spacing_m=1.0)
  )
  goals = sampler.create_gather_leg().get_goals(SCENE, _base_pose())
  assert all(goal is not None for goal in goals)
  assert math.hypot(goals[0].x - goals[1].x, goals[0].y - goals[1].y) == (
    pytest.approx(1.0)
  )
  assert goals[0].yaw == pytest.approx(goals[1].yaw)
  assert all(not collision.contains(TABLE, [g.x, g.y], MARGIN) for g in goals)
