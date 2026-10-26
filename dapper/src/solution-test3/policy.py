"""solution-test3: scans the room, then drives the robots through synthetic goals.

Each cycle is three legs, all with goals outside the scanned obstacles: DriveIndependent to a
random goal per robot, DriveIndependent to a side-by-side pair (so the formation starts close
together), then DriveFormation to a random pair-frame goal. Each leg's outcome is appended to
<debug dir>/drive_legs.jsonl, and scan_room writes under <debug dir>/scan_room/.

Run: ./run.sh python /project/dapper/scripts/run_solution.py \
    --solution-dir /project/dapper/src/solution-test3 --steps 6000
"""

import dataclasses
import os
import pathlib

from behaviors import base
from behaviors import drive_formation
from behaviors import drive_independent
from behaviors import execution_utils
from behaviors import scan_room
from behaviors import synthetic_goals
from vision import camera_geometry
from vision import detector_factory
from vision import scene_integrator

DEBUG_DIR_ENV = "DAPPER_DEBUG_DIR"
NUM_CYCLES = 3
LEG_TIMEOUT_S = 30.0


def load_policy(checkpoint, observation_space, action_space):
  del checkpoint, observation_space, action_space

  cfg = base.BehaviorCfg()
  integrator = scene_integrator.BasicSceneIntegrator(
    detector=detector_factory.make_omdet_turbo(),
    mounts=camera_geometry.load_mounts(),
    config=scene_integrator.GridConfig(
      update_every_n_calls=scan_room.UPDATE_EVERY_N_STEPS
    ),
  )
  debug_dir = os.environ.get(DEBUG_DIR_ENV)
  scan = scan_room.ScanRoom(
    integrator,
    cfg=cfg,
    debug_dir=pathlib.Path(debug_dir) / "scan_room" if debug_dir else None,
  )
  log_path = pathlib.Path(debug_dir) / "drive_legs.jsonl" if debug_dir else None

  sampler = synthetic_goals.SyntheticGoalSampler()
  independent_config = dataclasses.replace(
    drive_independent.Config(), timeout_s=LEG_TIMEOUT_S
  )
  formation_config = dataclasses.replace(
    drive_formation.Config(), timeout_s=LEG_TIMEOUT_S
  )
  behaviors = [scan]
  for cycle in range(NUM_CYCLES):
    independent_legs = [
      ("independent", sampler.create_independent_leg()),
      ("gather", sampler.create_gather_leg()),
    ]
    for kind, leg in independent_legs:
      drive = drive_independent.DriveIndependent(
        leg.get_goals, cfg=cfg, config=independent_config
      )
      behaviors.append(
        synthetic_goals.LegRecorder(drive, leg, f"cycle{cycle}-{kind}", log_path)
      )
    leg = sampler.create_formation_leg()
    drive = drive_formation.DriveFormation(
      leg.get_goal, cfg=cfg, config=formation_config
    )
    behaviors.append(
      synthetic_goals.LegRecorder(drive, leg, f"cycle{cycle}-formation", log_path)
    )

  return execution_utils.SequencedPolicy(behaviors)
