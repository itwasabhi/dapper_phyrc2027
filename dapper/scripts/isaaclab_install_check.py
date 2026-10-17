"""Checks that the dapper-lab image launches Isaac Lab and has a usable torchrl wrapper.

Run: DAPPER_IMAGE=dapper-lab dapper/run_sim.sh python /project/dapper/scripts/isaaclab_install_check.py
"""

import torch
import torchrl

print("torch", torch.__version__, "torchrl", torchrl.__version__)

from isaaclab import app  # pylint: disable=wrong-import-position

launcher = app.AppLauncher(headless=True, enable_cameras=True)

import isaaclab.envs  # pylint: disable=wrong-import-position
import isaaclab.sensors  # pylint: disable=wrong-import-position
from isaaclab import assets  # pylint: disable=wrong-import-position

print("deformable cfg:", assets.DeformableObjectCfg.__module__)
print("direct env:", isaaclab.envs.DirectRLEnv)
try:
  from torchrl.envs.libs import isaac_lab  # pylint: disable=wrong-import-position

  print("IsaacLabWrapper:", isaac_lab.IsaacLabWrapper)
except ImportError as error:
  print("NO torchrl IsaacLabWrapper:", error)
launcher.app.close()
