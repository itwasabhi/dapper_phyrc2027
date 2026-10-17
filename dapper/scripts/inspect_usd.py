"""Prints the prims (type, applied schemas) under a path of a USD file.

Run: dapper/run_sim.sh python /project/dapper/scripts/inspect_usd.py FILE.usd /World/StretchFloor
"""

import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd  # pylint: disable=wrong-import-position

stage = Usd.Stage.Open(sys.argv[1])
for prim in Usd.PrimRange(stage.GetPrimAtPath(sys.argv[2])):
  print(
    "USD",
    prim.GetPath(),
    prim.GetTypeName(),
    list(prim.GetAppliedSchemas()),
    flush=True,
  )
app.close()
