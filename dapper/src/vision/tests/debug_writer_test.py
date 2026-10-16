import json

import numpy as np

from vision import debug_writer
from vision import detections


def test_writer_creates_nothing_until_first_write(tmp_path):
  out = tmp_path / "debug"
  writer = debug_writer.DebugWriter(out, video_fps=5)
  writer.close()
  assert not out.exists()


def test_writer_logs_detections_and_video(tmp_path):
  out = tmp_path / "debug"
  writer = debug_writer.DebugWriter(out, video_fps=5)
  detection = detections.Detection(0, "box", "white box", 0.5, (1.0, 2.0, 30.0, 40.0))
  writer.write(0, np.zeros((5, 256, 256, 3), np.uint8), [[detection], [], [], [], []])
  writer.close()
  [record] = [
    json.loads(line) for line in (out / "detections.jsonl").read_text().splitlines()
  ]
  assert record["step"] == 0
  assert record["detections"][0]["class_name"] == "box"
  assert (out / "annotated.mp4").is_file()
  assert (out / "step_00000.png").is_file()
