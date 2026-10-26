"""solution-test2: base translation check; both robots trace an asterisk, out and back along 8 spokes."""

from behaviors import base
from behaviors import execution_utils
from behaviors import translate_star


def load_policy(checkpoint, observation_space, action_space):
  del checkpoint, observation_space, action_space
  return execution_utils.SequencedPolicy(
    [translate_star.TranslateStar(cfg=base.BehaviorCfg())]
  )
