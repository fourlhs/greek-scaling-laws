"""Learning rate schedules.

Kept separate from train.py so the schedule can be tested without importing the
training script, which parses arguments and loads the corpus at import time.
"""

import math


def make_lr_multiplier(schedule, warmup_steps, total_steps, min_lr_frac):
    """Return an LR multiplier in [0, 1] as a function of step index.

    'constant' is a flat 1.0 and reproduces the committed results.csv.
    'cosine' is linear warmup followed by cosine decay to min_lr_frac.

    Warmup matters more here than the decay: AdamW on a transformer with no
    warmup takes its largest steps while the parameters are still noise.

    The schedule also interacts with the scaling law measurement. Under a
    constant lr a larger D buys more optimizer steps as well as more data, so
    the D axis conflates the two -- which is why the published b_D of -0.254 is
    far steeper than Kaplan's -0.095. Decaying to a floor by the end of each run
    makes runs of differing D more nearly comparable, since each finishes in the
    same part of its schedule rather than being cut off mid-descent.
    """
    if schedule == "constant":
        return lambda current_step: 1.0

    if schedule != "cosine":
        raise ValueError(f"unknown schedule: {schedule!r}")

    def multiplier(current_step):
        if current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return multiplier
