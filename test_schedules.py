"""Checks on the learning rate schedule.

    python test_schedules.py
"""

import math

from schedules import make_lr_multiplier


def test_constant_is_flat():
    f = make_lr_multiplier("constant", warmup_steps=0, total_steps=1000, min_lr_frac=0.1)
    assert all(f(s) == 1.0 for s in [0, 1, 500, 999, 5000])


def test_warmup_ramps_from_near_zero_to_peak():
    f = make_lr_multiplier("cosine", warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert f(0) == 0.01                     # first step is not a full-size step
    assert f(49) == 0.5
    assert f(99) == 1.0                     # peak exactly at the end of warmup
    # monotonically increasing through warmup
    vals = [f(s) for s in range(100)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_decay_is_monotonic_and_hits_the_floor():
    f = make_lr_multiplier("cosine", warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    vals = [f(s) for s in range(100, 1001)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))
    assert math.isclose(f(1000), 0.1, abs_tol=1e-12)     # exactly min_lr_frac at the end
    assert math.isclose(f(550), 0.55, abs_tol=1e-9)      # midpoint of the cosine


def test_never_leaves_bounds_even_past_the_end():
    """The loop breaks on tokens, not steps, so it can overshoot total_steps."""
    f = make_lr_multiplier("cosine", warmup_steps=10, total_steps=100, min_lr_frac=0.1)
    for s in range(0, 500):
        assert 0.0 <= f(s) <= 1.0, f"out of bounds at step {s}: {f(s)}"
    assert math.isclose(f(400), 0.1, abs_tol=1e-12)      # clamped, not oscillating back up


def test_degenerate_short_runs_do_not_divide_by_zero():
    f = make_lr_multiplier("cosine", warmup_steps=1, total_steps=1, min_lr_frac=0.0)
    assert f(0) == 1.0
    assert 0.0 <= f(1) <= 1.0


def test_zero_floor_decays_to_zero():
    f = make_lr_multiplier("cosine", warmup_steps=10, total_steps=100, min_lr_frac=0.0)
    assert math.isclose(f(100), 0.0, abs_tol=1e-12)


def test_unknown_schedule_is_rejected():
    try:
        make_lr_multiplier("linear", 10, 100, 0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown schedule")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
