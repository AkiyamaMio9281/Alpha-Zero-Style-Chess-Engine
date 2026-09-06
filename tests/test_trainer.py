import pytest

# trainer imports torch (and model.py) at module level; skip cleanly without it.
pytest.importorskip("torch")

from trainer import _parse_milestones, lr_at_step

BASE = 0.2
FLOOR = 1e-5
MS = [20000, 40000]


def test_lr_is_flat_before_the_first_milestone():
    assert lr_at_step(BASE, 0, MS, FLOOR) == pytest.approx(0.2)
    assert lr_at_step(BASE, 19999, MS, FLOOR) == pytest.approx(0.2)


def test_lr_drops_10x_at_each_milestone():
    assert lr_at_step(BASE, 20000, MS, FLOOR) == pytest.approx(0.02)
    assert lr_at_step(BASE, 39999, MS, FLOOR) == pytest.approx(0.02)
    assert lr_at_step(BASE, 40000, MS, FLOOR) == pytest.approx(0.002)


def test_lr_depends_only_on_total_steps_not_on_how_often_training_resumed():
    """The old schedule keyed off the resumed checkpoint's epoch counter, so the
    same amount of training yielded an ever-smaller lr purely as a function of
    how many times it had been resumed. Replays autoloopexpert.py's pattern --
    one resume per iteration, 2 epochs of 2000 steps each -- and checks the lr
    only ever tracks the step count."""
    step = 0
    seen = []
    for _iteration in range(10):          # 10 resumes
        for _epoch in range(2):
            seen.append((step, lr_at_step(BASE, step, MS, FLOOR)))
            step += 2000

    for s, lr in seen:
        assert lr == pytest.approx(lr_at_step(BASE, s, MS, FLOOR))

    # 20k steps in, the schedule has taken exactly one drop and stayed there.
    # The old code would have been at 0.2 / 10**19 by this same point.
    assert seen[-1][0] == 38000
    assert seen[-1][1] == pytest.approx(0.02)


def test_lr_never_collapses_to_zero():
    """The actual regression: repeated --resume used to drive lr to 2e-20 and
    below while the logs still looked healthy."""
    aggressive = list(range(1000, 100000, 1000))  # 99 milestones
    lr = lr_at_step(BASE, 10**9, aggressive, FLOOR)
    assert lr == pytest.approx(FLOOR)
    assert lr > 0


def test_empty_milestones_means_constant_lr():
    assert lr_at_step(BASE, 10**9, [], FLOOR) == pytest.approx(BASE)


def test_parse_milestones_handles_commas_whitespace_and_ordering():
    assert _parse_milestones("20000,40000") == [20000, 40000]
    assert _parse_milestones("40000, 20000") == [20000, 40000]   # sorted
    assert _parse_milestones(" 100   50 ") == [50, 100]
    assert _parse_milestones("") == []
