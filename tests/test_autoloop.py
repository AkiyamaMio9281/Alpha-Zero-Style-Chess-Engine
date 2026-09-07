import sys
from pathlib import Path

import pytest

import autoloop


def _fake_run_stream(calls, ck_state):
    """Stands in for run_stream: records each command and reproduces just enough
    trainer.py / arena.py output for autoloop's parsing to work.

    The checkpoint naming mirrors trainer.py exactly -- model_ep{epoch}_step{step}.pt
    with both counters carried over from --resume -- because the filename
    collision that naming causes when --resume is missing is half of what these
    tests guard.
    """
    def run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        joined = " ".join(cmd)
        if "trainer.py" in joined:
            spe = int(cmd[cmd.index("--steps-per-epoch") + 1])
            out = cmd[cmd.index("--out") + 1]
            if "--resume" in cmd:
                epoch, step = ck_state[cmd[cmd.index("--resume") + 1]]
            else:
                epoch, step = 0, 0
            epoch, step = epoch + 1, step + spe
            path = str(Path(out) / f"model_ep{epoch}_step{step}.pt")
            ck_state[path] = (epoch, step)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).touch()
            return f"[ckpt] saved: {path}   stats: {{}}\n"
        if "arena.py" in joined:
            new = cmd[cmd.index("--new") + 1]
            old = cmd[cmd.index("--old") + 1]
            # a checkpoint evaluated against itself scores ~50%, i.e. no promotion
            win = 50.0 if new == old else 82.5
            return f"[arena] NEW vs OLD  W:0  L:0  D:0  Win%={win}%\n"
        return ""
    return run


@pytest.fixture
def loop(tmp_path, monkeypatch):
    calls, ck_state = [], {}
    monkeypatch.setattr(autoloop, "run_stream", _fake_run_stream(calls, ck_state))
    monkeypatch.setattr(autoloop, "save_state", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "autoloop.py", "--rounds", "3",
        "--data-root", str(tmp_path / "data"),
        "--ckpt-root", str(tmp_path / "ckpt"),
        "--steps-per-epoch", "2000", "--games", "4", "--eval-games", "10",
    ])
    autoloop.main()
    return calls, ck_state


def _cmds(calls, script):
    return [c for c in calls if script in " ".join(c)]


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_round_one_cold_starts(loop):
    calls, _ = loop
    assert "--resume" not in _cmds(calls, "trainer.py")[0]


def test_later_rounds_resume_from_the_promoted_checkpoint(loop):
    """Without --resume every round trained a fresh random model, so the arena
    could never promote it and the loop spun in place forever. Round 3 resuming
    from round 2's output is also the assertion that round 2 was promoted."""
    calls, _ = loop
    trainers = _cmds(calls, "trainer.py")
    assert len(trainers) == 3
    assert all("--resume" in c for c in trainers[1:])
    assert Path(_arg(trainers[1], "--resume")).name == "model_ep1_step2000.pt"
    assert Path(_arg(trainers[2], "--resume")).name == "model_ep2_step4000.pt"


def test_each_round_produces_a_distinct_checkpoint(loop):
    """trainer.py names checkpoints from epoch/step. With no --resume both
    counters reset every round, so all three rounds wrote the same filename and
    each silently overwrote the previous generation's weights."""
    _, ck_state = loop
    names = [Path(p).name for p in ck_state]
    assert sorted(names) == ["model_ep1_step2000.pt",
                             "model_ep2_step4000.pt",
                             "model_ep3_step6000.pt"]


def test_arena_never_compares_a_checkpoint_against_itself(loop):
    calls, _ = loop
    arenas = _cmds(calls, "arena.py")
    assert len(arenas) == 2
    for cmd in arenas:
        assert _arg(cmd, "--new") != _arg(cmd, "--old")
