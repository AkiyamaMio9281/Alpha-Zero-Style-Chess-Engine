"""Flags declared on the autoloop wrappers must actually reach the subprocess.

This has gone wrong three times: --channels never reached arena (so any
non-default model size crashed at evaluation), --batch-max and --batch-wait-ms
were never forwarded to selfplay at all, and --cuda-graph existed nowhere. Each
time the flag was visible in --help and did nothing. These tests run the real
main() with the subprocess runner replaced by a recorder, and assert on the
commands it built.
"""

import os
import sys
from pathlib import Path

import pytest

import autoloop
import autoloopexpert


def _cmds(calls, script):
    return [c for c in calls if script in " ".join(c)]


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


# ----- autoloop.py -----

def _fake_run_stream(calls, ck_state):
    def run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        joined = " ".join(cmd)
        if "trainer.py" in joined:
            spe = int(_arg(cmd, "--steps-per-epoch"))
            out = _arg(cmd, "--out")
            epoch, step = ck_state.get(_arg(cmd, "--resume"), (0, 0)) if "--resume" in cmd else (0, 0)
            epoch, step = epoch + 1, step + spe
            path = str(Path(out) / f"model_ep{epoch}_step{step}.pt")
            ck_state[path] = (epoch, step)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).touch()
            return f"[ckpt] saved: {path}   stats: {{}}\n"
        if "arena.py" in joined:
            return "[arena] NEW vs OLD  W:0 L:0 D:0  Win%=82.5%\n"
        return ""
    return run


@pytest.fixture
def autoloop_calls(tmp_path, monkeypatch):
    calls, ck_state = [], {}
    monkeypatch.setattr(autoloop, "run_stream", _fake_run_stream(calls, ck_state))
    monkeypatch.setattr(autoloop, "save_state", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "autoloop.py", "--rounds", "2",
        "--data-root", str(tmp_path / "data"), "--ckpt-root", str(tmp_path / "ckpt"),
        "--steps-per-epoch", "2000", "--games", "4", "--eval-games", "10",
        "--channels", "64", "--resblocks", "4", "--batch-wait-ms", "0", "--batch-max", "64",
    ])
    autoloop.main()
    return calls


def test_autoloop_forwards_batching_flags_to_selfplay(autoloop_calls):
    sp = _cmds(autoloop_calls, "selfplay.py")[0]
    assert _arg(sp, "--batch-max") == "64"
    assert _arg(sp, "--batch-wait-ms") == "0"


def test_autoloop_forwards_cuda_graph_to_selfplay_and_arena(autoloop_calls):
    # on by default: the wrappers exist to run the pipeline well, and the graph
    # is bit-identical to eager with an automatic fallback
    assert "--cuda-graph" in _cmds(autoloop_calls, "selfplay.py")[0]
    assert "--cuda-graph" in _cmds(autoloop_calls, "arena.py")[0]


def test_autoloop_arena_agrees_with_trainer_on_model_size(autoloop_calls):
    """A mismatch here is a hard shape error after the expensive stages have
    already run."""
    arena = _cmds(autoloop_calls, "arena.py")[0]
    trainer = _cmds(autoloop_calls, "trainer.py")[0]
    for flag in ("--channels", "--resblocks", "--in-planes"):
        assert _arg(arena, flag) == _arg(trainer, flag)


def test_autoloop_no_cuda_graph_is_honoured(tmp_path, monkeypatch):
    calls, ck_state = [], {}
    monkeypatch.setattr(autoloop, "run_stream", _fake_run_stream(calls, ck_state))
    monkeypatch.setattr(autoloop, "save_state", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "autoloop.py", "--rounds", "1",
        "--data-root", str(tmp_path / "d"), "--ckpt-root", str(tmp_path / "c"),
        "--steps-per-epoch", "10", "--games", "2", "--no-cuda-graph",
    ])
    autoloop.main()
    assert "--cuda-graph" not in _cmds(calls, "selfplay.py")[0]


# ----- autoloopexpert.py -----

@pytest.fixture
def expert_calls(tmp_path, monkeypatch):
    calls = []

    def run_cmd(args, cwd=None):
        args = [str(a) for a in args]
        calls.append(args)
        joined = " ".join(args)
        if "trainer.py" in joined:
            out = Path(_arg(args, "--out"))
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"model_{len(calls)}.pt"
            path.touch()
            # latest_ckpt picks by mtime, and a real iteration takes minutes so
            # they always differ. Here every checkpoint lands in the same second,
            # so stamp them explicitly or which one is "latest" is a coin flip.
            os.utime(path, (1_000_000 + len(calls), 1_000_000 + len(calls)))
            return 0, ""
        if "arena.py" in joined:
            return 0, "[arena] Win%=82.5%\n"
        return 0, ""

    monkeypatch.setattr(autoloopexpert, "run_cmd", run_cmd)
    monkeypatch.setattr(autoloopexpert, "save_state", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "autoloopexpert.py", "--stockfish", "sf.exe",
        "--ckpt-dir", str(tmp_path / "ckpt"), "--data-root", str(tmp_path / "data"),
        "--iters", "2", "--games", "4", "--steps-per-epoch", "10",
        "--curriculum", "--pure-iters", "1",
        "--eval-batch-size", "8", "--batch-wait-ms", "0", "--expert-cp-scale", "120",
        "--do-arena", "--arena-games", "2",
    ])
    autoloopexpert.main()
    return calls


def test_expert_loop_forwards_search_and_batching_flags(expert_calls):
    sp = _cmds(expert_calls, "selfplay_uci.py")[0]
    assert _arg(sp, "--eval-batch-size") == "8"
    assert _arg(sp, "--batch-wait-ms") == "0"
    # forwarded as a float, so compare numerically rather than by string
    assert float(_arg(sp, "--expert-cp-scale")) == 120.0
    assert "--cuda-graph" in sp


def test_expert_loop_forwards_cuda_graph_to_arena(expert_calls):
    arenas = _cmds(expert_calls, "arena.py")
    assert arenas, "no arena run -- --do-arena should have triggered one"
    assert all("--cuda-graph" in cmd for cmd in arenas)
