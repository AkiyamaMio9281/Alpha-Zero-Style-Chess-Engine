import numpy as np
import pytest

torch = pytest.importorskip("torch")     # trainer imports torch at module level

from trainer import SelfPlayDataset

ACTION_SIZE = 4672


@pytest.fixture
def shards(tmp_path):
    """20 shards, each one 'game' whose samples are tagged with its shard id so
    a batch can be traced back to the shards it came from."""
    rng = np.random.default_rng(0)
    for shard in range(20):
        n = 12
        feats = np.zeros((n, 102, 8, 8), np.float32)
        feats[:, 0, 0, 0] = shard                 # tag
        pi = rng.random((n, ACTION_SIZE)).astype(np.float32)
        pi /= pi.sum(1, keepdims=True)
        z = rng.choice([-1.0, 0.0, 1.0], n).astype(np.float32)
        np.savez_compressed(tmp_path / f"s{shard}.npz", feats=feats, pi=pi, z=z, result="1-0")
    return str(tmp_path)


def _shards_touched(dataset, batch_size, files_per_batch):
    seen = []
    original = dataset._load_file
    dataset._load_file = lambda path: (seen.append(path), original(path))[1]
    x, _, _ = dataset.sample(batch_size, files_per_batch=files_per_batch)
    dataset._load_file = original
    return x, seen


def test_batch_shapes_are_unchanged(shards):
    ds = SelfPlayDataset(shards, in_planes=102)
    x, pi, z = ds.sample(64, files_per_batch=8)
    assert x.shape == (64, 102, 8, 8)
    assert pi.shape == (64, ACTION_SIZE)
    assert z.shape == (64,)


@pytest.mark.parametrize("batch_size, files_per_batch", [(64, 8), (64, 16), (10, 3), (7, 16), (1, 16)])
def test_batch_is_always_exactly_batch_size(shards, batch_size, files_per_batch):
    """The remainder has to go somewhere: 64 over 16 shards divides evenly, 7
    over 16 does not, and a short batch silently changes the effective
    learning rate."""
    ds = SelfPlayDataset(shards, in_planes=102)
    x, _, _ = ds.sample(batch_size, files_per_batch=files_per_batch)
    assert len(x) == batch_size


def test_fewer_shard_reads_than_samples(shards):
    """The whole point: a batch of 64 used to decompress up to 64 shards to
    keep one position out of each."""
    ds = SelfPlayDataset(shards, in_planes=102)
    _, seen = _shards_touched(ds, batch_size=64, files_per_batch=8)
    assert len(seen) == 8


def test_files_per_batch_equal_to_batch_size_restores_old_behaviour(shards):
    # 16 of the fixture's 20 shards, so the clamp to len(files) is not what is
    # being measured here -- one read per sample is.
    ds = SelfPlayDataset(shards, in_planes=102)
    _, seen = _shards_touched(ds, batch_size=16, files_per_batch=16)
    assert len(seen) == 16


def test_a_batch_still_spans_several_games(shards):
    """Grouping trades diversity for reads. It has to stay a trade, not a
    collapse onto one game."""
    ds = SelfPlayDataset(shards, in_planes=102)
    x, _, _ = ds.sample(64, files_per_batch=8)
    tags = {float(v) for v in x[:, 0, 0, 0]}
    assert len(tags) > 1


def test_files_per_batch_is_clamped_to_what_exists(shards):
    """Asking for more shards than the batch has samples, or than the dataset
    has files, must not produce empty groups or a short batch."""
    ds = SelfPlayDataset(shards, in_planes=102)
    x, seen = _shards_touched(ds, batch_size=4, files_per_batch=999)
    assert len(x) == 4
    assert len(seen) == 4
