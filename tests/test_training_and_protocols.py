import copy

import numpy as np
import pytest
import torch

from pea_vr.config import DEFAULTS
from pea_vr.data.prepare import PreparedDataset, identity_split
from pea_vr.data.protocols import EpisodeSampler, validate_episode
from pea_vr.experiments import method_config
from pea_vr.models.factory import build_model
from pea_vr.models.matching import Matcher
from pea_vr.training.engine import adapted_parameters, train


def test_episode_disjoint_balanced_and_state(prepared_fixture):
    data = PreparedDataset(prepared_fixture)
    sampler = EpisodeSampler(data, seed=7)
    state = copy.deepcopy(sampler.state_dict())
    first = sampler.sample(2, 3, 3, balanced=True)
    validate_episode(data, first)
    for group in ['support', 'query']:
        for label in [0, 1]:
            modes = [data.rows[data.by_id[sid]]['mode'] for sid, y in zip(first[group], first[group + '_labels']) if y == label]
            assert sorted(modes) == ['H', 'L', 'T']
    sampler.load_state_dict(state)
    assert sampler.sample(2, 3, 3, balanced=True) == first
    with pytest.raises(ValueError, match='eligible'):
        sampler.sample(4, 2, 2)


def test_split_strict_and_identity_disjoint():
    split = identity_split(list(map(str, range(100))), [50, 20, 30], 20260814)
    assert not (set(split['train']) & set(split['test']))
    assert split == identity_split(list(map(str, range(100))), [50, 20, 30], 20260814)
    with pytest.raises(ValueError):
        identity_split(['a', 'b'], [50, 20, 30], 0)


def small_config():
    config = copy.deepcopy(DEFAULTS)
    config['model'].update(channels=[8, 12], tokens=6, global_dim=16, projection_hidden=24, alignment_dim=6)
    config['training'].update(episodes=4, ways=2, shots=2, queries=2, device='cpu', threads=2, query_chunk=2,
                              save_every=10, log_every=1)
    return config


def test_exact_training_resume(prepared_fixture, tmp_path):
    config = small_config()
    whole, resumed = tmp_path / 'whole', tmp_path / 'resumed'
    train(config, prepared_fixture, whole)
    train(config, prepared_fixture, resumed, max_steps=2)
    train(config, prepared_fixture, resumed, resume=resumed / 'latest.pt')
    a = torch.load(whole / 'latest.pt', weights_only=False)
    b = torch.load(resumed / 'latest.pt', weights_only=False)
    assert a['step'] == b['step'] == 4
    for name, value in a['model'].items():
        torch.testing.assert_close(value, b['model'][name], rtol=0, atol=0)
    assert a['sampler_rng'] == b['sampler_rng']
    assert a['view_rng'] == b['view_rng']
    torch.testing.assert_close(a['rng']['torch'], b['rng']['torch'])


@pytest.mark.parametrize('method', ['protonet', 'deepmetric', 'coda', 'cl_metaflow', 'transformer', 'global_amp', 'diagonal', 'soft_dtw', 'pea_vr'])
def test_method_backward(method, prepared_fixture):
    torch.set_num_threads(2)
    data = PreparedDataset(prepared_fixture)
    config = method_config(method)
    if config['model']['kind'] in ['pea_vr', 'global_amp']:
        config['model'].update(channels=[8, 12], tokens=6, global_dim=16, projection_hidden=24, alignment_dim=6)
    model = build_model(config)
    inputs = data.batch([0, 1, 12, 13], torch.device('cpu'))
    fp = model(inputs)
    scores, _ = Matcher(config['matching']).candidates(fp.index([1, 3]), fp.index([0, 2]), torch.tensor([0, 1]))
    scores.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if method == 'cl_metaflow':
        fast, _ = adapted_parameters(model, inputs, torch.tensor([0, 0, 1, 1]), config)
        assert any(not torch.equal(fast[k], v) for k, v in model.named_parameters())


@pytest.mark.parametrize('method', ['protonet', 'deepmetric', 'coda', 'cl_metaflow', 'transformer', 'global_amp', 'diagonal', 'soft_dtw'])
def test_method_training_objectives(method, prepared_fixture, tmp_path):
    config = method_config(method)
    config['model'].update(channels=[8, 12], tokens=6, global_dim=16, projection_hidden=24, alignment_dim=6)
    config['training'].update(episodes=2, ways=2, shots=2, queries=2, device='cpu', threads=2, log_every=1)
    config['baseline']['pretrain_episodes'] = 1
    status = train(config, prepared_fixture, tmp_path / method)
    assert status['complete']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_deterministic_full_objective(prepared_fixture, tmp_path):
    config = small_config()
    config['training'].update(episodes=1, device='cuda')
    config['model']['tokens'] = 30
    status = train(config, prepared_fixture, tmp_path / 'cuda')
    assert status['complete']
