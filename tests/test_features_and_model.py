import numpy as np
import pytest
import torch

from pea_vr.data.features import Normalizer, packet_features, validate_scales
from pea_vr.models.amp import AMPEncoder
from pea_vr.models.matching import Matcher
from pea_vr.models.partial_views import prefix_view, warp_fingerprint, alignment_target
from pea_vr.config import DEFAULTS


def test_features_boundaries_counts_and_observation():
    result = packet_features([0., .1, .5, 60], [0, 0, 1, 0], [10, 20, 30, 100])
    assert result[100]['x'][0, 0] == 10
    assert result[500]['x'][0, 0] == 30
    assert result[500]['x'][0, 4] == 15
    assert result[500]['x'][1, 1] == 30
    assert result[100]['x'][:, :2].sum() == 60
    assert result[100]['mask'].min() == 1
    assert packet_features([], [], [], observed_until=0)[100]['mask'].max() == 0
    with pytest.raises(ValueError):
        validate_scales([100, 700])


def test_normalizer_training_only_and_empty_observed_bins():
    inputs = {100: {'x': np.array([[[0.] * 5, [2.] * 5], [[999.] * 5, [999.] * 5]], dtype=np.float32),
                    'mask': np.ones((2, 2), dtype=np.float32)}}
    normalizer = Normalizer().fit(inputs, [0])
    assert normalizer.state['scales']['100']['mean'][0] == pytest.approx(np.log(3) / 2)
    assert normalizer.state['fit_indices'] == [0]
    inputs[100]['mask'][:] = 0
    with pytest.raises(ValueError, match='No observed'):
        Normalizer().fit(inputs, [0])


def make_inputs(batch=4):
    return {s: {'x': torch.rand(batch, 60000 // s, 5), 'mask': torch.ones(batch, 60000 // s)}
            for s in [100, 500, 2000]}


def test_all_parameters_receive_gradient_and_missing_values_are_inert():
    torch.set_num_threads(2)
    model = AMPEncoder(channels=(8, 12), global_dim=16, projection_hidden=24, alignment_dim=6, tokens=6, dropout=0)
    inputs = make_inputs()
    fractions = torch.tensor([1., .6, .4, .3])
    partial = prefix_view(inputs, fractions)
    fp = model(partial)
    matcher = Matcher(DEFAULTS['matching'])
    scores, classes = matcher.candidates(fp.index([2, 3]), fp.index([0, 1]), torch.tensor([0, 1]))
    scores.square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    altered = {s: {'x': x['x'].clone(), 'mask': x['mask']} for s, x in partial.items()}
    for x in altered.values():
        x['x'][x['mask'] == 0] = float('nan')
    torch.testing.assert_close(model(altered).z, fp.z)
    empty = prefix_view(inputs, torch.zeros(4))
    efp = model(empty)
    assert (efp.z == 0).all() and (efp.r == 0).all()


def test_warp_identity_and_targets():
    model = AMPEncoder(channels=(8, 12), global_dim=16, projection_hidden=24, alignment_dim=6, dropout=0)
    fp = model(make_inputs(2))
    warped, coordinates = warp_fingerprint(fp, torch.zeros(2), model.local_projection)
    torch.testing.assert_close(fp.u, warped.u, atol=1e-5, rtol=1e-5)
    target, valid = alignment_target(fp.r, warped.r, coordinates)
    torch.testing.assert_close(target.sum(1), torch.ones_like(fp.r))
    assert valid.all()


def test_single_support_aggregation_and_overlap_bound():
    model = AMPEncoder(channels=(8, 12), global_dim=16, projection_hidden=24, alignment_dim=6, tokens=6, dropout=0)
    fp = model(make_inputs(2))
    for rule in ['mean', 'support_max', 'uniform_lse', 'similarity', 'overlap', 'compatibility']:
        matcher = Matcher({**DEFAULTS['matching'], 'aggregation': rule})
        scores, _ = matcher.candidates(fp.index([0]), fp.index([1]), torch.tensor([3]))
        pair = matcher.pair(fp.index([0]), fp.index([1]))
        torch.testing.assert_close(scores[0, 0], pair['score'][0])
        assert 0 <= pair['overlap'].item() <= 1 + 1e-6


def test_support_permutation_and_soft_max_limit():
    model = AMPEncoder(channels=(8, 12), global_dim=16, projection_hidden=24, alignment_dim=6, tokens=6, dropout=0)
    fp = model(make_inputs(5))
    labels = torch.tensor([0, 0, 1, 1])
    query, support = fp.index([4]), fp.index([0, 1, 2, 3])
    matcher = Matcher(DEFAULTS['matching'])
    score, _ = matcher.candidates(query, support, labels)
    order = torch.tensor([3, 1, 2, 0])
    permuted, _ = matcher.candidates(query, support.index(order), labels[order])
    torch.testing.assert_close(score, permuted)
    soft = Matcher({**DEFAULTS['matching'], 'aggregation': 'uniform_lse', 'tau_s': 1e-5})
    hard = Matcher({**DEFAULTS['matching'], 'aggregation': 'support_max'})
    torch.testing.assert_close(soft.candidates(query, support, labels)[0], hard.candidates(query, support, labels)[0], atol=1e-4, rtol=1e-4)


def test_chunked_query_gradients_match_whole_batch():
    torch.manual_seed(16)
    a = AMPEncoder(channels=(8, 12), global_dim=16, projection_hidden=24, alignment_dim=6, tokens=6, dropout=0).double()
    b = AMPEncoder(channels=(8, 12), global_dim=16, projection_hidden=24, alignment_dim=6, tokens=6, dropout=0).double()
    b.load_state_dict(a.state_dict())
    inputs = {s: {k: value.double() for k, value in item.items()} for s, item in make_inputs(8).items()}
    matcher = Matcher(DEFAULTS['matching'])
    labels = torch.tensor([0, 0, 1, 1])
    fa, fb = a(inputs), b(inputs)
    score, _ = matcher.candidates(fa.index(slice(4, None)), fa.index(slice(None, 4)), labels)
    torch.nn.functional.cross_entropy(score / .1, labels).backward()
    for i in range(4):
        score, _ = matcher.candidates(fb.index(slice(4+i, 5+i)), fb.index(slice(None, 4)), labels)
        loss = torch.nn.functional.cross_entropy(score / .1, labels[i:i+1]) / 4
        loss.backward(retain_graph=i < 3)
    for (name, p), q in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-9, rtol=1e-8, msg=name)
