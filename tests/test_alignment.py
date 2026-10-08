import torch
import pytest

from pea_vr.models.pma import partial_monotone_alignment, pma_reference_score, soft_dtw_mass


def test_pma_derivative_and_mass_constraints():
    torch.manual_seed(7)
    affinity = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    valid = torch.rand(2, 4, 5) > .2
    score, mass = partial_monotone_alignment(affinity, valid, .1, .2)
    reference = pma_reference_score(affinity, valid, .1, .2)
    derivative = torch.autograd.grad(reference.sum(), affinity, create_graph=True)[0]
    torch.testing.assert_close(score, reference)
    torch.testing.assert_close(mass, derivative)
    assert (mass >= 0).all() and (mass.sum(-1) <= 1 + 1e-10).all()
    assert (mass.sum(-2) <= 1 + 1e-10).all()
    assert (mass[~valid] == 0).all()
    torch.autograd.grad((mass * affinity).sum(), affinity)


def test_pma_higher_order_gradient():
    x = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    function = lambda a: partial_monotone_alignment(a, gap=.1, gamma=.3)[1]
    assert torch.autograd.gradcheck(function, (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (x,), fast_mode=True)


def test_all_invalid_alignment_and_single_match():
    x = torch.tensor([[[.7]]], dtype=torch.float64, requires_grad=True)
    _, a = partial_monotone_alignment(x, torch.zeros_like(x, dtype=torch.bool))
    assert a.item() == 0
    _, a = partial_monotone_alignment(x)
    assert 0 < a.item() < 1


def soft_dtw_reference(cost, gamma):
    cells = [[torch.tensor(float('inf'), dtype=cost.dtype) for _ in range(cost.shape[1] + 1)]
             for _ in range(cost.shape[0] + 1)]
    cells[0][0] = cost.new_zeros(())
    for i in range(1, len(cells)):
        for j in range(1, len(cells[0])):
            cells[i][j] = cost[i-1, j-1] - gamma * torch.logsumexp(
                -torch.stack([cells[i-1][j-1], cells[i-1][j], cells[i][j-1]]) / gamma, 0)
    return cells[-1][-1]


def test_soft_dtw_matches_cost_derivative_with_holes():
    torch.manual_seed(2)
    similarity = torch.randn(1, 4, 5, dtype=torch.float64, requires_grad=True)
    ra, rb = torch.tensor([[1., 0, 1, 1]]), torch.tensor([[0., 1, 1, 0, 1]])
    mass = soft_dtw_mass(similarity, ra, rb, .2)
    cost = (1 - similarity[0, [0, 2, 3]][:, [1, 2, 4]]).detach().requires_grad_()
    expected = torch.autograd.grad(soft_dtw_reference(cost, .2), cost)[0]
    torch.testing.assert_close(mass[0, [0, 2, 3]][:, [1, 2, 4]], expected)
    assert torch.isfinite(torch.autograd.grad((mass * similarity).sum(), similarity)[0]).all()
    assert (mass[0, 1] == 0).all() and (mass[0, :, [0, 3]] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_pma_cpu_cuda():
    x = torch.randn(3, 8, 7)
    cpu = partial_monotone_alignment(x)
    gpu = partial_monotone_alignment(x.cuda())
    for a, b in zip(cpu, gpu):
        torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=2e-5)
