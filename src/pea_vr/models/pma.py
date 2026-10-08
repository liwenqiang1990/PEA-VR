from __future__ import annotations

from functools import lru_cache

import torch


def _read(diagonals, starts, d, rows, batch, default=0.0):
    if d < 0 or d >= len(diagonals) or diagonals[d] is None:
        return batch.new_full((batch.shape[0], len(rows)), default)
    values = diagonals[d]
    return _gather(values, starts[d], rows, default)


def _gather(values, start, rows, default=0.0):
    index = rows - start
    valid = (index >= 0) & (index < values.shape[-1])
    value = values[:, index.clamp(0, values.shape[-1] - 1)]
    return torch.where(valid[None], value, torch.full_like(value, default))


@lru_cache(maxsize=32)
def _layout(m, n, device):
    rows, starts, flat = [], [], []
    for d in range(m + n - 1):
        start, end = max(0, d - n + 1), min(m - 1, d)
        row = torch.arange(start, end + 1, device=device)
        rows.append(row)
        starts.append(start)
        flat.append(row * n + d - row)
    return rows, starts, torch.argsort(torch.cat(flat))


def partial_monotone_alignment(affinity: torch.Tensor, valid: torch.Tensor | None = None,
                               gap: float = 0.10, gamma: float = 0.10):
    """Return soft endpoint score and d(score)/d(affinity), preserving higher derivatives."""
    if gamma <= 0 or gap < 0 or affinity.ndim < 2:
        raise ValueError('Invalid PMA arguments')
    shape, m, n = affinity.shape[:-2], affinity.shape[-2], affinity.shape[-1]
    if min(m, n) < 1:
        raise ValueError('PMA requires nonempty token axes')
    x = affinity.reshape(-1, m, n)
    valid = torch.ones_like(x, dtype=torch.bool) if valid is None else valid.expand_as(affinity).reshape_as(x).bool()
    rows, starts, order = _layout(m, n, x.device)
    forward, probabilities = [], []
    for d, row in enumerate(rows):
        col = d - row
        diagonal = _read(forward, starts, d - 2, row - 1, x) + x[:, row, col]
        diagonal = diagonal.masked_fill(~valid[:, row, col], -torch.inf)
        top = _read(forward, starts, d - 1, row - 1, x) - gap
        left = _read(forward, starts, d - 1, row, x) - gap
        transitions = torch.stack((torch.zeros_like(top), diagonal, top, left), -1)
        probabilities.append(torch.softmax(transitions / gamma, -1))
        forward.append(gamma * torch.logsumexp(transitions / gamma, -1))
    score = gamma * torch.logsumexp(torch.cat(forward, -1) / gamma, -1)
    backward = [None] * len(rows)
    mass = [None] * len(rows)
    for d in range(len(rows) - 1, -1, -1):
        row = rows[d]
        endpoint = ((forward[d] - score[:, None]) / gamma).exp()
        downstream = endpoint
        for shift, offset, transition in ((1, 1, 2), (1, 0, 3), (2, 1, 1)):
            following = d + shift
            if following < len(rows):
                flow = backward[following] * probabilities[following][..., transition]
                downstream = downstream + _gather(flow, starts[following], row + offset)
        backward[d] = downstream
        mass[d] = downstream * probabilities[d][..., 1]
    result = torch.cat(mass, -1)[:, order].reshape(*shape, m, n)
    return score.reshape(shape), result


def pma_reference_score(affinity, valid=None, gap=0.10, gamma=0.10):
    m, n = affinity.shape[-2:]
    zero = torch.zeros_like(affinity[..., 0, 0])
    table = [[zero for _ in range(n + 1)] for _ in range(m + 1)]
    cells = []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = table[i - 1][j - 1] + affinity[..., i - 1, j - 1]
            if valid is not None:
                match = match.masked_fill(~valid[..., i - 1, j - 1], -torch.inf)
            table[i][j] = gamma * torch.logsumexp(torch.stack(
                (zero, match, table[i - 1][j] - gap, table[i][j - 1] - gap), -1) / gamma, -1)
            cells.append(table[i][j])
    return gamma * torch.logsumexp(torch.stack(cells, -1) / gamma, -1)


def soft_dtw_mass(similarity, ra, rb, gamma=0.10):
    """Soft-DTW occupancy on compacted valid token sequences."""
    if gamma <= 0:
        raise ValueError('Soft-DTW gamma must be positive')
    batch, m, n = similarity.shape
    ia = torch.argsort((ra <= 0).to(torch.int64), dim=-1, stable=True)
    ib = torch.argsort((rb <= 0).to(torch.int64), dim=-1, stable=True)
    cost = 1 - similarity.gather(1, ia[..., None].expand(-1, -1, n)).gather(2, ib[:, None].expand(-1, m, -1))
    la, lb = (ra > 0).sum(-1), (rb > 0).sum(-1)
    rows, starts, order = _layout(m, n, cost.device)
    forward, probabilities, available = [], [], []
    for d, row in enumerate(rows):
        col = d - row
        diagonal = _read(forward, starts, d - 2, row - 1, cost, torch.inf)
        diagonal = torch.where(((row == 0) & (col == 0))[None], torch.zeros_like(diagonal), diagonal)
        top = _read(forward, starts, d - 1, row - 1, cost, torch.inf)
        left = _read(forward, starts, d - 1, row, cost, torch.inf)
        transitions = torch.stack((diagonal, top, left), -1)
        reachable = torch.isfinite(transitions).any(-1)
        safe = torch.where(reachable[..., None], transitions, torch.zeros_like(transitions))
        valid = (row[None] < la[:, None]) & (col[None] < lb[:, None]) & reachable
        probabilities.append(torch.softmax(-safe / gamma, -1) * valid[..., None])
        forward.append(torch.where(valid, cost[:, row, col] - gamma * torch.logsumexp(-safe / gamma, -1), torch.inf))
        available.append(valid)
    backward = [None] * len(rows)
    mass = [None] * len(rows)
    for d in range(len(rows) - 1, -1, -1):
        row, col = rows[d], d - rows[d]
        downstream = ((row[None] == la[:, None] - 1) & (col[None] == lb[:, None] - 1)).to(cost.dtype)
        for shift, offset, transition in ((1, 1, 1), (1, 0, 2), (2, 1, 0)):
            following = d + shift
            if following < len(rows):
                flow = backward[following] * probabilities[following][..., transition]
                downstream = downstream + _gather(flow, starts[following], row + offset)
        backward[d] = downstream * available[d]
        mass[d] = backward[d]
    compact = torch.cat(mass, -1)[:, order].reshape(batch, m, n)
    inverse_a, inverse_b = torch.argsort(ia, -1), torch.argsort(ib, -1)
    return compact.gather(1, inverse_a[..., None].expand(-1, -1, n)).gather(2, inverse_b[:, None].expand(-1, m, -1))
