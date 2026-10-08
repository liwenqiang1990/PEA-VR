from __future__ import annotations

import torch
from torch.nn import functional as F

from .amp import Fingerprint


def token_centers(count: int, window: float, device=None, dtype=None):
    return (torch.arange(count, device=device, dtype=dtype) + 0.5) * window / count


def prefix_view(inputs, fractions, window=60.0):
    result = {}
    for scale, item in inputs.items():
        x, mask = item['x'], item['mask']
        times = token_centers(x.shape[1], window, x.device, x.dtype)
        visible = (times[None] <= fractions[:, None] * window).to(x.dtype)
        result[scale] = {'x': x * visible[..., None], 'mask': mask * visible}
    return result


def warped_coordinates(eta, count, window=60.0):
    time = token_centers(count, window, eta.device, eta.dtype)[None]
    middle = window * (0.5 + eta[:, None])
    return torch.where(time <= window / 2, time * middle / (window / 2),
                       middle + (time - window / 2) * (window - middle) / (window / 2))


def interpolate_tokens(values, coordinates, window=60.0):
    count = values.shape[1]
    position = (coordinates * count / window - 0.5).clamp(0, count - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(count - 1)
    weight = position - left.to(position.dtype)
    if values.ndim == 3:
        left = left[..., None].expand(-1, -1, values.shape[-1])
        right = right[..., None].expand_as(left)
        weight = weight[..., None]
    return values.gather(1, left) * (1 - weight) + values.gather(1, right) * weight


def warp_fingerprint(fingerprint: Fingerprint, eta, projection, window=60.0):
    coordinates = warped_coordinates(eta, fingerprint.u.shape[1], window)
    u = interpolate_tokens(fingerprint.u, coordinates, window)
    r = interpolate_tokens(fingerprint.r, coordinates, window)
    u = torch.where((r > 0)[..., None], u, torch.zeros_like(u))
    v = F.normalize(projection(u), dim=-1) if projection is not None else None
    if v is not None:
        v = torch.where((r > 0)[..., None], v, torch.zeros_like(v))
    return Fingerprint(fingerprint.z, u, r, v), coordinates


def alignment_target(source_r, transformed_r, coordinates, sigma=2.0, window=60.0):
    centers = token_centers(source_r.shape[1], window, source_r.device, source_r.dtype)
    distance = centers[None, :, None] - coordinates[:, None, :]
    weights = source_r[..., :, None] * transformed_r[..., None, :] * torch.exp(-distance.square() / (2 * sigma**2))
    normalizer = weights.sum(-2, keepdim=True)
    return weights / normalizer.clamp_min(1e-8), (transformed_r > 0) & (normalizer.squeeze(-2) > 0)

