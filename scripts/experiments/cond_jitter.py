#!/usr/bin/env python3
"""Condition jitter: train-time augmentation that simulates a bandwidth change.

Why this rather than changing the input representation: `persample_norm` showed that removing the
absolute byte scale helps YDMS cross-bandwidth (+11.15) but costs LongEnough accuracy everywhere
(-2.30 to -7.15), because when the conditions are controlled the absolute scale carries identity
information. An augmentation leaves the representation intact and instead teaches invariance, so it
can help the cross-condition scenarios without discarding what the other scenarios use.

Two operations, both applied to the ALREADY NORMALIZED multiscale views:

  scale jitter    a bandwidth change multiplies the byte and packet counts of a session by roughly
                  a constant factor c. The views are log1p-then-z-scored, so for the non-empty bins
                  log1p(c*x) ~= log(c) + log1p(x): a multiplicative change in raw space is an
                  ADDITIVE shift in the normalized space. We therefore add a per-sample shift to
                  the count channels (down/up bytes, down/up packets), leaving the mean-packet-length
                  channel alone because it does not scale with bandwidth.

  time dilation   a lower bandwidth spreads the same content over more wall-clock time. We resample
                  the bin axis by a random factor and pad or crop back to the original length, which
                  simulates that dilation without touching the channel semantics.

Documented approximation: the shift is applied in normalized space with a magnitude in normalized
units, rather than by rebuilding the views from raw counts; the two coincide up to the per-feature
sigma of the base-split normalization.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

COUNT_CHANNELS = (0, 1, 2, 3)          # down_bytes, up_bytes, down_packets, up_packets


def condition_jitter(inputs: dict[int, torch.Tensor], scale_std: float, dilate: float,
                     generator: torch.Generator | None = None) -> dict[int, torch.Tensor]:
    """Per-sample scale shift (std `scale_std`) and time dilation (factor in [1-d, 1+d])."""
    if scale_std <= 0 and dilate <= 0:
        return inputs
    any_scale = next(iter(inputs.values()))
    b = any_scale.shape[0]
    device = any_scale.device
    shift = (torch.randn(b, device=device, generator=generator) * scale_std
             if scale_std > 0 else None)
    factor = (1.0 + (torch.rand(b, device=device, generator=generator) * 2 - 1) * dilate
              if dilate > 0 else None)
    out = {}
    for scale, x in inputs.items():
        v = x.clone()
        if shift is not None:
            nonempty = v.abs().sum(dim=2, keepdim=True) > 0
            add = torch.zeros_like(v)
            add[:, :, COUNT_CHANNELS] = shift[:, None, None]
            v = torch.where(nonempty, v + add, v)
        if factor is not None:
            n = v.shape[1]
            src = v.transpose(1, 2)                                  # (B, C, L)
            resampled = torch.empty_like(src)
            for i in range(b):
                m = max(2, int(round(n * float(factor[i]))))
                r = F.interpolate(src[i:i + 1], size=m, mode="linear", align_corners=False)
                if m >= n:
                    resampled[i] = r[0, :, :n]
                else:
                    resampled[i, :, :m] = r[0]
                    resampled[i, :, m:] = 0.0
            v = resampled.transpose(1, 2)
        out[scale] = v
    return out
