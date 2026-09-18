#!/usr/bin/env python3
"""Token-shuffle intervention: is AMP's gain the per-position cross-scale correspondence?

AMP tokenizes each temporal scale into the same number of aligned tokens and fuses ACROSS SCALES at
each token position (`observe()` concatenates along the feature axis, so position k of the 100 ms
view meets position k of the 2000 ms view). Global pooling instead collapses each scale over the
whole session before the scales ever meet. The local-neighbourhood analysis showed AMP's advantage
is a wider cross-condition margin; this intervention asks whether that alignment is the cause.

Two modes, both applied during training and inference:

  independent : each scale's token axis is permuted with its OWN random permutation, redrawn per
                sample. The multiset of tokens per scale is untouched and every later stage is
                order-invariant, so the only thing destroyed is which token meets which.
  shared      : one permutation applied to ALL scales. Alignment is preserved, only the order
                changes. In this configuration (`temporal_encoder="identity"`, mean+amax pooling)
                nothing downstream reads the order, so the forward pass is provably identical --
                verified numerically at 3.7e-08 max elementwise difference. Its trained accuracy
                nevertheless differs, which can only come from the RNG stream, so:
  rngonly     : draws the same random numbers and discards them. This separates "the intervention
                changed the model" from "the extra draws moved the training trajectory", and is
                the control the alignment effect must be measured against.

A fixed permutation would not test anything: a deterministic relabelling of positions is one the
model can simply learn, so the permutation is redrawn per sample.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from train_longenough_offset0_swapcanonical import AMPEncoder


class TokenShuffledAMP(AMPEncoder):
    """AMPEncoder whose per-scale token axes are permuted before cross-scale fusion."""

    def __init__(self, *args, shuffle: str = "independent", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if shuffle not in {"independent", "shared", "rngonly"}:
            raise ValueError(f"unknown shuffle mode: {shuffle}")
        self.shuffle = shuffle

    def observe(self, inputs: dict[int, torch.Tensor]) -> torch.Tensor:
        aligned = [self.tokenizers[str(scale)](inputs[scale]) for scale in self.scales]
        b, t, _ = aligned[0].shape
        device = aligned[0].device
        if self.shuffle == "rngonly":
            # draw exactly as many random numbers as the other modes, then discard them: an
            # operation that is provably identity BOTH in the forward pass and in the token
            # ordering. Any difference this produces can only come from the RNG stream.
            for _ in aligned:
                torch.rand(b, t, device=device)
            return self.observation_fusion(torch.cat(aligned, dim=2))
        if self.shuffle == "shared":
            perm = torch.argsort(torch.rand(b, t, device=device), dim=1)
            perms = [perm] * len(aligned)
        else:
            perms = [torch.argsort(torch.rand(b, t, device=device), dim=1) for _ in aligned]
        aligned = [a.gather(1, p.unsqueeze(-1).expand(-1, -1, a.shape[2]))
                   for a, p in zip(aligned, perms)]
        return self.observation_fusion(torch.cat(aligned, dim=2))
