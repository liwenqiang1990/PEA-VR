#!/usr/bin/env python3
"""Make the global-pooling encoder speak the AMPEncoder interface.

`e2_train_repr.py` is the harness that produced the reported PEA-VR numbers, but it could only
build `AMPEncoder`, so the encoder ablation would otherwise have to run in a separate harness.
Running the same configuration under two harnesses is not a matched comparison -- their
augmentation RNG streams differ, which alone moved the result by 1.18 points. Wrapping the
global-pooling encoder in the same dict interface lets both arms run inside e2, so the encoder
margin is read within one harness.

`MultiScaleTemporalEncoder` already exposes `.projection`, which `extract_embeddings` needs, so
only `forward` has to be adapted.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from train_longenough_offset0_multiscale_baseline import MultiScaleTemporalEncoder


class GlobalPoolAdapter(MultiScaleTemporalEncoder):
    """Same encoder, dict output: {'embedding', 'identity_logits', 'observed', 'content', 'style'}."""

    def forward(self, inputs: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding, pooled, logits = super().forward(inputs)
        return {"embedding": embedding, "identity_logits": logits,
                "observed": pooled, "content": pooled, "style": pooled}


def batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Batch-hard triplet on L2-normalized embeddings, as in the objective ablation."""
    z = torch.nn.functional.normalize(emb, dim=1)
    d = torch.cdist(z, z)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    hardest_pos = torch.where(same & ~eye, d, torch.zeros_like(d)).max(1).values
    hardest_neg = torch.where(~same, d, torch.full_like(d, 1e9)).min(1).values
    return torch.nn.functional.relu(hardest_pos - hardest_neg + margin).mean()
