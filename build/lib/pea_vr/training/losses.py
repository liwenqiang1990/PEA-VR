from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive(z, labels, temperature=0.10):
    z = F.normalize(z, dim=-1)
    same = labels[:, None] == labels[None]
    eye = torch.eye(len(labels), device=z.device, dtype=torch.bool)
    positive = same & ~eye
    if not torch.all(positive.any(-1)):
        raise ValueError('Each contrastive anchor requires a positive sample')
    logits = z @ z.T / temperature
    denominator = torch.logsumexp(logits.masked_fill(eye, -torch.inf), -1, keepdim=True)
    return -((logits - denominator) * positive).sum(-1).div(positive.sum(-1)).mean()


def alignment_cross_entropy(mass, target, valid, epsilon=1e-8):
    probability = mass / (mass.sum(-2, keepdim=True) + epsilon)
    loss = -(target * (probability + epsilon).log()).sum(-2)
    per_pair = (loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    usable = valid.any(-1)
    return per_pair[usable].mean() if usable.any() else mass.sum() * 0


def batch_hard_triplet(z, labels, margin=0.2):
    distance = torch.cdist(F.normalize(z, dim=-1), F.normalize(z, dim=-1))
    same = labels[:, None] == labels[None]
    eye = torch.eye(len(labels), device=z.device, dtype=torch.bool)
    positive, negative = same & ~eye, ~same
    if not positive.any(-1).all() or not negative.any(-1).all():
        raise ValueError('Triplet batches need positive and negative samples')
    hardest_positive = distance.masked_fill(~positive, -torch.inf).amax(-1)
    hardest_negative = distance.masked_fill(~negative, torch.inf).amin(-1)
    return F.relu(hardest_positive - hardest_negative + margin).mean()

