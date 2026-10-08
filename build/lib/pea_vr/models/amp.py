from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .pooling import AdaptiveMeanPool1d


@dataclass
class Fingerprint:
    z: torch.Tensor
    u: torch.Tensor
    r: torch.Tensor
    v: torch.Tensor | None = None

    def index(self, indices) -> 'Fingerprint':
        return Fingerprint(self.z[indices], self.u[indices], self.r[indices],
                           self.v[indices] if self.v is not None else None)

    def to(self, *args, **kwargs) -> 'Fingerprint':
        return Fingerprint(self.z.to(*args, **kwargs), self.u.to(*args, **kwargs),
                           self.r.to(*args, **kwargs), self.v.to(*args, **kwargs) if self.v is not None else None)

    def detach(self) -> 'Fingerprint':
        return Fingerprint(self.z.detach(), self.u.detach(), self.r.detach(),
                           self.v.detach() if self.v is not None else None)

    @classmethod
    def cat(cls, parts: list['Fingerprint']) -> 'Fingerprint':
        return cls(*(torch.cat([getattr(p, key) for p in parts]) if getattr(parts[0], key) is not None else None
                     for key in ('z', 'u', 'r', 'v')))


class AMPEncoder(nn.Module):
    def __init__(self, scales=(100, 500, 2000), channels=(128, 256), tokens=30,
                 global_dim=512, projection_hidden=1024, alignment_dim=128,
                 dropout=0.1, fusion='learned', local=True):
        super().__init__()
        self.scales = tuple(scales)
        self.tokens = int(tokens)
        self.fusion = fusion
        self.local = local
        self.pool = AdaptiveMeanPool1d(tokens)
        dim = channels[-1]
        self.branches = nn.ModuleDict({str(s): nn.Sequential(
            nn.Conv1d(6, channels[0], 5, padding=2), nn.GELU(),
            nn.Conv1d(channels[0], dim, 5, padding=2), nn.GELU(), nn.Dropout(dropout)) for s in scales})
        self.projections = nn.ModuleDict({str(s): nn.Linear(dim, dim) for s in scales})
        self.gates = nn.ModuleDict({str(s): nn.Linear(dim, 1) for s in scales}) if fusion == 'learned' else None
        self.norm = nn.LayerNorm(dim)
        self.global_projection = nn.Sequential(nn.Linear(2 * dim, projection_hidden), nn.GELU(),
                                               nn.Dropout(dropout), nn.Linear(projection_hidden, global_dim))
        self.local_projection = nn.Linear(dim, alignment_dim) if local else None

    def forward(self, inputs: dict[int, dict[str, torch.Tensor]]) -> Fingerprint:
        hidden, reliability, logits = [], [], []
        for scale in self.scales:
            x, mask = inputs[scale]['x'], inputs[scale]['mask']
            if x.ndim != 3 or x.shape[-1] != 5 or mask.shape != x.shape[:2]:
                raise ValueError('Expected features [B,L,5] and mask [B,L]')
            mask = mask.to(x.dtype)
            x = torch.where(mask[..., None] > 0, x * mask[..., None], torch.zeros_like(x))
            h = self.branches[str(scale)](torch.cat((x, mask[..., None]), -1).transpose(1, 2))
            h = self.pool(h).transpose(1, 2)
            r = self.pool(mask)
            hidden.append(self.projections[str(scale)](h))
            reliability.append(r)
            logits.append(self.gates[str(scale)](h).squeeze(-1) if self.gates is not None else torch.zeros_like(r))
        rs = torch.stack(reliability, -1)
        scores = torch.stack(logits, -1)
        scores = scores - scores.max(-1, keepdim=True).values
        weights = rs * scores.exp()
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)
        r = rs.amax(-1)
        u = self.norm(F.gelu((torch.stack(hidden, -2) * weights[..., None]).sum(-2)))
        u = torch.where((r > 0)[..., None], u, torch.zeros_like(u))
        average = (u * r[..., None]).sum(1) / (r.sum(1, keepdim=True) + 1e-8)
        maximum = u.masked_fill((r == 0)[..., None], -torch.inf).amax(1)
        maximum = torch.where((r.sum(1) > 0)[:, None], maximum, torch.zeros_like(maximum))
        z = F.normalize(self.global_projection(torch.cat((average, maximum), -1)), dim=-1)
        z = torch.where((r.sum(1) > 0)[:, None], z, torch.zeros_like(z))
        v = F.normalize(self.local_projection(u), dim=-1) if self.local_projection is not None else None
        if v is not None:
            v = torch.where((r > 0)[..., None], v, torch.zeros_like(v))
        return Fingerprint(z, u, r, v)
