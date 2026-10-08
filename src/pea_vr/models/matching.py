from __future__ import annotations

import torch
from torch.nn import functional as F

from .amp import Fingerprint
from .pma import partial_monotone_alignment, soft_dtw_mass


class Matcher:
    def __init__(self, config: dict):
        self.config = dict(config)

    def pair(self, a: Fingerprint, b: Fingerprint, return_mass=False):
        c = self.config
        global_similarity = (a.z * b.z).sum(-1)
        if c['alignment'] == 'none':
            return {'score': global_similarity, 'local': torch.zeros_like(global_similarity),
                    'overlap': torch.zeros_like(global_similarity), 'mass': None}
        if a.v is None or b.v is None:
            raise ValueError('Local matching requires projected tokens')
        similarity = a.v @ b.v.transpose(-1, -2)
        valid = (a.r > 0)[..., :, None] & (b.r > 0)[..., None, :]
        if c['alignment'] == 'pma':
            _, mass = partial_monotone_alignment(similarity - c['delta'], valid, c['gap'], c['gamma'])
        elif c['alignment'] == 'diagonal':
            mass = torch.eye(similarity.shape[-2], similarity.shape[-1], device=similarity.device,
                             dtype=similarity.dtype)[None] * valid
        else:
            mass = soft_dtw_mass(similarity, a.r, b.r, c['soft_dtw_gamma'])
        eps = c['epsilon']
        local = (mass * similarity).sum((-2, -1)) / (mass.sum((-2, -1)) + eps)
        if c['alignment'] == 'pma':
            numerator = (mass * torch.minimum(a.r[..., :, None], b.r[..., None, :])).sum((-2, -1))
            overlap = numerator / (torch.minimum(a.r.sum(-1), b.r.sum(-1)) + eps)
        else:
            overlap = torch.zeros_like(local)
        score = c['lambda_z'] * global_similarity + (1 - c['lambda_z']) * local
        return {'score': score, 'local': local, 'overlap': overlap, 'mass': mass if return_mass else None}

    def matrix(self, query: Fingerprint, support: Fingerprint, return_details=False):
        nq, ns = query.z.shape[0], support.z.shape[0]
        scores, locals_, overlaps = [], [], []
        total = nq * ns
        for start in range(0, total, self.config['pair_chunk']):
            index = torch.arange(start, min(start + self.config['pair_chunk'], total), device=query.z.device)
            result = self.pair(query.index(index // ns), support.index(index % ns))
            scores.append(result['score'])
            locals_.append(result['local'])
            overlaps.append(result['overlap'])
        details = {key: torch.cat(parts).reshape(nq, ns) for key, parts in
                   (('pair', scores), ('local', locals_), ('overlap', overlaps))}
        return details if return_details else details['pair']

    def candidates(self, query: Fingerprint, support: Fingerprint, support_labels: torch.Tensor,
                   return_details=False):
        classes = torch.unique(support_labels, sorted=True)
        if len(classes) == 0 or support.z.shape[0] != len(support_labels):
            raise ValueError('Nonempty labeled supports are required')
        c = self.config
        if c['aggregation'] == 'prototype':
            prototypes = torch.stack([F.normalize(support.z[support_labels == label].mean(0), dim=0) for label in classes])
            scores = query.z @ prototypes.T
            return (scores, classes, {}) if return_details else (scores, classes)
        details = self.matrix(query, support, True)
        columns, responsibilities = [], []
        for label in classes:
            selected = support_labels == label
            pair = details['pair'][:, selected]
            kind = c['aggregation']
            local, overlap = details['local'][:, selected], details['overlap'][:, selected]
            if kind == 'mean':
                score = pair.mean(-1)
                omega = torch.full_like(pair, 1 / pair.shape[-1])
            elif kind == 'support_max':
                score, arg = pair.max(-1)
                omega = F.one_hot(arg, pair.shape[-1]).to(pair.dtype)
            else:
                compatibility = torch.zeros_like(pair)
                if kind in ('compatibility', 'similarity'):
                    compatibility = compatibility + c['beta_similarity'] * local
                if kind in ('compatibility', 'overlap') and c['alignment'] == 'pma':
                    compatibility = compatibility + c['beta_overlap'] * overlap
                log_omega = F.log_softmax(compatibility / c['tau_c'], -1)
                omega = log_omega.exp()
                score = c['tau_s'] * torch.logsumexp(log_omega + pair / c['tau_s'], -1)
            columns.append(score)
            responsibilities.append(omega)
        scores = torch.stack(columns, -1)
        if return_details:
            details['responsibility'] = responsibilities
            return scores, classes, details
        return scores, classes

