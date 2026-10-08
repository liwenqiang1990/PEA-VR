from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .amp import Fingerprint
from .pooling import AdaptiveMeanPool1d


def fingerprint(z):
    z = F.normalize(z, dim=-1)
    return Fingerprint(z, z[:, None], z.new_ones((len(z), 1)))


class ConvEmbedding(nn.Module):
    def __init__(self, kind='protonet', embedding=512):
        super().__init__()
        self.scale = 100
        if kind == 'deepmetric':
            channels, kernel, pooling = [128, 256, 512], 7, 2
        else:
            channels, kernel, pooling = [64, 64, 64, 64], 5, 2
        layers, previous = [], 6
        for i, width in enumerate(channels):
            for _ in range(2 if kind == 'deepmetric' else 1):
                layers.extend([nn.ConstantPad1d(((kernel - 1) // 2, kernel // 2), 0),
                               nn.Conv1d(previous, width, kernel)])
                layers.extend([nn.ReLU(), nn.BatchNorm1d(width)] if kind == 'deepmetric' else [nn.BatchNorm1d(width), nn.ReLU()])
                previous = width
            if kind != 'deepmetric' or i < len(channels) - 1:
                layers.extend([nn.MaxPool1d(pooling, ceil_mode=True), nn.Dropout(.1)])
        if kind == 'deepmetric':
            self.encoder = nn.Sequential(*layers, AdaptiveMeanPool1d(1), nn.Flatten(), nn.Linear(previous, 1024),
                                         nn.ReLU(), nn.Dropout(.1), nn.Linear(1024, embedding))
        else:
            self.encoder = nn.Sequential(*layers, AdaptiveMeanPool1d(8), nn.Flatten(), nn.Linear(previous * 8, embedding))

    def forward(self, inputs):
        x, mask = inputs[self.scale]['x'], inputs[self.scale]['mask']
        values = torch.cat((torch.where(mask[..., None] > 0, x, 0), mask[..., None]), -1).transpose(1, 2)
        return fingerprint(self.encoder(values))


class MetaFlowEncoder(nn.Module):
    def __init__(self, embedding=512):
        super().__init__()
        self.views = [(0, 1), (2, 3), (4,), (0, 1, 2, 3, 4)]
        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Conv1d(len(view) + 1, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), AdaptiveMeanPool1d(8), nn.Flatten(),
            nn.Linear(1024, 128)) for view in self.views])
        self.fusion = nn.Sequential(nn.Linear(512, embedding), nn.GELU(), nn.LayerNorm(embedding))

    def forward(self, inputs):
        x, mask = inputs[100]['x'], inputs[100]['mask']
        x = torch.where(mask[..., None] > 0, x, 0)
        views = [encoder(torch.cat((x[..., list(view)], mask[..., None]), -1).transpose(1, 2))
                 for encoder, view in zip(self.encoders, self.views)]
        return fingerprint(self.fusion(torch.cat(views, -1)))


class TransformerEmbedding(nn.Module):
    def __init__(self, embedding=512):
        super().__init__()
        self.input = nn.Linear(6, 128)
        self.position = nn.Parameter(torch.randn(1, 120, 128) * .02)
        block = nn.TransformerEncoderLayer(128, 4, 512, .1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, 3, enable_nested_tensor=False)
        self.output = nn.Linear(128, embedding)

    def forward(self, inputs):
        x, mask = inputs[500]['x'], inputs[500]['mask']
        values = torch.cat((torch.where(mask[..., None] > 0, x, 0), mask[..., None]), -1)
        hidden = self.input(values) + self.position[:, :len(x[0])]
        padding = mask <= 0
        padding = padding.clone()
        padding[:, 0] = False
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return fingerprint(self.output(pooled))
