import torch
from torch import nn


class AdaptiveMeanPool1d(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.count = int(count)
        self.register_buffer('_weights', torch.empty(0), persistent=False)

    def forward(self, values):
        length = values.shape[-1]
        if self._weights.shape != (self.count, length) or self._weights.device != values.device or self._weights.dtype != values.dtype:
            rows = torch.arange(self.count, device=values.device)[:, None]
            columns = torch.arange(length, device=values.device)[None]
            start = rows * length // self.count
            end = ((rows + 1) * length + self.count - 1) // self.count
            self._weights = ((columns >= start) & (columns < end)).to(values.dtype) / (end - start)
        return values @ self._weights.T
