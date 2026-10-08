from __future__ import annotations

import numpy as np

FEATURES = ['down_bytes', 'up_bytes', 'down_packets', 'up_packets', 'down_mean_packet_length']


def validate_scales(scales, window=60.0):
    duration = int(round(window * 1000))
    scales = tuple(sorted(set(int(s) for s in scales)))
    if window <= 0 or not scales or any(s <= 0 or duration % s for s in scales):
        raise ValueError('Every positive scale must divide the observation window')
    if any(s % scales[0] for s in scales):
        raise ValueError('Scales must be multiples of the finest scale')
    return scales


def packet_features(times, directions, lengths, scales=(100, 500, 2000), window=60.0,
                    observed_until=None, start=0.0):
    scales = validate_scales(scales, window)
    times = np.asarray(times, dtype=np.float64) - start
    directions, lengths = np.asarray(directions), np.asarray(lengths, dtype=np.float64)
    if times.ndim != 1 or not (len(times) == len(directions) == len(lengths)):
        raise ValueError('Packet arrays must be one-dimensional and aligned')
    if not np.isfinite(times).all() or not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError('Invalid packet values')
    if not np.isin(directions, [0, 1]).all():
        raise ValueError('Direction must be 0 (down) or 1 (up)')
    observed_until = window if observed_until is None else float(observed_until)
    if not 0 <= observed_until <= window:
        raise ValueError('Invalid observation horizon')
    selected = (times >= 0) & (times < observed_until)
    times, directions, lengths = times[selected], directions[selected], lengths[selected]
    output = {}
    for scale in scales:
        count = int(round(window * 1000)) // scale
        bins = np.floor(times * 1000 / scale + 1e-9).astype(np.int64)
        x = np.zeros((count, 5), dtype=np.float64)
        for direction in (0, 1):
            pick = directions == direction
            x[:, direction] = np.bincount(bins[pick], weights=lengths[pick], minlength=count)
            x[:, 2 + direction] = np.bincount(bins[pick], minlength=count)
        np.divide(x[:, 0], x[:, 2], out=x[:, 4], where=x[:, 2] > 0)
        centers = (np.arange(count) + 0.5) * scale / 1000
        mask = (centers <= observed_until).astype(np.float32)
        output[scale] = {'x': x.astype(np.float32), 'mask': mask}
    return output


class Normalizer:
    def __init__(self, state=None):
        self.state = state

    def fit(self, inputs, training_indices):
        indices = np.asarray(training_indices, dtype=np.int64)
        if not len(indices):
            raise ValueError('Normalization requires training sessions')
        state = {'features': FEATURES, 'fit_indices': indices.tolist(), 'scales': {}}
        for scale, item in inputs.items():
            values = np.log1p(item['x'][indices].astype(np.float64))
            mask = item['mask'][indices] > 0
            if not mask.any():
                raise ValueError('No observed bins available for normalization')
            values = values[mask]
            mean, std = values.mean(0), values.std(0)
            std[std < 1e-6] = 1.0
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise ValueError('Non-finite normalization statistics')
            state['scales'][str(scale)] = {'mean': mean.tolist(), 'std': std.tolist()}
        self.state = state
        return self

    def transform(self, inputs):
        if self.state is None:
            raise ValueError('Normalizer has not been fitted')
        result = {}
        for scale, item in inputs.items():
            stats = self.state['scales'][str(scale)]
            x = item['x']
            if not np.isfinite(x).all() or (x < 0).any():
                raise ValueError('Expected finite nonnegative raw features')
            transformed = (np.log1p(x.astype(np.float64)) - stats['mean']) / stats['std']
            transformed = np.where(item['mask'][..., None] > 0, transformed, 0)
            result[scale] = {'x': transformed.astype(np.float32), 'mask': item['mask'].astype(np.float32)}
        return result
