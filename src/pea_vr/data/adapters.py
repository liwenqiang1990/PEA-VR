from __future__ import annotations

import csv
import ipaddress
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ITAG_HEIGHT = {18: 360, 22: 720, 133: 240, 134: 360, 135: 480, 136: 720, 137: 1080,
               138: 2160, 160: 144, 242: 240, 243: 360, 244: 480, 247: 720, 248: 1080,
               264: 1440, 266: 2160, 271: 1440, 272: 2160, 278: 144, 298: 720, 299: 1080,
               302: 720, 303: 1080, 308: 1440, 313: 2160, 315: 2160, 330: 144, 331: 240,
               332: 360, 333: 480, 334: 720, 335: 1080, 336: 1440, 337: 2160,
               394: 144, 395: 240, 396: 360, 397: 480, 398: 720, 399: 1080, 400: 1440, 401: 2160}


def _rows(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def epoch_seconds(value):
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def read_ydms_packets(path, retain_seconds=120.0):
    rows = list(_rows(path))
    if not rows:
        raise ValueError('Empty traffic trace')
    counts = Counter(address for row in rows for address in (row['ipSrc'], row['ipDst']))
    private = Counter({text: count for text, count in counts.items()
                       if ipaddress.ip_address(text).is_private and not ipaddress.ip_address(text).is_loopback})
    if not private:
        raise ValueError('No identifiable client address')
    client = private.most_common(1)[0][0]
    selected = [r for r in rows if r['payloadProtocolNumber'] in ('6', '17', '6.0', '17.0')
                and client in (r['ipSrc'], r['ipDst'])]
    times = np.array([float(r['timestamp']) for r in selected], dtype=np.float64)
    direction = np.array([int(r['ipSrc'] == client) for r in selected], dtype=np.uint8)
    lengths = np.array([float(r['tcpLen'] or 0) + float(r['udpLen'] or 0) for r in selected], dtype=np.float32)
    if not len(times):
        raise ValueError('No TCP/UDP video packets')
    order = np.argsort(times, kind='stable')
    times, direction, lengths = times[order], direction[order], lengths[order]
    origin, duration = float(times[0]), float(times[-1] - times[0])
    times = times - origin
    keep = times < retain_seconds
    return times[keep], direction[keep], lengths[keep], origin, duration


def read_longenough_packets(path, retain_seconds=120.0):
    times, directions, lengths = [], [], []
    origin = None
    last = 0.
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = line.strip().split(',')
            if not row or not row[0]:
                continue
            if len(row) < 3:
                raise ValueError('Malformed LongEnough packet row')
            t = float(row[0]) / 1e9
            if origin is None:
                first = t
                origin = float(row[3]) / 1000 if len(row) > 3 else 0.
            t -= first
            last = max(last, t)
            if t >= retain_seconds:
                continue
            direction = row[1].strip().lower()
            if direction not in ('s', 'r', 'sent', 'received', 'send', 'recv'):
                raise ValueError(f'Unknown direction: {direction}')
            times.append(t)
            directions.append(int(direction in ('s', 'sent', 'send')))
            lengths.append(float(row[2]))
    if origin is None:
        raise ValueError('Empty traffic trace')
    order = np.argsort(times, kind='stable')
    return np.asarray(times)[order], np.asarray(directions, dtype=np.uint8)[order], np.asarray(lengths, dtype=np.float32)[order], origin, last


def step_sample(times, values, targets, max_initial_gap=5., initial=None):
    order = np.argsort(times, kind='stable')
    times, values = np.asarray(times)[order], np.asarray(values)[order]
    if not len(times) or not np.isfinite(values).all() or (initial is None and times[0] - targets[0] > max_initial_gap):
        raise ValueError('Missing player quality observations')
    indices = np.searchsorted(times, targets, side='right') - 1
    result = values[np.maximum(indices, 0)].copy()
    if initial is not None:
        result[indices < 0] = initial
    return result


def read_ydms_player(path, origin, window=60.):
    rows = list(_rows(path))
    identities = {r['videoid'].strip() for r in rows if r.get('videoid', '').strip()}
    if len(identities) != 1:
        raise ValueError('Player log must contain exactly one video identity')
    times = np.array([epoch_seconds(r['timestamp']) - origin for r in rows])
    if not np.any((times >= 0) & (times < window)):
        raise ValueError('No player observations within the traffic window')
    quality = []
    for row in rows:
        fmt = int(float(row['fmt']))
        if fmt not in ITAG_HEIGHT:
            raise ValueError(f'Unsupported video itag: {fmt}')
        quality.append(ITAG_HEIGHT[fmt])
    targets = (np.arange(int(window / 2)) + .5) * 2
    q = step_sample(times, quality, targets, initial=0.)
    buffer = np.array([float(r['bh']) / 1000 for r in rows])
    stalled = np.array([float(r['stalling']) for r in rows])
    phase = np.array([r.get('phase', '').lower() for r in rows])
    state = np.isin(phase, ['stalling', 'depletion']).astype(float)
    b = step_sample(times, buffer, targets, initial=0.)
    s = step_sample(times, stalled, targets, initial=1.)
    p = step_sample(times, state, targets, initial=1.)
    features = [float(s.mean()), float((b < 1).mean()), float(np.median(b)),
                float(b[-5:].mean()), float(p.mean()), float(np.mean(q <= 360)),
                float(np.log1p(q).mean()), float(np.mean(q[1:] != q[:-1]))]
    return next(iter(identities)), q.astype(float), np.asarray(features), {
        'times': times.tolist(), 'quality': quality, 'buffer_seconds': buffer.tolist(),
        'stalling': stalled.tolist(), 'phase': phase.tolist()}


def read_longenough_player(path, origin, window=60.):
    events = {}
    state_events = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = line.strip().split(',', 2)
            if len(row) < 2:
                continue
            try:
                t = float(row[0]) / 1000 - origin
            except ValueError:
                continue
            name = row[1].strip()
            if t > 120:
                continue
            if name.startswith('playback') or name in ('bufferStalled', 'bufferLoaded'):
                state_events.append([t, name])
            if len(row) < 3:
                continue
            try:
                value = float(row[2])
            except ValueError:
                continue
            if not np.isfinite(value):
                continue
            if name.lower() in ('btr', 'bitrate', 'buf', 'pbr'):
                events.setdefault(name.lower(), []).append((t, value))
    quality_events = events.get('btr', events.get('bitrate', []))
    if not quality_events:
        raise ValueError('Player log has no bitrate trajectory')
    times, values = zip(*quality_events)
    targets = (np.arange(int(window / 2)) + .5) * 2
    quality = step_sample(times, values, targets)
    q = np.log1p(quality)
    features = np.r_[q, q.mean(), q.std(), q[0], q[-1], np.mean(q[1:] != q[:-1]),
                     np.mean(q <= np.quantile(q, .25)), q[-5:].std()]
    return quality, features, {'events': {k: v for k, v in events.items()}, 'state_events': state_events}


def discover(dataset, root, offsets=(0,)):
    root = Path(root).resolve()
    if dataset == 'ydms':
        for path in sorted(root.rglob('video_traffic.csv')):
            if (path.parent / 'application_data.csv').exists():
                yield {'source_id': path.parent.name, 'traffic': str(path),
                       'player': str(path.parent / 'application_data.csv'), 'condition': 'native'}
    elif dataset == 'longenough':
        for path in sorted(root.rglob('*.log')):
            match = re.fullmatch(r'(\d{4})-(\d{4})-(\d{4})\.log', path.name)
            if not match or int(match[2]) not in offsets:
                continue
            player = path.with_name(path.stem + '.qoe.log')
            if not player.exists():
                raise ValueError(f'Missing player log: {player}')
            bw = path.with_suffix('.bw')
            if not bw.exists():
                raise ValueError(f'Missing bandwidth metadata: {bw}')
            import json
            scale = json.loads(bw.read_text(encoding='utf-8'))['scale']
            yield {'source_id': '/'.join(path.relative_to(root).parts), 'traffic': str(path), 'player': str(player),
                   'identity': match[1], 'offset': int(match[2]), 'repeat': int(match[3]),
                   'condition': f'bw{float(scale):g}'}
    else:
        raise ValueError(dataset)
