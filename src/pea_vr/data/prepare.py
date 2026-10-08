from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..utils import canonical_hash, read_json, read_jsonl, sha256, write_json, write_jsonl
from .adapters import discover, read_longenough_packets, read_longenough_player, read_ydms_packets, read_ydms_player
from .features import Normalizer, packet_features, validate_scales


def signature(times, direction, lengths, quantum):
    selected = times < 60
    t = np.rint(times[selected] / quantum).astype('<i8')
    d = direction[selected].astype('u1')
    length = np.rint(lengths[selected]).astype('<i4')
    return hashlib.sha256(t.tobytes() + d.tobytes() + length.tobytes()).hexdigest()


def identity_split(identities, counts, seed):
    identities = np.array(sorted(set(identities)))
    if len(counts) != 3 or counts[0] < 1 or any(not isinstance(c, (int, np.integer)) or c < 0 for c in counts):
        raise ValueError('Split counts must contain positive train and nonnegative validation/test integers')
    if len(identities) != sum(counts):
        raise ValueError(f'Expected {sum(counts)} identities, found {len(identities)}')
    shuffled = np.random.default_rng(seed).permutation(identities).tolist()
    a, b = counts[:2]
    return {'train': shuffled[:a], 'validation': shuffled[a:a+b], 'test': shuffled[a+b:]}


def mode_fit(features, train_indices, dataset, clusters=3):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _mode_fit(features, train_indices, dataset, clusters)


def _mode_fit(features, train_indices, dataset, clusters=3):
    train = features[train_indices]
    mean, std = train.mean(0), train.std(0)
    std[std < 1e-6] = 1.
    standardized = (features - mean) / std
    diagnostics = {}
    if dataset == 'longenough':
        for k in range(2, 11):
            if len(train) <= k:
                break
            km = KMeans(k, random_state=20260814, n_init=20).fit(standardized[train_indices])
            if len(set(km.labels_)) == k:
                diagnostics[str(k)] = float(silhouette_score(standardized[train_indices], km.labels_,
                                                            sample_size=min(5000, len(train)), random_state=20260814))
    km = KMeans(clusters, random_state=20260814 if dataset == 'longenough' else 20260816,
                n_init=20).fit(standardized[train_indices])
    raw_centers = km.cluster_centers_ * std + mean
    if dataset == 'longenough':
        quality_order = np.argsort(raw_centers[:, :30].mean(1), kind='stable')
        names = ['L', 'T', 'H']
    else:
        starved = int(np.argmax(raw_centers[:, 0] + raw_centers[:, 1]))
        rest = [k for k in range(3) if k != starved]
        high = max(rest, key=lambda k: raw_centers[k, 6] + .02 * raw_centers[k, 2] - raw_centers[k, 7])
        quality_order = np.array([starved, next(k for k in rest if k != high), high])
        names = ['S', 'T', 'H']
    lookup = {int(k): names[i] for i, k in enumerate(quality_order)}
    modes = [lookup[int(k)] for k in km.predict(standardized)]
    return modes, {'mean': mean.tolist(), 'std': std.tolist(), 'centers': km.cluster_centers_.tolist(),
                   'names': lookup, 'fit_indices': list(map(int, train_indices)),
                   'silhouette': diagnostics, 'silhouette_selected_k': max(diagnostics, key=diagnostics.get) if diagnostics else None}


def prepare_dataset(dataset, source, destination, scales=(100, 500, 2000), exclude_condition=None,
                    counts=None, strict=True, offsets=(0,)):
    expected_counts = [50, 20, 30] if dataset == 'longenough' else [96, 38, 58]
    if strict and counts is not None and list(counts) != expected_counts:
        raise ValueError('Different split counts require --allow-cohort-difference')
    if strict and dataset == 'longenough' and list(offsets) != [0]:
        raise ValueError('The primary LongEnough cohort uses only offset 0')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'dataset.json').exists():
        raise FileExistsError('Prepared dataset exists; choose a new output directory')
    scales = validate_scales(scales)
    packets_dir = destination / 'packets'
    packets_dir.mkdir(exist_ok=True)
    players_dir = destination / 'players'
    players_dir.mkdir(exist_ok=True)
    records_dir = destination / 'records'
    records_dir.mkdir(exist_ok=True)
    parser_hash = canonical_hash([sha256(Path(__file__).with_name('adapters.py')), 2])
    rows, excluded = [], []
    iterator = list(discover(dataset, source, offsets))
    if not iterator:
        raise ValueError('No supported traffic/player pairs were discovered')
    for index, item in enumerate(iterator):
        try:
            sid = canonical_hash([dataset, item['source_id']])[:24]
            record_path = records_dir / (sid + '.json')
            if record_path.exists():
                cached = read_json(record_path)
                if (cached.pop('_parser_hash', None) == parser_hash and
                    cached['traffic_sha256'] == sha256(item['traffic']) and
                    cached['player_sha256'] == sha256(item['player']) and
                    cached['packet_sha256'] == sha256(destination / cached['packet_path']) and
                    cached['player_cache_sha256'] == sha256(destination / cached['player_path'])):
                    rows.append(cached)
                    if index % 100 == 0:
                        print(f'[prepare] {index + 1}/{len(iterator)}; valid={len(rows)} excluded={len(excluded)}', flush=True)
                    continue
            parser = read_ydms_packets if dataset == 'ydms' else read_longenough_packets
            times, directions, lengths, origin, duration = parser(item['traffic'])
            if duration < 60:
                raise ValueError('Traffic duration below 60 seconds')
            if dataset == 'ydms':
                identity, quality, mode_features, player = read_ydms_player(item['player'], origin)
            else:
                identity = item['identity']
                quality, mode_features, player = read_longenough_player(item['player'], origin)
            packet_path = packets_dir / (sid + '.npz')
            player_path = players_dir / (sid + '.json')
            np.savez_compressed(packet_path, times=times, directions=directions, lengths=lengths)
            write_json(player_path, player)
            rows.append({'id': sid, 'identity': str(identity), 'condition': item['condition'],
                         'source_id': item['source_id'], 'duration': duration, 'origin': origin,
                         'offset': item.get('offset'), 'repeat': item.get('repeat'),
                         'packet_path': packet_path.relative_to(destination).as_posix(),
                         'packet_sha256': sha256(packet_path), 'traffic_sha256': sha256(item['traffic']),
                         'player_sha256': sha256(item['player']), 'quality': quality.tolist(),
                         'mode_features': mode_features.tolist(), 'player_path': player_path.relative_to(destination).as_posix(),
                         'player_cache_sha256': sha256(player_path),
                         'exact_signature': signature(times, directions, lengths, 1e-6),
                         'near_signature': signature(times, directions, lengths, 1e-3)})
            write_json(record_path, {'_parser_hash': parser_hash, **rows[-1]})
        except (ValueError, KeyError, OSError) as error:
            excluded.append({'source_id': item['source_id'], 'reason': str(error)})
        if index % 100 == 0 or index == len(iterator) - 1:
            print(f'[prepare] {index + 1}/{len(iterator)}; valid={len(rows)} excluded={len(excluded)}', flush=True)
    # Duplicate groups are formed before identity splitting.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in ('exact_signature', 'near_signature'):
            value = (key, row[key])
            if value in seen:
                parent[find(i)] = find(seen[value])
            seen[value] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    kept = []
    for indices in groups.values():
        conflict = len({rows[i]['identity'] for i in indices}) > 1
        for j, i in enumerate(indices):
            if conflict or j > 0:
                excluded.append({'source_id': rows[i]['source_id'], 'reason': 'conflicting_duplicate' if conflict else 'duplicate'})
            else:
                kept.append(rows[i])
    frequency = Counter(row['identity'] for row in kept)
    minimum = 11 if dataset == 'ydms' else 1
    rows = sorted([row for row in kept if frequency[row['identity']] >= minimum], key=lambda row: row['id'])
    excluded += [{'source_id': row['source_id'], 'reason': 'identity_below_minimum_sessions'}
                 for row in kept if frequency[row['identity']] < minimum]
    write_jsonl(destination / 'excluded.jsonl', excluded)
    write_json(destination / 'cohort_inventory.json', {'discovered': len(iterator), 'sessions': len(rows),
               'identity_session_counts': dict(Counter(row['identity'] for row in rows)),
               'condition_counts': dict(Counter(row['condition'] for row in rows)),
               'exclusion_counts': dict(Counter(row['reason'] for row in excluded))})
    if counts is None:
        if strict:
            counts = [50, 20, 30] if dataset == 'longenough' else [96, 38, 58]
        else:
            total = len({row['identity'] for row in rows})
            training, validation = max(1, total // 2), total // 5
            counts = [training, validation, total - training - validation]
    if strict and dataset == 'longenough' and len(rows) != 4000:
        write_json(destination / 'cohort_failure.json', {'sessions': len(rows), 'identities': len(set(r['identity'] for r in rows)),
                                                       'exclusions': excluded})
        raise ValueError(f'LongEnough cohort requires 4000 sessions, found {len(rows)}; see cohort_failure.json')
    if strict and dataset == 'longenough':
        for identity in {row['identity'] for row in rows}:
            members = [row for row in rows if row['identity'] == identity]
            if Counter(row['condition'] for row in members) != Counter({f'bw{k}': 10 for k in [1, 2, 4, 8]}):
                raise ValueError(f'Incomplete per-identity bandwidth/repeat cohort: {identity}')
            for condition in ['bw1', 'bw2', 'bw4', 'bw8']:
                if sorted(row['repeat'] for row in members if row['condition'] == condition) != list(range(10)):
                    raise ValueError(f'Invalid repeat indices for {identity}, {condition}')
    splits = identity_split([row['identity'] for row in rows], counts, 20260814 if dataset == 'longenough' else 20260816)
    training_indices = [i for i, row in enumerate(rows) if row['identity'] in splits['train']
                        and row['condition'] != exclude_condition]
    if not training_indices:
        raise ValueError('Empty training cohort')
    features = {s: {'x': [], 'mask': []} for s in scales}
    for row in rows:
        with np.load(destination / row['packet_path']) as packets:
            values = packet_features(packets['times'], packets['directions'], packets['lengths'], scales)
        for s in scales:
            for key in ('x', 'mask'):
                features[s][key].append(values[s][key])
    features = {s: {k: np.stack(v) for k, v in item.items()} for s, item in features.items()}
    normalizer = Normalizer().fit(features, training_indices)
    modes, mode_state = mode_fit(np.array([row['mode_features'] for row in rows]), training_indices, dataset)
    for row, mode in zip(rows, modes):
        row['mode'] = mode
        row['split'] = next(name for name, ids in splits.items() if row['identity'] in ids)
    np.savez_compressed(destination / 'features.npz', **{f'{s}_{k}': v for s, item in features.items() for k, v in item.items()})
    write_jsonl(destination / 'sessions.jsonl', rows)
    write_jsonl(destination / 'excluded.jsonl', excluded)
    write_json(destination / 'split.json', splits)
    write_json(destination / 'normalizer.json', normalizer.state)
    write_json(destination / 'modes.json', mode_state)
    cohort = {'dataset': dataset, 'window': 60., 'scales': list(scales), 'sessions': len(rows),
              'identities': len({row['identity'] for row in rows}), 'exclude_condition': exclude_condition, 'counts': counts,
              'mode_counts': dict(Counter(modes)), 'strict': strict,
              'files': {name: sha256(destination / name) for name in
                        ['features.npz', 'sessions.jsonl', 'split.json', 'normalizer.json', 'modes.json']}}
    cohort['fingerprint'] = canonical_hash(cohort)
    write_json(destination / 'dataset.json', cohort)
    return cohort


class PreparedDataset:
    def __init__(self, root, verify=True):
        self.root = Path(root).resolve()
        self.metadata = read_json(self.root / 'dataset.json')
        if verify:
            if self.metadata['fingerprint'] != canonical_hash({key: value for key, value in self.metadata.items() if key != 'fingerprint'}):
                raise ValueError('Prepared dataset metadata fingerprint mismatch')
            for name, expected in self.metadata['files'].items():
                if sha256(self.root / name) != expected:
                    raise ValueError(f'Prepared dataset checksum mismatch: {name}')
        self.rows = read_jsonl(self.root / 'sessions.jsonl')
        if verify:
            for row in self.rows:
                for path_key, hash_key in [('packet_path', 'packet_sha256'), ('player_path', 'player_cache_sha256')]:
                    if hash_key in row and sha256(self.root / row[path_key]) != row[hash_key]:
                        raise ValueError(f'Session cache checksum mismatch: {row["id"]}, {path_key}')
        self.by_id = {row['id']: i for i, row in enumerate(self.rows)}
        self.normalizer = Normalizer(read_json(self.root / 'normalizer.json'))
        with np.load(self.root / 'features.npz') as values:
            raw = {s: {key: values[f'{s}_{key}'] for key in ['x', 'mask']} for s in self.metadata['scales']}
        self.inputs = self.normalizer.transform(raw)

    def batch(self, indices, device, fraction=1., start=0., window=60.):
        import torch
        indices = list(indices)
        if fraction == 1. and start == 0. and window == 60.:
            values = {s: {k: v[indices] for k, v in item.items()} for s, item in self.inputs.items()}
        else:
            collected = {s: {'x': [], 'mask': []} for s in self.metadata['scales']}
            for index in indices:
                with np.load(self.root / self.rows[index]['packet_path']) as packets:
                    raw = packet_features(packets['times'], packets['directions'], packets['lengths'],
                                          self.metadata['scales'], window=window, start=start, observed_until=window * fraction)
                for s, item in raw.items():
                    for k, value in item.items():
                        collected[s][k].append(value)
            values = self.normalizer.transform({s: {k: np.stack(v) for k, v in item.items()} for s, item in collected.items()})
        return {s: {k: torch.from_numpy(v).to(device) for k, v in item.items()} for s, item in values.items()}

    def indices(self, split, condition=None, mode=None):
        return [i for i, row in enumerate(self.rows) if row['split'] == split
                and (condition is None or row['condition'] == condition) and (mode is None or row['mode'] == mode)]
