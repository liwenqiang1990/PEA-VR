from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils import derived_seed, read_json, sha256, write_json, write_jsonl


class EpisodeSampler:
    def __init__(self, dataset, split='train', seed=20260814, exclude_condition=None):
        self.dataset, self.split = dataset, split
        self.rng = np.random.default_rng(seed)
        self.rows = [i for i, r in enumerate(dataset.rows) if r['split'] == split and r['condition'] != exclude_condition]
        self.groups = defaultdict(list)
        for i in self.rows:
            self.groups[dataset.rows[i]['identity']].append(i)

    def state_dict(self):
        return self.rng.bit_generator.state

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state

    def sample(self, ways, shots, queries, support_mode=None, query_mode=None,
               support_condition=None, query_condition=None, balanced=False):
        supports, query_pools = {}, {}
        for identity, indices in self.groups.items():
            supports[identity] = [i for i in indices if
                                  (support_mode is None or self.dataset.rows[i]['mode'] == support_mode) and
                                  (support_condition is None or self.dataset.rows[i]['condition'] == support_condition)]
            query_pools[identity] = [i for i in indices if
                                     (query_mode is None or self.dataset.rows[i]['mode'] == query_mode) and
                                     (query_condition is None or self.dataset.rows[i]['condition'] == query_condition)]
        eligible = []
        for identity in sorted(self.groups):
            a, b = set(supports[identity]), set(query_pools[identity])
            feasible = len(a) >= shots and len(b) >= queries and len(a | b) >= shots + queries
            if balanced:
                feasible = all(sum(self.dataset.rows[i]['mode'] == m for i in self.groups[identity]) >= 2 for m in ['L', 'T', 'H'])
            if feasible:
                eligible.append(identity)
        if len(eligible) < ways:
            raise ValueError(f'Protocol needs {ways} eligible identities, found {len(eligible)}; '
                             f'support={support_mode or support_condition}, query={query_mode or query_condition}')
        identities = self.rng.choice(eligible, ways, replace=False).tolist()
        support, query, sy, qy = [], [], [], []
        for label, identity in enumerate(identities):
            if balanced:
                selected_support, selected_query = [], []
                for mode in ['L', 'T', 'H']:
                    candidates = [i for i in self.groups[identity] if self.dataset.rows[i]['mode'] == mode]
                    a, b = self.rng.choice(candidates, 2, replace=False).tolist()
                    selected_support.append(a)
                    selected_query.append(b)
            else:
                a, b = supports[identity], query_pools[identity]
                # Reserve query-only records before sampling the shared pool.
                b_only = [i for i in b if i not in set(a)]
                shared = [i for i in b if i in set(a)]
                query_only_count = min(len(b_only), queries)
                selected_query = self.rng.choice(b_only, query_only_count, replace=False).tolist() if query_only_count else []
                remainder = queries - query_only_count
                selected_query += self.rng.choice(shared, remainder, replace=False).tolist() if remainder else []
                available = [i for i in a if i not in set(selected_query)]
                selected_support = self.rng.choice(available, shots, replace=False).tolist()
            support += selected_support
            query += selected_query
            sy += [label] * len(selected_support)
            qy += [label] * len(selected_query)
        return {'identities': identities, 'support': [self.dataset.rows[i]['id'] for i in support],
                'query': [self.dataset.rows[i]['id'] for i in query], 'support_labels': sy, 'query_labels': qy,
                'support_mode': support_mode, 'query_mode': query_mode,
                'support_condition': support_condition, 'query_condition': query_condition}


def protocol_specs(dataset, heldout=None, bandwidth_matrix=False):
    if dataset == 'ydms':
        return {'random_mixed': [(300, 10, 5, 5, {})],
                'T_to_H': [(500, 6, 5, 5, {'support_mode': 'T', 'query_mode': 'H'})],
                'H_to_T': [(500, 6, 5, 5, {'support_mode': 'H', 'query_mode': 'T'})]}
    if heldout:
        return {'heldout_' + heldout: [(300, 5, 2, 2, {'support_condition': heldout, 'query_condition': heldout})]}
    modes = ['L', 'T', 'H']
    cross = [(50, 5, 2, 2, {'support_mode': a, 'query_mode': b}) for a in modes for b in modes if a != b]
    same = [(100, 5, 2, 2, {'support_mode': m, 'query_mode': m}) for m in modes]
    conditions = [('bw1', 'bw8'), ('bw8', 'bw1'), ('bw1', 'bw2'), ('bw2', 'bw1')]
    specs = {'cross_mode': cross, 'same_mode': same,
             'mode_balanced': [(300, 5, 3, 3, {'balanced': True})],
             'random_mixed': [(300, 5, 5, 5, {})],
             'cross_bandwidth': [(75, 5, 2, 2, {'support_condition': a, 'query_condition': b}) for a, b in conditions]}
    if bandwidth_matrix:
        for a in [1, 2, 4, 8]:
            for b in [1, 2, 4, 8]:
                specs[f'bandwidth_{a}_to_{b}'] = [(300, 5, 2, 2, {'support_condition': f'bw{a}', 'query_condition': f'bw{b}'})]
    return specs


def build_protocols(dataset, destination, split='test', seed=20260817, bandwidth_matrix=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    heldout = dataset.metadata['exclude_condition']
    specs = protocol_specs(dataset.metadata['dataset'], heldout, bandwidth_matrix)
    manifest = {'dataset_fingerprint': dataset.metadata['fingerprint'], 'split': split, 'seed': seed,
                'protocols': {}, 'heldout': heldout, 'files': {}}
    for scenario, parts in specs.items():
        episodes = []
        for partition, (count, ways, shots, queries, filters) in enumerate(parts):
            sampler = EpisodeSampler(dataset, split, derived_seed(seed, dataset.metadata['dataset'], split, scenario, partition))
            for _ in range(count):
                episode = sampler.sample(ways, shots, queries, **filters)
                episode.update({'scenario': scenario, 'index': len(episodes), 'partition': partition})
                episodes.append(episode)
        write_jsonl(destination / (scenario + '.jsonl'), episodes)
        manifest['protocols'][scenario] = len(episodes)
        manifest['files'][scenario + '.jsonl'] = sha256(destination / (scenario + '.jsonl'))
    if dataset.metadata['dataset'] == 'longenough' and split == 'test' and heldout is None:
        manifest['open_set'] = build_open_partitions(dataset, destination, seed)
        manifest['files']['open_set.jsonl'] = sha256(destination / 'open_set.jsonl')
    write_json(destination / 'protocols.json', manifest)
    return manifest


def build_open_partitions(dataset, destination, seed=20260817):
    identities = sorted({r['identity'] for r in dataset.rows if r['split'] == 'test'})
    if len(identities) != 30:
        raise ValueError('Open-set protocol requires 30 test identities')
    records = []
    for partition in range(5):
        rng = np.random.default_rng(derived_seed(seed, 'longenough', 'open', partition))
        known = set(rng.choice(identities, 15, replace=False).tolist())
        supports, queries, sy, qy = [], [], [], []
        labels = {identity: i for i, identity in enumerate(sorted(known))}
        for identity in identities:
            indices = [i for i, row in enumerate(dataset.rows) if row['identity'] == identity]
            gallery = rng.choice(indices, 5, replace=False).tolist() if identity in known else []
            supports += [dataset.rows[i]['id'] for i in gallery]
            if gallery:
                sy += [labels[identity]] * len(gallery)
            remaining = [i for i in indices if i not in gallery]
            queries += [dataset.rows[i]['id'] for i in remaining]
            qy += [labels.get(identity, -1)] * len(remaining)
        records.append({'partition': partition, 'known': sorted(known), 'unknown': sorted(set(identities) - known),
                        'support': supports, 'support_labels': sy, 'query': queries, 'query_labels': qy})
    write_jsonl(Path(destination) / 'open_set.jsonl', records)
    return len(records)


def validate_episode(dataset, episode):
    support, query = episode['support'], episode['query']
    if len(set(support)) != len(support) or len(set(query)) != len(query) or set(support) & set(query):
        raise ValueError('Repeated or overlapping support/query records')
    for group, label_key in [('support', 'support_labels'), ('query', 'query_labels')]:
        if len(episode[group]) != len(episode[label_key]):
            raise ValueError('Unaligned episode labels')
        for sid, label in zip(episode[group], episode[label_key]):
            if label < (-1 if group == 'query' and 'known' in episode else 0):
                raise ValueError('Invalid episode class label')
            if sid not in dataset.by_id:
                raise ValueError(f'Unknown session id {sid}')
            if label >= 0 and 'identities' in episode:
                if dataset.rows[dataset.by_id[sid]]['identity'] != episode['identities'][label]:
                    raise ValueError('Episode label/identity mismatch')
            row = dataset.rows[dataset.by_id[sid]]
            if 'known' in episode:
                expected_label = episode['known'].index(row['identity']) if row['identity'] in episode['known'] else -1
                if label != expected_label:
                    raise ValueError('Open-set identity/label mismatch')
            for field in ['mode', 'condition']:
                expected = episode.get(group + '_' + field)
                if expected is not None and row[field] != expected:
                    raise ValueError(f'Episode {group} {field} mismatch')


def verify_protocol_manifest(dataset, directory):
    manifest = read_json(Path(directory) / 'protocols.json')
    if manifest['dataset_fingerprint'] != dataset.metadata['fingerprint']:
        raise ValueError('Protocol dataset fingerprint mismatch')
    for name, expected in manifest['files'].items():
        if sha256(Path(directory) / name) != expected:
            raise ValueError(f'Protocol checksum mismatch: {name}')
    return manifest
