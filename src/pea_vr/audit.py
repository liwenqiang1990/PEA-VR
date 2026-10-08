from collections import Counter
from pathlib import Path

from .data.prepare import PreparedDataset
from .data.protocols import validate_episode, verify_protocol_manifest
from .utils import read_json, read_jsonl, sha256, write_json


def audit_data(data, protocols=None, output=None):
    dataset = PreparedDataset(data, verify=True)
    identities = {split: {r['identity'] for r in dataset.rows if r['split'] == split}
                  for split in ['train', 'validation', 'test']}
    for a in identities:
        for b in identities:
            if a != b and identities[a] & identities[b]:
                raise ValueError('Identity leakage across dataset splits')
    for state in [dataset.normalizer.state, read_json(dataset.root / 'modes.json')]:
        for index in state['fit_indices']:
            row = dataset.rows[index]
            if row['split'] != 'train' or row['condition'] == dataset.metadata['exclude_condition']:
                raise ValueError('Training-only transform contains held-out data')
    signatures = set()
    for row in dataset.rows:
        if sha256(dataset.root / row['packet_path']) != row['packet_sha256']:
            raise ValueError(f'Packet cache checksum mismatch: {row["id"]}')
        if sha256(dataset.root / row['player_path']) != row['player_cache_sha256']:
            raise ValueError(f'Player cache checksum mismatch: {row["id"]}')
        for key in ['exact_signature', 'near_signature']:
            signature = (key, row[key])
            if signature in signatures:
                raise ValueError('Duplicate packet signatures retained')
            signatures.add(signature)
    episodes = 0
    if protocols:
        manifest = verify_protocol_manifest(dataset, protocols)
        for scenario, count in manifest['protocols'].items():
            rows = read_jsonl(Path(protocols) / (scenario + '.jsonl'))
            if len(rows) != count:
                raise ValueError('Episode count mismatch')
            for episode in rows:
                validate_episode(dataset, episode)
                for sid in episode['support'] + episode['query']:
                    if dataset.rows[dataset.by_id[sid]]['split'] != manifest['split']:
                        raise ValueError('Protocol split mismatch')
            episodes += len(rows)
    result = {'passed': True, 'dataset_fingerprint': dataset.metadata['fingerprint'], 'sessions': len(dataset.rows),
              'identity_counts': {k: len(v) for k, v in identities.items()}, 'episodes_checked': episodes,
              'mode_counts': dict(Counter(r['mode'] for r in dataset.rows))}
    if output:
        write_json(output, result)
    return result
