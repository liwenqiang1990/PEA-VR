from pathlib import Path

import numpy as np
import pytest

from pea_vr.data.features import Normalizer, packet_features
from pea_vr.utils import canonical_hash, sha256, write_json, write_jsonl


@pytest.fixture
def prepared_fixture(tmp_path):
    root = tmp_path / 'prepared'
    (root / 'packets').mkdir(parents=True)
    rng = np.random.default_rng(8)
    rows, arrays = [], {s: {'x': [], 'mask': []} for s in [100, 500, 2000]}
    for identity in range(6):
        for repeat in range(12):
            sid = f'{identity:02d}_{repeat:02d}'
            times = np.sort(rng.uniform(0, 60, 80))
            directions = rng.integers(0, 2, 80)
            lengths = rng.integers(50, 1500, 80)
            path = root / 'packets' / (sid + '.npz')
            np.savez_compressed(path, times=times, directions=directions, lengths=lengths)
            rows.append({'id': sid, 'identity': str(identity), 'condition': f'bw{[1, 2, 4, 8][repeat % 4]}',
                         'mode': ['L', 'T', 'H'][repeat % 3], 'split': 'train' if identity < 3 else 'validation' if identity == 3 else 'test',
                         'duration': 100., 'packet_path': 'packets/' + path.name})
            values = packet_features(times, directions, lengths)
            for s in arrays:
                for k in arrays[s]:
                    arrays[s][k].append(values[s][k])
    arrays = {s: {k: np.stack(v) for k, v in item.items()} for s, item in arrays.items()}
    norm = Normalizer().fit(arrays, range(36))
    np.savez_compressed(root / 'features.npz', **{f'{s}_{k}': v for s, item in arrays.items() for k, v in item.items()})
    write_jsonl(root / 'sessions.jsonl', rows)
    write_json(root / 'normalizer.json', norm.state)
    metadata = {'dataset': 'longenough', 'window': 60., 'scales': [100, 500, 2000],
                'exclude_condition': None, 'sessions': len(rows), 'files': {
                    n: sha256(root / n) for n in ['sessions.jsonl', 'normalizer.json', 'features.npz']}}
    metadata['fingerprint'] = canonical_hash(metadata)
    write_json(root / 'dataset.json', metadata)
    return root
