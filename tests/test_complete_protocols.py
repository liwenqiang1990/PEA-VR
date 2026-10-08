from types import SimpleNamespace

from pea_vr.data.protocols import build_protocols, verify_protocol_manifest
from pea_vr.utils import read_jsonl


def test_complete_longenough_protocol_counts_and_content(tmp_path):
    rows = []
    for identity in range(30):
        for condition in [1, 2, 4, 8]:
            for repeat in range(10):
                rows.append({'id': f'{identity}_{condition}_{repeat}', 'identity': str(identity),
                             'condition': f'bw{condition}', 'mode': ['L', 'T', 'H'][repeat % 3], 'split': 'test'})
    data = SimpleNamespace(rows=rows, by_id={r['id']: i for i, r in enumerate(rows)},
                           metadata={'dataset': 'longenough', 'fingerprint': 'protocol-test', 'exclude_condition': None})
    manifest = build_protocols(data, tmp_path, bandwidth_matrix=True)
    assert all(manifest['protocols'][name] == 300 for name in ['cross_mode', 'same_mode', 'mode_balanced', 'random_mixed', 'cross_bandwidth'])
    assert len([name for name in manifest['protocols'] if name.startswith('bandwidth_')]) == 16
    cross = read_jsonl(tmp_path / 'cross_mode.jsonl')
    pairs = {}
    for episode in cross:
        key = episode['support_mode'], episode['query_mode']
        pairs[key] = pairs.get(key, 0) + 1
        assert key[0] != key[1]
        assert len(episode['support']) == len(episode['query']) == 10
        assert not set(episode['support']) & set(episode['query'])
    assert len(pairs) == 6 and set(pairs.values()) == {50}
    bw_pairs = {}
    for episode in read_jsonl(tmp_path / 'cross_bandwidth.jsonl'):
        key = episode['support_condition'], episode['query_condition']
        bw_pairs[key] = bw_pairs.get(key, 0) + 1
    assert bw_pairs == {('bw1', 'bw8'): 75, ('bw8', 'bw1'): 75, ('bw1', 'bw2'): 75, ('bw2', 'bw1'): 75}
    opened = read_jsonl(tmp_path / 'open_set.jsonl')
    assert len(opened) == 5
    for partition in opened:
        assert len(partition['known']) == len(partition['unknown']) == 15
        assert len(partition['support']) == 75 and len(partition['query']) == 1125
        assert not set(partition['support']) & set(partition['query'])
        assert sum(y == -1 for y in partition['query_labels']) == 600
    verify_protocol_manifest(data, tmp_path)


def test_complete_ydms_protocol_counts_and_modes(tmp_path):
    rows = []
    for identity in range(58):
        for repeat, mode in enumerate(['S'] + ['T'] * 5 + ['H'] * 5):
            rows.append({'id': f'{identity}_{repeat}', 'identity': str(identity), 'condition': 'native', 'mode': mode, 'split': 'test'})
    data = SimpleNamespace(rows=rows, by_id={r['id']: i for i, r in enumerate(rows)},
                           metadata={'dataset': 'ydms', 'fingerprint': 'protocol-test', 'exclude_condition': None})
    manifest = build_protocols(data, tmp_path)
    assert manifest['protocols'] == {'random_mixed': 300, 'T_to_H': 500, 'H_to_T': 500}
    for name in ['T_to_H', 'H_to_T']:
        for episode in read_jsonl(tmp_path / (name + '.jsonl')):
            assert len(episode['identities']) == 6
            assert len(episode['support']) == len(episode['query']) == 30
            assert not set(episode['support']) & set(episode['query'])
            for sid in episode['support'] + episode['query']:
                assert data.rows[data.by_id[sid]]['mode'] in ['T', 'H']
