from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import load_config
from .utils import read_json, write_json


def select_soft_dtw(matrix, work_root, output):
    matrix, work_root = Path(matrix).resolve(), Path(work_root)
    manifest = read_json(matrix)
    candidates = defaultdict(lambda: defaultdict(list))
    for row in manifest['runs']:
        if 'soft_dtw_validation' not in row['analyses']:
            continue
        status = read_json(work_root / 'runs' / row['name'] / 'status.json')
        if not status['complete'] or status['best']['step'] is None:
            raise ValueError(f'Incomplete smoothing-validation run: {row["name"]}')
        config = load_config(matrix.parent / row['config'])
        candidates[row['dataset']][config['matching']['soft_dtw_gamma']].append(
            (row['seed'], status['best']['accuracy'], row['name']))
    selected = {}
    for dataset, gammas in candidates.items():
        for gamma, records in gammas.items():
            if sorted(r[0] for r in records) != manifest['seeds']:
                raise ValueError(f'Missing matched smoothing-validation seeds: {dataset}, gamma={gamma}')
        if sorted(gammas) != [.01, .05, .1, .2, .5]:
            raise ValueError(f'Incomplete smoothing grid: {dataset}')
        chosen = max(sorted(gammas), key=lambda gamma: np.mean([r[1] for r in gammas[gamma]]))
        selected[dataset] = chosen
    if set(selected) != {'longenough', 'ydms'}:
        raise ValueError('Both datasets require smoothing validation')
    for row in manifest['runs']:
        config_path = (matrix.parent / row['config']).resolve()
        config = load_config(config_path)
        row['config'] = str(config_path)
        if config['matching']['alignment'] == 'soft_dtw':
            if config['matching']['soft_dtw_gamma'] != selected[row['dataset']]:
                row['analyses'] = [a for a in row['analyses'] if a not in ['table2', 'table3', 'truncation']]
            else:
                row['analyses'] = sorted(set(row['analyses'] + ['table2', 'truncation'] + (['table3'] if row['dataset'] == 'longenough' else [])))
    manifest['soft_dtw_selected_gamma'] = selected
    write_json(output, manifest)
    return selected
