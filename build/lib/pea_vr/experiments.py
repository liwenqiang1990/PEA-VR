from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from .config import DEFAULTS, validate
from .utils import write_json


SEEDS = [20260814, 20260815, 20260816]


def method_config(method, dataset='longenough', seed=20260814):
    c = deepcopy(DEFAULTS)
    c['data']['dataset'] = dataset
    c['training']['seed'] = seed
    if method in ('protonet', 'deepmetric', 'coda', 'cl_metaflow', 'transformer', 'global_amp'):
        c['model']['kind'] = method
        c['matching'].update(alignment='none', lambda_z=1., aggregation='prototype' if method in ('protonet', 'coda', 'cl_metaflow', 'transformer') else 'support_max')
        c['training']['lambda_ali'] = 0.
        c['training']['lambda_sup'] = .2 if method in ('coda', 'cl_metaflow', 'global_amp') else 0.
        c['views']['warp_probability'] = 0.
        if method != 'global_amp':
            c['views']['prefix_probability'] = 0.
        if method == 'deepmetric':
            c['model']['global_dim'] = 1024
    elif method in ('diagonal', 'soft_dtw'):
        c['matching']['alignment'] = method
        c['matching']['beta_overlap'] = 0.
        c['training']['lambda_ali'] = 0.
    elif method == 'uniform_fusion':
        c['model']['fusion'] = 'uniform'
    elif method == 'global_only':
        c = method_config('global_amp', dataset, seed)
    elif method == 'no_prefix':
        c['views']['prefix_probability'] = 0.
    elif method == 'no_warp':
        c['views']['warp_probability'] = 0.
    elif method == 'no_ali':
        c['training']['lambda_ali'] = 0.
    elif method == 'no_sup':
        c['training']['lambda_sup'] = 0.
    elif method.startswith('aggregation_'):
        c['matching']['aggregation'] = method[len('aggregation_'):]
    elif method != 'pea_vr':
        raise ValueError(method)
    validate(c)
    return c


def experiment_matrix(destination):
    destination = Path(destination)
    configs = destination / 'configs'
    configs.mkdir(parents=True, exist_ok=True)
    runs = {}
    def add(name, config, analyses):
        key = (config['data']['dataset'], config['training']['seed'], yaml.safe_dump(config, sort_keys=True))
        for row in runs.values():
            if row['_key'] == key:
                row['analyses'] = sorted(set(row['analyses'] + analyses))
                return
        filename = name + '.yaml'
        (configs / filename).write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        runs[name] = {'name': name, 'config': f'configs/{filename}', 'dataset': config['data']['dataset'],
                      'seed': config['training']['seed'], 'heldout': config['data']['exclude_condition'],
                      'analyses': analyses, '_key': key}
    methods = ['protonet', 'deepmetric', 'coda', 'cl_metaflow', 'global_amp', 'diagonal', 'soft_dtw', 'pea_vr', 'transformer']
    for dataset in ['longenough', 'ydms']:
        for method in methods:
            for seed in SEEDS:
                add(f'{dataset}_{method}_{seed}', method_config(method, dataset, seed),
                    ['table2', 'truncation'] + (['table3', 'table6_open', 'table7'] if dataset == 'longenough' else []) +
                    (['figure4', 'real_overlap'] if dataset == 'longenough' and method == 'pea_vr' else ['figure4'] if dataset == 'longenough' and method == 'global_amp' else []))
    for method in ['uniform_fusion', 'global_only', 'no_prefix', 'no_warp', 'no_ali', 'no_sup', 'pea_vr']:
        for seed in SEEDS:
            add(f'longenough_{method}_{seed}', method_config(method, seed=seed), ['table4'] + (['table3'] if method in ['no_ali', 'no_warp'] else []))
    for rule in ['mean', 'support_max', 'uniform_lse', 'similarity', 'overlap', 'compatibility']:
        for seed in SEEDS:
            add(f'longenough_aggregation_{rule}_{seed}', method_config('aggregation_' + rule, seed=seed), ['table5'])
    for condition in ['bw1', 'bw8']:
        for method in ['protonet', 'deepmetric', 'coda', 'cl_metaflow', 'global_amp', 'pea_vr']:
            for seed in SEEDS:
                c = method_config(method, seed=seed)
                c['data']['exclude_condition'] = condition
                add(f'longenough_{method}_heldout_{condition}_{seed}', c, ['table6_heldout'])
    for field, values in [('tokens', [15, 20, 30, 40, 60]), ('gap', [0., .05, .1, .2, .4]), ('lambda_z', [0., .25, .5, .75, 1.])]:
        for value in values:
            for seed in SEEDS[:2]:
                c = method_config('pea_vr', seed=seed)
                c['model' if field == 'tokens' else 'matching'][field] = value
                add(f'longenough_{field}_{value}_{seed}', c, ['figure9'])
    for dataset in ['longenough', 'ydms']:
        for gamma in [.01, .05, .1, .2, .5]:
            for seed in SEEDS:
                c = method_config('soft_dtw', dataset, seed)
                c['matching']['soft_dtw_gamma'] = gamma
                add(f'{dataset}_soft_dtw_gamma_{gamma}_{seed}', c, ['soft_dtw_validation'])
    clean = [{k: v for k, v in row.items() if k != '_key'} for row in runs.values()]
    write_json(destination / 'matrix.json', {'seeds': SEEDS, 'runs': clean})
    return clean
