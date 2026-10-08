from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from .utils import canonical_hash

DEFAULTS = {
    'schema_version': 2,
    'data': {'dataset': 'longenough', 'window': 60.0, 'scales': [100, 500, 2000], 'exclude_condition': None},
    'model': {'kind': 'pea_vr', 'channels': [128, 256], 'tokens': 30, 'global_dim': 512,
              'projection_hidden': 1024, 'alignment_dim': 128, 'dropout': 0.1, 'fusion': 'learned'},
    'matching': {'alignment': 'pma', 'aggregation': 'compatibility', 'delta': 0.20, 'gap': 0.10,
                 'gamma': 0.10, 'soft_dtw_gamma': 0.10, 'lambda_z': 0.50,
                 'beta_overlap': 1.0, 'beta_similarity': 1.0, 'tau_c': 0.20, 'tau_s': 0.10,
                 'epsilon': 1e-8, 'pair_chunk': 128},
    'views': {'prefix_probability': 0.5, 'prefix_min': 0.3, 'warp_probability': 0.5, 'warp_max': 0.3},
    'training': {'seed': 20260814, 'episodes': 30000, 'ways': 10, 'shots': 2, 'queries': 4,
                 'lr': 1e-4, 'min_lr': 1e-6, 'weight_decay': 1e-4, 'betas': [0.9, 0.999],
                 'adam_epsilon': 1e-8, 'gradient_clip': 1.0, 'tau_e': 0.10, 'tau_sup': 0.10,
                 'lambda_ali': 0.50, 'lambda_sup': 0.20, 'sigma_w': 2.0,
                 'eval_every': 500, 'save_every': 1000, 'keep_checkpoints': 3, 'log_every': 10, 'query_chunk': 4,
                 'alignment_chunk': 8, 'deterministic': True, 'device': 'auto', 'threads': 4},
    'baseline': {'triplet_margin': 0.2, 'prototype_weight': 0.2, 'domain_weight': 0.1,
                 'pretrain_episodes': 10000, 'inner_steps': 5, 'inner_lr': 0.01},
}


def merge(base: dict, update: dict) -> dict:
    value = deepcopy(base)
    for key, item in update.items():
        if key not in value:
            raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(value[key], dict) and not isinstance(item, dict):
            raise ValueError(f'Expected a mapping for configuration key: {key}')
        value[key] = merge(value[key], item) if isinstance(value[key], dict) else deepcopy(item)
    return value


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    config = deepcopy(DEFAULTS)
    if path is not None:
        with Path(path).open(encoding='utf-8') as stream:
            config = merge(config, yaml.safe_load(stream) or {})
    if overrides:
        config = merge(config, overrides)
    validate(config)
    return config


def validate(c: dict) -> None:
    from .data.features import validate_scales
    import math
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite(c):
        raise ValueError('Configuration values must be finite')
    validate_scales(c['data']['scales'], c['data']['window'])
    m, v, t = c['matching'], c['views'], c['training']
    if c['schema_version'] != 2 or c['data']['window'] != 60.:
        raise ValueError('Schema 2 uses a 60-second observation window')
    if c['data']['exclude_condition'] not in (None, 'bw1', 'bw8'):
        raise ValueError('Unsupported held-out bandwidth')
    model = c['model']
    if len(model['channels']) != 2 or min(model['channels']) < 1 or min(model[k] for k in ['tokens', 'global_dim', 'projection_hidden', 'alignment_dim']) < 1 or not 0 <= model['dropout'] < 1:
        raise ValueError('Invalid encoder dimensions or dropout')
    if c['data']['dataset'] not in ('longenough', 'ydms'):
        raise ValueError('dataset must be longenough or ydms')
    if c['model']['kind'] not in ('pea_vr', 'global_amp', 'protonet', 'deepmetric', 'coda', 'cl_metaflow', 'transformer'):
        raise ValueError('Unknown model kind')
    if c['model']['kind'] in ('protonet', 'deepmetric', 'coda', 'cl_metaflow') and 100 not in c['data']['scales']:
        raise ValueError('This adapted baseline requires the 100 ms input scale')
    if c['model']['kind'] == 'transformer' and 500 not in c['data']['scales']:
        raise ValueError('Transformer requires the 500 ms input scale')
    if c['model']['fusion'] not in ('learned', 'uniform'):
        raise ValueError('Unknown fusion')
    if m['alignment'] not in ('pma', 'diagonal', 'soft_dtw', 'none'):
        raise ValueError('Unknown alignment')
    if m['aggregation'] not in ('compatibility', 'mean', 'support_max', 'uniform_lse', 'similarity', 'overlap', 'prototype'):
        raise ValueError('Unknown aggregation')
    for key in ('gamma', 'soft_dtw_gamma', 'tau_c', 'tau_s', 'epsilon'):
        if m[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if m['pair_chunk'] < 1:
        raise ValueError('pair_chunk must be positive')
    if not 0 <= m['lambda_z'] <= 1 or min(m['gap'], m['beta_overlap'], m['beta_similarity']) < 0:
        raise ValueError('Invalid matching weights')
    for key in ('prefix_probability', 'warp_probability'):
        if not 0 <= v[key] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    if not 0 < v['prefix_min'] <= 1 or not 0 <= v['warp_max'] < 0.5:
        raise ValueError('Invalid view ranges')
    for key in ('episodes', 'ways', 'shots', 'queries', 'query_chunk', 'alignment_chunk', 'eval_every', 'save_every', 'keep_checkpoints', 'log_every', 'threads'):
        if t[key] < 1:
            raise ValueError(f'{key} must be positive')
    if t['ways'] < 2 or t['lr'] <= 0 or not 0 < t['min_lr'] <= t['lr']:
        raise ValueError('Invalid training configuration')
    if t['gradient_clip'] <= 0 or t['adam_epsilon'] <= 0 or t['weight_decay'] < 0 or len(t['betas']) != 2 or any(not 0 <= b < 1 for b in t['betas']):
        raise ValueError('Invalid optimizer configuration')
    if c['baseline']['inner_steps'] < 1 or c['baseline']['inner_lr'] <= 0 or c['baseline']['pretrain_episodes'] < 0:
        raise ValueError('Invalid baseline adaptation configuration')
    if min(t['lambda_ali'], t['lambda_sup']) < 0 or min(t['tau_e'], t['tau_sup'], t['sigma_w']) <= 0:
        raise ValueError('Invalid objective configuration')
    if c['model']['kind'] != 'pea_vr' and t['lambda_ali'] != 0:
        raise ValueError('Alignment supervision requires pea_vr')
    if m['alignment'] != 'pma' and t['lambda_ali'] != 0:
        raise ValueError('Alignment supervision requires PMA')


def config_hash(config: dict) -> str:
    return canonical_hash(config)
