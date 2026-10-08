from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .utils import canonical_hash, read_json, read_jsonl, write_json, write_jsonl


def collect_results(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    groups, open_groups, diagnostic_groups = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for path in sorted(root.rglob('*.json')):
        try:
            report = read_json(path)
        except (ValueError, UnicodeDecodeError):
            continue
        if 'config' not in report:
            continue
        config = report['config']
        settings = {key: value for key, value in config.items() if key != 'training'}
        train_settings = {key: value for key, value in config['training'].items() if key not in ['seed', 'device', 'threads']}
        group = canonical_hash([settings, train_settings])
        seed = config['training']['seed']
        if 'accuracy' in report and report.get('split') == 'test':
            key = (group, report['scenario'], report['fraction'])
            if seed in groups[key] and groups[key][seed]['value'] != report['accuracy']:
                raise ValueError(f'Conflicting results for the same configuration/seed: {path}')
            groups[key][seed] = {'value': report['accuracy'], 'config': config, 'source': str(path)}
        elif 'partitions' in report and 'auroc' in report:
            for record in report['partitions']:
                open_groups[group][(seed, record['partition'])] = {'auroc': record['auroc'], 'fpr95': record['fpr95'], 'config': config}
        elif 'alignment_mae_seconds' in report or 'encode_ms' in report or 'spearman_ci95' in report:
            kind = 'correspondence' if 'alignment_mae_seconds' in report else 'benchmark' if 'encode_ms' in report else 'real_overlap'
            report['_source'] = str(path)
            diagnostic_groups[(group, kind)][seed] = report
    rows = []
    for (group, scenario, fraction), records in groups.items():
        values = np.array([r['value'] for _, r in sorted(records.items())])
        config = next(iter(records.values()))['config']
        rows.append({'configuration': group[:12], 'dataset': config['data']['dataset'], 'method': config['model']['kind'],
                     'alignment': config['matching']['alignment'], 'aggregation': config['matching']['aggregation'],
                     'scenario': scenario, 'fraction': fraction, 'seeds': len(values),
                     'mean_accuracy_percent': float(100 * values.mean()),
                     'sample_std_percent': float(100 * values.std(ddof=1)) if len(values) > 1 else None,
                     'complete_three_seeds': len(values) == 3, 'per_seed': {str(seed): r['value'] for seed, r in sorted(records.items())},
                     'config': config})
    open_rows = []
    for group, records in open_groups.items():
        row = {'configuration': group[:12], 'evaluations': len(records), 'complete_15_evaluations': len(records) == 15}
        for key in ['auroc', 'fpr95']:
            values = np.array([r[key] for r in records.values()])
            row[key + '_mean'] = float(values.mean())
            row[key + '_sample_std'] = float(values.std(ddof=1)) if len(values) > 1 else None
        open_rows.append(row)
    write_json(destination / 'recognition.json', rows)
    write_json(destination / 'open_set.json', open_rows)
    diagnostic_rows = []
    fields = {'correspondence': ['alignment_mae_seconds', 'align_at_2s', 'overlap_mae'],
              'benchmark': ['parameters', 'storage_raw_kib', 'storage_serialized_kib', 'encode_ms', 'match_ms'],
              'real_overlap': ['spearman', 'mae']}
    for (group, kind), records in diagnostic_groups.items():
        row = {'configuration': group[:12], 'kind': kind, 'seeds': len(records),
               'config': next(iter(records.values()))['config'], 'sources': [r['_source'] for r in records.values()]}
        for key in fields[kind]:
            values = np.array([r[key] for r in records.values() if r.get(key) is not None])
            row[key + '_mean'] = float(values.mean()) if len(values) else None
            row[key + '_sample_std'] = float(values.std(ddof=1)) if len(values) > 1 else None
        diagnostic_rows.append(row)
        if kind == 'real_overlap':
            for seed, record in records.items():
                pairs = Path(record['_source']).with_suffix('.pairs.jsonl')
                if pairs.exists():
                    write_jsonl(destination / 'real_overlap_pairs' / f'{group[:12]}_{seed}.jsonl', read_jsonl(pairs))
    write_json(destination / 'diagnostics.json', diagnostic_rows)
    if rows:
        with (destination / 'recognition.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            fields = [key for key in rows[0] if key not in ['config', 'per_seed']]
            writer = csv.DictWriter(stream, fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    return {'recognition_configurations': len(rows), 'open_set_configurations': len(open_rows)}


def plot_results(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = read_json(Path(summary) / 'recognition.json')
    for field in ['tokens', 'gap', 'lambda_z']:
        selected = []
        for row in rows:
            c = row['config']
            if row['dataset'] != 'longenough' or row['scenario'] != 'cross_mode' or row['fraction'] != 1 or c['model']['kind'] != 'pea_vr':
                continue
            default_fields = {'tokens': 30, 'gap': .1, 'lambda_z': .5}
            values = {'tokens': c['model']['tokens'], 'gap': c['matching']['gap'], 'lambda_z': c['matching']['lambda_z']}
            if all(values[k] == v for k, v in default_fields.items() if k != field) and c['matching']['aggregation'] == 'compatibility' and c['views']['prefix_probability'] == .5 and c['views']['warp_probability'] == .5 and c['training']['lambda_ali'] == .5 and c['training']['lambda_sup'] == .2 and c['model']['fusion'] == 'learned':
                matched = [row['per_seed'][str(seed)] for seed in [20260814, 20260815] if str(seed) in row['per_seed']]
                if len(matched) == 2:
                    selected.append((values[field], 100 * np.mean(matched), 100 * np.std(matched, ddof=1)))
        if selected:
            selected = sorted(set(selected))
            x, y, error = np.array(selected).T
            fig, ax = plt.subplots(figsize=(4.8, 3.2))
            ax.errorbar(x, y, yerr=error, marker='o', capsize=3)
            ax.set(xlabel=field, ylabel='Cross-mode accuracy (%)')
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(destination / f'sensitivity_{field}.pdf')
            fig.savefig(destination / f'sensitivity_{field}.png', dpi=200)
            plt.close(fig)
    truncation = defaultdict(list)
    for row in rows:
        if row['dataset'] == 'longenough' and row['scenario'] == 'cross_mode':
            from .experiments import method_config
            import copy
            actual = copy.deepcopy(row['config'])
            method = row['alignment'] if row['method'] == 'pea_vr' and row['alignment'] != 'pma' else row['method']
            expected = method_config(method, row['dataset'], actual['training']['seed'])
            if method == 'soft_dtw':
                expected['matching']['soft_dtw_gamma'] = actual['matching']['soft_dtw_gamma']
            for value in [actual, expected]:
                value['training'].pop('device', None)
                value['training'].pop('threads', None)
            if canonical_hash(actual) != canonical_hash(expected):
                continue
            label = f"{row['method']}/{row['alignment']}/{row['aggregation']}"
            truncation[label].append((row['fraction'], row['mean_accuracy_percent']))
    if truncation:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, values in truncation.items():
            values = sorted(set(values))
            if len(values) > 1:
                x, y = np.array(values).T
                ax.plot(x, y, marker='o', label=label)
        ax.set(xlabel='Retained query fraction', ylabel='Accuracy (%)')
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(destination / 'query_truncation.pdf')
        fig.savefig(destination / 'query_truncation.png', dpi=200)
        plt.close(fig)
    for path in (Path(summary) / 'real_overlap_pairs').glob('*.jsonl'):
        pairs = read_jsonl(path)
        true = np.array([r['reference'] for r in pairs])
        predicted = np.array([r['prediction'] for r in pairs])
        errors = np.sort(np.abs(true - predicted))
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
        axes[0].scatter(true, predicted, s=9, alpha=.4)
        axes[0].plot([0, 1], [0, 1], '--', color='black')
        axes[0].set(xlabel='Player-derived overlap', ylabel='PMA overlap', xlim=(0, 1), ylim=(0, 1))
        axes[1].plot(errors, (np.arange(len(errors)) + 1) / len(errors))
        axes[1].set(xlabel='Absolute overlap error', ylabel='Empirical CDF')
        fig.savefig(destination / f'real_overlap_{path.stem}.pdf')
        fig.savefig(destination / f'real_overlap_{path.stem}.png', dpi=200)
        plt.close(fig)
    bandwidth = defaultdict(dict)
    for row in rows:
        c = row['config']
        if not row['scenario'].startswith('bandwidth_') or row['fraction'] != 1:
            continue
        if c['model']['kind'] not in ('pea_vr', 'global_amp') or c['model']['tokens'] != 30 or c['model']['fusion'] != 'learned':
            continue
        if c['model']['kind'] == 'pea_vr' and (c['matching']['alignment'] != 'pma' or c['matching']['aggregation'] != 'compatibility' or c['training']['lambda_ali'] != .5):
            continue
        parts = row['scenario'].split('_')
        a, b = [1, 2, 4, 8].index(int(parts[1])), [1, 2, 4, 8].index(int(parts[3]))
        bandwidth[c['model']['kind']][(a, b)] = row['mean_accuracy_percent']
    for kind, cells in bandwidth.items():
        matrix = np.full((4, 4), np.nan)
        for (a, b), value in cells.items():
            matrix[a, b] = value
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap='viridis')
        for (a, b), value in cells.items():
            text = f'{value:.1f}'
            if kind == 'pea_vr' and (a, b) in bandwidth.get('global_amp', {}):
                text += f'\n{value-bandwidth["global_amp"][(a,b)]:+.1f}'
            ax.text(b, a, text, ha='center', va='center', color='white')
        ax.set(xticks=range(4), yticks=range(4), xticklabels=[1, 2, 4, 8], yticklabels=[1, 2, 4, 8],
               xlabel='Query bandwidth factor', ylabel='Support bandwidth factor')
        fig.colorbar(im, ax=ax, label='Accuracy (%)')
        fig.tight_layout()
        fig.savefig(destination / f'bandwidth_{kind}.pdf')
        fig.savefig(destination / f'bandwidth_{kind}.png', dpi=200)
        plt.close(fig)
