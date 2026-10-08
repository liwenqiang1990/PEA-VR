from collections import defaultdict
from pathlib import Path
import csv

import numpy as np

from .config import load_config
from .utils import canonical_hash, read_json, write_json


def display_method(c):
    names = {'protonet': 'ProtoNet', 'deepmetric': 'DeepMetric', 'coda': 'CoDA',
             'cl_metaflow': 'CL-MetaFlow', 'transformer': 'Enhanced Transformer',
             'global_amp': 'Global AMP + SupportMax'}
    if c['model']['kind'] in names:
        return names[c['model']['kind']]
    return {'diagonal': 'AMP + Diagonal', 'soft_dtw': 'AMP + Soft-DTW', 'pma': 'PEA-VR'}[c['matching']['alignment']]


def ablation_name(c):
    if c['model']['kind'] == 'global_amp':
        return 'Global-only matching'
    if c['model']['fusion'] == 'uniform':
        return 'Uniform scale fusion'
    if c['views']['prefix_probability'] == 0:
        return 'Without prefix'
    if c['views']['warp_probability'] == 0:
        return 'Without progress warp'
    if c['training']['lambda_ali'] == 0:
        return 'Without Lali'
    if c['training']['lambda_sup'] == 0:
        return 'Without Lsup'
    return 'Full'


def export_tables(matrix, root, destination, strict=True):
    matrix, root, destination = Path(matrix).resolve(), Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    groups, missing = defaultdict(dict), []
    def checked_report(path, config):
        report = read_json(path)
        left, right = read_json_copy(config), read_json_copy(report['config'])
        for value in [left, right]:
            value['training'].pop('device', None)
            value['training'].pop('threads', None)
        if canonical_hash(left) != canonical_hash(right):
            raise ValueError(f'Report configuration differs from experiment matrix: {path}')
        return report
    def add(table, dataset, label, scenario, metric, value, scale, identity, source, expected=3):
        if value is not None:
            groups[(table, dataset, label, scenario, metric, scale, expected)][identity] = (float(value) * scale, str(source))
    for row in read_json(matrix)['runs']:
        run = root / row['name']
        if not (run / 'status.json').exists() or not read_json(run / 'status.json')['complete']:
            missing.append({'run': row['name'], 'reason': 'training_incomplete'})
            continue
        c = load_config(matrix.parent / row['config'])
        if read_json(run / 'status.json')['episode'] != c['training']['episodes']:
            raise ValueError(f'Training budget does not match the experiment matrix: {run}')
        seed, dataset = row['seed'], row['dataset']
        reports = run / 'evaluation'
        method = display_method(c)
        for analysis in row['analyses']:
            specs = []
            if analysis == 'table2':
                scenarios = ['cross_mode', 'same_mode', 'mode_balanced', 'cross_bandwidth'] if dataset == 'longenough' else ['random_mixed', 'T_to_H', 'H_to_T']
                specs = [(method, scenario) for scenario in scenarios]
            elif analysis == 'table4':
                specs = [(ablation_name(c), 'cross_mode')]
            elif analysis == 'table5':
                specs = [(c['matching']['aggregation'], scenario) for scenario in ['cross_mode', 'mode_balanced', 'random_mixed']]
            elif analysis == 'table6_heldout':
                specs = [(method, 'heldout_' + row['heldout'])]
            elif analysis == 'figure9' and seed in [20260814, 20260815]:
                values = {'tokens': c['model']['tokens'], 'gap': c['matching']['gap'], 'lambda_z': c['matching']['lambda_z']}
                defaults = {'tokens': 30, 'gap': .1, 'lambda_z': .5}
                for field, value in values.items():
                    if all(values[key] == default for key, default in defaults.items() if key != field):
                        specs.append((f'{field}={value:g}', 'cross_mode'))
            for label, scenario in specs:
                path = reports / 'recognition' / f'{scenario}_rho1.json'
                if path.exists():
                    report = checked_report(path, c)
                    expected_episodes = 500 if scenario in ['T_to_H', 'H_to_T'] else 300
                    if report['episodes'] != expected_episodes:
                        missing.append({'run': row['name'], 'analysis': analysis, 'reason': 'episode_count_mismatch', 'file': str(path)})
                        continue
                    add(analysis, dataset, label, scenario, 'accuracy_percent', report['accuracy'], 100, seed, path,
                        expected=2 if analysis == 'figure9' else 3)
                else:
                    missing.append({'run': row['name'], 'analysis': analysis, 'file': str(path)})
            if analysis == 'table3' and c['matching']['alignment'] != 'none':
                path = reports / 'correspondence.json'
                if path.exists():
                    report = checked_report(path, c)
                    label = method if c['matching']['alignment'] != 'pma' else 'PMA ' + ablation_name(c)
                    for metric, scale in [('alignment_mae_seconds', 1), ('align_at_2s', 100), ('overlap_mae', 1)]:
                        add('table3', dataset, label, 'transformed_views', metric, report[metric], scale, seed, path)
                else:
                    missing.append({'run': row['name'], 'analysis': analysis, 'file': str(path)})
            if analysis == 'table6_open':
                path = reports / 'open_set.json'
                if path.exists():
                    report = checked_report(path, c)
                    if len(report['partitions']) != 5:
                        missing.append({'run': row['name'], 'analysis': analysis, 'reason': 'open_partition_count_mismatch'})
                    for partition in report['partitions']:
                        for metric in ['auroc', 'fpr95']:
                            add('table6_open', dataset, method, 'open_set', metric, partition[metric], 1,
                                (seed, partition['partition']), path, expected=15)
                else:
                    missing.append({'run': row['name'], 'analysis': analysis, 'file': str(path)})
            if analysis == 'table7' and c['model']['kind'] in ['deepmetric', 'global_amp', 'pea_vr'] and c['matching']['alignment'] in ['none', 'pma']:
                path = reports / 'benchmark.json'
                if path.exists():
                    report = checked_report(path, c)
                    for metric in ['parameters', 'storage_raw_kib', 'storage_serialized_kib', 'encode_ms', 'match_ms']:
                        add('table7', dataset, method, report['environment']['gpu'] or 'CPU', metric, report[metric], 1, seed, path)
                else:
                    missing.append({'run': row['name'], 'analysis': analysis, 'file': str(path)})
    output = []
    for (table, dataset, method, scenario, metric, scale, expected), values in sorted(groups.items()):
        array = np.array([item[0] for item in values.values()])
        output.append({'table': table, 'dataset': dataset, 'method': method, 'scenario': scenario, 'metric': metric,
                       'mean': float(array.mean()), 'sample_std': float(array.std(ddof=1)) if len(array) > 1 else None,
                       'measurements': len(array), 'expected_measurements': expected, 'complete': len(array) == expected,
                       'sources': sorted({item[1] for item in values.values()})})
    write_json(destination / 'tables.json', output)
    write_json(destination / 'missing.json', missing)
    for table in sorted({r['table'] for r in output}):
        records = [r for r in output if r['table'] == table]
        fields = [key for key in records[0] if key != 'sources']
        with (destination / (table + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
    complete = bool(output) and not missing and all(row['complete'] for row in output)
    write_json(destination / 'status.json', {'complete': complete, 'missing': len(missing), 'incomplete_rows': sum(not r['complete'] for r in output)})
    if strict and not complete:
        raise ValueError(f'Publication tables require complete training seeds and episode counts; see {destination / "missing.json"}')
    return {'measurements': len(output), 'missing_runs_or_reports': len(missing)}


def read_json_copy(value):
    import copy
    return copy.deepcopy(value)
