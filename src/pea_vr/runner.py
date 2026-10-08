from pathlib import Path

from .config import load_config
from .data.prepare import PreparedDataset, prepare_dataset
from .data.protocols import build_protocols
from .evaluation import benchmark, correspondence, evaluate, evaluate_open, real_overlap
from .training.engine import train
from .utils import read_json, write_json


def run_matrix(matrix, raw_root, work_root, names=None, analyses=None, device='auto', max_runs=None):
    matrix = Path(matrix).resolve()
    raw_root, work_root = Path(raw_root), Path(work_root)
    rows = read_json(matrix)['runs']
    selected = [r for r in rows if (names is None or r['name'] in names) and
                (analyses is None or set(r['analyses']) & set(analyses))]
    if max_runs is not None:
        selected = selected[:max_runs]
    if not selected:
        raise ValueError('No matrix runs selected')
    for row in selected:
        config = load_config(matrix.parent / row['config'], {'training': {'device': device}})
        cohort = row['dataset'] + ('_heldout_' + row['heldout'] if row['heldout'] else '')
        prepared = work_root / 'prepared' / cohort
        if not (prepared / 'dataset.json').exists():
            prepare_dataset(row['dataset'], raw_root / row['dataset'], prepared, exclude_condition=row['heldout'])
        dataset = PreparedDataset(prepared)
        protocol_dirs = {}
        for split in ['validation', 'test']:
            directory = work_root / 'protocols' / cohort / split
            if not (directory / 'protocols.json').exists():
                build_protocols(dataset, directory, split, bandwidth_matrix=split == 'test' and row['dataset'] == 'longenough' and row['heldout'] is None)
            protocol_dirs[split] = directory
        run = work_root / 'runs' / row['name']
        status_file = run / 'status.json'
        if not status_file.exists() or not read_json(status_file)['complete']:
            checkpoint = run / 'latest.pt'
            train(config, prepared, run, protocol_dirs['validation'], resume=checkpoint if checkpoint.exists() else None)
        checkpoint = run / 'best.pt'
        if not checkpoint.exists():
            raise ValueError(f'No validation-selected checkpoint: {run}')
        reports = run / 'evaluation'
        requested = set(row['analyses']) if analyses is None else set(row['analyses']) & set(analyses)
        completed = set(read_json(reports / 'complete.json')['analyses']) if (reports / 'complete.json').exists() else set()
        if not requested <= completed:
            if requested == {'soft_dtw_validation'}:
                write_json(reports / 'complete.json', {'run': row['name'], 'analyses': sorted(completed | requested)})
                continue
            from .data.protocols import protocol_specs
            scenarios = list(protocol_specs(row['dataset'], row['heldout'], bandwidth_matrix='figure4' in requested))
            missing = [scenario for scenario in scenarios if not (reports / 'recognition' / f'{scenario}_rho1.json').exists()]
            if missing:
                evaluate(checkpoint, prepared, protocol_dirs['test'], reports / 'recognition', scenarios=missing, device=device)
            primary = 'heldout_' + row['heldout'] if row['heldout'] else ('cross_mode' if row['dataset'] == 'longenough' else 'random_mixed')
            if 'truncation' in requested and 'truncation' not in completed:
                evaluate(checkpoint, prepared, protocol_dirs['test'], reports / 'truncation', [primary], [.2, .3, .5, .7, 1.], device)
            if 'table6_open' in requested and not (reports / 'open_set.json').exists() and row['dataset'] == 'longenough' and row['heldout'] is None:
                evaluate_open(checkpoint, prepared, protocol_dirs['test'], reports / 'open_set.json', device)
            if 'table3' in requested and not (reports / 'correspondence.json').exists() and config['matching']['alignment'] != 'none':
                correspondence(checkpoint, prepared, reports / 'correspondence.json', device=device)
            if 'table7' in requested and not (reports / 'benchmark.json').exists() and config['model']['kind'] in ['deepmetric', 'global_amp', 'pea_vr'] and config['matching']['alignment'] in ['none', 'pma']:
                benchmark(checkpoint, prepared, reports / 'benchmark.json', device=device)
            if 'real_overlap' in requested and not (reports / 'real_overlap.json').exists() and config['matching']['alignment'] == 'pma':
                real_overlap(checkpoint, prepared, reports / 'real_overlap.json', device=device)
            write_json(reports / 'complete.json', {'run': row['name'], 'analyses': sorted(requested | completed)})
    return {'completed_runs': len(selected)}
