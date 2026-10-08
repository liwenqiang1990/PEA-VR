from __future__ import annotations

import argparse
import json
import sys


def parser():
    p = argparse.ArgumentParser(prog='pea-vr', description='PEA-VR training, evaluation, and enrollment')
    p.add_argument('--version', action='version', version='PEA-VR 2.0.0')
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('download', help='Acquire public datasets with checksums and resume')
    c.add_argument('dataset', choices=['longenough', 'ydms'])
    c.add_argument('--output', required=True)
    c.add_argument('--no-extract', action='store_true')
    c.add_argument('--url')
    c.add_argument('--sha256')
    c.add_argument('--offsets', type=int, nargs='+', default=[0])
    c.add_argument('--videos', type=int, nargs='+')
    c = commands.add_parser('extract', help='Safely extract a downloaded archive')
    c.add_argument('archive')
    c.add_argument('--output', required=True)
    c = commands.add_parser('prepare', help='Parse packets/player logs and freeze training-only transforms')
    c.add_argument('dataset', choices=['longenough', 'ydms'])
    c.add_argument('--source', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--exclude-condition', choices=['bw1', 'bw8'])
    c.add_argument('--scales', nargs='+', type=int, default=[100, 500, 2000])
    c.add_argument('--counts', nargs=3, type=int)
    c.add_argument('--offsets', nargs='+', type=int, default=[0])
    c.add_argument('--allow-cohort-difference', action='store_true')
    c = commands.add_parser('protocols', help='Build fixed support/query episodes')
    c.add_argument('--data', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--split', choices=['validation', 'test'], default='test')
    c.add_argument('--seed', type=int, default=20260817)
    c.add_argument('--bandwidth-matrix', action='store_true')
    c = commands.add_parser('matrix', help='Write the complete experiment configuration matrix')
    c.add_argument('--output', required=True)
    c = commands.add_parser('train', help='Train an episodic model or resume an exact checkpoint')
    c.add_argument('--config', required=True)
    c.add_argument('--data', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--validation-protocols', required=True)
    c.add_argument('--resume')
    c.add_argument('--max-steps', type=int)
    c.add_argument('--device', default='auto')
    c = commands.add_parser('run-matrix', help='Prepare, train, and evaluate selected complete experiments')
    c.add_argument('--matrix', required=True)
    c.add_argument('--raw-root', required=True)
    c.add_argument('--work-root', required=True)
    c.add_argument('--names', nargs='+')
    c.add_argument('--analyses', nargs='+')
    c.add_argument('--max-runs', type=int)
    c.add_argument('--device', default='auto')
    c = commands.add_parser('select-soft-dtw', help='Freeze smoothing from matched validation runs')
    c.add_argument('--matrix', required=True)
    c.add_argument('--work-root', required=True)
    c.add_argument('--output', required=True)
    for name in ['evaluate', 'open-set', 'calibrate', 'correspondence', 'real-overlap', 'benchmark']:
        c = commands.add_parser(name)
        c.add_argument('--checkpoint', required=True)
        c.add_argument('--data', required=True)
        c.add_argument('--output', required=True)
        c.add_argument('--device', default='auto')
        if name in ['evaluate', 'open-set', 'calibrate']:
            c.add_argument('--protocols', required=True)
        if name == 'evaluate':
            c.add_argument('--scenarios', nargs='+')
            c.add_argument('--fractions', type=float, nargs='+', default=[1.])
        elif name == 'calibrate':
            c.add_argument('--scenario')
        elif name == 'benchmark':
            c.add_argument('--warmup', type=int, default=100)
            c.add_argument('--queries', type=int, default=1000)
        elif name == 'real-overlap':
            c.add_argument('--start', type=float, default=30.)
            c.add_argument('--bootstrap', type=int, default=2000)
    c = commands.add_parser('enroll', help='Encode labeled packet traces into a gallery')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--manifest', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--device', default='auto')
    c = commands.add_parser('recognize', help='Recognize a packet-only query from an enrolled gallery')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--gallery', required=True)
    c.add_argument('--query', required=True)
    c.add_argument('--format', choices=['npz', 'longenough', 'ydms'], required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--threshold')
    c.add_argument('--fraction', type=float, default=1.)
    c.add_argument('--device', default='auto')
    c = commands.add_parser('collect', help='Aggregate matched seeds without replacing missing measurements')
    c.add_argument('--root', required=True)
    c.add_argument('--output', required=True)
    c = commands.add_parser('tables', help='Export manuscript tables from measured matrix runs')
    c.add_argument('--matrix', required=True)
    c.add_argument('--root', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--allow-incomplete', action='store_true')
    c = commands.add_parser('case-study', help='Export local correspondence and support responsibility figures')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--data', required=True)
    c.add_argument('--protocols', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--scenario', default='mode_balanced')
    c.add_argument('--episode-index', type=int, default=0)
    c.add_argument('--query-index', type=int, default=0)
    c.add_argument('--device', default='auto')
    c = commands.add_parser('plot', help='Render measured sensitivity and truncation curves')
    c.add_argument('--summary', required=True)
    c.add_argument('--output', required=True)
    c = commands.add_parser('audit', help='Validate the prepared data and fixed protocols')
    c.add_argument('--data', required=True)
    c.add_argument('--protocols')
    c.add_argument('--output', required=True)
    return p


def execute(a):
    c = a.command
    if c == 'download':
        from .data.download import download_dataset
        if a.dataset == 'longenough' and a.url is None:
            from .data.remote_zip import download_longenough_cohort
            result = download_longenough_cohort(a.output, a.offsets, a.videos)
            return {'files': len(result['files']), 'output': a.output}
        return download_dataset(a.dataset, a.output, not a.no_extract, a.url, a.sha256)
    if c == 'extract':
        from .data.download import extract_archive
        return extract_archive(a.archive, a.output)
    if c == 'prepare':
        from .data.prepare import prepare_dataset
        return prepare_dataset(a.dataset, a.source, a.output, a.scales, a.exclude_condition, a.counts,
                               not a.allow_cohort_difference, a.offsets)
    if c == 'protocols':
        from .data.prepare import PreparedDataset
        from .data.protocols import build_protocols
        return build_protocols(PreparedDataset(a.data), a.output, a.split, a.seed, a.bandwidth_matrix)
    if c == 'matrix':
        from .experiments import experiment_matrix
        return {'runs': len(experiment_matrix(a.output))}
    if c == 'train':
        from .config import load_config
        from .training.engine import train
        return train(load_config(a.config, {'training': {'device': a.device}}), a.data, a.output,
                     a.validation_protocols, a.resume, a.max_steps)
    if c == 'run-matrix':
        from .runner import run_matrix
        return run_matrix(a.matrix, a.raw_root, a.work_root, a.names, a.analyses, a.device, a.max_runs)
    if c == 'select-soft-dtw':
        from .selection import select_soft_dtw
        return select_soft_dtw(a.matrix, a.work_root, a.output)
    if c in ['evaluate', 'open-set', 'calibrate', 'correspondence', 'real-overlap', 'benchmark']:
        from . import evaluation as e
        if c == 'evaluate':
            return e.evaluate(a.checkpoint, a.data, a.protocols, a.output, a.scenarios, a.fractions, a.device)
        if c == 'open-set':
            return e.evaluate_open(a.checkpoint, a.data, a.protocols, a.output, a.device)
        if c == 'calibrate':
            return e.calibrate_threshold(a.checkpoint, a.data, a.protocols, a.output, a.scenario, a.device)
        if c == 'correspondence':
            return e.correspondence(a.checkpoint, a.data, a.output, device=a.device)
        if c == 'real-overlap':
            return e.real_overlap(a.checkpoint, a.data, a.output, start=a.start, bootstrap=a.bootstrap, device=a.device)
        return e.benchmark(a.checkpoint, a.data, a.output, a.warmup, a.queries, a.device)
    if c == 'enroll':
        from .inference import enroll
        return enroll(a.checkpoint, a.manifest, a.output, a.device)
    if c == 'recognize':
        from .inference import recognize
        return recognize(a.checkpoint, a.gallery, a.query, a.format, a.output, a.threshold, a.fraction, a.device)
    if c == 'collect':
        from .reporting import collect_results
        return collect_results(a.root, a.output)
    if c == 'tables':
        from .tables import export_tables
        return export_tables(a.matrix, a.root, a.output, strict=not a.allow_incomplete)
    if c == 'case-study':
        from .case_study import case_study
        return case_study(a.checkpoint, a.data, a.protocols, a.output, a.scenario, a.episode_index, a.query_index, a.device)
    if c == 'plot':
        from .reporting import plot_results
        return plot_results(a.summary, a.output)
    if c == 'audit':
        from .audit import audit_data
        return audit_data(a.data, a.protocols, a.output)
    raise ValueError(c)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    a = parser().parse_args(argv)
    try:
        result = execute(a)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    if result is not None:
        if isinstance(result, list):
            result = {'results': len(result)}
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
