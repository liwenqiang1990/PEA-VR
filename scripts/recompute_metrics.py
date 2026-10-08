from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auc_and_fpr95(known, scores):
    known, scores = np.asarray(known, bool), np.asarray(scores, float)
    positive, negative = scores[known], scores[~known]
    if not len(positive) or not len(negative):
        raise ValueError('Known and unknown queries are required')
    # Pairwise comparison is independent of sklearn's ROC implementation.
    auc = 0.
    for start in range(0, len(positive), 256):
        differences = positive[start:start+256, None] - negative[None]
        auc += (differences > 0).sum() + .5 * (differences == 0).sum()
    auc /= len(positive) * len(negative)
    for threshold in sorted(set(scores), reverse=True):
        if np.mean(positive >= threshold) >= .95:
            return float(auc), float(np.mean(negative >= threshold))
    raise ValueError('Invalid ROC inputs')


def recompute(root):
    checks = []
    for path in sorted(Path(root).rglob('*.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(report, dict):
            continue
        if 'accuracy' in report and 'episodes' in report:
            raw_path = path.with_name(path.stem + '_episodes.jsonl')
            if not raw_path.exists():
                continue
            rows = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines() if line]
            accuracy = float(np.mean([np.mean(np.array(r['prediction']) == np.array(r['target'])) for r in rows]))
            valid = len(rows) == report['episodes'] and abs(accuracy - report['accuracy']) < 1e-12
            checks.append({'file': str(path), 'metric': 'accuracy', 'recomputed': accuracy, 'passed': valid})
        elif 'partitions' in report and 'auroc' in report:
            for partition in report['partitions']:
                auc, fpr = auc_and_fpr95(partition['known'], partition['confidence'])
                checks.append({'file': str(path), 'partition': partition['partition'], 'metric': 'open_set',
                               'auroc': auc, 'fpr95': fpr,
                               'passed': abs(auc-partition['auroc']) < 1e-12 and abs(fpr-partition['fpr95']) < 1e-12})
        elif 'alignment_mae_seconds' in report:
            raw_path = path.with_suffix('.pairs.jsonl')
            rows = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines() if line]
            errors = np.concatenate([row['absolute_errors'] for row in rows])
            checks.append({'file': str(path), 'metric': 'correspondence',
                           'passed': abs(float(errors.mean())-report['alignment_mae_seconds']) < 1e-12 and
                                     abs(float(np.mean(errors <= 2))-report['align_at_2s']) < 1e-12})
    return {'checks': checks, 'passed': bool(checks) and all(c['passed'] for c in checks)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = recompute(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"{len(result['checks'])} independent metric checks; passed={result['passed']}")
    raise SystemExit(0 if result['passed'] else 1)
