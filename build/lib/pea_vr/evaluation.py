from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from .data.prepare import PreparedDataset
from .data.protocols import validate_episode, verify_protocol_manifest
from .models.matching import Matcher
from .models.partial_views import token_centers, warp_fingerprint
from .training.engine import encode_episode, evaluate_episodes, load_model
from .utils import derived_seed, environment, read_json, read_jsonl, sha256, write_json, write_jsonl


def require_dataset(checkpoint, dataset):
    if checkpoint['dataset']['fingerprint'] != dataset.metadata['fingerprint']:
        raise ValueError('Checkpoint and prepared dataset fingerprints differ')


def evaluate(checkpoint, data, protocols, output, scenarios=None, fractions=(1.,), device='auto'):
    model, state, device = load_model(checkpoint, device)
    dataset = PreparedDataset(data)
    require_dataset(state, dataset)
    manifest = verify_protocol_manifest(dataset, protocols)
    scenarios = scenarios or list(manifest['protocols'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for scenario in scenarios:
        if scenario not in manifest['protocols']:
            raise ValueError(f'Unknown scenario {scenario}')
        episodes = read_jsonl(Path(protocols) / (scenario + '.jsonl'))
        for fraction in fractions:
            if not 0 < fraction <= 1:
                raise ValueError('Query fraction must be in (0,1]')
            result = evaluate_episodes(model, dataset, episodes, state['config'], device, fraction, return_records=True)
            records = result.pop('records')
            result.update({'scenario': scenario, 'split': manifest['split'], 'training_seed': state['config']['training']['seed'],
                           'checkpoint_step': state['step'], 'config': state['config'],
                           'dataset_fingerprint': dataset.metadata['fingerprint']})
            name = f'{scenario}_rho{fraction:g}'
            write_json(output / (name + '.json'), result)
            write_jsonl(output / (name + '_episodes.jsonl'), records)
            reports.append(result)
            print(f'[evaluate] {name}: {100 * result["accuracy"]:.3f}% ({result["episodes"]} episodes)', flush=True)
    return reports


def open_set_metrics(known, confidence):
    known = np.asarray(known, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    if not known.any() or known.all() or not np.isfinite(confidence).all():
        raise ValueError('Open-set metrics require finite known and unknown scores')
    fpr, tpr, threshold = roc_curve(known, confidence, drop_intermediate=False)
    indices = np.flatnonzero(tpr >= .95)
    return {'auroc': float(roc_auc_score(known, confidence)), 'fpr95': float(fpr[indices[0]])}


def evaluate_open(checkpoint, data, protocols, output, device='auto'):
    model, state, device = load_model(checkpoint, device)
    dataset = PreparedDataset(data)
    require_dataset(state, dataset)
    manifest = verify_protocol_manifest(dataset, protocols)
    if manifest['split'] != 'test':
        raise ValueError('Open-set evaluation requires test partitions')
    records = []
    matcher = Matcher(state['config']['matching'])
    for episode in read_jsonl(Path(protocols) / 'open_set.jsonl'):
        validate_episode(dataset, episode)
        support, query, labels = encode_episode(model, dataset, episode, device, state['config'])
        scores = []
        with torch.no_grad():
            for start in range(0, len(query.z), 16):
                candidates, _ = matcher.candidates(query.index(slice(start, start+16)), support, labels)
                scores.extend(candidates.max(-1).values.cpu().tolist())
        known = np.array(episode['query_labels']) >= 0
        records.append({'partition': episode['partition'], **open_set_metrics(known, scores),
                        'known_count': int(known.sum()), 'unknown_count': int((~known).sum()),
                        'query': episode['query'], 'known': known.tolist(), 'confidence': scores})
    report = {'seed': state['config']['training']['seed'], 'config': state['config'], 'partitions': records,
              'auroc': float(np.mean([r['auroc'] for r in records])),
              'fpr95': float(np.mean([r['fpr95'] for r in records]))}
    write_json(output, report)
    return report


def calibrate_threshold(checkpoint, data, protocols, output, scenario=None, device='auto'):
    model, state, device = load_model(checkpoint, device)
    dataset = PreparedDataset(data)
    require_dataset(state, dataset)
    manifest = verify_protocol_manifest(dataset, protocols)
    if manifest['split'] != 'validation':
        raise ValueError('Rejection calibration requires validation identities')
    scenario = scenario or ('cross_mode' if dataset.metadata['dataset'] == 'longenough' else 'random_mixed')
    confidence = []
    matcher = Matcher(state['config']['matching'])
    for episode in read_jsonl(Path(protocols) / (scenario + '.jsonl')):
        support, query, labels = encode_episode(model, dataset, episode, device, state['config'])
        with torch.no_grad():
            scores, _ = matcher.candidates(query, support, labels)
            confidence += scores.max(-1).values.cpu().tolist()
    threshold = float(np.quantile(confidence, .05, method='lower'))
    result = {'threshold': threshold, 'target_known_tpr': .95, 'empirical_known_tpr': float(np.mean(np.array(confidence) >= threshold)),
              'split': 'validation', 'scenario': scenario, 'dataset_fingerprint': dataset.metadata['fingerprint'],
              'config': state['config'], 'checkpoint_step': state['step'], 'checkpoint_sha256': sha256(checkpoint)}
    write_json(output, result)
    return result


def correspondence(checkpoint, data, output, seed=20260817, device='auto'):
    model, state, device = load_model(checkpoint, device)
    config = state['config']
    if config['matching']['alignment'] == 'none':
        raise ValueError('Correspondence diagnostics require a local alignment operator')
    dataset = PreparedDataset(data)
    require_dataset(state, dataset)
    rng = np.random.default_rng(derived_seed(seed, 'correspondence'))
    matcher = Matcher(config['matching'])
    records = []
    with torch.no_grad():
        for index in dataset.indices('test'):
            inputs = dataset.batch([index], device)
            while True:
                lengths = rng.uniform(.3, 1., 2)
                starts = rng.uniform(0., 1. - lengths)
                ends = starts + lengths
                common = (max(starts), min(ends))
                if common[1] > common[0]:
                    break
            views, coords = [], []
            for a in range(2):
                masked = {}
                for scale, item in inputs.items():
                    centers = token_centers(item['x'].shape[1], 60., device, item['x'].dtype) / 60
                    visible = ((centers >= starts[a]) & (centers < ends[a])).to(item['x'].dtype)[None]
                    masked[scale] = {'x': item['x'] * visible[..., None], 'mask': item['mask'] * visible}
                fp = model(masked)
                eta = torch.tensor(rng.uniform(-.3, .3, 1), dtype=torch.float32, device=device)
                warped, coordinates = warp_fingerprint(fp, eta, model.local_projection)
                views.append(warped)
                coords.append(coordinates)
            result = matcher.pair(*views, return_mass=True)
            mass = result['mass']
            probability = mass / (mass.sum(-2, keepdim=True) + config['matching']['epsilon'])
            predicted = (probability * coords[0][:, :, None]).sum(-2)
            valid = ((coords[1] >= common[0] * 60) & (coords[1] <= common[1] * 60) &
                     (views[1].r > 0) & (mass.sum(-2) > 0))
            if not valid.any():
                continue
            errors = (predicted - coords[1]).abs()[valid].cpu().numpy()
            reference = (common[1] - common[0]) / min(lengths)
            records.append({'session': dataset.rows[index]['id'], 'tokens': len(errors), 'absolute_errors': errors.tolist(),
                            'overlap_reference': float(reference),
                            'overlap_prediction': result['overlap'].item() if config['matching']['alignment'] == 'pma' else None,
                            'visibility': [[float(starts[a]), float(ends[a])] for a in range(2)]})
    if not records:
        raise ValueError('No valid correspondence pairs')
    errors = np.concatenate([r['absolute_errors'] for r in records])
    report = {'seed': config['training']['seed'], 'config': config, 'pairs': len(records), 'tokens': len(errors),
              'alignment_mae_seconds': float(errors.mean()), 'align_at_2s': float((errors <= 2).mean()),
              'overlap_mae': float(np.mean([abs(r['overlap_prediction'] - r['overlap_reference']) for r in records]))
              if config['matching']['alignment'] == 'pma' else None}
    write_json(output, report)
    write_jsonl(Path(output).with_suffix('.pairs.jsonl'), records)
    return report


def player_frontier(row, targets, max_gap=4.):
    player = row['player']
    events = player['events']
    buffers = np.array(events.get('buf', []), dtype=float)
    rates = np.array(events.get('pbr', []), dtype=float)
    states = player.get('state_events', [])
    if len(buffers) < 2 or len(rates) < 2 or not states:
        raise ValueError('Insufficient player playback/buffer dynamics')
    updates = [(float(t), 'rate', float(v)) for t, v in rates] + [(float(t), 'state', str(v)) for t, v in states]
    updates = sorted(updates, key=lambda x: x[0])
    start = min(0., updates[0][0])
    initial_position = float(row.get('offset') or 0) * 60.
    last, position, rate, playing, seeks = start, initial_position, 1., False, 0
    playing_events = {'playbackPlaying', 'playbackStarted'}
    stopped_events = {'playbackWaiting', 'playbackPaused', 'playbackEnded', 'playbackStalled', 'playbackSeeking'}
    positions, cursor = [], 0
    for target in sorted(targets):
        while cursor < len(updates) and updates[cursor][0] <= target:
            t, kind, value = updates[cursor]
            position += max(0., t - last) * rate * int(playing)
            last = t
            if kind == 'rate':
                if not 0 <= value <= 4:
                    raise ValueError('Invalid player playback rate')
                rate = value
            elif value in playing_events:
                playing = True
            elif value in stopped_events:
                playing = False
            elif value == 'playbackSeeked':
                if seeks > 0 or position - initial_position > 2:
                    raise ValueError('Unannotated seek in player diagnostic')
                position = initial_position
                seeks += 1
            cursor += 1
        position += max(0., target - last) * rate * int(playing)
        last = target
        bindex = np.searchsorted(buffers[:, 0], target, side='right') - 1
        if bindex < 0 or target - buffers[bindex, 0] > max_gap or buffers[bindex, 1] < 0:
            raise ValueError('Player buffer log gap exceeds diagnostic limit')
        if np.searchsorted(rates[:, 0], target, side='right') == 0:
            raise ValueError('No recorded playback rate before diagnostic window')
        positions.append(position + buffers[bindex, 1])
    positions = np.array(positions)
    if np.min(np.diff(positions)) < -2:
        raise ValueError('Nonmonotone player delivery frontier')
    return np.maximum.accumulate(positions)


def real_overlap(checkpoint, data, output, start=30., seed=20260817, device='auto', bootstrap=2000):
    model, state, device = load_model(checkpoint, device)
    if state['config']['matching']['alignment'] != 'pma':
        raise ValueError('Real overlap validation requires PMA')
    dataset = PreparedDataset(data)
    require_dataset(state, dataset)
    if dataset.metadata['dataset'] != 'longenough':
        raise ValueError('Real overlap diagnostic is defined for LongEnough')
    identities = sorted({row['identity'] for row in dataset.rows if row['split'] == 'test'})
    if len(identities) != 30:
        raise ValueError('Real diagnostic requires all 30 held-out identities')
    rng = np.random.default_rng(derived_seed(seed, 'real_overlap'))
    selected, rejected = {}, []
    for identity in identities:
        for condition in ['bw1', 'bw2', 'bw4', 'bw8']:
            eligible = []
            for index, row in enumerate(dataset.rows):
                if row['identity'] != identity or row['condition'] != condition or row['duration'] < start + 60:
                    continue
                try:
                    path = dataset.root / row['player_path']
                    if sha256(path) != row['player_cache_sha256']:
                        raise ValueError('Player cache checksum mismatch')
                    curve = player_frontier({**row, 'player': read_json(path)}, np.arange(start, start + 60.001, .5))
                    if curve[-1] <= curve[0]:
                        raise ValueError('Zero player-derived observed media span')
                    eligible.append((index, [float(curve[0]), float(curve[-1])]))
                except ValueError as error:
                    rejected.append({'id': row['id'], 'reason': str(error)})
            if len(eligible) < 3:
                write_json(Path(output).with_suffix('.rejected.json'), rejected)
                raise ValueError(f'Insufficient valid real windows for identity={identity}, condition={condition}')
            choice = rng.choice(len(eligible), 3, replace=False)
            selected[(identity, condition)] = [eligible[i] for i in choice]
    matcher, records = Matcher(state['config']['matching']), []
    with torch.no_grad():
        for identity in identities:
            for repeat in range(3):
                fingerprints, intervals = [], []
                for condition in ['bw1', 'bw2', 'bw4', 'bw8']:
                    index, interval = selected[(identity, condition)][repeat]
                    fingerprints.append(model(dataset.batch([index], device, start=start)))
                    intervals.append(interval)
                for a in range(4):
                    for b in range(a + 1, 4):
                        left, right = intervals[a], intervals[b]
                        truth = max(0., min(left[1], right[1]) - max(left[0], right[0])) / min(left[1]-left[0], right[1]-right[0])
                        prediction = matcher.pair(fingerprints[a], fingerprints[b])['overlap'].item()
                        records.append({'identity': identity, 'repeat': repeat, 'conditions': [2**a, 2**b],
                                        'reference': truth, 'prediction': prediction, 'intervals': [left, right]})
    true, pred = np.array([r['reference'] for r in records]), np.array([r['prediction'] for r in records])
    group_indices = {identity: [i for i, r in enumerate(records) if r['identity'] == identity] for identity in identities}
    boot = []
    for _ in range(bootstrap):
        sampled = rng.choice(identities, len(identities), replace=True)
        index = np.concatenate([group_indices[identity] for identity in sampled])
        rho = float(spearmanr(true[index], pred[index]).statistic)
        if np.isfinite(rho):
            boot.append([rho, float(np.abs(pred[index] - true[index]).mean())])
    rho = float(spearmanr(true, pred).statistic)
    if not np.isfinite(rho) or not boot:
        raise ValueError('Real overlap correlation is undefined')
    ci = np.quantile(boot, [.025, .975], axis=0)
    result = {'windows': 360, 'pairs': len(records), 'spearman': rho, 'mae': float(np.abs(pred-true).mean()),
              'spearman_ci95': ci[:, 0].tolist(), 'mae_ci95': ci[:, 1].tolist(), 'bootstrap': bootstrap,
              'start': start, 'config': state['config'], 'reference': 'player playback-rate integration and buffer frontier'}
    write_json(output, result)
    write_jsonl(Path(output).with_suffix('.pairs.jsonl'), records)
    return result


def benchmark(checkpoint, data, output, warmup=100, queries=1000, device='auto'):
    model, state, device = load_model(checkpoint, device)
    dataset = PreparedDataset(data)
    require_dataset(state, dataset)
    rows = dataset.indices('test')
    labels = sorted({dataset.rows[i]['identity'] for i in rows})[:5]
    selected = [next(i for i in rows if dataset.rows[i]['identity'] == label and i != excluded)
                for label in labels for excluded in [-1, next(i for i in rows if dataset.rows[i]['identity'] == label)]]
    if len(selected) != 10:
        raise ValueError('Benchmark needs five identities with two supports each')
    query_index = next(i for i in rows if i not in selected)
    matcher = Matcher(state['config']['matching'])
    query_input = dataset.batch([query_index], device)
    with torch.no_grad():
        support = model(dataset.batch(selected, device))
        query = model(query_input)
    support_labels = torch.arange(5, device=device).repeat_interleave(2)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def measure(function):
        for _ in range(warmup):
            function()
        sync()
        started = time.perf_counter()
        for _ in range(queries):
            function()
        sync()
        return 1000 * (time.perf_counter() - started) / queries
    with torch.no_grad():
        encode_ms = measure(lambda: model(query_input))
        match_ms = measure(lambda: matcher.candidates(query, support, support_labels))
    stored = {'z': query.z.cpu()}
    if state['config']['matching']['alignment'] != 'none':
        stored.update(u=query.u.cpu(), r=query.r.cpu())
    raw = {k: value.numel() * value.element_size() for k, value in stored.items()}
    buffer = io.BytesIO()
    torch.save(stored, buffer)
    result = {'parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'storage_raw_kib': sum(raw.values()) / 1024, 'storage_components_bytes': raw,
              'storage_serialized_kib': buffer.tell() / 1024, 'encode_ms': encode_ms, 'match_ms': match_ms,
              'warmup': warmup, 'queries': queries, 'batch': 1, 'supports': 10, 'dtype': 'float32',
              'device': str(device), 'environment': environment(), 'config': state['config']}
    write_json(output, result)
    return result
