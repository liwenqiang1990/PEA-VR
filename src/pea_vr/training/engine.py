from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from ..data.prepare import PreparedDataset
from ..data.protocols import EpisodeSampler, validate_episode, verify_protocol_manifest
from ..models.amp import Fingerprint
from ..models.factory import build_model
from ..models.matching import Matcher
from ..models.partial_views import alignment_target, prefix_view, warp_fingerprint
from ..utils import canonical_hash, device_for, environment, implementation_hash, read_jsonl, restore_rng, rng_state, seed_all, write_json
from .losses import alignment_cross_entropy, batch_hard_triplet, supervised_contrastive


def subset_inputs(inputs, indices):
    return {s: {k: v[indices] for k, v in item.items()} for s, item in inputs.items()}


def adapted_parameters(model, support_inputs, labels, config, training=False):
    fast = dict(model.named_parameters())
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    classes = torch.unique(labels, sorted=True)
    for _ in range(config['baseline']['inner_steps']):
        with torch.enable_grad():
            fp = functional_call(model, (fast, buffers), (support_inputs,))
            similarity = fp.z @ fp.z.T
            columns = []
            for label in classes:
                weights = (labels == label).to(fp.z.dtype)[None].expand(len(labels), -1).clone()
                weights.fill_diagonal_(0)
                columns.append((similarity * weights).sum(1) / weights.sum(1).clamp_min(1))
            logits = torch.stack(columns, -1)
            targets = torch.searchsorted(classes, labels)
            loss = F.cross_entropy(logits / config['training']['tau_e'], targets)
            grads = torch.autograd.grad(loss, tuple(fast.values()), create_graph=False)
            fast = {key: value - config['baseline']['inner_lr'] * grad.detach()
                    for (key, value), grad in zip(fast.items(), grads)}
            if not training:
                fast = {key: value.detach().requires_grad_() for key, value in fast.items()}
    return fast, buffers


def encode_episode(model, dataset, episode, device, config, fraction=1.):
    support_indices = [dataset.by_id[sid] for sid in episode['support']]
    query_indices = [dataset.by_id[sid] for sid in episode['query']]
    support_inputs = dataset.batch(support_indices, device)
    query_inputs = dataset.batch(query_indices, device, fraction=fraction)
    labels = torch.tensor(episode['support_labels'], device=device)
    if config['model']['kind'] == 'cl_metaflow':
        fast, buffers = adapted_parameters(model, support_inputs, labels, config)
        with torch.no_grad():
            encode = lambda inputs: functional_call(model, (fast, buffers), (inputs,))
            support = encode_batches(encode, support_inputs)
            query = encode_batches(encode, query_inputs)
    else:
        with torch.no_grad():
            support, query = encode_batches(model, support_inputs), encode_batches(model, query_inputs)
    return support, query, labels


def encode_batches(encode, inputs, batch_size=16):
    count = len(next(iter(inputs.values()))['x'])
    return Fingerprint.cat([encode(subset_inputs(inputs, slice(start, start + batch_size)))
                            for start in range(0, count, batch_size)])


def evaluate_episodes(model, dataset, episodes, config, device, fraction=1., return_records=False):
    previous = model.training
    model.eval()
    matcher = Matcher(config['matching'])
    values, records = [], []
    for episode in episodes:
        validate_episode(dataset, episode)
        support, query, labels = encode_episode(model, dataset, episode, device, config, fraction)
        with torch.no_grad():
            scores, classes = matcher.candidates(query, support, labels)
            prediction = classes[scores.argmax(-1)].cpu().numpy()
        target = np.array(episode['query_labels'])
        accuracy = float((prediction == target).mean())
        values.append(accuracy)
        if return_records:
            records.append({'episode': episode.get('index', len(records)), 'accuracy': accuracy,
                            'prediction': prediction.tolist(), 'target': target.tolist(),
                            'confidence': scores.max(-1).values.cpu().tolist()})
    model.train(previous)
    array = np.array(values)
    return {'episodes': len(values), 'accuracy': float(array.mean()),
            'episode_std': float(array.std(ddof=1)) if len(array) > 1 else 0.,
            'episode_sem': float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.,
            'fraction': fraction, 'records': records if return_records else None}


def save_checkpoint(path, model, optimizer, scheduler, step, config, dataset, sampler, view_rng, best, validation_protocols=None):
    payload = {'format_version': 2, 'step': step, 'config': config, 'config_hash': canonical_hash(config),
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'rng': rng_state(), 'sampler_rng': sampler.state_dict(), 'view_rng': view_rng.bit_generator.state,
               'best': best, 'normalizer': dataset.normalizer.state,
               'dataset': dataset.metadata, 'environment': environment(), 'implementation_hash': implementation_hash(),
               'validation_protocols': validation_protocols}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(checkpoint, device='auto'):
    device = device_for(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get('format_version') != 2:
        raise ValueError('Unsupported checkpoint format')
    if payload.get('implementation_hash') is not None and payload['implementation_hash'] != implementation_hash():
        raise ValueError('Checkpoint requires the original implementation version')
    config = payload['config']
    model = build_model(config).to(device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    return model, payload, device


def prototype_alignment(z, labels, support_count):
    losses = []
    for label in torch.unique(labels):
        s = z[:support_count][labels[:support_count] == label]
        q = z[support_count:][labels[support_count:] == label]
        if len(s) and len(q):
            losses.append((s.mean(0) - q.mean(0)).square().sum())
    return torch.stack(losses).mean()


def domain_alignment(z, labels, modes):
    losses = []
    for label in torch.unique(labels):
        centers = []
        for mode in sorted(set(modes)):
            selected = (labels == label) & torch.tensor([m == mode for m in modes], device=z.device)
            if selected.any():
                centers.append(z[selected].mean(0))
        for a in range(len(centers)):
            for b in range(a):
                losses.append((centers[a] - centers[b]).square().sum())
    return torch.stack(losses).mean() if losses else z.sum() * 0


def train(config, data, output, protocols=None, resume=None, max_steps=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'latest.pt').exists() and resume is None:
        raise FileExistsError('Training output already contains a checkpoint; use --resume')
    tc = config['training']
    torch.set_num_threads(tc['threads'])
    seed_all(tc['seed'], tc['deterministic'])
    device = device_for(tc['device'])
    dataset = PreparedDataset(data)
    if dataset.metadata['dataset'] != config['data']['dataset'] or dataset.metadata['exclude_condition'] != config['data']['exclude_condition']:
        raise ValueError('Configuration and prepared dataset cohort differ')
    if dataset.metadata['scales'] != config['data']['scales']:
        raise ValueError('Configuration and prepared dataset scales differ')
    validation, validation_manifest = None, None
    if protocols:
        manifest = verify_protocol_manifest(dataset, protocols)
        validation_manifest = manifest
        if manifest['split'] != 'validation':
            raise ValueError('Checkpoint selection requires validation-identity protocols')
        scenario = 'heldout_' + config['data']['exclude_condition'] if config['data']['exclude_condition'] else (
            'cross_mode' if config['data']['dataset'] == 'longenough' else 'random_mixed')
        validation = read_jsonl(Path(protocols) / (scenario + '.jsonl'))
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'],
                                  betas=tuple(tc['betas']), eps=tc['adam_epsilon'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tc['episodes'], eta_min=tc['min_lr'])
    sampler = EpisodeSampler(dataset, 'train', tc['seed'], config['data']['exclude_condition'])
    view_rng = np.random.default_rng(tc['seed'] + 100000)
    matcher = Matcher(config['matching'])
    step, best = 0, {'accuracy': -1., 'step': None}
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state['config_hash'] != canonical_hash(config) or state['dataset']['fingerprint'] != dataset.metadata['fingerprint']:
            raise ValueError('Exact resume requires the same configuration and prepared dataset')
        if state.get('implementation_hash') is not None and state['implementation_hash'] != implementation_hash():
            raise ValueError('Exact resume requires the original implementation version')
        if state.get('validation_protocols') != validation_manifest:
            raise ValueError('Exact resume requires identical validation protocols')
        current_environment = environment()
        keys = ['torch', 'numpy', 'python'] + (['gpu', 'cuda'] if device.type == 'cuda' else [])
        if any(state['environment'][key] != current_environment[key] for key in keys):
            raise ValueError('Exact resume requires the original numerical runtime and device')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step, best = state['step'], state['best']
        sampler.load_state_dict(state['sampler_rng'])
        view_rng.bit_generator.state = state['view_rng']
        restore_rng(state['rng'])
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', environment())
    write_json(output / 'dataset.json', dataset.metadata)
    stop = min(tc['episodes'], max_steps) if max_steps else tc['episodes']
    started = time.perf_counter()
    model.train()
    while step < stop:
        episode = sampler.sample(tc['ways'], tc['shots'], tc['queries'])
        support_count = len(episode['support'])
        ids = episode['support'] + episode['query']
        indices = [dataset.by_id[sid] for sid in ids]
        labels = torch.tensor(episode['support_labels'] + episode['query_labels'], device=device)
        inputs = dataset.batch(indices, device)
        optimizer.zero_grad(set_to_none=True)
        kind = config['model']['kind']
        pretrain = kind == 'cl_metaflow' and step < config['baseline']['pretrain_episodes']
        if kind == 'cl_metaflow' and not pretrain:
            fast, buffers = adapted_parameters(model, subset_inputs(inputs, slice(None, support_count)),
                                               labels[:support_count], config, training=True)
            original = functional_call(model, (fast, buffers), (inputs,))
        else:
            original = model(inputs)
        vc = config['views']
        fractions = np.where(view_rng.random(len(ids)) < vc['prefix_probability'],
                             view_rng.uniform(vc['prefix_min'], 1., len(ids)), 1.)
        if pretrain:
            fractions = view_rng.uniform(.7, 1., len(ids))
        fractions_t = torch.tensor(fractions, dtype=torch.float32, device=device)
        partial = model(prefix_view(inputs, fractions_t)) if np.any(fractions < 1) else original
        eta = np.where(view_rng.random(len(ids)) < vc['warp_probability'],
                       view_rng.uniform(-vc['warp_max'], vc['warp_max'], len(ids)), 0.)
        transformed, coordinates = partial, None
        if config['matching']['alignment'] != 'none':
            transformed, coordinates = warp_fingerprint(partial, torch.tensor(eta, dtype=torch.float32, device=device),
                                                        model.local_projection)
        auxiliary = original.z.sum() * 0
        logged = {'epi': 0., 'sup': 0., 'ali': 0.}
        if pretrain:
            auxiliary = supervised_contrastive(torch.cat([original.z, partial.z]), labels.repeat(2), tc['tau_sup'])
            logged['sup'] = auxiliary.detach().item()
        elif kind == 'deepmetric':
            auxiliary = batch_hard_triplet(original.z, labels, config['baseline']['triplet_margin'])
            logged['triplet'] = auxiliary.detach().item()
        else:
            support = transformed.index(slice(None, support_count))
            query = transformed.index(slice(support_count, None))
            query_labels = labels[support_count:]
            for offset in range(0, len(query_labels), tc['query_chunk']):
                end = min(offset + tc['query_chunk'], len(query_labels))
                scores, classes = matcher.candidates(query.index(slice(offset, end)), support, labels[:support_count])
                loss = F.cross_entropy(scores / tc['tau_e'], torch.searchsorted(classes, query_labels[offset:end]), reduction='sum') / len(query_labels)
                logged['epi'] += loss.detach().item()
                loss.backward(retain_graph=True)
                del loss, scores
            if tc['lambda_sup']:
                sup = supervised_contrastive(torch.cat([original.z, transformed.z]), labels.repeat(2), tc['tau_sup'])
                logged['sup'] = sup.detach().item()
                auxiliary = auxiliary + tc['lambda_sup'] * sup
            if kind == 'coda':
                proto = prototype_alignment(original.z, labels, support_count)
                domain = domain_alignment(original.z, labels, [dataset.rows[i]['mode'] for i in indices])
                logged['prototype'], logged['domain'] = proto.detach().item(), domain.detach().item()
                auxiliary = auxiliary + config['baseline']['prototype_weight'] * proto + config['baseline']['domain_weight'] * domain
        if tc['lambda_ali'] and not pretrain:
            for offset in range(0, len(ids), tc['alignment_chunk']):
                end = min(offset + tc['alignment_chunk'], len(ids))
                left, right = original.index(slice(offset, end)), transformed.index(slice(offset, end))
                result = matcher.pair(left, right, return_mass=True)
                target, valid = alignment_target(left.r, right.r, coordinates[offset:end], tc['sigma_w'])
                ali = alignment_cross_entropy(result['mass'], target, valid) * (end - offset) / len(ids)
                logged['ali'] += ali.detach().item()
                (tc['lambda_ali'] * ali).backward(retain_graph=True)
                del result, ali
        auxiliary.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f'Nonfinite gradient at episode {step + 1}')
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), tc['gradient_clip']).item()
        optimizer.step()
        scheduler.step()
        step += 1
        if step % tc['log_every'] == 0 or step == 1:
            total = logged['sup'] if pretrain else logged.get('triplet', logged['epi'] + tc['lambda_sup'] * logged['sup'] +
                    tc['lambda_ali'] * logged['ali'] + config['baseline']['prototype_weight'] * logged.get('prototype', 0.) +
                    config['baseline']['domain_weight'] * logged.get('domain', 0.))
            row = {'episode': step, **logged, 'lr': scheduler.get_last_lr()[0], 'gradient_norm': gradient_norm,
                   'total': total, 'elapsed_seconds': time.perf_counter() - started, 'phase': 'pretrain' if pretrain else 'episodic'}
            with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row), flush=True)
        if validation and (step % tc['eval_every'] == 0 or step == tc['episodes']):
            # Validation must not alter the training random stream.
            random_state = rng_state()
            result = evaluate_episodes(model, dataset, validation, config, device)
            restore_rng(random_state)
            write_json(output / f'validation_{step:06d}.json', result)
            if result['accuracy'] > best['accuracy']:
                best = {'accuracy': result['accuracy'], 'step': step}
                save_checkpoint(output / 'best.pt', model, optimizer, scheduler, step, config, dataset, sampler, view_rng, best, validation_manifest)
        if step % tc['save_every'] == 0 or step == stop:
            save_checkpoint(output / 'latest.pt', model, optimizer, scheduler, step, config, dataset, sampler, view_rng, best, validation_manifest)
            if step % tc['save_every'] == 0:
                save_checkpoint(output / f'episode_{step:06d}.pt', model, optimizer, scheduler, step, config, dataset, sampler, view_rng, best, validation_manifest)
                expired = sorted(output.glob('episode_[0-9][0-9][0-9][0-9][0-9][0-9].pt'))[:-tc.get('keep_checkpoints', 3)]
                for path in expired:
                    if path.resolve().parent != output.resolve():
                        raise ValueError('Checkpoint retention target is outside the run directory')
                    path.unlink()
    result = {'episode': step, 'complete': step == tc['episodes'], 'best': best, 'validation_enabled': validation is not None}
    write_json(output / 'status.json', result)
    return result
