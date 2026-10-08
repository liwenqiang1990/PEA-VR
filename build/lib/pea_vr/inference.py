from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data.adapters import read_longenough_packets, read_ydms_packets
from .data.features import Normalizer, packet_features
from .models.amp import Fingerprint
from .models.matching import Matcher
from .training.engine import load_model
from .utils import canonical_hash, read_json, sha256, write_json


def encode_file(model, state, path, format, device, fraction=1.):
    if not 0 < fraction <= 1:
        raise ValueError('Query observation fraction must be in (0,1]')
    if format == 'npz':
        with np.load(path, allow_pickle=False) as raw:
            times, directions, lengths = raw['times'], raw['directions'], raw['lengths']
            if not len(times):
                raise ValueError('A packet trace must contain at least one packet')
            origin = float(np.min(times))
            times = times - origin
    elif format in ['ydms', 'longenough']:
        parser = read_ydms_packets if format == 'ydms' else read_longenough_packets
        times, directions, lengths, _, _ = parser(path)
    else:
        raise ValueError('format must be npz, ydms, or longenough')
    c = state['config']['data']
    inputs = packet_features(times, directions, lengths, c['scales'], c['window'], c['window'] * fraction)
    values = Normalizer(state['normalizer']).transform(inputs)
    tensors = {s: {k: torch.from_numpy(v)[None].to(device) for k, v in item.items()} for s, item in values.items()}
    with torch.no_grad():
        return model(tensors)


def enroll(checkpoint, manifest, output, device='auto'):
    model, state, device = load_model(checkpoint, device)
    fingerprints, labels, sources = [], [], []
    manifest = Path(manifest).resolve()
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            path = Path(row['path'])
            if not path.is_absolute():
                path = manifest.parent / path
            fingerprints.append(encode_file(model, state, path, row['format'], device).to('cpu'))
            labels.append(row['label'])
            sources.append({'path': str(path), 'sha256': sha256(path)})
    if not labels:
        raise ValueError('Empty enrollment manifest')
    fp = Fingerprint.cat(fingerprints)
    stored = {'z': fp.z}
    if state['config']['matching']['alignment'] != 'none':
        stored.update(u=fp.u, r=fp.r)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 2, 'checkpoint_sha256': sha256(checkpoint), 'labels': labels,
                'sources': sources, 'fingerprints': stored}, output)
    return {'sessions': len(labels), 'identities': len(set(labels)), 'gallery': str(output)}


def recognize(checkpoint, gallery, query, format, output, threshold=None, fraction=1., device='auto'):
    model, state, device = load_model(checkpoint, device)
    stored = torch.load(gallery, map_location=device, weights_only=False)
    if stored['format_version'] != 2 or stored['checkpoint_sha256'] != sha256(checkpoint):
        raise ValueError('Gallery was encoded by a different checkpoint')
    keys = sorted(set(stored['labels']))
    labels = torch.tensor([keys.index(label) for label in stored['labels']], device=device)
    raw = stored['fingerprints']
    z = raw['z']
    u = raw.get('u', z[:, None])
    r = raw.get('r', z.new_ones((len(z), 1)))
    with torch.no_grad():
        v = F.normalize(model.local_projection(u), dim=-1) if hasattr(model, 'local_projection') and model.local_projection is not None else None
        if v is not None:
            v = torch.where(r[..., None] > 0, v, 0)
        support = Fingerprint(z, u, r, v)
        query_fp = encode_file(model, state, query, format, device, fraction)
        scores, classes = Matcher(state['config']['matching']).candidates(query_fp, support, labels)
    confidence, index = scores[0].max(0)
    rejected = False
    threshold_value = None
    if threshold:
        calibrated = read_json(threshold)
        if calibrated['split'] != 'validation' or calibrated['checkpoint_step'] != state['step'] or calibrated['checkpoint_sha256'] != sha256(checkpoint) or canonical_hash(calibrated['config']) != canonical_hash(state['config']):
            raise ValueError('Rejection threshold does not match the checkpoint')
        threshold_value = calibrated['threshold']
        rejected = confidence.item() < threshold_value
    result = {'label': None if rejected else keys[int(classes[index])], 'rejected': rejected,
              'confidence': confidence.item(), 'threshold': threshold_value,
              'scores': {keys[int(label)]: float(score) for label, score in zip(classes, scores[0].cpu())}}
    write_json(output, result)
    return result
