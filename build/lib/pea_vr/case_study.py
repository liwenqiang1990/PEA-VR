from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data.prepare import PreparedDataset
from .data.protocols import verify_protocol_manifest
from .models.matching import Matcher
from .models.pooling import AdaptiveMeanPool1d
from .training.engine import encode_episode, load_model
from .utils import read_jsonl, write_json


def case_study(checkpoint, data, protocols, output, scenario='mode_balanced', episode_index=0, query_index=0, device='auto'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model, state, device = load_model(checkpoint, device)
    dataset = PreparedDataset(data)
    if dataset.metadata['fingerprint'] != state['dataset']['fingerprint']:
        raise ValueError('Case-study dataset differs from checkpoint')
    verify_protocol_manifest(dataset, protocols)
    episode = read_jsonl(Path(protocols) / (scenario + '.jsonl'))[episode_index]
    support, query, labels = encode_episode(model, dataset, episode, device, state['config'])
    if state['config']['matching']['alignment'] == 'none':
        raise ValueError('Alignment case study requires a local matching model')
    query = query.index(slice(query_index, query_index + 1))
    matcher = Matcher(state['config']['matching'])
    with torch.no_grad():
        scores, classes, details = matcher.candidates(query, support, labels, return_details=True)
    label = episode['query_labels'][query_index]
    indices = torch.nonzero(labels == label).flatten().tolist()
    responsibility = details['responsibility'][torch.nonzero(classes == label).item()][0]
    rows = []
    for offset, index in enumerate(indices):
        rows.append({'session': episode['support'][index], 'local_similarity': details['local'][0, index].item(),
                     'overlap': details['overlap'][0, index].item(), 'pair_score': details['pair'][0, index].item(),
                     'responsibility': responsibility[offset].item()})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target_index = indices[int(responsibility.argmax())]
    with torch.no_grad():
        mass = matcher.pair(query, support.index([target_index]), return_mass=True)['mass'][0]
    matrices = {'pma_mass': mass.cpu().numpy()}
    qid, sid = episode['query'][query_index], episode['support'][target_index]
    inputs = dataset.batch([dataset.by_id[qid], dataset.by_id[sid]], device)
    with torch.no_grad():
        pool = AdaptiveMeanPool1d(query.u.shape[1]).to(device)
        raw = pool(inputs[100]['x'].transpose(1, 2)).transpose(1, 2)
        matrices['raw_similarity'] = (F.normalize(raw[0], dim=-1) @ F.normalize(raw[1], dim=-1).T).cpu().numpy()
        single_inputs = {scale: {'x': item['x'], 'mask': item['mask'] if scale == 100 else torch.zeros_like(item['mask'])}
                         for scale, item in inputs.items()}
        single = model(single_inputs)
        matrices['single_scale_similarity'] = (single.v[0] @ single.v[1].T).cpu().numpy()
        matrices['multiscale_similarity'] = (query.v[0] @ support.v[target_index].T).cpu().numpy()
    np.savez_compressed(output / 'matrices.npz', **matrices)
    write_json(output / 'supports.json', {'query': qid, 'true_label': label, 'support_details': rows,
                                         'candidate_scores': scores[0].cpu().tolist(), 'classes': classes.cpu().tolist(),
                                         'selected_support': sid, 'config': state['config']})
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
    for ax, (name, matrix) in zip(axes, matrices.items()):
        im = ax.imshow(matrix, origin='lower', aspect='auto', extent=[0, 60, 0, 60], cmap='viridis')
        ax.set(title=name.replace('_', ' '), xlabel='Support time (s)', ylabel='Query time (s)')
        fig.colorbar(im, ax=ax, shrink=.75)
    fig.savefig(output / 'correspondence.pdf')
    fig.savefig(output / 'correspondence.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for i, title in enumerate(['Query', 'Support']):
        axes[0].plot((np.arange(600) + .5) / 10, inputs[100]['x'][i, :, 0].cpu(), label=title)
    axes[0].set(xlabel='Session time (s)', ylabel='Normalized downstream bytes')
    axes[0].legend()
    for i, row in enumerate(rows):
        axes[1].scatter(row['overlap'], row['local_similarity'], s=100 + 1000 * row['responsibility'])
        axes[1].annotate(f'Support {i+1}', (row['overlap'], row['local_similarity']))
    axes[1].set(xlabel='Estimated overlap', ylabel='Aligned local similarity')
    fig.savefig(output / 'support_evidence.pdf')
    fig.savefig(output / 'support_evidence.png', dpi=200)
    plt.close(fig)
    return {'query': qid, 'supports': len(rows), 'output': str(output)}
