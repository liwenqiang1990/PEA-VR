from .amp import AMPEncoder
from .baselines import ConvEmbedding, MetaFlowEncoder, TransformerEmbedding


def build_model(config):
    m = config['model']
    kind = m['kind']
    if kind in ('pea_vr', 'global_amp'):
        return AMPEncoder(scales=config['data']['scales'], channels=m['channels'], tokens=m['tokens'],
                          global_dim=m['global_dim'], projection_hidden=m['projection_hidden'],
                          alignment_dim=m['alignment_dim'], dropout=m['dropout'], fusion=m['fusion'],
                          local=config['matching']['alignment'] != 'none')
    if kind in ('protonet', 'deepmetric', 'coda'):
        return ConvEmbedding('deepmetric' if kind == 'deepmetric' else 'protonet', m['global_dim'])
    if kind == 'cl_metaflow':
        return MetaFlowEncoder(m['global_dim'])
    if kind == 'transformer':
        return TransformerEmbedding(m['global_dim'])
    raise ValueError(kind)
