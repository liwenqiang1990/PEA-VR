import importlib.util
from pathlib import Path

import numpy as np
import requests
import pytest

from pea_vr.data.download import download_file, file_digest
from pea_vr.data.remote_zip import HTTPRangeFile
from pea_vr.evaluation import open_set_metrics
from pea_vr.tables import export_tables
from pea_vr.utils import write_json, read_json
from pea_vr.experiments import method_config
from pea_vr.reporting import collect_results, plot_results


class Response:
    def __init__(self, data, start, total, etag='stable'):
        self.status_code = 206
        self.content = data
        self.headers = {'Content-Range': f'bytes {start}-{start+len(data)-1}/{total}',
                        'Content-Length': str(len(data)), 'ETag': etag}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.content


def test_transfer_resume_retry_and_fresh_figshare_redirect(tmp_path, monkeypatch):
    content = b'0123456789abcdefghijklmnopqrstuvwxyz'
    target = tmp_path / 'file.bin'
    target.with_suffix('.bin.part').write_bytes(content[:10])
    calls = []
    def get(url, headers, **kwargs):
        calls.append((url, headers))
        if len(calls) == 1:
            raise requests.ConnectionError('temporary interruption')
        assert headers['Range'] == f'bytes=10-{len(content)-1}'
        return Response(content[10:], 10, len(content))
    monkeypatch.setattr('pea_vr.data.download.requests.get', get)
    monkeypatch.setattr('pea_vr.data.download.time.sleep', lambda _: None)
    import hashlib
    download_file('https://ndownloader.figshare.com/files/1', target, len(content), hashlib.md5(content).hexdigest())
    assert target.read_bytes() == content and len(calls) == 2
    assert all('pea_vr_request=' in url for url, _ in calls)
    assert not target.with_suffix('.bin.part').exists()


def test_http_range_cache_and_seek():
    content = bytes(range(100))
    class Session:
        calls = 0
        def get(self, url, headers, **kwargs):
            self.calls += 1
            start, end = map(int, headers['Range'][6:].split('-'))
            return Response(content[start:end+1], start, len(content))
    session = Session()
    stream = HTTPRangeFile(session, 'https://example.invalid/archive.zip', len(content))
    stream.seek(-10, 2)
    assert stream.read(3) == content[90:93]
    assert stream.read(3) == content[93:96] and session.calls == 1
    stream.seek(0)
    assert stream.read(4) == content[:4] and session.calls == 2


def test_independent_open_set_recomputation_with_ties():
    path = Path(__file__).parents[1] / 'scripts' / 'recompute_metrics.py'
    spec = importlib.util.spec_from_file_location('independent_metrics', path)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    rng = np.random.default_rng(73)
    scores = rng.integers(0, 8, 100)
    known = np.arange(100) < 40
    expected = open_set_metrics(known, scores)
    auc, fpr = independent.auc_and_fpr95(known, scores)
    assert auc == expected['auroc'] and fpr == expected['fpr95']


def test_paper_tables_reject_incomplete_runs(tmp_path):
    matrix = tmp_path / 'matrix.json'
    write_json(matrix, {'runs': [{'name': 'not_trained'}]})
    with pytest.raises(ValueError, match='complete training'):
        export_tables(matrix, tmp_path / 'runs', tmp_path / 'tables')
    assert read_json(tmp_path / 'tables' / 'status.json')['complete'] is False
    result = export_tables(matrix, tmp_path / 'runs', tmp_path / 'inspection', strict=False)
    assert result['missing_runs_or_reports'] == 1


def test_truncation_plot_does_not_mix_representation_ablations(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    captured = []
    original = Axes.plot
    def capture(self, x, y, *args, **kwargs):
        captured.append((list(x), list(y)))
        return original(self, x, y, *args, **kwargs)
    monkeypatch.setattr(Axes, 'plot', capture)
    for method, accuracy in [('pea_vr', .8), ('no_ali', .2)]:
        for fraction in [.2, 1.]:
            write_json(tmp_path / 'runs' / f'{method}_{fraction}.json', {
                'config': method_config(method), 'scenario': 'cross_mode', 'fraction': fraction,
                'accuracy': accuracy, 'split': 'test'})
    collect_results(tmp_path / 'runs', tmp_path / 'summary')
    plot_results(tmp_path / 'summary', tmp_path / 'figures')
    assert captured == [([.2, 1.], [80., 80.])]
    assert (tmp_path / 'figures' / 'query_truncation.pdf').exists()
