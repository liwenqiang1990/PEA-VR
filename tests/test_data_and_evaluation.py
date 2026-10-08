import copy
import csv
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from pea_vr.data.adapters import read_longenough_packets, read_longenough_player, read_ydms_packets, read_ydms_player
from pea_vr.data.download import extract_archive
from pea_vr.data.prepare import PreparedDataset
from pea_vr.data.protocols import EpisodeSampler, verify_protocol_manifest
from pea_vr.evaluation import benchmark, calibrate_threshold, correspondence, evaluate, open_set_metrics, player_frontier
from pea_vr.inference import enroll, recognize
from pea_vr.models.pooling import AdaptiveMeanPool1d
from pea_vr.training.engine import train
from pea_vr.utils import sha256, write_json, write_jsonl
from test_training_and_protocols import small_config


def test_official_longenough_schema_and_nan_qoe(tmp_path):
    packets = tmp_path / 'trace.log'
    packets.write_text('0,s,60,100000,0,0\n100000000,r,1500,100100,1,1\n61000000000,r,100,161000,1,1\n')
    times, direction, lengths, origin, duration = read_longenough_packets(packets)
    assert origin == 100 and duration == 61
    assert direction.tolist() == [1, 0, 0]
    player = tmp_path / 'trace.qoe.log'
    player.write_text('100000,btr,1000\n100000,lat,NaN\n100100,playbackPlaying\n100100,pbr,1\n')
    quality, features, states = read_longenough_player(player, origin)
    assert np.isfinite(features).all() and (quality == 1000).all()
    json.dumps(states, allow_nan=False)


def test_official_ydms_schema_udp_and_player_states(tmp_path):
    packets = tmp_path / 'video_traffic.csv'
    packets.write_text('timestamp,ipSrc,ipDst,tcpLen,udpLen,payloadProtocolNumber\n100,10.0.0.2,8.8.8.8,0,100,17\n161,8.8.8.8,10.0.0.2,0,1000,17\n')
    t, d, l, origin, duration = read_ydms_packets(packets)
    assert d.tolist() == [1, 0] and l.tolist() == [100, 1000] and duration == 61
    player = tmp_path / 'application_data.csv'
    player.write_text('timestamp,fmt,bh,videoid,stalling,phase\n100,248,10000,video1,0,filling\n130,242,100,video1,1,stalling\n')
    identity, quality, features, states = read_ydms_player(player, origin)
    assert identity == 'video1' and quality[0] == 1080 and quality[-1] == 240
    assert features[0] == .5 and states['buffer_seconds'][0] == 10


def test_archive_traversal_rejected(tmp_path):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape.txt', 'bad')
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / 'output')
    assert not (tmp_path / 'escape.txt').exists()


def test_ydms_startup_wait_does_not_backfill_future_quality(tmp_path):
    player = tmp_path / 'application_data.csv'
    player.write_text('timestamp,fmt,bh,videoid,stalling,phase\n110,248,10000,video1,0,filling\n170,248,10000,video1,0,filling\n')
    _, quality, features, _ = read_ydms_player(player, 100.)
    assert (quality[:5] == 0).all() and (quality[5:] == 1080).all()
    assert features[0] == pytest.approx(5 / 30)
    with pytest.raises(ValueError, match='No player observations'):
        read_ydms_player(player, 0.)


@pytest.mark.parametrize('length,count', [(600, 30), (30, 20), (30, 60), (120, 40), (3, 8)])
def test_deterministic_pool_matches_adaptive_definition(length, count):
    values = torch.randn(2, 3, length, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.adaptive_avg_pool1d(values, count)
    result = AdaptiveMeanPool1d(count)(values)
    torch.testing.assert_close(result, expected)
    a = torch.autograd.grad(result.square().sum(), values)[0]
    b = torch.autograd.grad(expected.square().sum(), values)[0]
    torch.testing.assert_close(a, b)


def test_metrics_and_player_event_hold():
    assert open_set_metrics([1, 1, 0, 0], [.9, .8, .2, .1]) == {'auroc': 1., 'fpr95': 0.}
    row = {'offset': 0, 'player': {'state_events': [[0, 'playbackPlaying']],
                                 'events': {'pbr': [[0, 1.], [120, 1.]],
                                            'buf': [[t, 10.] for t in range(121)]}}}
    np.testing.assert_allclose(player_frontier(row, [30, 60, 90]), [40, 70, 100])
    with pytest.raises(ValueError, match='gap'):
        player_frontier(row, [130])


def protocol_fixture(data, root, split):
    root.mkdir(parents=True)
    sampler = EpisodeSampler(data, split=split, seed=12)
    episodes = [sampler.sample(2 if split == 'test' else 1, 2, 2) for _ in range(2)]
    write_jsonl(root / 'random_mixed.jsonl', episodes)
    write_json(root / 'protocols.json', {'dataset_fingerprint': data.metadata['fingerprint'], 'split': split,
                                        'protocols': {'random_mixed': 2}, 'files': {'random_mixed.jsonl': sha256(root / 'random_mixed.jsonl')}})
    return root


def test_end_to_end_evaluation_enrollment_and_integrity(prepared_fixture, tmp_path):
    config = small_config()
    config['training']['episodes'] = 2
    run = tmp_path / 'run'
    train(config, prepared_fixture, run)
    data = PreparedDataset(prepared_fixture)
    protocols = protocol_fixture(data, tmp_path / 'protocols', 'test')
    report = evaluate(run / 'latest.pt', prepared_fixture, protocols, tmp_path / 'eval', fractions=[.2, 1.], device='cpu')
    assert len(report) == 2 and all(r['episodes'] == 2 for r in report)
    corr = correspondence(run / 'latest.pt', prepared_fixture, tmp_path / 'correspondence.json', device='cpu')
    assert corr['tokens'] > 0 and corr['overlap_mae'] >= 0
    manifest = tmp_path / 'enrollment.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['label', 'path', 'format'])
        for i in [48, 49, 60, 61]:
            writer.writerow([data.rows[i]['identity'], str(data.root / data.rows[i]['packet_path']), 'npz'])
    gallery = tmp_path / 'gallery.pt'
    result = enroll(run / 'latest.pt', manifest, gallery, 'cpu')
    assert result['identities'] == 2 and result['sessions'] == 4
    prediction = recognize(run / 'latest.pt', gallery, data.root / data.rows[50]['packet_path'], 'npz', tmp_path / 'prediction.json', device='cpu')
    assert set(prediction['scores']) == {'4', '5'} and prediction['label'] in ['4', '5']
    (protocols / 'random_mixed.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        verify_protocol_manifest(data, protocols)
