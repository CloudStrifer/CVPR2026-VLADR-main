"""Compare the optimized metric with the original scalar protocol, including ties."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from lreid_dataset.category_stream import EvaluationView, StreamSample, _path_key
from reid.evaluation.category_oracle import oracle_retrieval_metrics


def legacy_metrics(qf, gf, view):
    qf, gf = F.normalize(qf.float(), dim=1), F.normalize(gf.float(), dim=1)
    aps, rank1 = [], []
    for query, feature in zip(view.query, qf):
        order = torch.argsort(gf @ feature, descending=True, stable=True).tolist()
        matches = []
        for i in order:
            gallery = view.gallery[i]
            if _path_key(query.path) == _path_key(gallery.path):
                continue
            same = query.identity_key == gallery.identity_key
            if same and view.protocol == 'cross_camera' and query.camid == gallery.camid:
                continue
            matches.append(int(same))
        if not any(matches):
            raise ValueError('no valid positive')
        matches = torch.tensor(matches, dtype=torch.float64)
        precision = matches.cumsum(0) / torch.arange(1, len(matches) + 1, dtype=torch.float64)
        aps.append((precision * matches).sum().item() / matches.sum().item())
        rank1.append(matches[0].item())
    return dict(mAP=100 * sum(aps) / len(aps), Rank1=100 * sum(rank1) / len(rank1))


def sample(path, pid, camera='0', category='person', source='source'):
    return StreamSample(path, category, source, pid, camera, (category, source, pid), '', 'gallery')


class FastCategoryMetricTests(unittest.TestCase):
    def fixture(self):
        query = tuple(sample('q{}.png'.format(i), str(i)) for i in range(4))
        gallery = []
        for i, q in enumerate(query):
            gallery += [replace(q, path='./' + q.path), sample('same{}.png'.format(i), str(i)),
                        sample('positive{}.png'.format(i), str(i), '1'),
                        sample('other_category{}.png'.format(i), str(i), '1', category='vehicle'),
                        sample('other_source{}.png'.format(i), str(i), '1', source='another')]
        return EvaluationView('fixture', 'person', 'test', 'cross_camera', query, tuple(gallery))

    def test_exact_agreement_random_features_and_stable_ties_both_protocols(self):
        view = self.fixture()
        generator = torch.Generator().manual_seed(42)
        for protocol in ('cross_camera', 'exclude_self'):
            for tied in (False, True):
                for trial in range(3):
                    qf = torch.randn(4, 8, generator=generator)
                    gf = torch.randn(20, 8, generator=generator)
                    if tied:
                        gf[:] = 1
                    current = replace(view, protocol=protocol)
                    expected = legacy_metrics(qf, gf, current)
                    actual = oracle_retrieval_metrics(qf, gf, current)
                    self.assertEqual(expected, {k: actual[k] for k in expected})

    def test_paths_resolved_once_per_image_and_callback_does_not_change_metrics(self):
        view = self.fixture()
        qf, gf = torch.ones(4, 8), torch.ones(20, 8)
        events = []
        with patch('reid.evaluation.category_oracle._path_key', wraps=_path_key) as resolve:
            actual = oracle_retrieval_metrics(qf, gf, view, progress=events.append)
        self.assertEqual(resolve.call_count, len(view.query) + len(view.gallery))
        self.assertEqual(actual, oracle_retrieval_metrics(qf, gf, view))
        self.assertEqual(events[-1]['phase'], 'ranking_complete')
        self.assertEqual(events[-1]['queries'], len(view.query))

    def test_missing_positive_and_invalid_features_remain_errors(self):
        view = self.fixture()
        current = replace(view, gallery=(sample('negative.png', 'unknown'),))
        with self.assertRaisesRegex(ValueError, 'no valid gallery positive'):
            oracle_retrieval_metrics(torch.ones(4, 8), torch.ones(1, 8), current)
        with self.assertRaises(FloatingPointError):
            oracle_retrieval_metrics(torch.zeros(4, 8), torch.ones(20, 8), view)


if __name__ == '__main__':
    unittest.main()
