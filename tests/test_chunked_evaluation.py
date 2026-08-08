import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from reid.evaluation.fast_test import _rank


class ChunkedEvaluationTests(unittest.TestCase):
    def test_chunked_market1501_metrics_match_full_matrix(self):
        torch.manual_seed(7)
        query_features = F.normalize(torch.randn(6, 16), dim=1)
        gallery_features = F.normalize(
            torch.cat(
                [
                    query_features + 0.01 * torch.randn(6, 16),
                    torch.randn(6, 16),
                ],
                dim=0,
            ),
            dim=1,
        )
        query_pids = torch.arange(6, dtype=torch.long)
        gallery_pids = torch.cat(
            [
                torch.arange(6, dtype=torch.long),
                torch.arange(6, 12, dtype=torch.long),
            ]
        )
        query_cameras = torch.zeros(6, dtype=torch.long)
        gallery_cameras = torch.ones(12, dtype=torch.long)

        full_args = SimpleNamespace(
            save_evaluation=False,
            eval_full_matrix_max_elements=-1,
            eval_query_chunk_size=2,
        )
        chunked_args = SimpleNamespace(
            save_evaluation=False,
            eval_full_matrix_max_elements=0,
            eval_query_chunk_size=2,
        )
        full = _rank(
            query_features,
            gallery_features,
            query_pids,
            gallery_pids,
            query_cameras,
            gallery_cameras,
            full_args,
        )
        chunked = _rank(
            query_features,
            gallery_features,
            query_pids,
            gallery_pids,
            query_cameras,
            gallery_cameras,
            chunked_args,
        )

        for metric in ('mAP', 'Rank1', 'Rank5', 'Rank10'):
            self.assertAlmostEqual(full[metric], chunked[metric], places=6)


if __name__ == '__main__':
    unittest.main()
