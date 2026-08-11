import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.evaluation.adapter_fusion import (
    adapter_routing_scores,
    routing_bank_to,
    semantic_debiased_residual,
    topk_adapter_weights,
    validate_adapter_routing_bank,
)
from reid.evaluation.fast_test import (
    _osaf_descriptor,
    _use_osaf_for_domain,
)


def _routing_bank():
    return {
        'person': {
            'semantic_key': torch.tensor([1.0, 0.0, 0.0, 0.0]),
            'visual_prototypes': torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ),
            'category_basis': torch.tensor(
                [
                    [1.0],
                    [0.0],
                    [0.0],
                    [0.0],
                ]
            ),
        },
        'vehicle': {
            'semantic_key': torch.tensor([0.0, 1.0, 0.0, 0.0]),
            'visual_prototypes': torch.tensor(
                [[0.0, 1.0, 0.0, 0.0]]
            ),
            'category_basis': torch.tensor(
                [
                    [0.0],
                    [1.0],
                    [0.0],
                    [0.0],
                ]
            ),
        },
    }


class AdapterFusionTests(unittest.TestCase):
    def test_osaf_routing_scope_controls_seen_domains(self):
        bank = _routing_bank()
        learned = {'person'}

        unseen_args = SimpleNamespace(adapter_routing_scope='unseen')
        self.assertFalse(
            _use_osaf_for_domain(bank, 'person', learned, unseen_args)
        )
        self.assertTrue(
            _use_osaf_for_domain(bank, 'vehicle', learned, unseen_args)
        )

        all_args = SimpleNamespace(adapter_routing_scope='all')
        self.assertTrue(
            _use_osaf_for_domain(bank, 'person', learned, all_args)
        )
        self.assertTrue(
            _use_osaf_for_domain(bank, 'vehicle', learned, all_args)
        )

        legacy_args = SimpleNamespace()
        self.assertFalse(
            _use_osaf_for_domain(bank, 'person', learned, legacy_args)
        )
        self.assertFalse(
            _use_osaf_for_domain(None, 'vehicle', learned, all_args)
        )

    def test_validate_and_move_routing_bank(self):
        bank = _routing_bank()
        self.assertEqual(validate_adapter_routing_bank(bank), 4)
        moved = routing_bank_to(bank, device='cpu', dtype=torch.float64)
        self.assertEqual(moved['person']['semantic_key'].dtype, torch.float64)
        self.assertEqual(validate_adapter_routing_bank(moved), 4)

    def test_semantic_debias_removes_basis_component(self):
        residual = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        basis = _routing_bank()['person']['category_basis']
        debiased = semantic_debiased_residual(
            residual,
            basis,
            strength=1.0,
        )
        self.assertTrue(
            torch.equal(
                debiased,
                torch.tensor([[0.0, 3.0, 4.0, 5.0]]),
            )
        )
        unchanged = semantic_debiased_residual(
            residual,
            basis,
            strength=0.0,
        )
        self.assertTrue(torch.equal(unchanged, residual))

    def test_routing_combines_semantic_and_visual_similarity(self):
        inputs = torch.tensor(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
            ]
        )
        names, scores = adapter_routing_scores(
            inputs,
            _routing_bank(),
            semantic_weight=0.5,
        )
        self.assertEqual(names, ('person', 'vehicle'))
        self.assertEqual(scores.argmax(dim=1).tolist(), [0, 1])

    def test_topk_weights_are_normalized_and_bounded(self):
        scores = torch.tensor(
            [
                [0.9, 0.1, 0.5],
                [0.2, 0.8, 0.4],
            ]
        )
        indices, weights = topk_adapter_weights(
            scores,
            topk=2,
            temperature=0.1,
        )
        self.assertEqual(tuple(indices.shape), (2, 2))
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        all_indices, all_weights = topk_adapter_weights(
            scores,
            topk=10,
            temperature=1.0,
        )
        self.assertEqual(tuple(all_indices.shape), (2, 3))
        self.assertEqual(tuple(all_weights.shape), (2, 3))

    def test_osaf_descriptor_routes_and_fuses_selected_adapter(self):
        class DummyAdapterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))
                self.active = None

            def domain_adapter_names(self):
                return ('person', 'vehicle')

            def get_active_adapter(self):
                return self.active

            def set_active_adapter(self, name=None):
                self.active = name

            def forward(self, inputs):
                descriptor = inputs.clone()
                if self.active == 'person':
                    descriptor[:, 0] += 2.0
                    descriptor[:, -2] += 1.0
                elif self.active == 'vehicle':
                    descriptor[:, 1] += 3.0
                    descriptor[:, -1] += 1.0
                return descriptor

        bank = {
            'person': {
                'semantic_key': torch.tensor([1.0, 0.0]),
                'visual_prototypes': torch.tensor([[1.0, 0.0]]),
                'category_basis': torch.tensor([[0.0], [1.0]]),
            },
            'vehicle': {
                'semantic_key': torch.tensor([0.0, 1.0]),
                'visual_prototypes': torch.tensor([[0.0, 1.0]]),
                'category_basis': torch.tensor([[1.0], [0.0]]),
            },
        }
        args = SimpleNamespace(
            adapter_semantic_weight=0.5,
            adapter_topk=1,
            adapter_routing_temperature=0.1,
            adapter_debias_strength=1.0,
            adapter_fusion_weight=1.0,
        )
        model = DummyAdapterModel()
        inputs = torch.tensor(
            [
                [0.2, 0.3, 1.0, 0.0],
                [0.2, 0.3, 0.0, 1.0],
            ]
        )
        descriptor, names, top1 = _osaf_descriptor(
            model,
            inputs,
            bank,
            args,
        )
        self.assertEqual(names, ('person', 'vehicle'))
        self.assertEqual(top1.tolist(), [0, 1])
        legacy_expected = F.normalize(
            torch.tensor(
                [
                    [0.2, 0.3, 2.0, 0.0],
                    [0.2, 0.3, 0.0, 2.0],
                ]
            ),
            dim=1,
        )
        self.assertTrue(
            torch.allclose(descriptor, legacy_expected, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                descriptor.norm(dim=1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertIsNone(model.get_active_adapter())

        args.adapter_main_fusion = 'top1'
        args.adapter_debias_strength = 0.0
        top1_descriptor, _, _ = _osaf_descriptor(
            model,
            inputs,
            bank,
            args,
        )
        oracle_expected = F.normalize(
            torch.tensor(
                [
                    [2.2, 0.3, 2.0, 0.0],
                    [0.2, 3.3, 0.0, 2.0],
                ]
            ),
            dim=1,
        )
        self.assertTrue(
            torch.allclose(top1_descriptor, oracle_expected, atol=1e-6)
        )
        self.assertIsNone(model.get_active_adapter())


if __name__ == '__main__':
    unittest.main()
