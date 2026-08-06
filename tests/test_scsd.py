import unittest

import torch

from reid.loss.scsd import (
    SCSDLoss,
    attribute_relation_matrix,
    global_relation_distillation,
    semantic_transfer_matrix,
)
from reid.models.attribute_pooling import TextConditionedAttributePooler


class AttributePoolingTests(unittest.TestCase):
    def test_text_query_selects_compatible_patch(self):
        pooler = TextConditionedAttributePooler(temperature=0.01)
        patches = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
        )
        texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        attributes = pooler(patches, texts)
        self.assertEqual(tuple(attributes.shape), (1, 2, 2))
        self.assertGreater(float(attributes[0, 0, 0]), 0.99)
        self.assertGreater(float(attributes[0, 1, 1]), 0.99)


class SCSDTests(unittest.TestCase):
    def test_relation_rows_are_probabilities(self):
        features = torch.randn(4, 3, 8)
        relations = attribute_relation_matrix(features, temperature=0.1)
        self.assertEqual(tuple(relations.shape), (3, 4, 4))
        self.assertTrue(
            torch.allclose(
                relations.sum(dim=-1),
                torch.ones(3, 4),
                atol=1e-6,
            )
        )

    def test_index_mode_is_identity_routing(self):
        similarity = torch.randn(3, 3)
        routing, confidence = semantic_transfer_matrix(
            similarity,
            mode='index',
            threshold=0.9,
        )
        self.assertTrue(torch.equal(routing, torch.eye(3)))
        self.assertTrue(torch.equal(confidence, torch.ones(3)))

    def test_incompatible_attributes_return_finite_zero(self):
        old_features = torch.randn(4, 2, 6)
        current_features = torch.randn(4, 3, 6, requires_grad=True)
        old_text = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        current_text = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        )
        loss_fn = SCSDLoss(
            mode='scsd',
            semantic_threshold=0.5,
        )
        loss, diagnostics = loss_fn(
            old_features,
            current_features,
            old_text,
            current_text,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(diagnostics['effective_attributes']), 0.0)
        loss.backward()
        self.assertIsNotNone(current_features.grad)

    def test_identical_relations_have_zero_index_loss(self):
        features = torch.randn(5, 3, 7)
        texts = torch.randn(3, 4)
        loss_fn = SCSDLoss(mode='index')
        loss, diagnostics = loss_fn(
            features,
            features.clone().requires_grad_(True),
            texts,
            texts,
        )
        self.assertLess(abs(float(loss.detach())), 1e-6)
        self.assertEqual(float(diagnostics['effective_attributes']), 3.0)

    def test_variable_attribute_counts_are_supported(self):
        old_features = torch.randn(6, 2, 5)
        current_features = torch.randn(6, 4, 5, requires_grad=True)
        old_text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        current_text = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [-1.0, 0.0],
            ]
        )
        loss_fn = SCSDLoss(
            mode='scsd',
            semantic_threshold=0.1,
        )
        loss, diagnostics = loss_fn(
            old_features,
            current_features,
            old_text,
            current_text,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(diagnostics['effective_attributes']), 3.0)
        loss.backward()
        self.assertTrue(torch.isfinite(current_features.grad).all())

    def test_global_relation_is_zero_for_identical_features(self):
        features = torch.randn(6, 8)
        loss = global_relation_distillation(features, features.clone())
        self.assertLess(abs(float(loss.detach())), 1e-6)

    def test_global_relation_updates_student_only(self):
        old_features = torch.randn(6, 8)
        current_features = torch.randn(6, 8, requires_grad=True)
        loss = global_relation_distillation(
            old_features,
            current_features,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(current_features.grad)
        self.assertTrue(torch.isfinite(current_features.grad).all())


if __name__ == '__main__':
    unittest.main()
