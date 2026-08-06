import unittest

import torch
import torch.nn as nn

from reid.utils.feature_tools import (
    build_cross_modal_identity_anchors,
    category_semantic_subspace,
    visual_identity_statistics,
)


class _IdentityProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x=None, get_image=False, **kwargs):
        del kwargs
        if not get_image:
            raise ValueError('test model only supports get_image=True')
        return x * self.scale


class IdentityAnchorTests(unittest.TestCase):
    def test_visual_statistics_average_normalized_features(self):
        model = _IdentityProjection()
        model.train()
        loader = [
            (
                torch.tensor(
                    [
                        [2.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 3.0],
                    ]
                ),
                None,
                torch.tensor([0, 0, 1]),
                None,
                None,
            )
        ]

        prototypes, center = visual_identity_statistics(
            model,
            loader,
            num_classes=2,
        )

        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(torch.allclose(prototypes, expected))
        self.assertTrue(
            torch.allclose(
                center,
                torch.tensor([[0.5, 0.5]]),
            )
        )
        self.assertTrue(model.training)

    def test_text_mode_matches_normalized_prompt_features(self):
        text = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
        anchors = build_cross_modal_identity_anchors(
            text,
            mode='text',
        )
        self.assertTrue(
            torch.allclose(
                anchors,
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            )
        )

    def test_centered_mode_is_invariant_to_common_visual_shift(self):
        text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prototypes = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
        center = prototypes.mean(dim=0, keepdim=True)
        anchors = build_cross_modal_identity_anchors(
            text,
            prototypes,
            center,
            mode='centered',
        )

        shift = torch.tensor([[2.0, -3.0]])
        shifted_anchors = build_cross_modal_identity_anchors(
            text,
            prototypes + shift,
            center + shift,
            mode='centered',
        )

        self.assertTrue(torch.allclose(anchors, shifted_anchors))
        self.assertTrue(
            torch.allclose(
                anchors.norm(dim=1),
                torch.ones(2),
            )
        )

    def test_prototype_and_centered_modes_are_distinct(self):
        text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prototypes = torch.tensor([[0.9, 0.4], [0.5, 0.8]])
        prototype_anchors = build_cross_modal_identity_anchors(
            text,
            prototypes,
            mode='prototype',
        )
        centered_anchors = build_cross_modal_identity_anchors(
            text,
            prototypes,
            mode='centered',
        )
        self.assertFalse(
            torch.allclose(prototype_anchors, centered_anchors)
        )

    def test_category_semantic_subspace_is_orthonormal(self):
        category_text = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        )
        basis = category_semantic_subspace(category_text)
        self.assertEqual(tuple(basis.shape), (3, 2))
        self.assertTrue(
            torch.allclose(
                basis.t() @ basis,
                torch.eye(2),
                atol=1e-6,
            )
        )

    def test_ocia_removes_category_subspace_variation(self):
        text = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        prototypes = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [-1.0, 0.0, -1.0],
            ]
        )
        category_basis = torch.tensor(
            [
                [1.0],
                [0.0],
                [0.0],
            ]
        )
        anchors = build_cross_modal_identity_anchors(
            text,
            prototypes,
            mode='ocia',
            category_basis=category_basis,
        )

        altered_prototypes = torch.tensor(
            [
                [6.0, 0.0, 1.0],
                [-8.0, 0.0, -1.0],
            ]
        )
        altered_anchors = build_cross_modal_identity_anchors(
            text,
            altered_prototypes,
            mode='ocia',
            category_basis=category_basis,
        )
        self.assertTrue(torch.allclose(anchors, altered_anchors))
        self.assertTrue(
            torch.allclose(
                anchors.norm(dim=1),
                torch.ones(2),
            )
        )

    def test_zero_ocia_residual_weight_matches_text_mode(self):
        text = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        prototypes = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
        category_basis = torch.tensor([[1.0], [0.0]])
        text_anchors = build_cross_modal_identity_anchors(
            text,
            mode='text',
        )
        ocia_anchors = build_cross_modal_identity_anchors(
            text,
            prototypes,
            category_basis=category_basis,
            mode='ocia',
            residual_weight=0.0,
        )
        self.assertTrue(torch.allclose(text_anchors, ocia_anchors))

    def test_ocia_requires_category_basis(self):
        with self.assertRaises(ValueError):
            build_cross_modal_identity_anchors(
                torch.eye(2),
                torch.eye(2),
                mode='ocia',
            )


if __name__ == '__main__':
    unittest.main()
