import unittest

import torch

from reid.models.CLIP_ReID.model.clip.model import (
    ResidualAttentionBlock,
    VisionTransformer,
)


class DomainAdapterTests(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_block_output(self):
        torch.manual_seed(1)
        block = ResidualAttentionBlock(d_model=8, n_head=2).eval()
        inputs = torch.randn(5, 3, 8)
        with torch.no_grad():
            baseline = block(inputs)

        block.add_domain_adapter('panda', bottleneck_dim=2, scale=1.0)
        block.set_active_domain_adapter('panda')
        with torch.no_grad():
            adapted = block(inputs)

        self.assertTrue(torch.allclose(baseline, adapted, atol=1e-7))

    def test_adapter_switch_selects_independent_parameters(self):
        block = ResidualAttentionBlock(d_model=8, n_head=2).eval()
        panda = block.add_domain_adapter('panda', bottleneck_dim=2)
        vehicle = block.add_domain_adapter('vehicle', bottleneck_dim=2)
        with torch.no_grad():
            panda.up.bias.fill_(0.25)
            vehicle.up.bias.fill_(-0.25)
        inputs = torch.randn(4, 2, 8)

        block.set_active_domain_adapter('panda')
        panda_output = block(inputs)
        block.set_active_domain_adapter('vehicle')
        vehicle_output = block(inputs)

        self.assertFalse(torch.allclose(panda_output, vehicle_output))

    def test_vision_adapter_bank_uses_only_last_blocks(self):
        vision = VisionTransformer(
            h_resolution=2,
            w_resolution=2,
            patch_size=2,
            stride_size=2,
            width=8,
            layers=4,
            heads=2,
            output_dim=4,
        )
        vision.add_domain_adapter(
            'panda',
            last_blocks=2,
            bottleneck_dim=2,
        )
        blocks = list(vision.transformer.resblocks)
        self.assertNotIn('panda', blocks[0].domain_adapters)
        self.assertNotIn('panda', blocks[1].domain_adapters)
        self.assertIn('panda', blocks[2].domain_adapters)
        self.assertIn('panda', blocks[3].domain_adapters)
        self.assertEqual(vision.domain_adapter_names(), ('panda',))

        parameter_count = sum(
            parameter.numel()
            for parameter in vision.domain_adapter_parameters('panda')
        )
        self.assertEqual(parameter_count, 84)

    def test_old_adapter_receives_no_gradient_when_current_is_active(self):
        block = ResidualAttentionBlock(d_model=8, n_head=2)
        old_adapter = block.add_domain_adapter('old', bottleneck_dim=2)
        current_adapter = block.add_domain_adapter(
            'current',
            bottleneck_dim=2,
        )
        for parameter in block.parameters():
            parameter.requires_grad = False
        for parameter in current_adapter.parameters():
            parameter.requires_grad = True
        block.set_active_domain_adapter('current')

        loss = block(torch.randn(4, 2, 8)).sum()
        loss.backward()

        self.assertTrue(
            all(parameter.grad is None for parameter in old_adapter.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in current_adapter.parameters()
            )
        )

    def test_adapter_state_dict_round_trip(self):
        source = VisionTransformer(
            h_resolution=2,
            w_resolution=2,
            patch_size=2,
            stride_size=2,
            width=8,
            layers=4,
            heads=2,
            output_dim=4,
        )
        source.add_domain_adapter('panda', 2, 2, 1.0)
        with torch.no_grad():
            for parameter in source.domain_adapter_parameters('panda'):
                parameter.add_(0.1)

        target = VisionTransformer(
            h_resolution=2,
            w_resolution=2,
            patch_size=2,
            stride_size=2,
            width=8,
            layers=4,
            heads=2,
            output_dim=4,
        )
        target.add_domain_adapter('panda', 2, 2, 1.0)
        target.load_state_dict(source.state_dict())

        source_params = list(source.domain_adapter_parameters('panda'))
        target_params = list(target.domain_adapter_parameters('panda'))
        self.assertTrue(
            all(
                torch.equal(source_param, target_param)
                for source_param, target_param in zip(
                    source_params,
                    target_params,
                )
            )
        )


if __name__ == '__main__':
    unittest.main()
