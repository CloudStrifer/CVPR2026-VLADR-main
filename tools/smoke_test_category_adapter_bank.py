"""Check the real local CLIP backbone with synthetic tensors, not ReID accuracy."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from reid.models.category_adapter_bank import CategoryAdapterBank
from reid.models.wrapper import make_category_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    model = make_category_model(reference_checkpoint=args.reference_checkpoint, device=args.device)
    model.add_category('person')
    model.add_category('vehicle')
    model.set_trainable_categories(['person'])
    images = torch.randn(2, 3, *model.reference.input_size, device=args.device)
    with torch.no_grad():
        reference_before = model.encode_reference(images)
        vehicle_before = model.encode_category(images, 'vehicle')
    model.train()
    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    use_amp = torch.device(args.device).type == 'cuda'
    with torch.autocast(device_type=torch.device(args.device).type, enabled=use_amp):
        features = model.encode_category(images, 'person')
        loss = (features.float() - torch.randn_like(features.float())).square().mean()
    loss.backward()
    finite_nonzero_gradients = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.adapter_parameters('person'))
    assert finite_nonzero_gradients, 'person adapter did not receive valid gradients'
    assert all(p.grad is None for p in model.adapter_parameters('vehicle'))
    optimizer.step()
    model.assert_reference_unchanged()
    with torch.no_grad():
        reference_after = model.encode_reference(images)
        vehicle_after = model.encode_category(images, 'vehicle')
        person_after = model.encode_category(images, 'person')
    assert torch.equal(reference_before, reference_after), 'reference features changed'
    assert torch.equal(vehicle_before, vehicle_after), 'absent category output changed'
    assert not torch.equal(reference_after, person_after), 'current category did not change'
    model.copy_category('person', 'panda')
    assert all(p.data_ptr() != q.data_ptr() for p, q in zip(
        model.adapter_parameters('person'), model.adapter_parameters('panda')))
    model.create_temporary_head('person', 2)
    restored = CategoryAdapterBank.from_checkpoint(model.export_checkpoint(), device=args.device)
    with torch.no_grad():
        restored_person = restored.encode_category(images, 'person')
    assert torch.allclose(person_after, restored_person, atol=1e-6, rtol=1e-6)
    report = {
        'status': 'passed', 'input': 'synthetic tensors, not a ReID experiment',
        'torch_version': str(torch.__version__), 'device': args.device,
        'amp_used': use_amp, 'input_size': list(model.reference.input_size),
        'feature_shape': list(reference_after.shape),
        'reference_weight_id': model.reference.weight_id,
        'reference_signature': model.reference_signature,
        'categories': list(model.categories),
        'reference_parameters': sum(p.numel() for n, p in model.visual.named_parameters()
                                    if '.domain_adapters.' not in n),
        'parameters_per_adapter': sum(p.numel() for p in model.adapter_parameters('person')),
        'reference_features_exactly_unchanged': True,
        'absent_vehicle_features_exactly_unchanged': True,
        'current_adapter_nonzero_gradient': True,
        'current_adapter_output_max_change': (person_after - reference_after).abs().max().item(),
        'restored_output_max_difference': (person_after - restored_person).abs().max().item(),
        'temporary_heads_restored': len(restored.temporary_heads),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
