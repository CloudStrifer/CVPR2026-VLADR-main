"""Real CLIP memory smoke test on generated images; not a ReID accuracy experiment."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw

from lreid_dataset.category_stream import load_category_stream
from lreid_dataset.category_stream_loaders import build_stage_prototype_loaders
from reid.memory.ecpm import IdentityPrototypeMemory
from reid.models.category_adapter_bank import build_category_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    model = build_category_model(reference_checkpoint=args.reference_checkpoint, device=args.device)
    reports = []
    reference_error = 0.
    with tempfile.TemporaryDirectory(prefix='identity_memory_') as temporary:
        directory = Path(temporary)
        example = ROOT / 'config/category_progressive_example.json'
        config = json.loads(example.read_text(encoding='utf-8'))
        config['data_root'] = str(directory / 'images')
        for stage in config['stages']:
            for category in stage['categories']:
                category['train_manifest'] = str((example.parent / category['train_manifest']).resolve())
        config['evaluation'] = []
        path = directory / 'stream.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        stream = load_category_stream(path)
        memory = IdentityPrototypeMemory(model, stream)
        old_rows = []
        for stage_id in ('t1', 't2'):
            stage = stream.stage(stage_id)
            for view in stage.categories:
                model.add_category(view.category)
                for i, sample in enumerate(view.samples):
                    image = Image.new('RGB', (48, 64), (10, 30, 50))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((5 + i, 5, 35, 55), fill=(30 + i * 40, 60, 200 - i * 30))
                    image_path = Path(sample.path)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
            model.set_trainable_categories([v.category for v in stage.categories])
            model.train()
            # Nonzero adapters exercise reference bypass while the student is trainable.
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(torch.randn_like(parameter) * .01)
            with torch.autocast(device_type=torch.device(args.device).type,
                                enabled=torch.device(args.device).type == 'cuda'):
                prepared = memory.prepare_stage(model, stage, batch_size=3)
            assert memory.processed_stages == tuple(r['processed_stages'][-1] for r in reports)
            # Independently compute one identity mean from raw reference features.
            view = stage.categories[0]
            loader = build_stage_prototype_loaders(stage, model.make_transforms()[1], batch_size=4)[view.category]
            batch = next(iter(loader))
            raw = model.encode_reference(batch['images'].to(args.device)).cpu()
            key = view.identity_keys[0]
            mean = raw[[i for i, k in enumerate(batch['identity_keys']) if k == key]].mean(0)
            expected = mean / mean.norm()
            actual = next(row['vector'] for row in prepared['rows'] if row['identity_key'] == key)
            reference_error = max(reference_error, (expected - actual).abs().max().item())
            assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
            reports.append(memory.commit_stage(prepared))
            for before, after in zip(old_rows, memory.state_dict()['rows']):
                assert torch.equal(before['vector'], after['vector']), 'old prototype changed'
            checkpoint = directory / (stage_id + '_memory.pt')
            memory.save(checkpoint)
            memory = IdentityPrototypeMemory.load(checkpoint, model, stream)
            old_rows = memory.state_dict()['rows']
            for view in stage.categories:
                for sample in view.samples:
                    Path(sample.path).unlink()
        assert memory.summary()['categories'] == {'person': 4, 'vehicle': 2, 'panda': 2}
        model.assert_reference_unchanged()
    report = dict(status='passed', data='generated rectangles; no real ReID accuracy claim',
                  device=args.device, outer_autocast=torch.device(args.device).type == 'cuda',
                  feature_dim=model.feature_dim, reference_signature=model.reference_signature,
                  reference_weights_unchanged=True, historical_prototypes_exactly_unchanged=True,
                  old_images_removed_before_next_stage=True, save_restore_each_stage=True,
                  hand_mean_max_absolute_error=reference_error, stages=reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
