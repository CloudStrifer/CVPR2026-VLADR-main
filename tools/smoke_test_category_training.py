"""Short real-CLIP CE+Triplet training on generated images; not ReID results."""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from lreid_dataset.category_stream import load_category_stream
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.models.wrapper import make_category_model
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = make_category_model(reference_checkpoint=args.reference_checkpoint, device=args.device)
    probe = torch.randn(2, 3, 224, 224, device=args.device)
    reference_before = model.encode_reference(probe)
    training = CategoryTrainingConfig(epochs=2, iterations_per_epoch=2,
                                      adapter_lr=0.0003, head_lr=0.01,
                                      amp=torch.device(args.device).type == 'cuda')
    reports = []
    with tempfile.TemporaryDirectory(prefix='category_training_') as directory:
        directory = Path(directory)
        example = ROOT / 'config/category_progressive_example.json'
        payload = json.loads(example.read_text(encoding='utf-8'))
        payload['data_root'] = str(directory / 'images')
        for stage in payload['stages']:
            for entry in stage['categories']:
                entry['train_manifest'] = str((example.parent / entry['train_manifest']).resolve())
        payload['evaluation'] = []
        config_path = directory / 'stream.json'
        config_path.write_text(json.dumps(payload), encoding='utf-8')
        stream = load_category_stream(config_path)
        vehicle_before = None
        for stage_id in ('t1', 't2'):
            stage = stream.stage(stage_id)
            # Create only the current stage's images; future images remain absent.
            for category in stage.categories:
                labels = category.label_map
                for sample in category.samples:
                    path = Path(sample.path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image = Image.new('RGB', (40, 48), (20, 20, 20))
                    draw = ImageDraw.Draw(image)
                    color = (220, 50, 30) if labels[sample.identity_key] == 0 else (30, 60, 220)
                    shift = int(sample.camid)
                    draw.rectangle((5 + shift, 5, 30 + shift, 40), fill=color)
                    image.save(path)
            train_transform, reference_transform = model.make_transforms(
                hflip_prob=0, crop_padding=0, erasing_prob=0)
            loaders = build_stage_loaders(stage, batch_size=4, num_instances=2,
                                          train_transform=train_transform, reference_transform=reference_transform)
            trainer = CategoryProgressiveTrainer(model, stage, loaders, training,
                                                  output.with_name(output.stem + '_' + stage_id + '.jsonl'))
            initial = {c: model.export_adapter(c) for c in trainer.categories}
            for epoch in range(training.epochs):
                trainer.train_epoch(epoch)
            summary = trainer.finish_stage()
            for category in summary['category_updates']:
                assert any(not torch.equal(v, model.export_adapter(category)['state'][k])
                           for k, v in initial[category]['state'].items()), 'current adapter did not update'
            reports.append(summary)
            assert torch.equal(reference_before, model.encode_reference(probe))
            with torch.no_grad():
                vehicle_now = model.encode_category(probe, 'vehicle')
            if vehicle_before is not None:
                assert torch.equal(vehicle_before, vehicle_now), 'absent vehicle changed in t2'
            vehicle_before = vehicle_now
    model.assert_reference_unchanged()
    report = {'status': 'passed', 'data': 'generated colored rectangles; no real ReID accuracy claim',
              'device': args.device, 'amp': training.amp, 'feature_dim': model.feature_dim,
              'reference_signature': model.reference_signature,
              'reference_features_exactly_unchanged': True,
              'absent_vehicle_features_exactly_unchanged_in_t2': True,
              'temporary_heads_after_finish': len(model.temporary_heads), 'stages': reports}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'stages'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
