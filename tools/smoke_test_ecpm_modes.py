"""Real CLIP three-stage ECPM smoke and a symmetric-distribution drift diagnostic."""

import argparse
import json
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
from reid.memory import ECPMMemory, FinchConfig
from reid.memory.ecpm_modes import _same
from reid.memory.finch_modes import aggregate_modes, finch_first_partition, prototype_drift
from reid.models.category_adapter_bank import build_category_model


def drift_diagnostic():
    first = torch.tensor([[1., 0.]] * 2)
    second = torch.tensor([[1., 0.]] * 2 + [[.6, .8]] * 2 + [[.6, -.8]] * 2)
    old_labels, _ = finch_first_partition(first)
    new_labels, _ = finch_first_partition(second)
    old_modes, old_center, _ = aggregate_modes(first, old_labels)
    new_modes, new_center, sizes = aggregate_modes(second, new_labels)
    result = dict(old_modes=len(old_modes), new_modes=len(new_modes), new_cluster_sizes=sizes.tolist(),
                  old_center=old_center.tolist(), new_center=new_center.tolist(),
                  drift=prototype_drift(old_center, new_center),
                  interpretation='Symmetric new modes can leave center drift zero; drift is not a complete distribution metric.')
    assert result['old_modes'] == 1 and result['new_modes'] == 3 and result['drift'] < 1e-6
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    model = build_category_model(reference_checkpoint=args.reference_checkpoint, device=args.device)
    reports = []
    with tempfile.TemporaryDirectory(prefix='ecpm_modes_') as temporary:
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
        memory = ECPMMemory(model, stream, FinchConfig(2))
        for stage_index, stage in enumerate(stream.stages):
            for view in stage.categories:
                model.add_category(view.category)
                for i, sample in enumerate(view.samples):
                    image = Image.new('RGB', (48, 64), (20 + 25 * stage_index, 30, 50))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((5 + i, 5 + stage_index, 35, 55), fill=(30 + i * 40, 60, 200 - i * 30))
                    image_path = Path(sample.path)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
            model.set_trainable_categories([v.category for v in stage.categories])
            model.train()
            old = memory.state_dict()
            with torch.autocast(device_type=torch.device(args.device).type,
                                enabled=torch.device(args.device).type == 'cuda'):
                prepared = memory.prepare_stage(model, stage, batch_size=3)
            assert _same(old, memory.state_dict()), 'prepare modified memory'
            assert set(prepared['old_categories']) == set(old['category_memory'])
            reports.append(memory.commit_stage(prepared))
            for category in stage.absent_categories:
                assert _same(old['category_memory'][category], memory.snapshot()[category])
            for before, after in zip(old['identity_memory']['rows'], memory.state_dict()['identity_memory']['rows']):
                assert _same(before, after), 'historical identity prototype changed'
            # Verify FINCH first-layer membership using the full upstream function.
            from finch import FINCH
            for category in prepared['category_updates']:
                rows = sorted(memory.category_prototypes(category), key=lambda row: row['identity_key'])
                vectors = torch.stack([row['vector'] for row in rows])
                labels, _, _ = FINCH(vectors.numpy(), distance='cosine', ensure_early_exit=False, verbose=False)
                actual = memory.snapshot()[category]['labels'].numpy()
                assert np.array_equal(actual[:, None] == actual, labels[:, 0, None] == labels[:, 0])
            checkpoint = directory / (stage.stage_id + '_ecpm.pt')
            memory.save(checkpoint)
            restored = ECPMMemory.load(checkpoint, model, stream)
            assert _same(memory.state_dict(), restored.state_dict()), 'restore changed ECPM state'
            memory = restored
            for view in stage.categories:
                for sample in view.samples:
                    Path(sample.path).unlink()
        assert memory.summary()['identities'] == 12
        model.assert_reference_unchanged()
    report = dict(status='passed', data='generated rectangles and hand vectors; no real ReID accuracy claim',
                  device=args.device, reference_signature=model.reference_signature,
                  reference_unchanged=True, official_first_partition_matches=True,
                  prepare_preserves_history=True, absent_categories_exactly_unchanged=True,
                  historical_identity_vectors_exactly_unchanged=True, save_restore_each_stage=True,
                  old_images_removed_before_next_stage=True, stages=reports, symmetric_drift_diagnostic=drift_diagnostic())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
