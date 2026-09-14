"""Three-stage real CLIP ECPM + recurring PGCA training on generated images."""

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
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.loss.pgca import PGCAConsistencyConfig
from reid.adaptation.pgca import PGCATransferConfig
from reid.memory import ECPMMemory, FinchConfig
from reid.memory.ecpm_modes import _same
from reid.models.category_adapter_bank import build_category_model, CategoryAdapterBank
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig


def main(default_init_mode='default'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    parser.add_argument('--init-mode', choices=('default', 'similarity'), default=default_init_mode)
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--delta', type=float, default=.5)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = build_category_model(reference_checkpoint=args.reference_checkpoint, device=args.device)
    probe = torch.randn(2, 3, 224, 224, device=args.device)
    reference = model.encode_reference(probe)
    training = CategoryTrainingConfig(epochs=2, iterations_per_epoch=2, adapter_lr=.0003,
                                      head_lr=.01, amp=torch.device(args.device).type == 'cuda')
    reports = []
    transfers_verified = 0
    with tempfile.TemporaryDirectory(prefix='pgca_recurring_') as temporary:
        directory = Path(temporary)
        example = ROOT / 'config/category_progressive_example.json'
        config = json.loads(example.read_text(encoding='utf-8'))
        config['data_root'] = str(directory / 'images')
        config['evaluation'] = []
        for stage in config['stages']:
            for category in stage['categories']:
                category['train_manifest'] = str((example.parent / category['train_manifest']).resolve())
        path = directory / 'stream.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        stream = load_category_stream(path)
        memory = ECPMMemory(model, stream, FinchConfig(2))
        for index, stage in enumerate(stream.stages):
            for view in stage.categories:
                for sample in view.samples:
                    color = (210, 40, 30) if view.label_map[sample.identity_key] == 0 else (30, 60, 210)
                    image = Image.new('RGB', (40, 48), (20 + 20 * index, 20, 20))
                    draw = ImageDraw.Draw(image)
                    shift = int(sample.camid)
                    draw.rectangle((5 + shift, 5, 30 + shift, 40), fill=color)
                    image_path = Path(sample.path)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
            old_memory = memory.state_dict()
            absent = {c: model.export_adapter(c) for c in stage.absent_categories}
            old_recurring = {c: model.export_adapter(c) for c in stage.recurring_categories}
            historical = {c: model.export_adapter(c) for c in stage.seen_before}
            candidate = memory.prepare_stage(model, stage, batch_size=3)
            transforms = model.make_transforms(hflip_prob=0., crop_padding=0, erasing_prob=0.)
            loaders = build_stage_loaders(stage, batch_size=4, num_instances=2,
                                          train_transform=transforms[0], reference_transform=transforms[1])
            trainer = CategoryProgressiveTrainer(model, stage, loaders, training,
                output.with_name(output.stem + '_' + stage.stage_id + '.jsonl'),
                consistency=PGCAConsistencyConfig('drift', 2., 3.), ecpm_memory=memory, ecpm_candidate=candidate,
                initialization=PGCATransferConfig(args.init_mode, args.alpha, args.delta))
            for target, decision in trainer.initialization.metadata()['decisions'].items():
                source = decision['selected_source']
                if source is not None:
                    assert source in historical, 'source was not historical at stage start'
                    assert _same(historical[source]['state'], model.export_adapter(target)['state']), 'wrong donor version'
                    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(
                        model.adapter_parameters(target), model.adapter_parameters(source))), 'shared donor storage'
                    assert trainer.consistency.weight(target) == 0, 'new category received source distillation'
                    transfers_verified += 1
            initial = {c: model.export_adapter(c) for c in trainer.categories}
            teacher = trainer.consistency.teacher
            teacher_probe = {}
            if teacher is not None:
                assert teacher.categories == tuple(sorted(stage.recurring_categories))
                for c in stage.recurring_categories:
                    assert _same(old_recurring[c], teacher.model.export_adapter(c))
                    teacher_probe[c] = teacher.encode(probe, c)
            for epoch in range(training.epochs):
                trainer.train_epoch(epoch)
            assert _same(old_memory, memory.state_dict()), 'training committed ECPM prematurely'
            for c in trainer.categories:
                assert not _same(initial[c], model.export_adapter(c)), 'current adapter did not learn'
            for c in absent:
                assert _same(absent[c], model.export_adapter(c)), 'absent adapter changed'
            if teacher is not None:
                teacher.assert_unchanged()
                for c in teacher_probe:
                    assert torch.equal(teacher_probe[c], teacher.encode(probe, c)), 'teacher features changed'
            report = trainer.finish_stage()
            assert trainer.consistency.teacher is None
            if stage.recurring_categories:
                assert any(e['categories'][c]['consistency'] > 0 for e in report['epochs'] for c in stage.recurring_categories)
            assert torch.equal(reference, model.encode_reference(probe)), 'reference features changed'
            memory_report = memory.commit_stage(candidate)
            reports.append(dict(training=report, ecpm=memory_report))
            # Exercise real state reconstruction between stages, without old images.
            checkpoint = directory / (stage.stage_id + '.pt')
            torch.save(dict(model=model.export_checkpoint(), ecpm=memory.state_dict()), checkpoint)
            if teacher is not None:
                del teacher
            restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = CategoryAdapterBank.from_checkpoint(restored['model'], device=args.device)
            memory = ECPMMemory.from_state_dict(restored['ecpm'], model, stream)
            for view in stage.categories:
                for sample in view.samples:
                    Path(sample.path).unlink()
        model.assert_reference_unchanged()
    if args.init_mode == 'similarity':
        assert transfers_verified > 0, 'smoke must exercise at least one accepted transfer'
    result = dict(status='passed', data='generated colored rectangles; no real ReID accuracy claim',
                  device=args.device, amp=training.amp, reference_signature=model.reference_signature,
                  reference_features_exactly_unchanged=True, teachers_exactly_unchanged=True,
                  recurring_teachers_match_stage_start=True, current_adapters_updated=True,
                  absent_adapters_exactly_unchanged=True, saved_and_restored_each_stage=True,
                  old_images_removed_before_next_stage=True, stages=reports)
    result.update(initialization_mode=args.init_mode, transfers_verified=transfers_verified,
                  transferred_adapters_match_historical_snapshot=(transfers_verified > 0),
                  transferred_adapters_have_independent_storage=(transfers_verified > 0))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'stages'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
