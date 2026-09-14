"""Matched-stage validation sweeps from one frozen pre-stage checkpoint."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.adaptation.pgca import PGCATransferConfig
from reid.evaluation.category_progressive import EvaluationConfig, check_evaluation_coverage, evaluate_stage
from reid.loss.pgca import PGCAConsistencyConfig
from reid.memory import ECPMMemory
from reid.models.category_adapter_bank import _tensor_digest
from reid.trainer_category_progressive import CategoryTrainingConfig
from reid.trainer_category_resumable import ResumableCategoryTrainer, model_from_bank
from reid.utils.progressive_checkpoint import atomic_json, capture_rng, exact_runtime, file_digest, restore_rng
from tools.evaluate_category_progressive import load_committed_run


def diagnose(run_dir, output, kind='sources', weights=(0., .1, 1., 10.), device='cpu'):
    runtime = exact_runtime(device)
    state, reference, stream, original, old_memory = load_committed_run(run_dir, device)
    if state['phase'] != 'between_stages' or state['stage_index'] >= len(stream.stages):
        raise ValueError('diagnostics need a committed boundary BEFORE the target stage')
    stage = stream.stages[state['stage_index']]
    settings = state['settings']
    config = EvaluationConfig('validation', 'per_dataset', settings.get('beta', .5), 'ecpm', settings.get('eval_batch_size', 128))
    check_evaluation_coverage(stream, stage.stage_id, config)
    training = CategoryTrainingConfig(**{k: settings[k] for k in CategoryTrainingConfig.__dataclass_fields__})
    if kind == 'sources':
        if not stage.new_categories or not stage.seen_before:
            raise ValueError('source diagnosis requires new categories and historical donors')
        variants = {'default': dict(mode='default'), 'similarity': dict(mode='similarity')}
        variants.update({'source_' + c: dict(mode='source', source=c) for c in sorted(stage.seen_before)})
    elif kind == 'drift':
        if not stage.recurring_categories:
            raise ValueError('drift diagnosis requires recurring categories')
        variants = {'drift': dict(mode='drift', lambda_con=settings['lambda_con'])}
        for weight in weights:
            PGCAConsistencyConfig('fixed', weight)
            variants['fixed_' + str(float(weight))] = dict(mode='fixed', lambda_con=float(weight))
    else:
        raise ValueError('diagnostic kind must be sources or drift')
    binding = dict(checkpoint_sha256=file_digest(Path(run_dir) / 'latest.pt'), kind=kind, stage_id=stage.stage_id,
                   variants=variants, training=asdict(training), evaluation=asdict(config), runtime=runtime)
    output = Path(output)
    completed = {}
    if output.exists():
        saved = json.loads(output.read_text(encoding='utf-8'))
        if saved['binding'] != binding:
            raise ValueError('diagnosis differs from existing output; use another output file')
        completed = saved['variants']
    restore_rng(state['rng'])
    candidate = old_memory.prepare_stage(original, stage, settings['prototype_batch_size'], 0)
    common_rng = capture_rng()
    del original
    for name, variant in variants.items():
        if name in completed:
            continue
        model = model_from_bank(reference, state['bank'], device)
        memory = ECPMMemory.from_state_dict(state['ecpm'], model, stream)
        train_transform, fixed_transform = model.make_transforms()
        loaders = build_stage_loaders(stage, settings['batch_size'], settings['num_instances'], workers=0,
            seed=settings['seed'], train_transform=train_transform, reference_transform=fixed_transform)
        initialization = dict(mode=settings['init_mode'], alpha=settings['alpha'], delta=settings['delta'],
                              summary=settings.get('control_summary', 'ecpm'), seed=settings['seed'])
        consistency = dict(mode=settings['consistency'], lambda_con=settings['lambda_con'], gamma=settings['gamma'],
                           summary=settings.get('control_summary', 'ecpm'))
        (initialization if kind == 'sources' else consistency).update(variant)
        # Every variant starts with identical reference, historical adapters,
        # current candidate, sampler seeds, augmentation and classifier RNG.
        restore_rng(common_rng)
        trainer = ResumableCategoryTrainer(model, stage, loaders, training,
            consistency=PGCAConsistencyConfig(**consistency), initialization=PGCATransferConfig(**initialization),
            ecpm_memory=memory, ecpm_candidate=candidate)
        for _ in range(training.epochs):
            trainer.train_epoch()
        training_report = trainer.finish_stage()
        memory.commit_stage(candidate)
        evaluation = evaluate_stage(model, memory, stream, stage.stage_id, config)
        completed[name] = dict(training=training_report, evaluation=evaluation,
                              adapter_sha256={c: _tensor_digest(model.export_adapter(c)['state']) for c in model.categories})
        atomic_json(dict(binding=binding, variants=completed, completed=False), output)
        del trainer, model, memory, loaders
    rows = []
    categories = stage.new_categories if kind == 'sources' else stage.recurring_categories
    for name, result in completed.items():
        for category in categories:
            oracle = result['evaluation']['retrieval']['per_dataset']['oracle']['per_category'][category]
            row = dict(variant=name, category=category, split='validation', **oracle)
            if kind == 'sources':
                decision = result['training']['initialization']['decisions'][category]
                donor = decision['selected_source']
                row.update(source=donor, similarity=None if donor is None else decision['candidates'][donor]['score'],
                    gain_vs_default_mAP=oracle['mAP'] - completed['default']['evaluation']['retrieval']['per_dataset']['oracle']['per_category'][category]['mAP'])
            else:
                row.update(drift=result['training']['pgca']['drifts'][category],
                           effective_weight=result['training']['pgca']['weights'][category])
            rows.append(row)
    report = dict(binding=binding, variants=completed, completed=True, empirical_rows=rows,
        interpretation='Validation only; compare observed mAP/Rank1 across matched-budget variants. '
                       'Repeat across seeds/stages before inferring that similarity or drift predicts transfer/retention.')
    atomic_json(report, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kind', choices=('sources', 'drift'), required=True)
    parser.add_argument('--weights', nargs='+', type=float, default=[0., .1, 1., 10.])
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    return diagnose(args.run_dir, args.output, args.kind, args.weights, args.device)


if __name__ == '__main__':
    main()
