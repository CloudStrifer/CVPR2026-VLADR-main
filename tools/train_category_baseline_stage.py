"""Run one CE+Triplet baseline stage; no ECPM/PGCA or within-stage resume."""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lreid_dataset.category_stream import load_category_stream
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.evaluation.category_oracle import evaluate_category_oracle
from reid.models.category_adapter_bank import CategoryAdapterBank
from reid.models.wrapper import make_category_model
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig


def main(argv=None, model_factory=make_category_model):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-config', required=True)
    parser.add_argument('--stage-id', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--previous-checkpoint', help='completed immediately preceding baseline stage')
    parser.add_argument('--reference-checkpoint', help='local CLIP weights for the first stage only')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--iterations-per-epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-instances', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--adapter-lr', type=float, default=0.0003)
    parser.add_argument('--head-lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--lambda-tri', type=float, default=1.0)
    parser.add_argument('--triplet-margin', type=float, default=0.3)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--eval-split', choices=('none', 'validation', 'test'), default='none')
    args = parser.parse_args(argv)
    config = CategoryTrainingConfig(args.epochs, args.iterations_per_epoch, args.adapter_lr,
                                    args.head_lr, args.weight_decay, args.lambda_tri,
                                    args.triplet_margin, args.amp)
    stream = load_category_stream(args.stream_config)
    stage = stream.stage(args.stage_id)
    stage_index = [s.stage_id for s in stream.stages].index(stage.stage_id)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output directory must be empty; this runner does not resume partial stages')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if stage_index == 0:
        if args.previous_checkpoint:
            raise ValueError('the first stage cannot take a previous-stage checkpoint')
        model = model_factory(reference_checkpoint=args.reference_checkpoint, device=args.device)
    else:
        if not args.previous_checkpoint or args.reference_checkpoint:
            raise ValueError('later stages require --previous-checkpoint, not fresh reference weights')
        checkpoint = torch.load(args.previous_checkpoint, map_location='cpu', weights_only=True)
        if (checkpoint.get('kind') != 'category_ce_triplet_baseline'
                or checkpoint.get('stage_id') != stream.stages[stage_index - 1].stage_id
                or checkpoint.get('stream_fingerprint') != stream.fingerprint
                or checkpoint.get('completed') is not True):
            raise ValueError('checkpoint must be the completed preceding stage of this exact stream')
        model = CategoryAdapterBank.from_checkpoint(checkpoint['model'], device=args.device)
    transforms = model.make_transforms()
    loaders = build_stage_loaders(stage, batch_size=args.batch_size, num_instances=args.num_instances,
                                  workers=args.workers, seed=args.seed, train_transform=transforms[0],
                                  reference_transform=transforms[1])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_config.json').write_text(json.dumps({
        'arguments': vars(args), 'training': asdict(config), 'stream_fingerprint': stream.fingerprint,
        'reference_signature': model.reference_signature,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    trainer = CategoryProgressiveTrainer(model, stage, loaders, config, output / 'training.jsonl')
    for epoch in range(config.epochs):
        report = trainer.train_epoch(epoch)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    summary = trainer.finish_stage()
    payload = {'kind': 'category_ce_triplet_baseline', 'schema_version': 1,
               'completed': True, 'stage_id': stage.stage_id,
               'stream_fingerprint': stream.fingerprint, 'model': model.export_checkpoint()}
    temporary = output / 'completed_model.pt.tmp'
    torch.save(payload, temporary)
    os.replace(str(temporary), str(output / 'completed_model.pt'))
    # The completed training checkpoint is durable before optional diagnostics.
    (output / 'training_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.eval_split != 'none':
        evaluation = [evaluate_category_oracle(model, view, workers=args.workers)
                      for view in stream.evaluations_at(stage.stage_id, split=args.eval_split)]
        (output / 'oracle_evaluation.json').write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
