"""One-stage PGCA training; transfer optional, no within-stage resume."""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lreid_dataset.category_stream import load_category_stream
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.loss.pgca import PGCAConsistencyConfig
from reid.adaptation.pgca import PGCATransferConfig
from reid.memory import ECPMMemory, FinchConfig
from reid.models.category_adapter_bank import CategoryAdapterBank, build_category_model
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig


def main(argv=None, model_factory=build_category_model, default_init_mode='default'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-config', required=True)
    parser.add_argument('--stage-id', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--previous-checkpoint', help='completed preceding PGCA checkpoint, or step-3 baseline')
    parser.add_argument('--previous-memory', help='required only when importing a step-3 baseline checkpoint')
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--iterations-per-epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-instances', type=int, default=4)
    parser.add_argument('--prototype-batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--adapter-lr', type=float, default=.0003)
    parser.add_argument('--head-lr', type=float, default=.0003)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--lambda-tri', type=float, default=1.)
    parser.add_argument('--triplet-margin', type=float, default=.3)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--consistency', choices=('off', 'fixed', 'drift'), default='drift')
    parser.add_argument('--lambda-con', type=float, default=1.)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--init-mode', choices=('default', 'similarity'), default=default_init_mode)
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--delta', type=float, default=.5)
    parser.add_argument('--finch-chunk-size', type=int, default=256)
    args = parser.parse_args(argv)
    training = CategoryTrainingConfig(args.epochs, args.iterations_per_epoch, args.adapter_lr, args.head_lr,
                                      args.weight_decay, args.lambda_tri, args.triplet_margin, args.amp)
    consistency = PGCAConsistencyConfig(args.consistency, args.lambda_con, args.gamma)
    initialization = PGCATransferConfig(args.init_mode, args.alpha, args.delta)
    stream = load_category_stream(args.stream_config)
    stage = stream.stage(args.stage_id)
    index = [s.stage_id for s in stream.stages].index(stage.stage_id)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use an empty output directory; partial-stage resume is not supported')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if index == 0:
        if args.previous_checkpoint or args.previous_memory:
            raise ValueError('first stage cannot load previous checkpoints')
        model = model_factory(reference_checkpoint=args.reference_checkpoint, device=args.device)
        memory = ECPMMemory(model, stream, FinchConfig(args.finch_chunk_size))
    else:
        if not args.previous_checkpoint or args.reference_checkpoint:
            raise ValueError('later stages require --previous-checkpoint, not fresh reference weights')
        previous = torch.load(args.previous_checkpoint, map_location='cpu', weights_only=True)
        if (previous.get('completed') is not True or previous.get('schema_version') != 1
                or previous.get('stage_id') != stream.stages[index - 1].stage_id
                or previous.get('stream_fingerprint') != stream.fingerprint):
            raise ValueError('checkpoint must be the completed preceding stage of this exact stream')
        model = CategoryAdapterBank.from_checkpoint(previous['model'], device=args.device)
        if previous.get('kind') in ('pgca_recurring_stage', 'pgca_stage'):
            if args.previous_memory:
                raise ValueError('PGCA checkpoint already contains the corresponding ECPM state')
            memory = ECPMMemory.from_state_dict(previous['ecpm'], model, stream)
        elif previous.get('kind') == 'category_ce_triplet_baseline' and args.previous_memory:
            memory = ECPMMemory.load(args.previous_memory, model, stream)
        else:
            raise ValueError('expected PGCA checkpoint or baseline plus --previous-memory')
        if memory.processed_stages != tuple(s.stage_id for s in stream.stages[:index]):
            raise ValueError('ECPM memory must end exactly at the preceding stage')
        if memory.clustering != FinchConfig(args.finch_chunk_size):
            raise ValueError('FINCH chunk size differs from previous memory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_config.json').write_text(json.dumps(dict(arguments=vars(args), stream_fingerprint=stream.fingerprint,
        reference_signature=model.reference_signature), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Current image prototypes precede training; no ECPM commit until success.
    candidate = memory.prepare_stage(model, stage, args.prototype_batch_size, args.workers)
    train_transform, reference_transform = model.make_transforms()
    loaders = build_stage_loaders(stage, batch_size=args.batch_size, num_instances=args.num_instances,
                                  workers=args.workers, seed=args.seed,
                                  train_transform=train_transform, reference_transform=reference_transform)
    trainer = CategoryProgressiveTrainer(model, stage, loaders, training, output / 'training.jsonl',
                                         consistency=consistency, ecpm_memory=memory, ecpm_candidate=candidate,
                                         initialization=initialization)
    for epoch in range(training.epochs):
        print(json.dumps(trainer.train_epoch(epoch), ensure_ascii=False), flush=True)
    training_report = trainer.finish_stage()
    memory_report = memory.commit_stage(candidate)
    payload = dict(kind='pgca_stage' if initialization.mode == 'similarity' else 'pgca_recurring_stage',
                   schema_version=1, completed=True, stage_id=stage.stage_id,
                   stream_fingerprint=stream.fingerprint, model=model.export_checkpoint(), ecpm=memory.state_dict(),
                   pgca=training_report['pgca'], initialization=training_report['initialization'], arguments=vars(args))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix='completed_stage.', suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / 'completed_stage.pt')
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    report = dict(training=training_report, ecpm=memory_report)
    (output / 'training_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
