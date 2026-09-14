"""Extract a full ECPM stage or explicitly upgrade saved step-4 identity memory."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lreid_dataset.category_stream import load_category_stream
from reid.memory import ECPMMemory, FinchConfig, IdentityPrototypeMemory
from reid.models.category_adapter_bank import CategoryAdapterBank, build_category_model


def main(argv=None, model_factory=build_category_model):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-config', required=True)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument('--stage-id', help='extract this current stage from images')
    task.add_argument('--identity-memory', help='upgrade completed step-4 stages from saved vectors only')
    parser.add_argument('--previous-memory', help='previous complete ECPM mode memory')
    parser.add_argument('--output-memory', required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--reference-checkpoint')
    source.add_argument('--model-checkpoint', help='completed baseline model (same or immediately previous stage)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--finch-chunk-size', type=int, default=256)
    args = parser.parse_args(argv)
    output = Path(args.output_memory).resolve()
    report_path = output.with_suffix(output.suffix + '.json')
    if output.exists() or report_path.exists():
        raise FileExistsError('output memory/report must not already exist')
    stream = load_category_stream(args.stream_config)
    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint, map_location='cpu', weights_only=True)
        if (checkpoint.get('kind') != 'category_ce_triplet_baseline' or checkpoint.get('schema_version') != 1
                or checkpoint.get('completed') is not True or checkpoint.get('stream_fingerprint') != stream.fingerprint):
            raise ValueError('invalid completed baseline checkpoint for this stream')
        if args.stage_id:
            index = [s.stage_id for s in stream.stages].index(args.stage_id)
            allowed = {stream.stages[i].stage_id for i in (index, max(0, index - 1))}
            if checkpoint['stage_id'] not in allowed:
                raise ValueError('baseline model must belong to this or immediately preceding stage')
        model = CategoryAdapterBank.from_checkpoint(checkpoint['model'], device=args.device)
    else:
        model = model_factory(reference_checkpoint=args.reference_checkpoint, device=args.device)
    config = FinchConfig(args.finch_chunk_size)
    memory = (ECPMMemory.load(args.previous_memory, model, stream) if args.previous_memory
              else ECPMMemory(model, stream, config))
    if memory.clustering != config:
        raise ValueError('use the same FINCH chunk size as the saved memory')
    print(json.dumps(dict(event='ecpm_start', previous_stages=memory.processed_stages), ensure_ascii=False), flush=True)
    if args.identity_memory:
        identity = IdentityPrototypeMemory.load(args.identity_memory, model, stream)
        reports = memory.extend_from_identity_memory(identity)
        if not reports:
            raise ValueError('identity memory contains no new completed stage')
    else:
        reports = [memory.update_stage(model, stream.stage(args.stage_id), args.batch_size, args.workers)]
    memory.save(output)
    report = dict(event='ecpm_saved', output_memory=str(output), stage_reports=reports)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    main()
