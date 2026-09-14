"""Extract and atomically save one stage of ECPM identity memory, without training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lreid_dataset.category_stream import load_category_stream
from reid.memory.ecpm import IdentityPrototypeMemory
from reid.models.category_adapter_bank import CategoryAdapterBank, build_category_model


def main(argv=None, model_factory=build_category_model):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-config', required=True)
    parser.add_argument('--stage-id', required=True)
    parser.add_argument('--output-memory', required=True, help='new destination; never overwrite an existing run')
    parser.add_argument('--previous-memory', help='memory through the immediately preceding stage')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--reference-checkpoint', help='local OpenAI CLIP weights (otherwise use local default cache)')
    source.add_argument('--model-checkpoint', help='completed baseline checkpoint for this or preceding stage')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)
    output = Path(args.output_memory).resolve()
    if output.exists():
        raise FileExistsError('output memory already exists; choose a new destination')
    stream = load_category_stream(args.stream_config)
    stage = stream.stage(args.stage_id)
    index = [s.stage_id for s in stream.stages].index(stage.stage_id)
    if bool(args.previous_memory) != (index > 0):
        raise ValueError('only later stages require --previous-memory')
    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint, map_location='cpu', weights_only=True)
        allowed = {stage.stage_id}
        if index:
            allowed.add(stream.stages[index - 1].stage_id)
        if (checkpoint.get('kind') != 'category_ce_triplet_baseline'
                or checkpoint.get('schema_version') != 1 or checkpoint.get('completed') is not True
                or checkpoint.get('stream_fingerprint') != stream.fingerprint
                or checkpoint.get('stage_id') not in allowed):
            raise ValueError('model checkpoint must belong to this or the preceding stage of this stream')
        model = CategoryAdapterBank.from_checkpoint(checkpoint['model'], device=args.device)
    else:
        model = model_factory(reference_checkpoint=args.reference_checkpoint, device=args.device)
    memory = (IdentityPrototypeMemory.load(args.previous_memory, model, stream)
              if args.previous_memory else IdentityPrototypeMemory(model, stream))
    print(json.dumps({'event': 'extract_start', 'stage_id': stage.stage_id,
                      'previous_stages': memory.processed_stages}, ensure_ascii=False), flush=True)
    summary = memory.update_stage(model, stage, batch_size=args.batch_size, workers=args.workers)
    memory.save(output)
    report = dict(event='memory_saved', output_memory=str(output), **summary)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    main()
