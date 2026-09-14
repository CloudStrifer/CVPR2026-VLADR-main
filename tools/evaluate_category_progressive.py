"""Evaluate one committed continuous checkpoint without changing training state."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from lreid_dataset.category_stream import load_category_stream
from reid.memory import ECPMMemory
from reid.trainer_category_resumable import model_from_bank
from reid.evaluation.category_progressive import EvaluationConfig, evaluate_stage
from reid.utils.progressive_checkpoint import atomic_json, exact_runtime, file_digest


def load_committed_run(directory, device='cpu'):
    directory = Path(directory)
    state = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=True)
    if (state.get('kind') != 'ecpm_pgca_continuous' or state.get('schema_version') != 1
            or state['phase'] == 'training' or state['stage_index'] < 1):
        raise ValueError('evaluation needs a completed stage checkpoint, not an in-training student')
    if file_digest(directory / 'reference.pt') != state['reference_sha256']:
        raise ValueError('reference checkpoint binding mismatch')
    stream = load_category_stream(state['settings']['stream_config'])
    if stream.fingerprint != state['stream_fingerprint']:
        raise ValueError('stream protocol changed')
    reference = torch.load(directory / 'reference.pt', map_location='cpu', weights_only=True)
    model = model_from_bank(reference, state['bank'], device)
    memory = ECPMMemory.from_state_dict(state['ecpm'], model, stream)
    return state, reference, stream, model, memory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--gallery', choices=('per_dataset', 'mixed', 'both'), default='both')
    parser.add_argument('--summary', choices=('ecpm', 'identity_mean'), default='ecpm')
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args(argv)
    runtime = exact_runtime(args.device)
    state, _, stream, model, memory = load_committed_run(args.run_dir, args.device)
    report = evaluate_stage(model, memory, stream, stream.stages[state['stage_index'] - 1].stage_id,
                           EvaluationConfig(args.split, args.gallery, args.beta, args.summary, args.batch_size))
    report['evaluation_runtime'] = runtime
    report['checkpoint_sha256'] = file_digest(Path(args.run_dir) / 'latest.pt')
    atomic_json(report, args.output)
    return report


if __name__ == '__main__':
    main()
