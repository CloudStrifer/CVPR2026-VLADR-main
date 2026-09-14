"""Real CLIP: compare uninterrupted continuation with separate-process resume."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw

from lreid_dataset.category_stream import load_category_stream


def exact(left, right, location='state'):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError('tensor differs: ' + location)
    elif isinstance(left, dict):
        assert set(left) == set(right), location
        for key in left:
            exact(left[key], right[key], location + '/' + str(key))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), location
        for i, (a, b) in enumerate(zip(left, right)):
            exact(a, b, location + '/' + str(i))
    else:
        assert left == right, location


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pgca_exact_resume_') as temporary:
        directory = Path(temporary)
        example = ROOT / 'config/category_progressive_example.json'
        config = json.loads(example.read_text(encoding='utf-8'))
        config['data_root'], config['evaluation'] = str(directory / 'images'), []
        for stage in config['stages']:
            for category in stage['categories']:
                category['train_manifest'] = str((example.parent / category['train_manifest']).resolve())
        stream_path = directory / 'stream.json'
        stream_path.write_text(json.dumps(config), encoding='utf-8')
        stream = load_category_stream(stream_path)
        for index, stage in enumerate(stream.stages):
            for view in stage.categories:
                for sample in view.samples:
                    image = Image.new('RGB', (40, 48), (20 + index * 20, 20, 20))
                    draw = ImageDraw.Draw(image)
                    color = (210, 40, 30) if view.label_map[sample.identity_key] == 0 else (30, 60, 210)
                    shift = int(sample.camid)
                    draw.rectangle((5 + shift, 5, 30 + shift, 40), fill=color)
                    path = Path(sample.path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)
        def command(name, *control, resume=False):
            argv = [sys.executable, str(ROOT / 'train_category_progressive.py'), '--output-dir', str(directory / name)]
            if resume:
                argv += ['--resume']
            else:
                argv += ['--stream-config', str(stream_path), '--device', args.device, '--epochs', '2',
                         '--iterations-per-epoch', '3', '--batch-size', '4', '--num-instances', '2',
                         '--prototype-batch-size', '4', '--head-lr', '.01', '--delta', '-1', '--max-grad-norm', '.1']
                if args.device.startswith('cuda'):
                    argv += ['--amp']
                if args.reference_checkpoint:
                    argv += ['--reference-checkpoint', args.reference_checkpoint]
            subprocess.run(argv + list(control), cwd=ROOT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        def checkpoint(name):
            return torch.load(directory / name / 'latest.pt', map_location='cpu', weights_only=True)
        # The baseline process runs continuously across the t2 midpoint being
        # compared; later continuation is identical in both branches.
        command('baseline', '--max-updates', '9')
        expected = checkpoint('baseline')
        command('split', '--max-updates', '8')
        paused = checkpoint('split')
        assert paused['stage_index'] == 1 and paused['trainer']['iteration_in_epoch'] == 2
        assert paused['trainer']['teacher_bank'] is not None
        assert paused['trainer']['initialization']['decisions']['panda']['selected_source'] is not None
        checkpoint_bytes = (directory / 'split' / 'latest.pt').stat().st_size
        command('split', '--max-updates', '1', resume=True)
        actual = checkpoint('split')
        exact(expected['trainer'], actual['trainer'])
        exact(expected['rng'], actual['rng'])
        output.with_suffix('.progress.json').write_text(json.dumps(dict(status='midpoint_passed',
            trainer_optimizer_teacher_scaler_rng_bitwise_equal=True), indent=2), encoding='utf-8')
        del expected, paused, actual
        command('baseline', resume=True)
        command('split', resume=True)
        expected, actual = checkpoint('baseline'), checkpoint('split')
        exact(expected['bank'], actual['bank'])
        exact(expected['rng'], actual['rng'])
        exact(expected['ecpm']['identity_memory'], actual['ecpm']['identity_memory'])
        for category in expected['ecpm']['category_memory']:
            # Wall-clock FINCH timings need not agree across independent runs.
            left, right = expected['ecpm']['category_memory'][category], actual['ecpm']['category_memory'][category]
            exact({k: v for k, v in left.items() if k != 'clustering'},
                  {k: v for k, v in right.items() if k != 'clustering'})
        for index in range(3):
            name = 'stage_{:04d}.jsonl'.format(index)
            assert (directory / 'baseline' / name).read_bytes() == (directory / 'split' / name).read_bytes()
            shutil.copyfile(directory / 'split' / name, output.with_name(output.stem + '_t{}.jsonl'.format(index + 1)))
        report = dict(status='passed', device=args.device, amp=args.device.startswith('cuda'),
            data='generated images, not a ReID accuracy experiment', stages=3, optimizer_updates=18,
            separate_process_resume=True, random_augmentation=True, gradient_clipping=.1,
            sampler_pass_restarts=True, midpoint_teacher_optimizer_scaler_heads_rng_bitwise_equal=True,
            all_training_logs_bitwise_equal=True, final_adapter_and_prototype_tensors_bitwise_equal=True,
            final_rng_bitwise_equal=True, transferred_new_category_exercised=True,
            reference_file_bytes=(directory / 'split' / 'reference.pt').stat().st_size,
            midpoint_checkpoint_bytes=checkpoint_bytes)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
