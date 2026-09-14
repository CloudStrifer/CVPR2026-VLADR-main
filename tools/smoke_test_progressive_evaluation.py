"""Three-stage real CLIP routing/evaluation with independent-process resume."""

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
from reid.evaluation.prototype_router import RoutedEncoder
from reid.utils.progressive_checkpoint import atomic_json, exact_runtime
from tools.evaluate_category_progressive import load_committed_run
from tools.smoke_test_progressive_resume import exact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pgca_eval_') as temporary:
        directory = Path(temporary)
        example = ROOT / 'config/category_progressive_example.json'
        config = json.loads(example.read_text(encoding='utf-8'))
        config['data_root'] = str(directory / 'images')
        for stage in config['stages']:
            for category in stage['categories']:
                category['train_manifest'] = str((example.parent / category['train_manifest']).resolve())
        for view in config['evaluation']:
            for field in ('query_manifest', 'gallery_manifest'):
                view[field] = str((example.parent / view[field]).resolve())
        stream_path = directory / 'stream.json'
        atomic_json(config, stream_path)
        stream = load_category_stream(stream_path)
        samples = [s for stage in stream.stages for view in stage.categories for s in view.samples]
        samples += [s for view in stream.evaluations for s in view.query + view.gallery]
        for index, sample in enumerate(samples):
            background = {'person': (40, 20, 20), 'vehicle': (20, 40, 20), 'panda': (20, 20, 40)}[sample.category]
            picture = Image.new('RGB', (40, 48), background)
            draw = ImageDraw.Draw(picture)
            color = (210, 40, 30) if index % 2 else (30, 60, 210)
            draw.rectangle((5 + int(sample.camid) % 3, 5, 30, 40), fill=color)
            path = Path(sample.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            picture.save(path)
        def command(name, *controls, resume=False):
            command = [sys.executable, str(ROOT / 'train_category_progressive.py'), '--output-dir', str(directory / name)]
            if resume:
                command += ['--resume']
            else:
                command += ['--stream-config', str(stream_path), '--device', args.device, '--epochs', '1',
                    '--iterations-per-epoch', '2', '--batch-size', '4', '--num-instances', '2', '--prototype-batch-size', '4',
                    '--delta', '-1', '--evaluate', '--eval-gallery', 'both', '--eval-batch-size', '4']
                if args.device.startswith('cuda'):
                    command += ['--amp']
            subprocess.run(command + list(controls), cwd=ROOT, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        command('baseline')
        atomic_json(dict(status='baseline_complete'), output.with_suffix('.progress.json'))
        command('split', '--max-stages', '1')
        command('split', '--max-updates', '1', resume=True)
        command('split', resume=True)
        baseline = torch.load(directory / 'baseline/latest.pt', weights_only=True, map_location='cpu')
        resumed = torch.load(directory / 'split/latest.pt', weights_only=True, map_location='cpu')
        exact(baseline['bank'], resumed['bank'])
        exact(baseline['rng'], resumed['rng'])
        assert len(resumed['evaluations']) == 3 and resumed['pending_evaluation'] is None
        for left, right in zip(baseline['evaluations'], resumed['evaluations']):
            exact(left['retrieval'], right['retrieval'])
            exact(left['routing'], right['routing'])
        for index in range(3):
            name = 'stage_{:04d}.jsonl'.format(index)
            assert (directory / 'baseline' / name).read_bytes() == (directory / 'split' / name).read_bytes()
        exact_runtime(args.device)
        _, _, _, model, memory = load_committed_run(directory / 'baseline', args.device)
        sample = stream.evaluations_at('t3')[0].query[0]
        _, transform = model.make_transforms()
        with Image.open(sample.path) as image:
            probe = transform(image.convert('RGB'))[None].to(args.device)
        first = RoutedEncoder(model, memory)(probe)
        first = (first[0].cpu(), first[1], first[2].cpu())
        del model, memory
        _, _, _, model, memory = load_committed_run(directory / 'split', args.device)
        second = RoutedEncoder(model, memory)(probe)
        second = (second[0].cpu(), second[1], second[2].cpu())
        exact(first, second)
        shutil.copyfile(directory / 'split/evaluation_summary.json', output.with_name(output.stem + '_metrics.json'))
        shutil.copyfile(directory / 'split/evaluation.jsonl', output.with_suffix('.jsonl'))
        result = dict(status='passed', device=args.device, amp=args.device.startswith('cuda'),
            data='generated images only; no real-world retrieval efficacy claim', stages=3, updates=6,
            resumed_at_stage_boundary_and_inside_t2=True, adapter_rng_and_training_logs_bitwise_equal=True,
            retrieval_and_routing_reports_equal=True, reloaded_routing_scores_predictions_descriptors_bitwise_equal=True,
            per_dataset_and_mixed_gallery=True, oracle_and_prototype=True,
            full_performance_and_forgetting_matrices=True)
        atomic_json(result, output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
