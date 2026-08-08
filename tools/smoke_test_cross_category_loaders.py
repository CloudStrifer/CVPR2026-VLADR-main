"""Decode one PK training batch and one evaluation batch for every domain."""

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reid.utils.data import Preprocessor
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomIdentityBatchSampler


def _load_manifest_module():
    module_path = (
        REPOSITORY_ROOT / 'lreid_dataset' / 'datasets' / 'manifest_reid.py'
    )
    spec = importlib.util.spec_from_file_location(
        'manifest_reid_standalone',
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--domain-config',
        type=Path,
        default=REPOSITORY_ROOT / 'config' / 'cross_category_five_domains.json',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=REPOSITORY_ROOT / 'data',
    )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-instances', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_module = _load_manifest_module()
    config = manifest_module.load_domain_config(str(args.domain_config))
    height, width = config.get('input_size') or (224, 224)
    resize_mode = config.get('resize_mode') or 'pad'
    resize = (
        T.ResizePad((height, width), interpolation=3)
        if resize_mode == 'pad'
        else T.Resize((height, width), interpolation=3)
    )
    transform = T.Compose(
        [
            resize,
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    for domain in config['train_domains']:
        dataset = manifest_module.ManifestReID(str(args.data_dir), domain)
        train = sorted(dataset.train)
        sampler = RandomIdentityBatchSampler(
            train,
            args.batch_size,
            args.num_instances,
        )
        train_loader = DataLoader(
            Preprocessor(
                train,
                root=dataset.images_dir,
                transform=transform,
            ),
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            drop_last=True,
        )
        train_images, _, train_pids, _, _ = next(iter(train_loader))
        pid_counts = Counter(train_pids.tolist())
        expected_pids = args.batch_size // args.num_instances
        if len(pid_counts) != expected_pids:
            raise RuntimeError(
                '{} batch has {} identities, expected {}'.format(
                    domain['name'],
                    len(pid_counts),
                    expected_pids,
                )
            )
        if set(pid_counts.values()) != {args.num_instances}:
            raise RuntimeError(
                '{} batch is not PK-balanced: {}'.format(
                    domain['name'],
                    sorted(pid_counts.values()),
                )
            )

        evaluation = sorted(
            list(set(dataset.query) | set(dataset.gallery))
        )
        evaluation_loader = DataLoader(
            Preprocessor(
                evaluation,
                root=dataset.images_dir,
                transform=transform,
            ),
            batch_size=min(args.batch_size, len(evaluation)),
            shuffle=False,
            num_workers=args.workers,
        )
        evaluation_images, _, _, _, _ = next(iter(evaluation_loader))
        print(
            '{}: train_batch={}, PK={}x{}, eval_batch={}'.format(
                domain['name'],
                tuple(train_images.shape),
                expected_pids,
                args.num_instances,
                tuple(evaluation_images.shape),
            )
        )

    for domain in config['test_domains']:
        dataset = manifest_module.ManifestReID(str(args.data_dir), domain)
        if dataset.train:
            raise RuntimeError(
                '{} is test-only but exposes {} training images'.format(
                    domain['name'],
                    len(dataset.train),
                )
            )
        evaluation = sorted(
            list(set(dataset.query) | set(dataset.gallery))
        )
        evaluation_loader = DataLoader(
            Preprocessor(
                evaluation,
                root=dataset.images_dir,
                transform=transform,
            ),
            batch_size=min(args.batch_size, len(evaluation)),
            shuffle=False,
            num_workers=args.workers,
        )
        images, _, _, _, _ = next(iter(evaluation_loader))
        print(
            '{} [unseen]: query={}, gallery={}, eval_batch={}'.format(
                domain['name'],
                len(dataset.query),
                len(dataset.gallery),
                tuple(images.shape),
            )
        )

    print('All cross-category loader smoke tests passed.')


if __name__ == '__main__':
    main()
