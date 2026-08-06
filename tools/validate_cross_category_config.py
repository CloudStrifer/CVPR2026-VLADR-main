"""Validate manifests and retrieval protocols before starting GPU training."""

import argparse
import importlib.util
from pathlib import Path


def _load_manifest_module():
    """Load the standalone adapter without importing every legacy dataset.

    The full dataset registry imports optional scipy-backed benchmarks, which
    should not be required merely to audit cross-category CSV manifests.
    """

    repository_root = Path(__file__).resolve().parents[1]
    module_path = (
        repository_root / 'lreid_dataset' / 'datasets' / 'manifest_reid.py'
    )
    spec = importlib.util.spec_from_file_location(
        'manifest_reid_standalone',
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain-config', required=True)
    parser.add_argument(
        '--data-dir',
        required=True,
        help='default root for image paths inside manifests',
    )
    args = parser.parse_args()

    manifest_module = _load_manifest_module()
    config = manifest_module.load_domain_config(args.domain_config)
    domains = config['train_domains'] + config['test_domains']
    total_images = 0
    for spec in domains:
        dataset = manifest_module.ManifestReID(args.data_dir, spec)
        total_images += (
            len(dataset.train) + len(dataset.query) + len(dataset.gallery)
        )

    print(
        'Validated {} domains and {} images.'.format(
            len(domains), total_images
        )
    )


if __name__ == '__main__':
    main()
