"""Build and audit the test-only MSMT17 unseen-domain manifest.

The unseen-domain protocol deliberately ignores ``bounding_box_train``.  It
uses the official query/gallery folders and parses identity/camera metadata
from filenames such as ``0000_c12_0033.jpg``.
"""

from __future__ import absolute_import

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


FIELDNAMES = (
    'path',
    'pid',
    'camid',
    'split',
    'category',
    'domain',
    'object_noun',
    'eval_protocol',
    'session_id',
    'shot_id',
    'frame_id',
    'source_group',
    'query_group',
    'augmentation',
)

DOMAIN_NAME = 'msmt17_unseen'
STANDARD_COUNTS = {
    'query_images': 11659,
    'gallery_images': 82161,
    'eval_identities': 3060,
    'cameras': 15,
}
MSMT17_PATTERN = re.compile(
    r'^(?P<pid>\d+)_c(?P<camid>\d+)_(?P<frame>\d+)\.jpg$',
    re.IGNORECASE,
)


class MSMT17ManifestError(RuntimeError):
    """Raised when the local MSMT17 copy violates the retrieval protocol."""


def _relative_posix(path, root):
    return Path(os.path.relpath(str(path), str(root))).as_posix()


def _image_files(directory):
    if not directory.is_dir():
        raise MSMT17ManifestError(
            'missing MSMT17 directory: {}'.format(directory)
        )
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == '.jpg'
        ),
        key=lambda path: path.name.casefold(),
    )


def _parse_image(path):
    match = MSMT17_PATTERN.fullmatch(path.name)
    if match is None:
        raise MSMT17ManifestError(
            'invalid MSMT17 image filename: {}'.format(path)
        )
    return (
        int(match.group('pid')),
        int(match.group('camid')),
        int(match.group('frame')),
    )


def build_msmt17_unseen_rows(root):
    """Return manifest rows and a protocol audit for one MSMT17 root."""

    root = Path(root).resolve()
    split_directories = (
        ('query', root / 'query'),
        ('gallery', root / 'bounding_box_test'),
    )
    rows = []
    seen_relative_paths = set()
    split_pid_counts = defaultdict(Counter)
    gallery_cameras = defaultdict(set)

    for split, directory in split_directories:
        for path in _image_files(directory):
            pid, camid, frame = _parse_image(path)
            relative_path = _relative_posix(path, root)
            if relative_path in seen_relative_paths:
                raise MSMT17ManifestError(
                    'image appears in more than one split: {}'.format(
                        relative_path
                    )
                )
            seen_relative_paths.add(relative_path)
            split_pid_counts[split][pid] += 1
            if split == 'gallery':
                gallery_cameras[pid].add(camid)
            rows.append(
                {
                    'path': relative_path,
                    'pid': str(pid),
                    'camid': str(camid),
                    'split': split,
                    'category': 'person',
                    'domain': DOMAIN_NAME,
                    'object_noun': 'person',
                    'eval_protocol': 'market1501',
                    'session_id': '',
                    'shot_id': '',
                    'frame_id': str(frame),
                    'source_group': directory.name,
                    'query_group': '',
                    'augmentation': '',
                }
            )

    query_rows = [row for row in rows if row['split'] == 'query']
    gallery_rows = [row for row in rows if row['split'] == 'gallery']
    if not query_rows or not gallery_rows:
        raise MSMT17ManifestError(
            'MSMT17 unseen protocol requires non-empty query and gallery'
        )

    query_pids = set(split_pid_counts['query'])
    gallery_pids = set(split_pid_counts['gallery'])
    missing_gallery_pids = query_pids - gallery_pids
    if missing_gallery_pids:
        raise MSMT17ManifestError(
            '{} query identities are absent from gallery'.format(
                len(missing_gallery_pids)
            )
        )

    invalid_queries = []
    for row in query_rows:
        pid = int(row['pid'])
        camid = int(row['camid'])
        if not any(
            gallery_camid != camid
            for gallery_camid in gallery_cameras[pid]
        ):
            invalid_queries.append(row['path'])
    if invalid_queries:
        raise MSMT17ManifestError(
            '{} query images have no cross-camera positive; examples: {}'
            .format(
                len(invalid_queries),
                ', '.join(invalid_queries[:5]),
            )
        )

    cameras = {
        int(row['camid'])
        for row in rows
    }
    audit = {
        'domain': DOMAIN_NAME,
        'root': str(root),
        'protocol': 'market1501',
        'training_images_used': 0,
        'query_images': len(query_rows),
        'gallery_images': len(gallery_rows),
        'query_identities': len(query_pids),
        'gallery_identities': len(gallery_pids),
        'eval_identities': len(query_pids | gallery_pids),
        'cameras': len(cameras),
        'camera_ids': sorted(cameras),
        'query_gallery_path_overlap': 0,
        'queries_without_cross_camera_positive': 0,
    }
    return rows, audit


def validate_standard_counts(audit):
    """Reject incomplete or non-standard MSMT17 V2 query/gallery copies."""

    mismatches = []
    for key, expected in STANDARD_COUNTS.items():
        actual = audit[key]
        if actual != expected:
            mismatches.append(
                '{}={} (expected {})'.format(key, actual, expected)
            )
    if mismatches:
        raise MSMT17ManifestError(
            'non-standard MSMT17 split: {}'.format('; '.join(mismatches))
        )


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    temporary_path.replace(path)


def build_evaluation_config(
    base_config_path,
    manifest_path,
    output_path,
    dataset_root=None,
):
    """Copy a train-only config and append MSMT17 as a test-only domain."""

    base_config_path = Path(base_config_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    dataset_root = (
        Path(dataset_root).resolve()
        if dataset_root is not None
        else manifest_path.parents[2] / 'data' / 'MSMT17'
    )
    with base_config_path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload.get('train_domains'), list) or not payload[
        'train_domains'
    ]:
        raise MSMT17ManifestError(
            'base config must contain non-empty train_domains'
        )

    existing_tests = payload.get('test_domains', [])
    if existing_tests:
        raise MSMT17ManifestError(
            'base config must be train-only; found {} test_domains'.format(
                len(existing_tests)
            )
        )
    relative_manifest = Path(
        os.path.relpath(str(manifest_path), str(output_path.parent))
    ).as_posix()
    relative_root = Path(
        os.path.relpath(
            str(dataset_root),
            str(output_path.parent),
        )
    ).as_posix()
    payload['test_domains'] = [
        {
            'name': DOMAIN_NAME,
            'category': 'person',
            'object_noun': 'person',
            'manifest': relative_manifest,
            'root': relative_root,
            'eval_protocol': 'market1501',
        }
    ]
    write_json(output_path, payload)


def parse_args():
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Build the test-only MSMT17 unseen-domain manifest.'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=repository_root / 'data' / 'MSMT17',
    )
    parser.add_argument(
        '--manifest-output',
        type=Path,
        default=(
            repository_root / 'config' / 'manifests' / 'msmt17_unseen.csv'
        ),
    )
    parser.add_argument(
        '--audit-output',
        type=Path,
        default=(
            repository_root
            / 'config'
            / 'manifests'
            / 'msmt17_unseen_audit.json'
        ),
    )
    parser.add_argument(
        '--base-domain-config',
        type=Path,
        default=(
            repository_root / 'config' / 'cross_category_five_domains.json'
        ),
    )
    parser.add_argument(
        '--evaluation-config-output',
        type=Path,
        default=(
            repository_root
            / 'config'
            / 'cross_category_five_domains_with_msmt17_unseen.json'
        ),
    )
    parser.add_argument(
        '--allow-nonstandard-counts',
        action='store_true',
        help='allow a protocol-valid subset instead of official MSMT17 V2 counts',
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='audit the dataset without writing manifest/config files',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, audit = build_msmt17_unseen_rows(args.root)
    if not args.allow_nonstandard_counts:
        validate_standard_counts(audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.check_only:
        return

    write_manifest(args.manifest_output, rows)
    write_json(args.audit_output, audit)
    build_evaluation_config(
        args.base_domain_config,
        args.manifest_output,
        args.evaluation_config_output,
        dataset_root=args.root,
    )
    print('Wrote manifest to {}'.format(args.manifest_output))
    print('Wrote audit to {}'.format(args.audit_output))
    print('Wrote evaluation config to {}'.format(
        args.evaluation_config_output
    ))


if __name__ == '__main__':
    main()
