"""Build and audit manifests for the five cross-category ReID domains.

The script reads the datasets in ``data/`` without moving or copying images.
It writes one UTF-8 CSV manifest per domain plus a JSON experiment config.

The generated manifests keep the legacy columns consumed by ``ManifestReID``:

    path,pid,camid,split

Additional columns preserve dataset-specific metadata for protocol-aware
evaluation in a later integration step.
"""

from __future__ import absolute_import

import argparse
import csv
import json
import os
import random
import re
import sys
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

DOMAIN_ORDER = ('market1501', 'ipanda50', 'veri', 'atrw', 'boat')

DOMAIN_INFO = {
    'market1501': {
        'category': 'person',
        'object_noun': 'person',
        'eval_protocol': 'market1501',
        'root_name': 'market1501',
        'hflip_prob': 0.5,
    },
    'ipanda50': {
        'category': 'panda',
        'object_noun': 'giant panda',
        'eval_protocol': 'session_reid',
        'root_name': 'iPanda50',
        'hflip_prob': 0.5,
    },
    'veri': {
        'category': 'vehicle',
        'object_noun': 'vehicle',
        'eval_protocol': 'veri776',
        'root_name': 'VeRi',
        'hflip_prob': 0.5,
    },
    'atrw': {
        'category': 'tiger',
        'object_noun': 'tiger',
        'eval_protocol': 'single_query',
        'root_name': 'atrw',
        'hflip_prob': 0.5,
    },
    'boat': {
        'category': 'boat',
        'object_noun': 'boat',
        'eval_protocol': 'cross_view',
        'root_name': 'boat',
        'hflip_prob': 0.5,
    },
}

MARKET_PATTERN = re.compile(r'^(-?\d+)_c(\d+)')
VERI_PATTERN = re.compile(r'^(\d+)_c(\d+)_')
BOAT_PATTERN = re.compile(
    r'^(?P<pid>\d+)_c(?P<view>\d+)_(?P<code>00(?:[0-5])?)_'
    r'(?P<index>\d*)\.jpg$',
    re.IGNORECASE,
)
IPANDA_DIR_PATTERN = re.compile(r'^(\d{2})_')
IPANDA_VIDEO_PATTERN = re.compile(
    r'^(?P<pid>\d{2})_v(?P<video>\d+)_f(?P<frame>\d+)$',
    re.IGNORECASE,
)


class ManifestError(RuntimeError):
    """Raised when raw data cannot produce a trustworthy ReID manifest."""


def _relative_posix(path, root):
    return Path(os.path.relpath(str(path), str(root))).as_posix()


def _image_files(directory):
    if not directory.is_dir():
        raise ManifestError('missing image directory: {}'.format(directory))
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == '.jpg'
        ),
        key=lambda path: path.name.casefold(),
    )


def _base_row(domain, path, pid, camid, split, **metadata):
    info = DOMAIN_INFO[domain]
    row = {
        'path': str(path),
        'pid': str(pid),
        'camid': str(int(camid)),
        'split': split,
        'category': info['category'],
        'domain': domain,
        'object_noun': info['object_noun'],
        'eval_protocol': info['eval_protocol'],
        'session_id': '',
        'shot_id': '',
        'frame_id': '',
        'source_group': '',
        'query_group': '',
        'augmentation': '',
    }
    for key, value in metadata.items():
        if key not in row:
            raise ManifestError('unsupported manifest field: {}'.format(key))
        row[key] = '' if value is None else str(value)
    return row


def build_market1501(root):
    rows = []
    skipped_junk = 0
    split_dirs = (
        ('train', 'bounding_box_train'),
        ('query', 'query'),
        ('gallery', 'bounding_box_test'),
    )
    for split, directory_name in split_dirs:
        directory = root / directory_name
        for path in _image_files(directory):
            match = MARKET_PATTERN.match(path.name)
            if not match:
                raise ManifestError(
                    'unrecognized Market1501 filename: {}'.format(path)
                )
            identity, camera = map(int, match.groups())
            if identity == -1:
                skipped_junk += 1
                continue
            rows.append(
                _base_row(
                    'market1501',
                    _relative_posix(path, root),
                    identity,
                    camera - 1,
                    split,
                    session_id='c{}'.format(camera),
                    source_group=path.stem,
                )
            )
    return rows, {'skipped_junk_images': skipped_junk}


def _read_name_list(path):
    if not path.is_file():
        raise ManifestError('missing list file: {}'.format(path))
    with path.open('r', encoding='utf-8-sig') as handle:
        return [line.strip() for line in handle if line.strip()]


def build_veri(root):
    rows = []
    train_directory = root / 'image_train'
    listed_train = set(_read_name_list(root / 'name_train.txt'))
    actual_train = {path.name for path in _image_files(train_directory)}
    missing_train = sorted(listed_train - actual_train)
    if missing_train:
        raise ManifestError(
            'VeRi name_train.txt references {} missing images'.format(
                len(missing_train)
            )
        )

    # This local public copy contains 32 valid training images omitted by
    # name_train.txt, including all 24 images of identity 389. Scanning the
    # directory recovers the 37,778 images / 576 identities stated in ReadMe.
    split_specs = (
        ('train', sorted(actual_train), 'image_train'),
        (
            'query',
            _read_name_list(root / 'name_query.txt'),
            'image_query',
        ),
        (
            'gallery',
            _read_name_list(root / 'name_test.txt'),
            'image_test',
        ),
    )
    for split, filenames, directory_name in split_specs:
        directory = root / directory_name
        for filename in filenames:
            path = directory / filename
            if not path.is_file():
                raise ManifestError(
                    '{} split references a missing image: {}'.format(
                        split,
                        path,
                    )
                )
            match = VERI_PATTERN.match(filename)
            if not match:
                raise ManifestError(
                    'unrecognized VeRi filename: {}'.format(path)
                )
            identity, camera = map(int, match.groups())
            rows.append(
                _base_row(
                    'veri',
                    _relative_posix(path, root),
                    identity,
                    camera - 1,
                    split,
                    session_id='c{}'.format(camera),
                    source_group=path.stem,
                )
            )
    return rows, {
        'name_train_listed_images': len(listed_train),
        'image_train_directory_images': len(actual_train),
        'unlisted_train_images_included': sorted(actual_train - listed_train),
    }


def _read_atrw_csv(path):
    if not path.is_file():
        raise ManifestError('missing ATRW list: {}'.format(path))
    records = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        for line_number, fields in enumerate(csv.reader(handle), start=1):
            if not fields:
                continue
            if len(fields) != 2:
                raise ManifestError(
                    '{}:{} expected pid,filename'.format(path, line_number)
                )
            try:
                identity = int(fields[0])
            except ValueError as exc:
                raise ManifestError(
                    '{}:{} invalid identity {!r}'.format(
                        path,
                        line_number,
                        fields[0],
                    )
                ) from exc
            records.append((identity, fields[1].strip()))
    return records


def _read_atrw_test_metadata(path):
    if not path.is_file():
        raise ManifestError('missing ATRW metadata: {}'.format(path))
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ManifestError('ATRW metadata must contain a JSON list')

    metadata = {}
    for index, item in enumerate(payload):
        frame = item.get('frame')
        if not isinstance(frame, list) or len(frame) != 3:
            raise ManifestError(
                'ATRW metadata row {} has invalid frame'.format(index)
            )
        filename = '{:06d}.jpg'.format(int(item['imgid']))
        if filename in metadata:
            raise ManifestError(
                'duplicate ATRW imgid for {}'.format(filename)
            )
        query_group = str(item.get('query', '')).strip().lower()
        if query_group not in ('sing', 'multi'):
            raise ManifestError(
                'ATRW {} has invalid query group {!r}'.format(
                    filename,
                    query_group,
                )
            )
        metadata[filename] = {
            'pid': int(item['entityid']),
            # Local grouping confirms frame[1] is the camera component:
            # every "multi" entity spans multiple values of frame[1].
            'shot_id': int(frame[0]),
            'camid': int(frame[1]),
            'frame_id': int(frame[2]),
            'query_group': query_group,
        }
    return metadata


def build_atrw(root):
    rows = []
    listed_train = _read_atrw_csv(root / 'reid_list_train.csv')
    listed_test = _read_atrw_csv(root / 'reid_list_test.csv')
    test_metadata = _read_atrw_test_metadata(root / 'gt_test_plain.json')

    for identity, filename in sorted(listed_train, key=lambda item: item[1]):
        path = root / 'train' / filename
        if not path.is_file():
            raise ManifestError(
                'ATRW train list references a missing image: {}'.format(path)
            )
        rows.append(
            _base_row(
                'atrw',
                _relative_posix(path, root),
                identity,
                0,
                'train',
                session_id='unknown',
                source_group=path.stem,
            )
        )

    test_pairs = []
    for identity, filename in listed_test:
        path = root / 'test' / filename
        if not path.is_file():
            raise ManifestError(
                'ATRW test list references a missing image: {}'.format(path)
            )
        metadata = test_metadata.get(filename)
        if metadata is None:
            raise ManifestError(
                'ATRW test image has no JSON metadata: {}'.format(filename)
            )
        if identity != metadata['pid']:
            raise ManifestError(
                'ATRW identity mismatch for {}: CSV={}, JSON={}'.format(
                    filename,
                    identity,
                    metadata['pid'],
                )
            )
        source_group = '{}:{}:{}'.format(
            metadata['shot_id'],
            metadata['camid'],
            metadata['frame_id'],
        )
        test_pairs.append((identity, path, metadata, source_group))

    listed_test_names = {filename for _, filename in listed_test}
    extra_metadata = sorted(set(test_metadata) - listed_test_names)
    if extra_metadata:
        raise ManifestError(
            'ATRW JSON contains {} images absent from test CSV'.format(
                len(extra_metadata)
            )
        )

    # Match the already validated PAD protocol: the first listed test image
    # of every identity is the query and all remaining images are gallery.
    # Pseudo cameras 0/1 make the standard Market-style evaluator retain all
    # valid positives without requiring an ATRW-specific temporal evaluator.
    first_path_by_pid = {}
    for identity, path, _, _ in test_pairs:
        first_path_by_pid.setdefault(identity, path)
    for identity, path, metadata, source_group in test_pairs:
        is_query = path == first_path_by_pid[identity]
        rows.append(
            _base_row(
                'atrw',
                _relative_posix(path, root),
                identity,
                0 if is_query else 1,
                'query' if is_query else 'gallery',
                session_id='{}:{}'.format(
                    metadata['shot_id'],
                    metadata['camid'],
                ),
                shot_id=metadata['shot_id'],
                frame_id=metadata['frame_id'],
                source_group=source_group,
                query_group=metadata['query_group'],
            )
        )

    train_jpgs = len(_image_files(root / 'train'))
    return rows, {
        'listed_train_images': len(listed_train),
        'unlisted_train_images_excluded': train_jpgs - len(listed_train),
        'atrw_local_test_version_images': len(listed_test),
    }


def _parse_ipanda_image(path, identity):
    stem = path.stem
    match = IPANDA_VIDEO_PATTERN.match(stem)
    if match:
        filename_pid = int(match.group('pid'))
        if filename_pid != identity:
            raise ManifestError(
                'iPanda identity mismatch for {}'.format(path)
            )
        session = 'v{:03d}'.format(int(match.group('video')))
        frame_id = int(match.group('frame'))
        return session, frame_id

    match = re.match(r'^(?P<pid>\d{2})_(?P<body>.+)$', stem)
    if not match or int(match.group('pid')) != identity:
        raise ManifestError('unrecognized iPanda filename: {}'.format(path))
    body = match.group('body')
    frame_match = re.match(r'^(?P<session>.*?)(?:_)?(?P<frame>\d+)$', body)
    if not frame_match:
        return 'legacy', ''
    session = frame_match.group('session').rstrip('_') or 'legacy'
    return session, int(frame_match.group('frame'))


def build_ipanda50(root, seed, train_id_count):
    by_identity = defaultdict(list)
    directories = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name != 'iPanda50-split'
        ),
        key=lambda path: path.name.casefold(),
    )
    for directory in directories:
        match = IPANDA_DIR_PATTERN.match(directory.name)
        if not match:
            raise ManifestError(
                'unrecognized iPanda identity directory: {}'.format(directory)
            )
        identity = int(match.group(1))
        for path in _image_files(directory):
            session, frame_id = _parse_ipanda_image(path, identity)
            by_identity[identity].append(
                {
                    'path': path,
                    'session': session,
                    'frame_id': frame_id,
                }
            )

    identities = sorted(by_identity)
    if len(identities) != 50:
        raise ManifestError(
            'expected 50 iPanda identities, found {}'.format(len(identities))
        )
    if not 1 <= train_id_count < len(identities):
        raise ManifestError(
            'iPanda train identity count must be in [1, 49]'
        )

    eligible_test_ids = []
    for identity in identities:
        sessions = {item['session'] for item in by_identity[identity]}
        if len(sessions) >= 2:
            eligible_test_ids.append(identity)
    test_id_count = len(identities) - train_id_count
    if len(eligible_test_ids) < test_id_count:
        raise ManifestError(
            'iPanda has only {} identities with at least two sessions, '
            'but {} test identities are required'.format(
                len(eligible_test_ids),
                test_id_count,
            )
        )
    shuffled = list(eligible_test_ids)
    random.Random(seed).shuffle(shuffled)
    test_ids = set(shuffled[:test_id_count])
    train_ids = set(identities) - test_ids

    session_keys = sorted(
        {
            '{}:{}'.format(identity, item['session'])
            for identity, items in by_identity.items()
            for item in items
        }
    )
    session_camids = {
        session_key: index for index, session_key in enumerate(session_keys)
    }

    rows = []
    query_paths = set()
    for identity in sorted(test_ids):
        by_session = defaultdict(list)
        for item in by_identity[identity]:
            by_session[item['session']].append(item)
        if len(by_session) < 2:
            raise ManifestError(
                'iPanda test identity {} has fewer than two sessions'.format(
                    identity
                )
            )
        for session, items in sorted(by_session.items()):
            items.sort(key=lambda item: item['path'].name.casefold())
            # PAD uses the lexicographically first frame from every video or
            # recording as the deterministic query.
            query_paths.add(items[0]['path'])

    for identity in sorted(by_identity):
        for item in sorted(
            by_identity[identity],
            key=lambda record: record['path'].name.casefold(),
        ):
            session_key = '{}:{}'.format(identity, item['session'])
            camid = session_camids[session_key]
            relative_path = _relative_posix(item['path'], root)
            metadata = {
                'session_id': item['session'],
                'frame_id': item['frame_id'],
                'source_group': session_key,
            }
            if identity in train_ids:
                rows.append(
                    _base_row(
                        'ipanda50',
                        relative_path,
                        '{:02d}'.format(identity),
                        camid,
                        'train',
                        **metadata
                    )
                )
            else:
                if item['path'] in query_paths:
                    rows.append(
                        _base_row(
                            'ipanda50',
                            relative_path,
                            '{:02d}'.format(identity),
                            camid,
                            'query',
                            **metadata
                        )
                    )
                else:
                    rows.append(
                        _base_row(
                            'ipanda50',
                            relative_path,
                            '{:02d}'.format(identity),
                            camid,
                            'gallery',
                            **metadata
                        )
                    )

    split_dir = root / 'iPanda50-split'
    original_split_overlap = {}
    for split_index in range(5):
        train_names = _read_name_list(
            split_dir / 'split{}_train.txt'.format(split_index)
        )
        test_names = _read_name_list(
            split_dir / 'split{}_test.txt'.format(split_index)
        )
        train_pids = {int(name[:2]) for name in train_names}
        test_pids = {int(name[:2]) for name in test_names}
        original_split_overlap[str(split_index)] = {
            'train_images': len(train_names),
            'test_images': len(test_names),
            'overlapping_identities': len(train_pids & test_pids),
        }

    return rows, {
        'identity_split_seed': seed,
        'train_identity_ids': sorted(train_ids),
        'test_identity_ids': sorted(test_ids),
        'original_five_splits': original_split_overlap,
    }


def build_boat(root, seed, num_test_ids):
    parsed = []
    for path in _image_files(root):
        match = BOAT_PATTERN.match(path.name)
        if not match:
            raise ManifestError(
                'unrecognized Boat filename: {}'.format(path)
            )
        identity = int(match.group('pid'))
        view = int(match.group('view'))
        code = match.group('code')
        progressive_index = match.group('index')
        parsed.append(
            {
                'path': path,
                'pid': identity,
                'view': view,
                'code': code,
                'index': progressive_index,
            }
        )

    identities = sorted({item['pid'] for item in parsed})
    if identities != list(range(107)):
        raise ManifestError(
            'Boat identities must be contiguous 0..106, found {} IDs'.format(
                len(identities)
            )
        )

    originals = [item for item in parsed if item['code'] == '00']
    eligible_test_ids = []
    for identity in identities:
        identity_originals = [
            item for item in originals if item['pid'] == identity
        ]
        if (
            len(identity_originals) >= 2
            and len({item['view'] for item in identity_originals}) >= 2
        ):
            eligible_test_ids.append(identity)
    if len(eligible_test_ids) < num_test_ids:
        raise ManifestError(
            'Boat has only {} identities eligible for testing, but {} were '
            'requested'.format(len(eligible_test_ids), num_test_ids)
        )
    shuffled = list(eligible_test_ids)
    random.Random(seed).shuffle(shuffled)
    test_ids = set(shuffled[:num_test_ids])
    train_ids = set(identities) - test_ids

    first_path_by_pid = {}
    for item in sorted(originals, key=lambda record: str(record['path'])):
        if item['pid'] in test_ids:
            first_path_by_pid.setdefault(item['pid'], item['path'])

    rows = []
    for item in sorted(parsed, key=lambda record: record['path'].name):
        identity = item['pid']
        view = item['view']
        code = item['code']
        relative_path = _relative_posix(item['path'], root)
        metadata = {
            'session_id': 'c{}'.format(view),
            'frame_id': item['index'],
            'source_group': '{}:c{}:{}:{}'.format(
                identity,
                view,
                code,
                item['index'] or 'missing',
            ),
            'augmentation': 'original' if code == '00' else code,
        }
        if identity in train_ids:
            rows.append(
                _base_row(
                    'boat',
                    relative_path,
                    identity,
                    view,
                    'train',
                    **metadata
                )
            )
        elif code == '00':
            split = (
                'query'
                if item['path'] == first_path_by_pid[identity]
                else 'gallery'
            )
            rows.append(
                _base_row(
                    'boat',
                    relative_path,
                    identity,
                    view,
                    split,
                    **metadata
                )
            )

    transform_counts = Counter(item['code'] for item in parsed)
    unusual_views = sorted(
        {
            item['view']
            for item in parsed
            if item['view'] not in set(range(1, 7))
        }
    )
    missing_progressive = sorted(
        item['path'].name for item in parsed if not item['index']
    )
    return rows, {
        'raw_images': len(parsed),
        'identity_split_seed': seed,
        'train_identity_ids': sorted(train_ids),
        'test_identity_ids': sorted(test_ids),
        'transform_code_counts': dict(sorted(transform_counts.items())),
        'unusual_view_ids': unusual_views,
        'missing_progressive_index_files': missing_progressive,
    }


def _rows_by_split(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['split']].append(row)
    return grouped


def audit_manifest(domain, root, rows):
    errors = []
    warnings = []
    grouped = _rows_by_split(rows)

    for split in ('train', 'query', 'gallery'):
        if not grouped[split]:
            errors.append('missing {} split'.format(split))

    seen_within_split = set()
    for row in rows:
        path = root / Path(row['path'])
        if not path.is_file():
            errors.append('missing image {}'.format(path))
        key = (row['split'], row['path'])
        if key in seen_within_split:
            errors.append(
                'duplicate path within {}: {}'.format(
                    row['split'],
                    row['path'],
                )
            )
        seen_within_split.add(key)
        try:
            camid = int(row['camid'])
        except ValueError:
            errors.append('non-integer camid for {}'.format(row['path']))
            continue
        if camid < 0:
            errors.append('negative camid for {}'.format(row['path']))

    train_pids = {row['pid'] for row in grouped['train']}
    eval_pids = {
        row['pid'] for row in grouped['query'] + grouped['gallery']
    }
    identity_overlap = train_pids & eval_pids
    if identity_overlap:
        errors.append(
            '{} train/eval identities overlap'.format(len(identity_overlap))
        )

    query_pids = {row['pid'] for row in grouped['query']}
    gallery_pids = {row['pid'] for row in grouped['gallery']}
    missing_gallery_pids = query_pids - gallery_pids
    if missing_gallery_pids:
        errors.append(
            '{} query identities absent from gallery'.format(
                len(missing_gallery_pids)
            )
        )
    query_paths = {row['path'] for row in grouped['query']}
    gallery_paths = {row['path'] for row in grouped['gallery']}
    eval_path_overlap = query_paths & gallery_paths
    if eval_path_overlap:
        errors.append(
            '{} images overlap between query and gallery'.format(
                len(eval_path_overlap)
            )
        )

    gallery_by_pid = defaultdict(list)
    for row in grouped['gallery']:
        gallery_by_pid[row['pid']].append(row)

    protocol = DOMAIN_INFO[domain]['eval_protocol']
    invalid_queries = 0
    for query in grouped['query']:
        candidates = gallery_by_pid[query['pid']]
        valid = any(
            int(gallery['camid']) != int(query['camid'])
            for gallery in candidates
        )
        if not valid:
            invalid_queries += 1
    if invalid_queries:
        errors.append(
            '{} queries have no valid positive gallery sample'.format(
                invalid_queries
            )
        )

    train_sources = {
        row['source_group']
        for row in grouped['train']
        if row['source_group']
    }
    eval_sources = {
        row['source_group']
        for row in grouped['query'] + grouped['gallery']
        if row['source_group']
    }
    source_overlap = train_sources & eval_sources
    if source_overlap:
        errors.append(
            '{} source groups leak from train to eval'.format(
                len(source_overlap)
            )
        )

    if domain == 'atrw':
        groups = Counter(row['query_group'] for row in grouped['query'])
        if not set(groups).issubset({'sing', 'multi'}):
            errors.append('ATRW query groups must be sing or multi')
        if len(grouped['query']) != len(query_pids):
            errors.append('ATRW must contain one query image per identity')

    if domain == 'boat':
        if any(
            row['augmentation'] != 'original'
            for row in grouped['query'] + grouped['gallery']
        ):
            errors.append('Boat evaluation contains augmented images')

    return {
        'status': 'ok' if not errors else 'failed',
        'protocol': protocol,
        'splits': {
            split: {
                'images': len(grouped[split]),
                'identities': len(
                    {row['pid'] for row in grouped[split]}
                ),
                'cameras_or_sessions': len(
                    {row['camid'] for row in grouped[split]}
                ),
            }
            for split in ('train', 'query', 'gallery')
        },
        'train_eval_identity_overlap': len(identity_overlap),
        'train_eval_source_overlap': len(source_overlap),
        'invalid_queries': invalid_queries,
        'warnings': warnings,
        'errors': errors,
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            extrasaction='raise',
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _domain_config(
    config_path,
    data_root,
    output_dir,
    split_seed,
    ipanda_train_ids,
    boat_test_ids,
):
    config_dir = config_path.parent
    domains = []
    for domain in DOMAIN_ORDER:
        info = DOMAIN_INFO[domain]
        root = data_root / info['root_name']
        manifest = output_dir / '{}.csv'.format(domain)
        domains.append(
            {
                'name': domain,
                'category': info['category'],
                'object_noun': info['object_noun'],
                'manifest': _relative_posix(manifest, config_dir),
                'root': _relative_posix(root, config_dir),
                'eval_protocol': info['eval_protocol'],
                'prompt_checkpoint': (
                    '../_CROSS_CATEGORY_PROMPTS/'
                    '{}_clipreid_prompt.pth'.format(domain)
                ),
                'max_train_ids': 0,
                'hflip_prob': info['hflip_prob'],
                'random_erasing_prob': 0.5,
                'crop_padding': 10,
            }
        )
    return {
        'input_size': [224, 224],
        'resize_mode': 'pad',
        'data_protocol': {
            'source': 'PAD object_reid loaders',
            'split_seed': int(split_seed),
            'ipanda_protocol': 'identity',
            'ipanda_train_ids': int(ipanda_train_ids),
            'boat_test_ids': int(boat_test_ids),
        },
        'train_domains': domains,
        'test_domains': [],
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    os.replace(str(temporary), str(path))


def _print_report(report):
    header = (
        '{:<12} {:<7} {:>9} {:>6} {:>9} {:>6} {:>9} {:>6}'.format(
            'domain',
            'status',
            'train_img',
            'train',
            'query_img',
            'query',
            'gallery',
            'g_ids',
        )
    )
    print(header)
    print('-' * len(header))
    for domain in DOMAIN_ORDER:
        audit = report['domains'][domain]['audit']
        splits = audit['splits']
        print(
            '{:<12} {:<7} {:>9} {:>6} {:>9} {:>6} {:>9} {:>6}'.format(
                domain,
                audit['status'],
                splits['train']['images'],
                splits['train']['identities'],
                splits['query']['images'],
                splits['query']['identities'],
                splits['gallery']['images'],
                splits['gallery']['identities'],
            )
        )


def parse_args(argv=None):
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Build and audit five cross-category ReID manifests.'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=repository_root / 'data',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=repository_root / 'config' / 'manifests',
    )
    parser.add_argument(
        '--config-output',
        type=Path,
        default=(
            repository_root / 'config' / 'cross_category_five_domains.json'
        ),
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=42,
        help='deterministic identity split seed used by iPanda and Boat',
    )
    parser.add_argument('--ipanda-train-ids', type=int, default=35)
    parser.add_argument('--boat-test-ids', type=int, default=27)
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='audit in memory without writing manifests or config',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    config_output = args.config_output.resolve()

    builders = {
        'market1501': lambda root: build_market1501(root),
        'ipanda50': lambda root: build_ipanda50(
            root,
            args.split_seed,
            args.ipanda_train_ids,
        ),
        'veri': lambda root: build_veri(root),
        'atrw': lambda root: build_atrw(root),
        'boat': lambda root: build_boat(
            root,
            args.split_seed,
            args.boat_test_ids,
        ),
    }

    report = {
        'data_root': str(data_root),
        'split_seed': args.split_seed,
        'ipanda_train_id_count': args.ipanda_train_ids,
        'boat_test_id_count': args.boat_test_ids,
        'domains': {},
    }
    failed = False
    for domain in DOMAIN_ORDER:
        info = DOMAIN_INFO[domain]
        root = data_root / info['root_name']
        if not root.is_dir():
            raise ManifestError(
                'missing dataset root for {}: {}'.format(domain, root)
            )
        rows, dataset_notes = builders[domain](root)
        audit = audit_manifest(domain, root, rows)
        report['domains'][domain] = {
            'root': str(root),
            'manifest': str(output_dir / '{}.csv'.format(domain)),
            'notes': dataset_notes,
            'audit': audit,
        }
        failed = failed or audit['status'] != 'ok'
        if not args.check_only:
            _write_csv(output_dir / '{}.csv'.format(domain), rows)

    if not args.check_only:
        config_payload = _domain_config(
            config_output,
            data_root,
            output_dir,
            args.split_seed,
            args.ipanda_train_ids,
            args.boat_test_ids,
        )
        _write_json(config_output, config_payload)
        _write_json(output_dir / 'audit_report.json', report)

    _print_report(report)
    for domain in DOMAIN_ORDER:
        audit = report['domains'][domain]['audit']
        for warning in audit['warnings']:
            print('WARNING [{}] {}'.format(domain, warning))
        for error in audit['errors']:
            print('ERROR [{}] {}'.format(domain, error), file=sys.stderr)

    if failed:
        return 1
    if args.check_only:
        print('Audit passed; check-only mode wrote no files.')
    else:
        print('Wrote manifests to {}'.format(output_dir))
        print('Wrote domain config to {}'.format(config_output))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ManifestError as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        sys.exit(1)
