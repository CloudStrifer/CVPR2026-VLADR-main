"""Convert existing audited domain manifests into a portable interleaved stream.

Standard library only. Original train/query/gallery manifests are never edited.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lreid_dataset.category_stream import load_category_stream


PROTOCOLS = dict(market1501='cross_camera', veri776='cross_camera', session_reid='cross_camera',
                 single_query='exclude_self', cross_view='cross_camera')
COLUMNS = ['path', 'original_pid', 'source_dataset', 'camid']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng(seed, category, purpose):
    key = '{}\0{}\0{}'.format(seed, category, purpose)
    return random.Random(hashlib.sha256(key.encode('utf-8')).hexdigest())


def shuffled(values, seed, category, purpose):
    values = sorted(values)
    rng(seed, category, purpose).shuffle(values)
    return values


def validation_rows(rows, protocol):
    # Boat manifests identify precomputed image transforms. Keep only originals
    # for validation, but remove the entire held-out identity from training.
    rows = sorted([r for r in rows if not r.get('augmentation') or r['augmentation'] == 'original'], key=lambda r: r['path'])
    if len(rows) < 2 or (protocol == 'cross_camera' and len({r['camid'] for r in rows}) < 2):
        return None
    return rows


def build_stream(source_config, recipe_path, output, data_root, check_images=False):
    source_config, recipe_path = Path(source_config).resolve(), Path(recipe_path).resolve()
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    if output.exists():
        raise FileExistsError('choose a new output directory; existing splits are never overwritten')
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    if set(recipe) != {'seed', 'validation_fraction', 'min_train_ids_per_stage', 'stages'}:
        raise ValueError('invalid recipe fields')
    seed, fraction, minimum = recipe['seed'], recipe['validation_fraction'], recipe['min_train_ids_per_stage']
    if (type(seed) is not int or type(fraction) not in (float, int) or not math.isfinite(fraction)
            or not 0 < fraction < 1 or type(minimum) is not int or minimum < 2):
        raise ValueError('invalid seed, validation fraction or minimum stage identities')
    stages = recipe['stages']
    if not isinstance(stages, list) or not stages:
        raise ValueError('recipe needs stages')
    occurrences = defaultdict(list)
    seen_stages = set()
    for index, stage in enumerate(stages):
        if set(stage) != {'stage_id', 'categories'} or not isinstance(stage['stage_id'], str) or not stage['stage_id'].strip():
            raise ValueError('invalid stage specification')
        if stage['stage_id'] in seen_stages:
            raise ValueError('duplicate stage_id')
        seen_stages.add(stage['stage_id'])
        categories = stage['categories']
        if not isinstance(categories, list) or not categories or any(not isinstance(c, str) or not c.strip() for c in categories):
            raise ValueError('stage needs category names')
        if len(categories) != len(set(categories)):
            raise ValueError('duplicate category in stage')
        for category in categories:
            occurrences[category].append(index)
    domains = json.loads(source_config.read_text(encoding='utf-8'))['train_domains']
    if len({d['category'] for d in domains}) != len(domains):
        raise ValueError('converter currently requires exactly one source dataset per category')
    if {d['category'] for d in domains} != set(occurrences):
        raise ValueError('recipe categories must exactly cover source training categories')
    payload = dict(schema_version=1, description='Generated interleaved stream; whole IDs allocated once; original test split preserved.',
        data_root=Path(os.path.relpath(data_root, output)).as_posix(),
        stages=[dict(stage_id=s['stage_id'], categories=[]) for s in stages], evaluation=[])
    manifests, assignments, source_report = {}, [], {}
    for domain_index, domain in enumerate(sorted(domains, key=lambda d: d['category'])):
        category, source = domain['category'], domain['name']
        protocol = PROTOCOLS.get(domain['eval_protocol'])
        if protocol is None:
            raise ValueError('unsupported evaluation protocol: ' + domain['eval_protocol'])
        source_root = (source_config.parent / domain['root']).resolve()
        prefix = source_root.relative_to(data_root).as_posix()
        manifest = (source_config.parent / domain['manifest']).resolve()
        with manifest.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'path', 'pid', 'camid', 'split'}.issubset(reader.fieldnames or []):
                raise ValueError('legacy manifest missing required fields')
            raw = list(reader)
        if not raw:
            raise ValueError('empty source manifest')
        grouped, split_rows, split_paths = defaultdict(list), defaultdict(list), defaultdict(set)
        for row in raw:
            if (row['split'] not in ('train', 'query', 'gallery') or not row['pid'].strip() or not row['camid'].strip()
                    or row.get('category', category) != category or row.get('domain', source) != source):
                raise ValueError('source row metadata disagrees with domain configuration')
            relative = row['path'].replace('\\', '/')
            if (not relative or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
                    or '..' in PurePosixPath(relative).parts):
                raise ValueError('source image paths must be safe relative paths')
            path = (PurePosixPath(prefix) / relative).as_posix()
            if path in split_paths[row['split']]:
                raise ValueError('duplicate source image path within split')
            split_paths[row['split']].add(path)
            converted = dict(path=path, original_pid=row['pid'], source_dataset=source,
                             camid=row['camid'], augmentation=row.get('augmentation', ''))
            split_rows[row['split']].append(converted)
            if row['split'] == 'train':
                grouped[row['pid']].append(converted)
        test_ids = {r['original_pid'] for s in ('query', 'gallery') for r in split_rows[s]}
        if set(grouped) & test_ids:
            raise ValueError('original source contains train/test identity overlap')
        eligible = {pid: validation_rows(rows, protocol) for pid, rows in grouped.items()}
        eligible = {pid: rows for pid, rows in eligible.items() if rows is not None}
        num_validation = math.ceil(len(grouped) * fraction)
        if len(eligible) < num_validation:
            raise ValueError('too few valid held-out identities for ' + category)
        validation_ids = set(shuffled(eligible, seed, category, 'validation')[:num_validation])
        train_ids = shuffled(set(grouped) - validation_ids, seed, category, 'training_stages')
        if len(train_ids) < len(occurrences[category]) * minimum:
            raise ValueError('not enough training identities per stage for ' + category)
        counts, discarded = {}, 0
        for position, stage_index in enumerate(occurrences[category]):
            # Balanced identity counts; remainder goes to earliest appearances.
            size, extra = divmod(len(train_ids), len(occurrences[category]))
            start = position * size + min(position, extra)
            ids = train_ids[start:start + size + int(position < extra)]
            rows = [r for pid in sorted(ids) for r in sorted(grouped[pid], key=lambda r: r['path'])]
            name = 'manifests/train_{:02d}_{:02d}.csv'.format(stage_index + 1, domain_index + 1)
            manifests[name] = rows
            payload['stages'][stage_index]['categories'].append(dict(category=category, train_manifest=name))
            counts[stages[stage_index]['stage_id']] = dict(identities=len(ids), images=len(rows))
            assignments.extend(dict(category=category, source_dataset=source, original_pid=pid,
                allocation='train', stage_id=stages[stage_index]['stage_id'], original_train_images=len(grouped[pid])) for pid in sorted(ids))
        query, gallery = [], []
        for pid in sorted(validation_ids):
            rows = eligible[pid]
            # One deterministically selected query, all other retained originals
            # as gallery. cross_camera filters same-ID same-camera positives.
            query_index = rng(seed, category, 'query/' + pid).randrange(len(rows))
            query.append(rows[query_index])
            gallery.extend(r for i, r in enumerate(rows) if i != query_index)
            discarded += len(grouped[pid]) - len(rows)
            assignments.append(dict(category=category, source_dataset=source, original_pid=pid,
                allocation='validation', stage_id='', original_train_images=len(grouped[pid])))
        for split, query_rows, gallery_rows in [('validation', query, gallery), ('test', split_rows['query'], split_rows['gallery'])]:
            if not query_rows or not gallery_rows:
                raise ValueError('missing query/gallery in ' + category + '/' + split)
            stem = 'manifests/eval_{:02d}_{}'.format(domain_index + 1, split)
            manifests[stem + '_query.csv'], manifests[stem + '_gallery.csv'] = query_rows, gallery_rows
            payload['evaluation'].append(dict(name=category + '_' + split, category=category, split=split,
                protocol=protocol, query_manifest=stem + '_query.csv', gallery_manifest=stem + '_gallery.csv'))
        source_report[category] = dict(source_dataset=source, source_manifest=str(manifest), source_manifest_sha256=digest(manifest),
            original_train_identities=len(grouped), original_train_images=len(split_rows['train']),
            validation_identities=len(validation_ids), validation_queries=len(query), validation_gallery=len(gallery),
            excluded_heldout_augmented_images=discarded, training_stages=counts,
            test_query_images=len(split_rows['query']), test_gallery_images=len(split_rows['gallery']),
            test_identities=len(test_ids), protocol=protocol, original_test_rows_preserved=True)
    # Preserve the user's display order; manifest filenames use safe indices.
    for stage, specification in zip(payload['stages'], stages):
        stage['categories'].sort(key=lambda v: specification['categories'].index(v['category']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stream_build_', dir=output.parent) as temporary:
        temporary = Path(temporary)
        for name, rows in manifests.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction='ignore', lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
        (temporary / 'main.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        with (temporary / 'identity_assignments.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['category', 'source_dataset', 'original_pid', 'allocation', 'stage_id', 'original_train_images'], lineterminator='\n')
            writer.writeheader()
            writer.writerows(assignments)
        # Temporary directory is a sibling, so relative image roots are identical.
        load_category_stream(temporary / 'main.json', check_images=check_images)
        report = dict(recipe=recipe, sources=source_report, source_config_sha256=digest(source_config),
            image_existence_checked=check_images, image_contents_decoded=False,
            output_manifest_sha256={name: digest(temporary / name) for name in sorted(manifests)},
            note='Audit paths/fingerprint are local; re-audit after moving to Ubuntu before starting a new run.')
        (temporary / 'preparation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, output)
    stream = load_category_stream(output / 'main.json', check_images=check_images)
    audit = stream.audit_report()
    audit.update(image_existence_checked=check_images, image_contents_decoded=False)
    (output / 'audit_report.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return stream, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', default=str(ROOT / 'config/cross_category_five_domains.json'))
    parser.add_argument('--recipe', default=str(ROOT / 'config/category_stream_recipe.json'))
    parser.add_argument('--output-dir', default=str(ROOT / 'config/category_progressive_real'))
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--check-images', action='store_true')
    args = parser.parse_args(argv)
    stream, _ = build_stream(args.source_config, args.recipe, args.output_dir, args.data_root, args.check_images)
    print(json.dumps(stream.audit_report(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
