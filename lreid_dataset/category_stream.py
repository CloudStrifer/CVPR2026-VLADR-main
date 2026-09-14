"""Category-interleaved ReID protocol. Metadata auditing needs only stdlib.

Training images are opened lazily by an explicitly selected stage's dataset.
This module does not import the legacy dataset registry or a model.
"""

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


class StreamProtocolError(ValueError):
    """Invalid stream, identity leakage, or invalid retrieval protocol."""


def _fail(where, message):
    raise StreamProtocolError('{}: {}'.format(where, message))


def _object(value, where, allowed):
    if not isinstance(value, dict):
        _fail(where, 'expected an object')
    unknown = set(value) - set(allowed)
    if unknown:
        _fail(where, 'unknown fields: {}'.format(sorted(unknown)))
    return value


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        _fail(where, 'expected a non-empty string')
    return value.strip()


def _identifier(value, where):
    # Preserve strings such as "001"; never silently merge them with "1".
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        _fail(where, 'expected a string or integer identifier')
    return _text(str(value), where)


def _items(value, where, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        _fail(where, 'expected {}list'.format('a non-empty ' if nonempty else 'a '))
    return value


def _path(value, base, where):
    path = Path(_text(value, where)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


@dataclass(frozen=True)
class StreamSample:
    path: str
    category: str
    source_dataset: str
    original_pid: str
    camid: str
    identity_key: tuple
    stage_id: str
    split: str


@dataclass(frozen=True)
class CategoryStage:
    stage_id: str
    category: str
    samples: tuple

    @property
    def identity_keys(self):
        return tuple(sorted({s.identity_key for s in self.samples}))

    @property
    def label_map(self):
        return {key: index for index, key in enumerate(self.identity_keys)}


@dataclass(frozen=True)
class StageView:
    stage_id: str
    categories: tuple
    new_categories: tuple
    recurring_categories: tuple
    absent_categories: tuple
    seen_before: tuple

    def category(self, name):
        for view in self.categories:
            if view.category == name:
                return view
        raise KeyError('category {!r} is not in stage {!r}'.format(name, self.stage_id))


@dataclass(frozen=True)
class EvaluationView:
    name: str
    category: str
    split: str
    protocol: str
    query: tuple
    gallery: tuple

    @property
    def label_map(self):
        keys = sorted({s.identity_key for s in self.query + self.gallery})
        return {key: index for index, key in enumerate(keys)}

    def is_positive(self, query, gallery):
        """Match rule shared by audit and future evaluation implementation."""
        if query.identity_key != gallery.identity_key or _path_key(query.path) == _path_key(gallery.path):
            return False
        if self.protocol == 'cross_camera':
            # For aliased identities across sources, curators must normalize
            # real camera IDs. Changing a source name cannot create a positive.
            return query.camid != gallery.camid
        return True


@dataclass(frozen=True)
class CategoryStream:
    config_path: str
    stages: tuple
    evaluations: tuple
    fingerprint: str

    def stage(self, stage_id):
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError('unknown stage {!r}'.format(stage_id))

    def evaluations_at(self, stage_id, split='test'):
        if split not in ('validation', 'test'):
            raise ValueError('evaluation split must be validation or test')
        stage = self.stage(stage_id)
        seen = set(stage.seen_before) | {c.category for c in stage.categories}
        return tuple(e for e in self.evaluations if e.split == split and e.category in seen)

    def audit_report(self):
        warnings = []
        categories = {c.category for s in self.stages for c in s.categories}
        for split in ('validation', 'test'):
            missing = categories - {e.category for e in self.evaluations if e.split == split}
            if missing:
                warnings.append('No {} set configured for: {}'.format(split, ', '.join(sorted(missing))))
        return {
            'schema_version': 1,
            'config_path': self.config_path,
            'protocol_fingerprint': self.fingerprint,
            'image_contents_read': False,
            'stages': [
                {
                    'stage_id': s.stage_id,
                    'new_categories': list(s.new_categories),
                    'recurring_categories': list(s.recurring_categories),
                    'absent_categories': list(s.absent_categories),
                    'seen_before': list(s.seen_before),
                    'categories': [
                        {'category': c.category, 'identities': len(c.identity_keys),
                         'images': len(c.samples)} for c in s.categories
                    ],
                } for s in self.stages
            ],
            'evaluation': [
                {'name': e.name, 'category': e.category, 'split': e.split,
                 'protocol': e.protocol, 'identities': len(e.label_map),
                 'query_images': len(e.query), 'gallery_images': len(e.gallery)}
                for e in self.evaluations
            ],
            'total_train_identities': sum(
                len(c.identity_keys) for s in self.stages for c in s.categories
            ),
            'total_train_images': sum(len(c.samples) for s in self.stages for c in s.categories),
            'warnings': warnings,
        }


def _read_rows(path):
    suffix = path.suffix.lower()
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            if suffix in ('.csv', '.tsv'):
                reader = csv.DictReader(handle, delimiter='\t' if suffix == '.tsv' else ',')
                fields = reader.fieldnames or []
                if len(set(fields)) != len(fields):
                    _fail(path, 'duplicate column names')
                required = {'path', 'original_pid', 'source_dataset', 'camid'}
                if not required.issubset(fields):
                    _fail(path, 'missing columns: {}'.format(sorted(required - set(fields))))
                rows = list(reader)
            elif suffix == '.jsonl':
                rows = [json.loads(line) for line in handle if line.strip()]
            else:
                _fail(path, 'manifest must be CSV, TSV or JSONL')
    except (OSError, ValueError) as exc:
        if isinstance(exc, StreamProtocolError):
            raise
        _fail(path, str(exc))
    if not rows:
        _fail(path, 'empty manifest; training cannot fall back to query/gallery')
    return rows


def _aliases(payload):
    result = {}
    for index, entry in enumerate(_items(payload, 'identity_aliases', nonempty=False)):
        where = 'identity_aliases[{}]'.format(index)
        _object(entry, where, ('category', 'source_dataset', 'original_pid', 'canonical_id'))
        key = (_text(entry.get('category'), where + '.category'),
               _text(entry.get('source_dataset'), where + '.source_dataset'),
               _identifier(entry.get('original_pid'), where + '.original_pid'))
        canonical = _identifier(entry.get('canonical_id'), where + '.canonical_id')
        if key in result:
            _fail(where, 'duplicate identity alias {}'.format(key))
        result[key] = (key[0], '__canonical__', canonical)
    return result


def _samples(manifest, root, category, stage_id, split, aliases, used_aliases, check_images):
    result = []
    paths = set()
    for index, row in enumerate(_read_rows(manifest), start=2):
        where = '{}:row {}'.format(manifest, index)
        if not isinstance(row, dict) or None in row:
            _fail(where, 'invalid manifest row')
        source = _text(row.get('source_dataset'), where + '.source_dataset')
        if source == '__canonical__':
            _fail(where, '__canonical__ is a reserved source namespace')
        pid = _identifier(row.get('original_pid'), where + '.original_pid')
        camid = _identifier(row.get('camid'), where + '.camid')
        for field, expected in (('category', category), ('stage_id', stage_id), ('split', split)):
            if field in row and row[field] not in (None, ''):
                if str(row[field]).strip() != expected:
                    _fail(where, '{} does not match parent configuration {!r}'.format(field, expected))
        image_path = _path(row.get('path'), root, where + '.path')
        image_key = _path_key(image_path)
        if image_key in paths:
            _fail(where, 'duplicate image path {}'.format(image_path))
        paths.add(image_key)
        if check_images and not image_path.is_file():
            _fail(where, 'image file does not exist: {}'.format(image_path))
        key = (category, source, pid)
        if key in aliases:
            used_aliases.add(key)
        result.append(StreamSample(str(image_path), category, source, pid, camid,
                                   aliases.get(key, key), stage_id, split))
    return tuple(result)


def _audit(stages, evaluations):
    identities = {}
    images = {}

    def register(sample, owner, allow_shared_eval_image=False):
        previous = identities.setdefault(sample.identity_key, owner)
        if previous != owner:
            _fail(owner, 'identity leakage: {} also belongs to {}'.format(sample.identity_key, previous))
        path_key = _path_key(sample.path)
        signature = (owner, sample.identity_key, sample.source_dataset, sample.camid)
        previous_image = images.get(path_key)
        if previous_image is not None:
            if not allow_shared_eval_image or previous_image != signature:
                _fail(owner, 'image leakage or conflicting metadata: {}'.format(sample.path))
        images[path_key] = signature

    for stage in stages:
        for category in stage.categories:
            for sample in category.samples:
                register(sample, 'train/{}'.format(stage.stage_id))
    for evaluation in evaluations:
        owner = '{}/{}'.format(evaluation.split, evaluation.name)
        for sample in evaluation.query + evaluation.gallery:
            register(sample, owner, allow_shared_eval_image=True)
        gallery_by_id = defaultdict(list)
        for sample in evaluation.gallery:
            gallery_by_id[sample.identity_key].append(sample)
        for query in evaluation.query:
            if not any(evaluation.is_positive(query, g) for g in gallery_by_id[query.identity_key]):
                _fail(owner, 'query has no valid {} positive: {}'.format(evaluation.protocol, query.path))


def load_category_stream(config_path, check_images=False):
    """Parse and audit all metadata; never decode training/evaluation images.

    `check_images` checks file existence only. All relative manifests and roots
    resolve against the config directory; image paths resolve against data_root.
    Stage order is the order in the JSON list, not lexicographic stage_id order.
    """
    config_path = Path(config_path).expanduser().resolve()
    try:
        payload = json.loads(config_path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        _fail(config_path, str(exc))
    _object(payload, 'config', ('schema_version', 'description', 'data_root', 'stages',
                               'evaluation', 'identity_aliases'))
    if type(payload.get('schema_version')) is not int or payload['schema_version'] != 1:
        _fail('schema_version', 'only integer schema_version=1 is supported')
    base = config_path.parent
    aliases = _aliases(payload.get('identity_aliases', []))
    used_aliases = set()
    seen = set()
    stage_ids = set()
    stages = []

    def root_for(spec, where):
        value = spec.get('data_root', payload.get('data_root'))
        return _path(value, base, where + '.data_root')

    for index, spec in enumerate(_items(payload.get('stages'), 'stages')):
        where = 'stages[{}]'.format(index)
        _object(spec, where, ('stage_id', 'categories'))
        stage_id = _text(spec.get('stage_id'), where + '.stage_id')
        if stage_id in stage_ids:
            _fail(where, 'duplicate stage_id {}'.format(stage_id))
        stage_ids.add(stage_id)
        categories = []
        current = set()
        for entry in _items(spec.get('categories'), where + '.categories'):
            _object(entry, where, ('category', 'train_manifest', 'data_root'))
            category = _text(entry.get('category'), where + '.category')
            if category in current:
                _fail(where, 'duplicate category {}'.format(category))
            current.add(category)
            manifest = _path(entry.get('train_manifest'), base, where + '.train_manifest')
            samples = _samples(manifest, root_for(entry, where), category, stage_id, 'train',
                               aliases, used_aliases, check_images)
            categories.append(CategoryStage(stage_id, category, samples))
        stages.append(StageView(stage_id, tuple(categories), tuple(sorted(current - seen)),
                                tuple(sorted(current & seen)), tuple(sorted(seen - current)),
                                tuple(sorted(seen))))
        seen.update(current)

    evaluations = []
    eval_names = set()
    for index, entry in enumerate(_items(payload.get('evaluation', []), 'evaluation', False)):
        where = 'evaluation[{}]'.format(index)
        _object(entry, where, ('name', 'category', 'split', 'protocol', 'query_manifest',
                               'gallery_manifest', 'data_root'))
        name = _text(entry.get('name'), where + '.name')
        if name in eval_names:
            _fail(where, 'duplicate evaluation name {}'.format(name))
        eval_names.add(name)
        category = _text(entry.get('category'), where + '.category')
        if category not in seen:
            _fail(where, 'evaluation category never appears in this training stream')
        split = entry.get('split')
        if split not in ('validation', 'test'):
            _fail(where, 'split must be validation or test')
        protocol = entry.get('protocol')
        if protocol not in ('cross_camera', 'exclude_self'):
            _fail(where, 'protocol must explicitly be cross_camera or exclude_self')
        groups = []
        for subset in ('query', 'gallery'):
            manifest = _path(entry.get(subset + '_manifest'), base, where + '.' + subset)
            groups.append(_samples(manifest, root_for(entry, where), category, '', subset,
                                   aliases, used_aliases, check_images))
        evaluations.append(EvaluationView(name, category, split, protocol, *groups))
    unused = set(aliases) - used_aliases
    if unused:
        _fail('identity_aliases', 'aliases do not match any manifest identity: {}'.format(sorted(unused)))
    _audit(stages, evaluations)
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8'))
    for stage in stages:
        for category in stage.categories:
            for sample in category.samples:
                digest.update(json.dumps(sample.__dict__, sort_keys=True).encode('utf-8'))
    for evaluation in evaluations:
        for sample in evaluation.query + evaluation.gallery:
            digest.update(json.dumps(sample.__dict__, sort_keys=True).encode('utf-8'))
    return CategoryStream(str(config_path), tuple(stages), tuple(evaluations), digest.hexdigest())
