"""Manifest-backed datasets for cross-category lifelong ReID experiments.

The original VLADR loaders encode the directory and filename conventions of
person ReID benchmarks.  This adapter provides one category-neutral format:

    path,pid,camid,split
    images/train/a.jpg,animal_001,0,train
    images/query/a.jpg,animal_101,0,query
    images/gallery/a.jpg,animal_101,1,gallery

CSV, TSV, and JSONL manifests are supported.  Optional ``category`` and
``domain`` columns are validated when present but are not required by the
legacy training tuple interface.
"""

from __future__ import absolute_import

import csv
import json
import os
import os.path as osp
from collections import defaultdict


REQUIRED_COLUMNS = ('path', 'pid', 'camid', 'split')
VALID_SPLITS = ('train', 'query', 'gallery')


def _read_json_or_yaml(path):
    suffix = osp.splitext(path)[1].lower()
    with open(path, 'r', encoding='utf-8') as handle:
        if suffix == '.json':
            return json.load(handle)
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                'YAML domain configs require PyYAML. Use JSON or install PyYAML.'
            ) from exc
        return yaml.safe_load(handle)


def _resolve_path(path, base_dir):
    path = osp.expanduser(str(path))
    if osp.isabs(path):
        return osp.normpath(path)
    return osp.normpath(osp.join(base_dir, path))


def load_domain_config(path):
    """Load and validate a cross-category experiment config.

    Relative manifest and prompt-checkpoint paths are resolved against the
    directory containing the config. Image paths inside a manifest remain
    relative to each domain's ``root`` (or the command-line ``--data-dir``).
    """

    path = osp.abspath(osp.expanduser(path))
    payload = _read_json_or_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError('domain config must contain a JSON/YAML object')

    config_dir = osp.dirname(path)
    train_domains = payload.get('train_domains')
    test_domains = payload.get('test_domains', [])
    if not isinstance(train_domains, list) or not train_domains:
        raise ValueError('domain config needs a non-empty train_domains list')
    if not isinstance(test_domains, list):
        raise ValueError('test_domains must be a list')

    seen_names = set()
    normalized = {'train_domains': [], 'test_domains': []}
    for split_name, domains in (
        ('train_domains', train_domains),
        ('test_domains', test_domains),
    ):
        for raw_spec in domains:
            if not isinstance(raw_spec, dict):
                raise ValueError('{} entries must be objects'.format(split_name))
            spec = dict(raw_spec)
            for key in ('name', 'category', 'manifest'):
                if not spec.get(key):
                    raise ValueError(
                        '{} entry is missing required field {!r}'.format(
                            split_name, key
                        )
                    )
            if spec['name'] in seen_names:
                raise ValueError(
                    'domain names must be unique; duplicate {!r}'.format(
                        spec['name']
                    )
                )
            seen_names.add(spec['name'])

            spec['manifest'] = _resolve_path(spec['manifest'], config_dir)
            if spec.get('prompt_checkpoint'):
                spec['prompt_checkpoint'] = _resolve_path(
                    spec['prompt_checkpoint'],
                    config_dir,
                )

            spec.setdefault('object_noun', spec['category'])
            category_generic_prompts = spec.get(
                'category_generic_prompts'
            )
            if category_generic_prompts is not None:
                if (
                    not isinstance(category_generic_prompts, list)
                    or not category_generic_prompts
                ):
                    raise ValueError(
                        '{} entry {!r} must define category_generic_prompts '
                        'as a non-empty list'.format(
                            split_name,
                            spec['name'],
                        )
                    )
                normalized_prompts = []
                for prompt_index, prompt in enumerate(
                    category_generic_prompts
                ):
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(
                            '{} entry {!r} has an invalid category-generic '
                            'prompt at index {}'.format(
                                split_name,
                                spec['name'],
                                prompt_index,
                            )
                        )
                    normalized_prompts.append(prompt.strip())
                if len(set(normalized_prompts)) != len(normalized_prompts):
                    raise ValueError(
                        '{} entry {!r} contains duplicate '
                        'category_generic_prompts'.format(
                            split_name,
                            spec['name'],
                        )
                    )
                spec['category_generic_prompts'] = normalized_prompts

            attributes = spec.get('attributes')
            if attributes is not None:
                if not isinstance(attributes, list) or not attributes:
                    raise ValueError(
                        '{} entry {!r} must define attributes as a '
                        'non-empty list'.format(split_name, spec['name'])
                    )
                normalized_attributes = []
                seen_attribute_names = set()
                for attribute_index, attribute in enumerate(attributes):
                    if isinstance(attribute, str):
                        name = 'attribute_{}'.format(attribute_index + 1)
                        prompt = attribute.strip()
                    elif isinstance(attribute, dict):
                        name = str(attribute.get('name', '')).strip()
                        prompt = str(attribute.get('prompt', '')).strip()
                    else:
                        raise ValueError(
                            '{} entry {!r} has an invalid attribute at '
                            'index {}'.format(
                                split_name,
                                spec['name'],
                                attribute_index,
                            )
                        )
                    if not name or not prompt:
                        raise ValueError(
                            '{} entry {!r} attribute {} needs non-empty '
                            'name and prompt'.format(
                                split_name,
                                spec['name'],
                                attribute_index,
                            )
                        )
                    if name in seen_attribute_names:
                        raise ValueError(
                            '{} entry {!r} contains duplicate attribute '
                            'name {!r}'.format(
                                split_name,
                                spec['name'],
                                name,
                            )
                        )
                    seen_attribute_names.add(name)
                    normalized_attributes.append(
                        {'name': name, 'prompt': prompt}
                    )
                spec['attributes'] = normalized_attributes
            spec['_config_dir'] = config_dir
            spec['_is_train_domain'] = split_name == 'train_domains'
            normalized[split_name].append(spec)

    normalized['input_size'] = payload.get('input_size')
    normalized['resize_mode'] = payload.get('resize_mode')
    normalized['source_path'] = path
    return normalized


class ManifestReID(object):
    """Expose a manifest through VLADR's legacy dataset attributes."""

    images_dir = None

    def __init__(self, datasets_root, spec, relabel=True, combineall=False):
        del combineall
        if not isinstance(spec, dict):
            raise TypeError('manifest dataset spec must be a dictionary')

        self.name = str(spec['name'])
        self.category = str(spec['category'])
        self.object_noun = str(spec.get('object_noun', self.category))
        self.manifest_path = osp.abspath(osp.expanduser(spec['manifest']))
        root = spec.get('root', datasets_root)
        config_dir = spec.get('_config_dir', osp.dirname(self.manifest_path))
        self.root = _resolve_path(root, config_dir) if root else config_dir
        self.relabel = relabel
        self.is_train_domain = bool(spec.get('_is_train_domain', True))

        rows = self._read_manifest(self.manifest_path)
        split_rows = defaultdict(list)
        for row_number, row in enumerate(rows, start=2):
            split_name = str(row['split']).strip().lower()
            if split_name not in VALID_SPLITS:
                raise ValueError(
                    '{}:{} has invalid split {!r}'.format(
                        self.manifest_path, row_number, split_name
                    )
                )

            if row.get('category') and str(row['category']) != self.category:
                raise ValueError(
                    '{}:{} category {!r} does not match domain category {!r}'.format(
                        self.manifest_path,
                        row_number,
                        row['category'],
                        self.category,
                    )
                )

            path = _resolve_path(row['path'], self.root)
            if not osp.isfile(path):
                raise FileNotFoundError(
                    '{}:{} image not found: {}'.format(
                        self.manifest_path, row_number, path
                    )
                )
            try:
                camid = int(row['camid'])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    '{}:{} camid must be an integer'.format(
                        self.manifest_path, row_number
                    )
                ) from exc
            if camid < 0:
                raise ValueError(
                    '{}:{} camid must be non-negative'.format(
                        self.manifest_path, row_number
                    )
                )

            split_rows[split_name].append(
                {
                    'path': path,
                    'pid': str(row['pid']),
                    'camid': camid,
                }
            )

        required_splits = (
            VALID_SPLITS if self.is_train_domain else ('query', 'gallery')
        )
        for split_name in required_splits:
            if not split_rows[split_name]:
                raise ValueError(
                    '{} contains no {} samples'.format(
                        self.manifest_path, split_name
                    )
                )

        train_pid_map = self._pid_map(split_rows['train'])
        eval_pid_map = self._pid_map(
            split_rows['query'] + split_rows['gallery']
        )
        self.train = self._as_tuples(split_rows['train'], train_pid_map)
        self.query = self._as_tuples(split_rows['query'], eval_pid_map)
        self.gallery = self._as_tuples(split_rows['gallery'], eval_pid_map)
        self.num_train_pids = len(train_pid_map)
        self.num_train_imgs = len(self.train)

        self._validate_eval_protocol(split_rows)
        self._show_info()

    @staticmethod
    def _read_manifest(path):
        if not osp.isfile(path):
            raise FileNotFoundError('manifest not found: {}'.format(path))
        suffix = osp.splitext(path)[1].lower()
        if suffix in ('.jsonl', '.ndjson'):
            rows = []
            with open(path, 'r', encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            '{}:{} is not valid JSON'.format(path, line_number)
                        ) from exc
        else:
            delimiter = '\t' if suffix in ('.tsv', '.txt') else ','
            with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle, delimiter=delimiter))

        if not rows:
            raise ValueError('manifest is empty: {}'.format(path))
        missing = [key for key in REQUIRED_COLUMNS if key not in rows[0]]
        if missing:
            raise ValueError(
                'manifest {} is missing columns: {}'.format(
                    path, ', '.join(missing)
                )
            )
        return rows

    @staticmethod
    def _pid_map(rows):
        def sort_key(pid):
            try:
                return 0, int(pid)
            except (TypeError, ValueError):
                return 1, str(pid)

        return {
            pid: index
            for index, pid in enumerate(
                sorted({row['pid'] for row in rows}, key=sort_key)
            )
        }

    @staticmethod
    def _as_tuples(rows, pid_map):
        tuples = []
        for row in rows:
            pid = pid_map[row['pid']]
            tuples.append((row['path'], pid, row['camid'], 0))
        return tuples

    def _validate_eval_protocol(self, split_rows):
        gallery_by_pid = defaultdict(list)
        for row in split_rows['gallery']:
            gallery_by_pid[row['pid']].append(row['camid'])

        invalid = []
        for row in split_rows['query']:
            gallery_cams = gallery_by_pid.get(row['pid'], [])
            if not any(camid != row['camid'] for camid in gallery_cams):
                invalid.append((row['pid'], row['camid']))
        if invalid:
            preview = ', '.join(
                '{}@cam{}'.format(pid, camid) for pid, camid in invalid[:5]
            )
            raise ValueError(
                '{} query identities have no cross-camera gallery match '
                '(examples: {})'.format(len(invalid), preview)
            )

    def _show_info(self):
        def stats(samples):
            return (
                len(samples),
                len({sample[1] for sample in samples}),
                len({sample[2] for sample in samples}),
            )

        print(
            '=> Loaded {} [{}] from {}'.format(
                self.name, self.category, self.manifest_path
            )
        )
        for split_name in VALID_SPLITS:
            count, identities, cameras = stats(getattr(self, split_name))
            print(
                '  {:7s}: {:6d} images, {:5d} IDs, {:3d} cameras'.format(
                    split_name, count, identities, cameras
                )
            )
