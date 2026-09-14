"""Step 4 of ECPM: append-only identity prototypes in frozen reference space.

Modes, category centers and drift belong to step 5. Prepared stages are
temporary candidates: prepare does not change memory, commit is all-or-nothing.
"""

import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from lreid_dataset.category_stream_loaders import build_stage_prototype_loaders


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


class IdentityPrototypeAccumulator:
    """FP32 sums of raw features, plus exact coverage checks for current rows."""

    def __init__(self, view, feature_dim):
        self.view = view
        self.feature_dim = feature_dim
        self.expected = {_path_key(s.path): s for s in view.samples}
        if not self.expected or len(self.expected) != len(view.samples):
            raise ValueError('empty or duplicate image rows')
        if any(s.split != 'train' or s.stage_id != view.stage_id or s.category != view.category
               for s in view.samples):
            raise ValueError('only current category training rows are allowed')
        self.seen = set()
        self.sums = {key: torch.zeros(feature_dim, dtype=torch.float32) for key in view.identity_keys}
        self.counts = Counter()

    def add(self, features, batch):
        fields = ('paths', 'identity_keys', 'categories', 'stage_ids', 'splits')
        n = len(batch['paths'])
        if any(len(batch[field]) != n for field in fields) or features.shape != (n, self.feature_dim):
            raise ValueError('feature/metadata shape mismatch')
        features = features.detach().to(device='cpu', dtype=torch.float32)
        if not torch.isfinite(features).all():
            bad = (~torch.isfinite(features).all(dim=1)).nonzero().flatten().tolist()
            raise FloatingPointError('non-finite reference features for identities: {}'.format(
                [batch['identity_keys'][i] for i in bad]))
        pending = []
        paths = set()
        for i in range(n):
            path = _path_key(batch['paths'][i])
            sample = self.expected.get(path)
            if sample is None or path in self.seen or path in paths:
                raise ValueError('unexpected or repeated prototype image')
            if (tuple(batch['identity_keys'][i]) != sample.identity_key
                    or batch['categories'][i] != sample.category
                    or batch['stage_ids'][i] != sample.stage_id or batch['splits'][i] != 'train'):
                raise ValueError('prototype row metadata does not match current training identity')
            paths.add(path)
            pending.append(sample.identity_key)
        for key, feature in zip(pending, features):
            self.sums[key].add_(feature)
            self.counts[key] += 1
        self.seen.update(paths)

    def finish(self):
        if self.seen != set(self.expected):
            raise ValueError('prototype extraction did not visit every current training image exactly once')
        rows = []
        for key in sorted(self.sums):
            mean = self.sums[key] / self.counts[key]
            norm = mean.norm()
            if not torch.isfinite(mean).all() or not torch.isfinite(norm) or norm <= 1e-12:
                raise FloatingPointError('identity mean is non-finite or has zero norm: {!r}'.format(key))
            rows.append(dict(identity_key=key, vector=mean / norm, image_count=self.counts[key],
                             first_stage=self.view.stage_id))
        return rows


class IdentityPrototypeMemory:
    """All historical identities, bound to one audited stream and reference.

    State stores only per-identity vectors/counts/keys and compatibility metadata.
    No model, stream, paths, images or per-image features are retained here.
    Caller must serialize prepare/commit operations (not a concurrent writer API).
    """

    def __init__(self, model, stream):
        model.assert_reference_unchanged()
        self._binding = dict(reference_signature=model.reference_signature,
                             reference=asdict(model.reference), feature_dim=model.feature_dim,
                             stream_fingerprint=stream.fingerprint,
                             formula='l2_normalize(mean(raw_projected_cls_fp32))')
        self._order = tuple(stage.stage_id for stage in stream.stages)
        self._stage_hashes = {stage.stage_id: _digest(asdict(stage)) for stage in stream.stages}
        # Metadata-only contract: enables full coverage validation also on restore.
        self._expected = {stage.stage_id: dict(Counter(
            s.identity_key for view in stage.categories for s in view.samples)) for stage in stream.stages}
        self._processed = ()
        self._rows = []

    @property
    def processed_stages(self):
        return self._processed

    @property
    def categories(self):
        return tuple(sorted({row['identity_key'][0] for row in self._rows}))

    def category_prototypes(self, category):
        """Return independent copies in arrival-stage, then identity-key order."""
        return copy.deepcopy([row for row in self._rows if row['identity_key'][0] == category])

    def _check_next(self, stage_id):
        index = len(self._processed)
        if index >= len(self._order) or stage_id != self._order[index]:
            raise ValueError('expected next unprocessed stage; repeated or out-of-order stage')

    def prepare_stage(self, model, stage, batch_size=128, workers=0):
        self._check_next(stage.stage_id)
        if self._stage_hashes[stage.stage_id] != _digest(asdict(stage)):
            raise ValueError('stage differs from the audited stream')
        if model.reference_signature != self._binding['reference_signature']:
            raise ValueError('reference signature mismatch')
        model.assert_reference_unchanged()
        loaders = build_stage_prototype_loaders(stage, model.make_transforms()[1], batch_size, workers)
        rows = []
        for view in stage.categories:
            accumulator = IdentityPrototypeAccumulator(view, self._binding['feature_dim'])
            for batch in loaders[view.category]:
                features = model.encode_reference(batch['images'].to(model.visual.proj.device))
                accumulator.add(features, batch)
            rows.extend(accumulator.finish())
        model.assert_reference_unchanged()
        return dict(binding=copy.deepcopy(self._binding), base_stages=self._processed,
                    stage_id=stage.stage_id, rows=rows)

    def _validate_rows(self, rows, processed):
        expected = {}
        for stage_id in processed:
            for key, count in self._expected[stage_id].items():
                if key in expected:
                    raise ValueError('identity repeated across stages')
                expected[key] = (stage_id, count)
        seen = set()
        for row in rows:
            if set(row) != {'identity_key', 'vector', 'image_count', 'first_stage'}:
                raise ValueError('invalid identity memory row schema')
            key = row['identity_key']
            if not isinstance(key, tuple) or key not in expected or key in seen:
                raise ValueError('unknown or duplicate identity key')
            if (type(row['image_count']) is not int
                    or (row['first_stage'], row['image_count']) != expected[key]):
                raise ValueError('identity stage or image count mismatch')
            vector = row['vector']
            if (not isinstance(vector, torch.Tensor) or vector.dtype != torch.float32
                    or vector.device.type != 'cpu' or vector.requires_grad
                    or vector.shape != (self._binding['feature_dim'],)):
                raise ValueError('prototype must be a detached CPU FP32 vector of reference dimension')
            if not torch.isfinite(vector).all() or not torch.isclose(vector.norm(), torch.tensor(1.), atol=1e-5, rtol=1e-5):
                raise ValueError('prototype must be finite and unit-normalized')
            seen.add(key)
        if seen != set(expected):
            raise ValueError('missing identity prototypes')

    def _validated_stage_rows(self, prepared):
        self._check_next(prepared['stage_id'])
        if prepared['binding'] != self._binding or prepared['base_stages'] != self._processed:
            raise ValueError('stale or incompatible prepared stage')
        processed = self._processed + (prepared['stage_id'],)
        additions = copy.deepcopy(prepared['rows'])
        self._validate_rows(additions, (prepared['stage_id'],))
        rows = self._rows + additions
        self._validate_rows(rows, processed)
        return rows, processed

    def commit_stage(self, prepared):
        rows, processed = self._validated_stage_rows(prepared)
        self._rows, self._processed = rows, processed
        return self.summary()

    def update_stage(self, model, stage, batch_size=128, workers=0):
        return self.commit_stage(self.prepare_stage(model, stage, batch_size, workers))

    def summary(self):
        return dict(processed_stages=list(self._processed), identities=len(self._rows),
                    images=sum(row['image_count'] for row in self._rows),
                    categories={c: sum(row['identity_key'][0] == c for row in self._rows) for c in self.categories},
                    feature_dim=self._binding['feature_dim'],
                    vector_bytes=len(self._rows) * self._binding['feature_dim'] * 4,
                    reference_signature=self._binding['reference_signature'],
                    stream_fingerprint=self._binding['stream_fingerprint'])

    def state_dict(self):
        return copy.deepcopy(dict(kind='ecpm_identity_memory', schema_version=1,
                                  binding=self._binding, stage_order=self._order,
                                  stage_hashes=self._stage_hashes, processed_stages=self._processed,
                                  rows=self._rows))

    @classmethod
    def from_state_dict(cls, state, model, stream):
        memory = cls(model, stream)
        if (state.get('kind') != 'ecpm_identity_memory' or state.get('schema_version') != 1
                or state.get('binding') != memory._binding or state.get('stage_order') != memory._order
                or state.get('stage_hashes') != memory._stage_hashes):
            raise ValueError('incompatible memory checkpoint, stream or reference')
        processed = state['processed_stages']
        if not isinstance(processed, tuple) or processed != memory._order[:len(processed)]:
            raise ValueError('processed stages must be an ordered prefix')
        rows = copy.deepcopy(state['rows'])
        memory._validate_rows(rows, processed)
        memory._rows, memory._processed = rows, processed
        return memory

    def save(self, path):
        """Atomic file replacement; interruption preserves the previous complete file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
                name = handle.name
                torch.save(self.state_dict(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path, model, stream):
        return cls.from_state_dict(torch.load(path, map_location='cpu', weights_only=True), model, stream)
