"""Complete ECPM memory: cumulative IDs, FINCH modes, snapshots and drift.

Candidates expose old_categories (all historical donors), category_updates
(current categories only), and drifts (None for new categories). No student
training, teacher copying or routing is performed in this module.
"""

import copy
import math
from dataclasses import asdict

import torch

from .ecpm import IdentityPrototypeMemory, _digest
from .finch_modes import FinchConfig, aggregate_modes, build_category_modes, finch_runtime, prototype_drift


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


class ECPMMemory(IdentityPrototypeMemory):
    def __init__(self, model, stream, clustering=FinchConfig()):
        super().__init__(model, stream)
        if not isinstance(clustering, FinchConfig):
            raise TypeError('clustering must be FinchConfig')
        self.clustering = clustering
        self._runtime = finch_runtime()
        self._category_memory = {}
        self._last_transition = None

    def snapshot(self):
        """Independent copies of all committed historical category modes/centers."""
        return copy.deepcopy(self._category_memory)

    @property
    def last_transition(self):
        return copy.deepcopy(self._last_transition)

    def prepare_stage(self, model, stage, batch_size=128, workers=0):
        identity_candidate = super().prepare_stage(model, stage, batch_size, workers)
        return self.prepare_identity_stage(identity_candidate)

    def prepare_identity_stage(self, identity_candidate):
        """Also accepts step-4 identity candidates, without re-reading any image."""
        rows, processed = self._validated_stage_rows(identity_candidate)
        old = self.snapshot()  # Before any current-category update.
        stage_id = processed[-1]
        current = sorted({r['identity_key'][0] for r in identity_candidate['rows']})
        updates, drifts = {}, {}
        for category in current:
            updates[category] = build_category_modes(
                [r for r in rows if r['identity_key'][0] == category], stage_id, self.clustering)
            drifts[category] = (prototype_drift(old[category]['category_prototype'],
                                              updates[category]['category_prototype']) if category in old else None)
        return dict(binding=copy.deepcopy(identity_candidate['binding']),
                    base_stages=identity_candidate['base_stages'], stage_id=stage_id,
                    rows=copy.deepcopy(identity_candidate['rows']), old_categories=old,
                    category_updates=updates, drifts=drifts, clustering=asdict(self.clustering),
                    runtime=copy.deepcopy(self._runtime))

    def _validate_categories(self, categories, rows, processed):
        expected = {r['identity_key'][0] for r in rows}
        if set(categories) != expected:
            raise ValueError('category memory coverage mismatch')
        for category, memory in categories.items():
            if set(memory) != {'identity_keys', 'labels', 'mode_prototypes', 'category_prototype',
                               'cluster_sizes', 'last_updated_stage', 'clustering'}:
                raise ValueError('invalid category memory schema')
            category_rows = sorted([r for r in rows if r['identity_key'][0] == category], key=lambda r: r['identity_key'])
            keys = tuple(r['identity_key'] for r in category_rows)
            last_stage = max((r['first_stage'] for r in category_rows), key=processed.index)
            if memory['identity_keys'] != keys or memory['last_updated_stage'] != last_stage:
                raise ValueError('category identity keys or last-updated stage mismatch')
            vectors = torch.stack([r['vector'] for r in category_rows])
            modes, center, sizes = aggregate_modes(vectors, memory['labels'])
            for field, expected_tensor in [('mode_prototypes', modes), ('category_prototype', center), ('cluster_sizes', sizes)]:
                tensor = memory[field]
                if (not isinstance(tensor, torch.Tensor) or tensor.dtype != expected_tensor.dtype
                        or tensor.device.type != 'cpu' or tensor.requires_grad or tensor.shape != expected_tensor.shape
                        or not torch.allclose(tensor, expected_tensor, atol=1e-6, rtol=1e-6)):
                    raise ValueError('inconsistent ' + field)
            details = memory['clustering']
            n = len(keys)
            fixed = dict(runtime=self._runtime, algorithm='FINCH', partition=0, distance='cosine',
                         neighbors='exact_chunked', tie_break='first_identity_key', chunk_size=self.clustering.chunk_size,
                         singleton=(n == 1), distance_block_bytes=0 if n == 1 else min(n, self.clustering.chunk_size) * n * 4)
            if set(details) != set(fixed) | {'clustering_seconds'} or any(details[k] != v for k, v in fixed.items()):
                raise ValueError('incompatible FINCH metadata')
            seconds = details['clustering_seconds']
            if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('invalid clustering time')

    def _validate_drifts(self, drifts, old, updates):
        if set(drifts) != set(updates):
            raise ValueError('drift coverage mismatch')
        for c in updates:
            if c not in old:
                if drifts[c] is not None:
                    raise ValueError('new category drift must be None')
            else:
                expected = prototype_drift(old[c]['category_prototype'], updates[c]['category_prototype'])
                if (type(drifts[c]) not in (int, float) or not math.isfinite(drifts[c])
                        or not 0 <= drifts[c] <= 2 or abs(drifts[c] - expected) > 1e-6):
                    raise ValueError('drift does not match old/new category centers')

    def _validated_mode_stage(self, prepared):
        rows, processed = self._validated_stage_rows(prepared)
        if prepared['clustering'] != asdict(self.clustering) or prepared['runtime'] != self._runtime:
            raise ValueError('FINCH configuration/runtime mismatch')
        if not _same(prepared['old_categories'], self._category_memory):
            raise ValueError('old category snapshot differs from committed memory')
        current = {r['identity_key'][0] for r in prepared['rows']}
        if set(prepared['category_updates']) != current:
            raise ValueError('only current categories may be updated')
        categories = copy.deepcopy(self._category_memory)
        categories.update(copy.deepcopy(prepared['category_updates']))
        self._validate_categories(categories, rows, processed)
        self._validate_drifts(prepared['drifts'], self._category_memory, prepared['category_updates'])
        transition = copy.deepcopy(dict(stage_id=prepared['stage_id'], old_categories=prepared['old_categories'],
                                        drifts=prepared['drifts']))
        return rows, processed, categories, transition

    def validate_candidate(self, prepared, model, stage):
        """Check PGCA's uncommitted current-stage input without changing memory."""
        if (prepared['stage_id'] != stage.stage_id
                or self._stage_hashes.get(stage.stage_id) != _digest(asdict(stage))
                or model.reference_signature != self._binding['reference_signature']):
            raise ValueError('ECPM candidate/model/stage mismatch')
        model.assert_reference_unchanged()
        self._validated_mode_stage(prepared)

    def commit_stage(self, prepared):
        rows, processed, categories, transition = self._validated_mode_stage(prepared)
        # Everything, including the old snapshot and all categories, validated first.
        self._rows, self._processed = rows, processed
        self._category_memory, self._last_transition = categories, transition
        return self.summary()

    def summary(self):
        report = super().summary()
        report['clustering_config'] = asdict(self.clustering)
        report['finch_runtime'] = copy.deepcopy(self._runtime)
        report['category_modes'] = {c: dict(identities=len(m['identity_keys']), modes=len(m['cluster_sizes']),
            cluster_sizes=m['cluster_sizes'].tolist(), last_updated_stage=m['last_updated_stage'],
            clustering_seconds=m['clustering']['clustering_seconds'],
            distance_block_bytes=m['clustering']['distance_block_bytes'],
            mode_and_center_bytes=(m['mode_prototypes'].numel() + m['category_prototype'].numel()) * 4,
            assignment_bytes=m['labels'].numel() * 8 + m['cluster_sizes'].numel() * 8)
            for c, m in self._category_memory.items()}
        report['drifts'] = {} if self._last_transition is None else copy.deepcopy(self._last_transition['drifts'])
        # Payload accounting excludes Python/string/serialization overhead and work buffers.
        def tensor_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(v) for v in value.values())
            return 0
        report['summary_tensor_bytes'] = tensor_bytes(self._category_memory)
        report['old_snapshot_tensor_bytes'] = tensor_bytes(self._last_transition)
        report['total_tensor_bytes'] = (report['vector_bytes'] + report['summary_tensor_bytes']
                                        + report['old_snapshot_tensor_bytes'])
        return report

    def state_dict(self):
        return copy.deepcopy(dict(kind='ecpm_mode_memory', schema_version=1,
            identity_memory=super().state_dict(), clustering=asdict(self.clustering), runtime=self._runtime,
            category_memory=self._category_memory, last_transition=self._last_transition))

    @classmethod
    def from_state_dict(cls, state, model, stream):
        if state.get('kind') != 'ecpm_mode_memory' or state.get('schema_version') != 1:
            raise ValueError('expected complete ECPM mode checkpoint; upgrade identity-only memory explicitly')
        identity = IdentityPrototypeMemory.from_state_dict(state['identity_memory'], model, stream)
        memory = cls(model, stream, FinchConfig(**state['clustering']))
        if state['runtime'] != memory._runtime:
            raise ValueError('FINCH runtime differs; restore the recorded dependency versions')
        categories = copy.deepcopy(state['category_memory'])
        memory._validate_categories(categories, identity._rows, identity._processed)
        transition = copy.deepcopy(state['last_transition'])
        if not identity._processed:
            if transition is not None:
                raise ValueError('empty memory cannot contain a transition')
        else:
            last_stage = identity._processed[-1]
            if not isinstance(transition, dict) or set(transition) != {'stage_id', 'old_categories', 'drifts'} or transition['stage_id'] != last_stage:
                raise ValueError('invalid last transition')
            old_rows = [r for r in identity._rows if r['first_stage'] != last_stage]
            old = transition['old_categories']
            memory._validate_categories(old, old_rows, identity._processed[:-1])
            current = {r['identity_key'][0] for r in identity._rows if r['first_stage'] == last_stage}
            if any(not _same(old[c], categories[c]) for c in set(old) - current):
                raise ValueError('absent category changed in last transition')
            memory._validate_drifts(transition['drifts'], old, {c: categories[c] for c in current})
        memory._rows, memory._processed = identity._rows, identity._processed
        memory._category_memory, memory._last_transition = categories, transition
        return memory

    def extend_from_identity_memory(self, identity):
        """Reconstruct chronological modes from step-4 vectors, never from images.

        With an empty receiver this explicitly upgrades all completed stages.
        With prior ECPM state it only processes the not-yet-committed suffix.
        """
        if (identity._binding != self._binding or identity._order != self._order
                or identity._stage_hashes != self._stage_hashes
                or identity._processed[:len(self._processed)] != self._processed):
            raise ValueError('identity memory is incompatible or behind ECPM memory')
        history = [r for r in identity._rows if r['first_stage'] in self._processed]
        if not _same(history, self._rows):
            raise ValueError('historical identity prototypes changed')
        reports = []
        for stage_id in identity._processed[len(self._processed):]:
            candidate = dict(binding=copy.deepcopy(self._binding), base_stages=self._processed,
                             stage_id=stage_id, rows=copy.deepcopy([r for r in identity._rows if r['first_stage'] == stage_id]))
            reports.append(self.commit_stage(self.prepare_identity_stage(candidate)))
        return reports
