"""PGCA recurring consistency and new-category prototype-guided initialization."""

import copy
import math
import hashlib
import random
from dataclasses import asdict, dataclass

import torch

from reid.loss.pgca import PGCAConsistencyConfig, consistency_weight
from reid.models.category_adapter_bank import _tensor_digest
from reid.memory.prototype_views import check_summary, control_view


@dataclass(frozen=True)
class PGCATransferConfig:
    mode: str = 'default'
    alpha: float = .5
    delta: float = .5
    summary: str = 'ecpm'
    seed: int = 42
    source: str = None

    def __post_init__(self):
        check_summary(self.summary)
        if self.mode not in ('default', 'similarity', 'random', 'source'):
            raise ValueError('initialization mode must be default, similarity, random or source')
        if type(self.seed) is not int or (self.mode == 'source' and (not isinstance(self.source, str) or not self.source)):
            raise ValueError('invalid transfer seed/source')
        for name, lower, upper in (('alpha', 0., 1.), ('delta', -1., 1.)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('{} must be finite in [{},{}]'.format(name, lower, upper))


def prototype_similarity(new_category, historical_category, alpha=.5):
    """Directional new -> historical coverage, mean over new-mode row maxima."""
    from reid.memory.finch_modes import unit_rows

    PGCATransferConfig(alpha=alpha)
    new_modes, old_modes = new_category['mode_prototypes'], historical_category['mode_prototypes']
    new_center, old_center = new_category['category_prototype'], historical_category['category_prototype']
    for matrix in (new_modes, old_modes, new_center.unsqueeze(0), old_center.unsqueeze(0)):
        unit_rows(matrix)
    if not (new_modes.shape[1] == old_modes.shape[1] == new_center.numel() == old_center.numel()):
        raise ValueError('category and mode dimensions must match')
    # Keep threshold decisions in FP32 even inside an ambient CPU autocast.
    with torch.autocast(device_type='cpu', enabled=False):
        new_modes = torch.nn.functional.normalize(new_modes, dim=1)
        old_modes = torch.nn.functional.normalize(old_modes, dim=1)
        mode_similarity = (new_modes @ old_modes.T).clamp(-1., 1.).max(dim=1).values.mean().item()
        global_similarity = torch.nn.functional.cosine_similarity(new_center[None], old_center[None]).clamp(-1., 1.).item()
    score = alpha * global_similarity + (1. - alpha) * mode_similarity
    return dict(global_similarity=global_similarity, mode_similarity=mode_similarity,
                score=min(1., max(-1., float(score))))


def select_transfer_source(new_category, historical_categories, config):
    """Pure scoring with stable exact-score tie breaks and inclusive threshold."""
    scores = {c: prototype_similarity(new_category, historical_categories[c], config.alpha)
              for c in sorted(historical_categories)}
    if not scores:
        return dict(candidates={}, best_source=None, best_score=None, selected_source=None, reason='no_history')
    best = min(scores, key=lambda c: (-scores[c]['score'], c))
    accepted = scores[best]['score'] >= config.delta
    return dict(candidates=scores, best_source=best, best_score=scores[best]['score'],
                selected_source=best if accepted else None,
                reason='threshold_met' if accepted else 'below_threshold')


class PrototypeGuidedInitialization:
    """Prepare all decisions and CPU donor snapshots before current training.

    apply() uses only frozen donor copies in sorted category order. It never
    changes recurring/absent adapters. An optional callback creates each fresh
    classifier after its adapter, preserving the baseline RNG call order.
    Construct optimizers/graphs only after the complete apply() call.
    """

    def __init__(self, model, stage, config=None, ecpm_memory=None, ecpm_candidate=None):
        self.config = config or PGCATransferConfig()
        if not isinstance(self.config, PGCATransferConfig):
            raise TypeError('initialization must be PGCATransferConfig')
        current = {view.category for view in stage.categories}
        self.categories = tuple(sorted(current))
        self.new_categories = tuple(sorted(current - set(stage.seen_before)))
        self.historical_categories = tuple(sorted(stage.seen_before))
        if set(stage.new_categories) != set(self.new_categories):
            raise ValueError('new category metadata disagrees with seen_before')
        self.stage_id = stage.stage_id
        self.reference_signature = model.reference_signature
        self.adapter_config = model.adapter_config
        self._snapshots = {}
        self._applied = False
        self._decisions = {c: dict(candidates={}, best_source=None, best_score=None,
                                  selected_source=None, reason='default_mode') for c in self.new_categories}
        if self.config.mode != 'default':
            if ecpm_memory is None or ecpm_candidate is None:
                raise ValueError('similarity initialization requires old ECPM memory and current candidate')
            ecpm_memory.validate_candidate(ecpm_candidate, model, stage)
            ecpm_candidate = control_view(ecpm_memory, ecpm_candidate, self.config.summary)
            if set(ecpm_candidate['old_categories']) != set(self.historical_categories):
                raise ValueError('transfer donor history disagrees with stage history')
            if not set(self.historical_categories).issubset(model.categories):
                raise ValueError('missing historical donor adapter')
            if set(self.new_categories) & set(model.categories):
                raise ValueError('similarity initialization requires unregistered new categories')
            if set(model.categories) != set(self.historical_categories):
                raise ValueError('adapter registry contains categories outside stage-start history')
            if model.temporary_heads:
                raise ValueError('prepare transfer before current temporary heads/training')
            for c in (self.historical_categories if self.new_categories else ()):
                snapshot = model.export_adapter(c)
                model._validate_adapter_snapshot(snapshot)
                self._snapshots[c] = snapshot
            for c in self.new_categories:
                self._decisions[c] = select_transfer_source(ecpm_candidate['category_updates'][c],
                                                            ecpm_candidate['old_categories'], self.config)
                if self.historical_categories and self.config.mode in ('random', 'source'):
                    if self.config.mode == 'random':
                        seed = hashlib.sha256('{}\0{}\0{}'.format(self.config.seed, self.stage_id, c).encode()).hexdigest()
                        selected = random.Random(seed).choice(self.historical_categories)
                    else:
                        selected = self.config.source
                        if selected not in self.historical_categories:
                            raise ValueError('forced source must be a historical category')
                    self._decisions[c].update(selected_source=selected, reason=self.config.mode + '_ablation')
        self._donor_hashes = {c: _tensor_digest(s['state']) for c, s in self._snapshots.items()}
        self.snapshot_tensor_bytes = sum(t.numel() * t.element_size() for s in self._snapshots.values() for t in s['state'].values())

    def apply(self, model, after_category=None):
        if self._applied:
            raise RuntimeError('initialization has already been applied')
        if model.reference_signature != self.reference_signature or model.adapter_config != self.adapter_config:
            raise ValueError('initialization target reference/architecture mismatch')
        if self.config.mode != 'default':
            if model.temporary_heads or set(self.new_categories) & set(model.categories):
                raise ValueError('apply before new categories, classifiers and optimizer are created')
            model.assert_reference_unchanged()
            # Validate every donor before mutating any target; copying uses snapshots,
            # never a possibly already-updated live model.copy_category(source,...).
            for c, snapshot in self._snapshots.items():
                model._validate_adapter_snapshot(snapshot)
                if _tensor_digest(snapshot['state']) != self._donor_hashes[c]:
                    raise ValueError('donor snapshot changed before initialization')
        for c in self.categories:
            if self.config.mode != 'default' and c in self.new_categories:
                decision = self._decisions[c]
                source = decision['selected_source']
                if source is None:
                    model.add_category(c)
                else:
                    model.load_adapter(c, self._snapshots[source])
                decision['initialized_adapter_sha256'] = _tensor_digest(model.export_adapter(c)['state'])
                decision['source_adapter_sha256'] = None if source is None else self._donor_hashes[source]
            else:
                model.add_category(c)
            if after_category is not None:
                after_category(c)
        self._applied = True
        # Only selected-source hashes/decisions persist; no historical parameter cache.
        self._snapshots.clear()
        return self.metadata()

    def metadata(self):
        return copy.deepcopy(dict(config=asdict(self.config), stage_id=self.stage_id,
            new_categories=list(self.new_categories), historical_categories=list(self.historical_categories),
            decisions=self._decisions, historical_adapter_sha256=self._donor_hashes,
            snapshot_tensor_bytes=self.snapshot_tensor_bytes, applied=self._applied))


class FrozenCategoryTeacher:
    """Independent stage-start visual bank. No student parameter swapping.

    One full reference tower copy carries the historical bank; forwards are
    restricted to recurring categories. Deepcopy does not consume RNG, so
    zero-consistency baselines keep identical student/head initialization.
    """

    def __init__(self, student, categories):
        self.categories = tuple(sorted(categories))
        if not self.categories or not set(self.categories).issubset(student.categories):
            raise ValueError('teacher needs registered historical recurring categories')
        if student.temporary_heads:
            raise ValueError('capture teacher before creating current stage heads')
        student.assert_reference_unchanged()
        self.model = copy.deepcopy(student)
        self.model.set_trainable_categories([])
        # The bank's train override enforces visual/head policies but does not
        # recursively set the empty ModuleDict container flag. Freeze every
        # submodule explicitly so the complete teacher is unambiguously eval.
        torch.nn.Module.train(self.model, False)
        self.signature = _tensor_digest(self.model.state_dict())
        self.reference_signature = student.reference_signature
        self.adapter_signatures = {c: _tensor_digest(student.export_adapter(c)['state']) for c in self.categories}

    def _check_policy(self):
        if any(m.training for m in self.model.modules()) or any(
                p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('historical teacher must remain frozen, eval and gradient-free')

    def encode(self, images, category):
        if category not in self.categories:
            raise ValueError('teacher is only for recurring categories')
        self._check_policy()
        with torch.no_grad():
            return self.model.encode_category(images, category).detach()

    def assert_unchanged(self):
        self._check_policy()
        if (self.model.reference_signature != self.reference_signature
                or _tensor_digest(self.model.state_dict()) != self.signature):
            raise RuntimeError('historical teacher changed after its stage-start snapshot')

    def metadata(self):
        return dict(categories=list(self.categories), state_sha256=self.signature,
                    adapter_sha256=dict(self.adapter_signatures), reference_signature=self.reference_signature,
                    tensor_bytes=sum(t.numel() * t.element_size() for t in self.model.state_dict().values()),
                    implementation='independent_frozen_visual_bank')


class RecurringCategoryConsistency:
    """Validated ECPM inputs and immutable-per-stage numeric loss weights."""

    def __init__(self, model, stage, config=None, ecpm_memory=None, ecpm_candidate=None):
        self.config = config or PGCAConsistencyConfig()
        if not isinstance(self.config, PGCAConsistencyConfig):
            raise TypeError('consistency config must be PGCAConsistencyConfig')
        self.categories = tuple(sorted(c.category for c in stage.categories))
        self.recurring = tuple(sorted(set(self.categories) & set(stage.seen_before)))
        if set(stage.recurring_categories) != set(self.recurring):
            raise ValueError('inconsistent recurring category metadata')
        if (ecpm_memory is None) != (ecpm_candidate is None):
            raise ValueError('provide both old ECPM memory and its uncommitted current candidate')
        if self.config.mode == 'drift' and ecpm_memory is None:
            raise ValueError('drift mode requires validated ECPM memory and current candidate')
        self._drifts = {c: None for c in self.categories}
        if ecpm_memory is not None:
            ecpm_memory.validate_candidate(ecpm_candidate, model, stage)
            ecpm_candidate = control_view(ecpm_memory, ecpm_candidate, self.config.summary)
            if set(ecpm_candidate['old_categories']) != set(stage.seen_before):
                raise ValueError('ECPM history does not match stage history')
            self._drifts = copy.deepcopy(ecpm_candidate['drifts'])
        self._weights = {c: consistency_weight(self.config, self._drifts[c], c in self.recurring)
                         for c in self.categories}
        active = [c for c in self.recurring if self._weights[c] > 0]
        # No teacher allocation/forward when every effective weight is zero.
        self.teacher = FrozenCategoryTeacher(model, self.recurring) if active else None
        self._teacher_metadata = self.teacher.metadata() if self.teacher is not None else None

    def weight(self, category):
        return self._weights[category]

    def drift(self, category):
        return self._drifts[category]

    def teacher_features(self, images, category):
        if self.weight(category) == 0:
            return None
        return self.teacher.encode(images, category)

    def assert_teacher_unchanged(self):
        if self.teacher is not None:
            self.teacher.assert_unchanged()

    def metadata(self):
        return dict(config=asdict(self.config), recurring_categories=list(self.recurring),
                    drifts=dict(self._drifts), weights=dict(self._weights),
                    teacher=copy.deepcopy(self._teacher_metadata))

    def release(self):
        self.teacher = None
