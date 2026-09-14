"""Stage-local CE + Triplet, optionally PGCA recurring-category consistency."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from lreid_dataset.category_stream import StageView
from reid.loss.category_triplet import CategoryBatchHardTriplet
from reid.models.category_adapter_bank import CategoryAdapterBank
from reid.adaptation.pgca import RecurringCategoryConsistency, PrototypeGuidedInitialization
from reid.loss.pgca import FeatureConsistencyLoss
from reid.utils.progressive_checkpoint import capture_rng, restore_rng


@dataclass(frozen=True)
class CategoryTrainingConfig:
    epochs: int = 10
    iterations_per_epoch: int = 100
    adapter_lr: float = 0.0003
    head_lr: float = 0.0003
    weight_decay: float = 0.0001
    lambda_tri: float = 1.0
    triplet_margin: float = 0.3
    amp: bool = False
    max_grad_norm: float = None

    def __post_init__(self):
        for name in ('epochs', 'iterations_per_epoch'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('{} must be a positive integer'.format(name))
        for name in ('adapter_lr', 'head_lr', 'weight_decay', 'lambda_tri', 'triplet_margin'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name.endswith('_lr') and value == 0):
                raise ValueError('invalid {}'.format(name))
        if type(self.amp) is not bool:
            raise ValueError('amp must be boolean')
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError('max_grad_norm must be positive or None')


class _CurrentCategoryIterator:
    def __init__(self, pair, epoch, iterations):
        self.pair = pair
        self.pass_index = epoch * (iterations + 1)
        self.restarts = 0
        self._restart()

    def _restart(self):
        self.pair.set_epoch(self.pass_index)
        self.iterator = iter(self.pair.train)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.pass_index += 1
            self.restarts += 1
            self._restart()
            try:
                return next(self.iterator)
            except StopIteration:
                raise ValueError('current category loader produced no batches') from None


class CategoryProgressiveTrainer:
    """One stage, one batch per current category per optimizer step.

    Categories contribute equally weighted *mean* losses; their gradients are
    summed, not divided by the number of categories. Call finish_stage before
    constructing the next stage's trainer/optimizer. Interrupted epochs are not
    resumable by this baseline; step 8 will implement full training resume.
    """

    def __init__(self, model, stage, loaders, config=None, log_path=None,
                 consistency=None, ecpm_memory=None, ecpm_candidate=None, initialization=None):
        if not isinstance(model, CategoryAdapterBank) or not isinstance(stage, StageView):
            raise TypeError('expected CategoryAdapterBank and one StageView')
        self.config = config or CategoryTrainingConfig()
        if not isinstance(self.config, CategoryTrainingConfig):
            raise TypeError('config must be CategoryTrainingConfig')
        self.model, self.stage = model, stage
        self.categories = tuple(sorted(c.category for c in stage.categories))
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ValueError('stage must contain distinct categories')
        if set(loaders) != set(self.categories):
            raise ValueError('loaders must match exactly the current stage categories')
        if not set(stage.seen_before).issubset(model.categories) or set(model.categories) - (
                set(stage.seen_before) | set(self.categories)):
            raise ValueError('model registry does not match this stage history (missing old or future category)')
        if model.temporary_heads:
            raise ValueError('finish/discard the previous stage temporary heads first')
        self.device = model.visual.proj.device
        if self.config.amp and self.device.type != 'cuda':
            raise ValueError('baseline AMP is supported only for CUDA; use amp=False on CPU')
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None and self.log_path.exists() and self.log_path.stat().st_size:
            raise FileExistsError('use a new log file; baseline cannot resume an existing training log')
        self._label_maps = {}
        self._samples_by_path = {}
        for category in self.categories:
            view = stage.category(category)
            pair = loaders[category]
            if pair.view != view or tuple(pair.train.dataset.samples) != view.samples:
                raise ValueError('loader contains a different stage/category dataset')
            if len(view.identity_keys) < 2 or len(pair.train) < 1:
                raise ValueError('each current category needs >=2 identities and a nonempty training loader')
            if pair.train.dataset.label_map != view.label_map:
                raise ValueError('loader local PID mapping differs from current stage')
            self._label_maps[category] = view.label_map
            self._samples_by_path[category] = {s.path: s for s in view.samples}
        # Capture every historical teacher before creating/updating any category.
        self.initialization = PrototypeGuidedInitialization(model, stage, initialization, ecpm_memory, ecpm_candidate)
        self.consistency = RecurringCategoryConsistency(model, stage, consistency, ecpm_memory, ecpm_candidate)
        self.initialization.apply(model, after_category=lambda c: model.create_temporary_head(c, len(self._label_maps[c])))
        self.consistency_loss = FeatureConsistencyLoss()
        self._pgca_logging = self.consistency.config.mode != 'off'
        self.loaders = dict(loaders)
        model.set_trainable_categories(self.categories)
        model.train()
        adapters = [p for c in self.categories for p in model.adapter_parameters(c)]
        heads = [p for p in model.temporary_heads.parameters()]
        self.optimizer = torch.optim.AdamW([
            {'params': adapters, 'lr': self.config.adapter_lr, 'name': 'adapters'},
            {'params': heads, 'lr': self.config.head_lr, 'name': 'temporary_heads'},
        ], weight_decay=self.config.weight_decay)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.amp)
        self.triplet = CategoryBatchHardTriplet(self.config.triplet_margin)
        self.completed_epochs = 0
        self.optimizer_steps = 0
        self.failed = False
        self.finished = False
        self.epoch_reports = []
        self._write_event({'event': 'stage_start', 'stage_id': stage.stage_id,
                           'config': asdict(self.config), 'categories': list(self.categories),
                           'identities': {c: len(m) for c, m in self._label_maps.items()},
                           'loss_components': ['ce', 'triplet'] + (['consistency'] if self._pgca_logging else []),
                           'pgca': self.consistency.metadata(),
                           'initialization': self.initialization.metadata(),
                           'reference_signature': model.reference_signature})

    def _write_event(self, event):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()

    def _validate_batch(self, category, batch):
        required = {'images', 'targets', 'identity_keys', 'paths', 'categories', 'stage_ids', 'splits'}
        if not isinstance(batch, dict) or not required.issubset(batch):
            raise ValueError('incomplete category training batch')
        images, targets = batch['images'], batch['targets']
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError('training images must be Tensor [B,C,H,W]')
        n = len(images)
        if not isinstance(targets, torch.Tensor) or targets.shape != (n,) or targets.dtype != torch.long:
            raise ValueError('training targets must be LongTensor [B]')
        if any(len(batch[name]) != n for name in required - {'images', 'targets'}):
            raise ValueError('batch metadata length mismatch')
        for index, target in enumerate(targets.tolist()):
            if (batch['categories'][index] != category or batch['stage_ids'][index] != self.stage.stage_id
                    or batch['splits'][index] != 'train'):
                raise ValueError('batch contains another category, stage, or evaluation split')
            sample = self._samples_by_path[category].get(batch['paths'][index])
            key = tuple(batch['identity_keys'][index])
            if sample is None or sample.identity_key != key or self._label_maps[category].get(key) != target:
                raise ValueError('batch identity/path/local label does not belong to the current category')
        _, counts = targets.unique(return_counts=True)
        if len(counts) < 2 or (counts < 2).any():
            raise ValueError('each category batch needs >=2 identities and >=2 samples per identity')

    def _train_batches(self, batches):
        for category, batch in batches.items():
            self._validate_batch(category, batch)
        # A retry is part of this same logical update: reuse augmented images
        # and stochastic forward state, and never advance the sampler/counters.
        rng = capture_rng() if self.scaler.is_enabled() else None
        max_retries = 20
        parameters = [p for group in self.optimizer.param_groups for p in group['params']]
        for attempt in range(max_retries + 1):
            values = self._backward_batches(batches)
            self.scaler.unscale_(self.optimizer)
            if self.scaler.is_enabled():
                finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters
                                      if p.grad is not None]).all().item()
                if not finite:
                    old_scale = self.scaler.get_scale()
                    # unscale_ recorded the invalid gradients. GradScaler.step
                    # therefore skips AdamW entirely (including weight decay).
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    new_scale = self.scaler.get_scale()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._write_event(dict(event='amp_overflow_retry', stage_id=self.stage.stage_id,
                                           optimizer_step=self.optimizer_steps + 1, attempt=attempt + 1,
                                           scale_before=old_scale, scale_after=new_scale))
                    if attempt == max_retries or not 0 < new_scale < old_scale:
                        raise FloatingPointError(
                            'AMP gradients remain non-finite after scale backoff in stage {} '
                            '(scale={}); inspect numerical stability or retry a fresh FP32 run '
                            'without --amp'.format(self.stage.stage_id, new_scale))
                    restore_rng(rng)
                    continue
            # Keep stable optimizer order for exact resume. Non-AMP errors and
            # aggregate norm overflow remain fatal; never clip NaN/Inf gradients.
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm or float('inf'),
                                                 error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return values, norm

    def _backward_batches(self, batches):
        self.optimizer.zero_grad(set_to_none=True)
        values = {}
        for category in self.categories:
            batch = batches[category]
            images = batch['images'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.config.amp):
                features = self.model.encode_category(images, category)
                logits = self.model.classify(features, category)
                teacher_features = self.consistency.teacher_features(images, category)
            ce = F.cross_entropy(logits.float(), targets)
            triplet, distances = self.triplet(features, targets)
            total = ce + self.config.lambda_tri * triplet
            weight = self.consistency.weight(category)
            con = self.consistency_loss(features, teacher_features) if teacher_features is not None else features.new_zeros((), dtype=torch.float32)
            if weight > 0:
                total = total + weight * con
            if not torch.isfinite(total):
                raise FloatingPointError('non-finite category loss: stage={}, category={}'.format(
                    self.stage.stage_id, category))
            # Independent category graphs allow immediate backward, keeping
            # only one category's activations until the shared optimizer step.
            self.scaler.scale(total).backward()
            values[category] = dict(ce=ce.detach().item(), triplet=triplet.detach().item(),
                                    weighted_triplet=(self.config.lambda_tri * triplet.detach()).item(),
                                    total=total.detach().item(), images=len(images),
                                    accuracy=(logits.argmax(1) == targets).float().mean().item(), **distances)
            if self._pgca_logging:
                values[category].update(consistency=con.detach().item(), consistency_weight=weight,
                                        weighted_consistency=(weight * con.detach()).item(),
                                        prototype_drift=self.consistency.drift(category))
        return values

    def train_epoch(self, epoch=None):
        epoch = self.completed_epochs if epoch is None else epoch
        if self.finished or self.failed or epoch != self.completed_epochs or epoch >= self.config.epochs:
            raise RuntimeError('invalid epoch or inactive/failed trainer; partial epoch resume is not supported')
        if self.model.trainable_categories != self.categories:
            raise RuntimeError('model training categories changed outside this trainer')
        self.model.train()
        totals = {c: dict(ce=0.0, triplet=0.0, weighted_triplet=0.0, total=0.0,
                          accuracy=0.0, positive_distance=0.0, negative_distance=0.0,
                          active_triplet_fraction=0.0, images=0) for c in self.categories}
        try:
            self.consistency.assert_teacher_unchanged()
            if self._pgca_logging:
                for category in self.categories:
                    totals[category].update(consistency=0., weighted_consistency=0., consistency_weight=0.)
            iterators = {c: _CurrentCategoryIterator(self.loaders[c], epoch, self.config.iterations_per_epoch)
                         for c in self.categories}
            for iteration in range(self.config.iterations_per_epoch):
                batches = {c: iterators[c].next() for c in self.categories}
                values, norm = self._train_batches(batches)
                self.optimizer_steps += 1
                for category in self.categories:
                    for key, value in values[category].items():
                        if key in totals[category]:
                            totals[category][key] += value
                self._write_event({'event': 'train_step', 'stage_id': self.stage.stage_id,
                                   'epoch': epoch, 'iteration': iteration, 'optimizer_step': self.optimizer_steps,
                                   'gradient_norm': norm.item(), 'categories': values,
                                   'summed_loss': sum(v['total'] for v in values.values())})
            self.consistency.assert_teacher_unchanged()
        except Exception as exc:
            self.failed = True
            self.optimizer.zero_grad(set_to_none=True)
            self._write_event({'event': 'stage_error', 'stage_id': self.stage.stage_id,
                               'epoch': epoch, 'completed_optimizer_steps': self.optimizer_steps,
                               'error': str(exc)})
            raise
        steps = self.config.iterations_per_epoch
        report = {'event': 'epoch_end', 'stage_id': self.stage.stage_id, 'epoch': epoch,
                  'optimizer_steps': steps, 'categories': {}}
        for category, values in totals.items():
            report['categories'][category] = {
                key: value if key == 'images' else value / steps for key, value in values.items()}
            report['categories'][category].update(updates=steps, loader_restarts=iterators[category].restarts)
            if self._pgca_logging:
                report['categories'][category]['prototype_drift'] = self.consistency.drift(category)
        self.completed_epochs += 1
        self.epoch_reports.append(report)
        self._write_event(report)
        return report

    def finish_stage(self):
        if self.finished or self.failed or self.completed_epochs != self.config.epochs:
            raise RuntimeError('cannot finish an incomplete, failed, or already completed stage')
        self.model.assert_reference_unchanged()
        self.consistency.assert_teacher_unchanged()
        self.model.set_trainable_categories([])
        self.model.discard_temporary_heads()
        self.model.eval()
        self.finished = True
        result = {'event': 'stage_end', 'stage_id': self.stage.stage_id,
                  'completed_epochs': self.completed_epochs, 'optimizer_steps': self.optimizer_steps,
                  'category_updates': {c: self.optimizer_steps for c in self.categories},
                  'config': asdict(self.config), 'epochs': self.epoch_reports,
                  'pgca': self.consistency.metadata(), 'initialization': self.initialization.metadata()}
        self._write_event(result)
        self.loaders.clear()
        self.optimizer = None
        self.scaler = None
        self.consistency.release()
        return result
