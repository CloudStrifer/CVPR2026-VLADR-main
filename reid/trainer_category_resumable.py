"""Step-level trainer: synchronous augmentation and explicit sampler cursors."""

import copy
from dataclasses import asdict
from pathlib import Path

import torch

from lreid_dataset.category_stream_loaders import CategoryIdentityBatchSampler, collate_category_samples
from reid.adaptation.pgca import (FrozenCategoryTeacher, PGCATransferConfig,
                                 PrototypeGuidedInitialization, RecurringCategoryConsistency)
from reid.loss.category_triplet import CategoryBatchHardTriplet
from reid.loss.pgca import FeatureConsistencyLoss, PGCAConsistencyConfig, consistency_weight
from reid.models.category_adapter_bank import CategoryAdapterBank
from reid.memory.prototype_views import control_view
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig


def model_from_bank(reference, bank, device):
    return CategoryAdapterBank.from_checkpoint(dict(reference, bank=bank), device=device)


class ResumableCategoryTrainer(CategoryProgressiveTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_in_epoch = 0
        self.partial_totals = {c: {} for c in self.categories}
        self._batch_cache = {}
        self.sampler_contract = self._sampler_contract()

    def _sampler_contract(self):
        result = {}
        for c in self.categories:
            pair = self.loaders[c]
            sampler = pair.train.batch_sampler
            if pair.train.num_workers != 0 or type(sampler) is not CategoryIdentityBatchSampler:
                raise ValueError('exact resume requires workers=0 and CategoryIdentityBatchSampler')
            result[c] = dict(seed=sampler.seed, batch_size=sampler.batch_size,
                             num_instances=sampler.num_instances)
        return result

    def _indices(self, c, epoch, iteration):
        # Batch counts can vary between passes for imbalanced identities. Replay
        # only local-RNG index construction, never skipped image augmentations.
        pair = self.loaders[c]
        pass_index = epoch * (self.config.iterations_per_epoch + 1)
        offset = iteration
        while True:
            key = (c, pass_index)
            if key not in self._batch_cache:
                pair.set_epoch(pass_index)
                self._batch_cache[key] = list(pair.train.batch_sampler)
            batches = self._batch_cache[key]
            if not batches:
                raise ValueError('empty sampling pass')
            if offset < len(batches):
                return batches[offset], dict(pass_index=pass_index, offset=offset)
            offset -= len(batches)
            pass_index += 1

    def sampling_cursor(self):
        if self.completed_epochs == self.config.epochs:
            return None
        return {c: self._indices(c, self.completed_epochs, self.iteration_in_epoch)[1] for c in self.categories}

    def train_step(self):
        if self.failed or self.finished or self.completed_epochs >= self.config.epochs:
            raise RuntimeError('inactive trainer')
        if self.model.trainable_categories != self.categories:
            raise RuntimeError('model training categories changed outside trainer')
        epoch, iteration = self.completed_epochs, self.iteration_in_epoch
        self.model.train()
        try:
            if iteration == 0:
                self.consistency.assert_teacher_unchanged()
            indices = {c: self._indices(c, epoch, iteration) for c in self.categories}
            batches = {c: collate_category_samples([self.loaders[c].train.dataset[i] for i in indices[c][0]])
                       for c in self.categories}
            values, norm = self._train_batches(batches)
            self.optimizer_steps += 1
            self.iteration_in_epoch += 1
            for c in self.categories:
                for key, value in values[c].items():
                    if key != 'prototype_drift':
                        self.partial_totals[c][key] = self.partial_totals[c].get(key, 0) + value
            event = dict(event='train_step', stage_id=self.stage.stage_id, epoch=epoch, iteration=iteration,
                         optimizer_step=self.optimizer_steps, gradient_norm=norm.item(), categories=values,
                         sampling={c: dict(indices=indices[c][0], **indices[c][1]) for c in self.categories},
                         summed_loss=sum(v['total'] for v in values.values()))
            self._write_event(event)
            if self.iteration_in_epoch == self.config.iterations_per_epoch:
                self.consistency.assert_teacher_unchanged()
                steps = self.config.iterations_per_epoch
                report = dict(event='epoch_end', stage_id=self.stage.stage_id, epoch=epoch,
                              optimizer_steps=steps, categories={})
                for c, totals in self.partial_totals.items():
                    report['categories'][c] = {k: v if k == 'images' else v / steps for k, v in totals.items()}
                    report['categories'][c].update(updates=steps,
                        loader_restarts=indices[c][1]['pass_index'] - epoch * (steps + 1))
                    if self._pgca_logging:
                        report['categories'][c]['prototype_drift'] = self.consistency.drift(c)
                self.completed_epochs += 1
                self.iteration_in_epoch = 0
                self.partial_totals = {c: {} for c in self.categories}
                self._batch_cache.clear()
                self.epoch_reports.append(report)
                self._write_event(report)
            return event
        except BaseException:
            # Includes Ctrl+C: caller must reload the last durable checkpoint,
            # never serialize possibly half-applied optimizer updates.
            self.failed = True
            self.optimizer.zero_grad(set_to_none=True)
            raise

    def train_epoch(self, epoch=None):
        if epoch is not None and epoch != self.completed_epochs:
            raise ValueError('epoch must match saved progress')
        current = self.completed_epochs
        while self.completed_epochs == current:
            self.train_step()
        return self.epoch_reports[-1]

    def state_dict(self):
        if self.failed or self.finished:
            raise RuntimeError('only active, successful optimizer boundaries can be saved')
        self.consistency.assert_teacher_unchanged()
        teacher = self.consistency.teacher
        return dict(schema_version=1, stage_id=self.stage.stage_id, config=asdict(self.config),
                    bank=self.model.export_bank(), heads=copy.deepcopy(self.model.temporary_heads.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()), scaler=self.scaler.state_dict(),
                    scheduler=None, completed_epochs=self.completed_epochs, iteration_in_epoch=self.iteration_in_epoch,
                    optimizer_steps=self.optimizer_steps, partial_totals=copy.deepcopy(self.partial_totals),
                    epoch_reports=copy.deepcopy(self.epoch_reports), sampler_contract=self.sampler_contract,
                    sampler_cursor=self.sampling_cursor(), pgca=self.consistency.metadata(),
                    teacher_bank=None if teacher is None else teacher.model.export_bank(),
                    initialization=self.initialization.metadata())

    @classmethod
    def from_state_dict(cls, state, reference, stage, loaders, memory, candidate, device, log_path=None):
        """Restore directly. Never apply transfer or derive a teacher from student."""
        if state.get('schema_version') != 1 or state.get('stage_id') != stage.stage_id or state.get('scheduler') is not None:
            raise ValueError('invalid trainer checkpoint')
        self = cls.__new__(cls)
        self.model = model_from_bank(reference, state['bank'], device)
        self.stage, self.config = stage, CategoryTrainingConfig(**state['config'])
        self.device = self.model.visual.proj.device
        if self.config.amp and self.device.type != 'cuda':
            raise ValueError('AMP checkpoint requires CUDA')
        memory.validate_candidate(candidate, self.model, stage)
        self.categories = tuple(sorted(c.category for c in stage.categories))
        if set(self.model.categories) != set(stage.seen_before) | set(self.categories) or set(loaders) != set(self.categories):
            raise ValueError('checkpoint/category history mismatch')
        self.loaders = dict(loaders)
        self._label_maps, self._samples_by_path = {}, {}
        for c in self.categories:
            view, pair = stage.category(c), loaders[c]
            if pair.view != view or tuple(pair.train.dataset.samples) != view.samples or pair.train.dataset.label_map != view.label_map:
                raise ValueError('restored loader differs from stage')
            self._label_maps[c] = view.label_map
            self._samples_by_path[c] = {s.path: s for s in view.samples}
            self.model.create_temporary_head(c, len(view.label_map))
        self.model.temporary_heads.load_state_dict(state['heads'], strict=True)
        self.model.set_trainable_categories(self.categories)
        self.model.train()
        self.optimizer = torch.optim.AdamW([
            dict(params=[p for c in self.categories for p in self.model.adapter_parameters(c)], lr=self.config.adapter_lr, name='adapters'),
            dict(params=list(self.model.temporary_heads.parameters()), lr=self.config.head_lr, name='temporary_heads')],
            weight_decay=self.config.weight_decay)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.amp)
        self.scaler.load_state_dict(state['scaler'])
        self.triplet = CategoryBatchHardTriplet(self.config.triplet_margin)
        self.consistency_loss = FeatureConsistencyLoss()
        meta = state['pgca']
        con = RecurringCategoryConsistency.__new__(RecurringCategoryConsistency)
        con.config = PGCAConsistencyConfig(**meta['config'])
        con.categories, con.recurring = self.categories, tuple(sorted(stage.recurring_categories))
        con._drifts, con._weights = dict(meta['drifts']), dict(meta['weights'])
        if (meta['recurring_categories'] != list(con.recurring)
                or con._drifts != control_view(memory, candidate, con.config.summary)['drifts']
                or con._weights != {c: consistency_weight(con.config, con._drifts[c], c in con.recurring) for c in self.categories}):
            raise ValueError('saved consistency metadata differs from stage candidate')
        con._teacher_metadata, con.teacher = copy.deepcopy(meta['teacher']), None
        active = any(con._weights[c] > 0 for c in con.recurring)
        if active != (state['teacher_bank'] is not None) or active != (meta['teacher'] is not None):
            raise ValueError('missing or unexpected historical teacher')
        if active:
            teacher = FrozenCategoryTeacher.__new__(FrozenCategoryTeacher)
            teacher.model = model_from_bank(reference, state['teacher_bank'], device)
            teacher.model.set_trainable_categories([])
            torch.nn.Module.train(teacher.model, False)
            teacher.categories = con.recurring
            teacher.signature = meta['teacher']['state_sha256']
            teacher.reference_signature = meta['teacher']['reference_signature']
            teacher.adapter_signatures = dict(meta['teacher']['adapter_sha256'])
            if set(teacher.model.categories) != set(stage.seen_before) or teacher.metadata() != meta['teacher']:
                raise ValueError('historical teacher metadata mismatch')
            teacher.assert_unchanged()
            con.teacher = teacher
        self.consistency, self._pgca_logging = con, con.config.mode != 'off'
        init = state['initialization']
        if (not init['applied'] or init['stage_id'] != stage.stage_id
                or init['new_categories'] != sorted(stage.new_categories)
                or init['historical_categories'] != sorted(stage.seen_before)):
            raise ValueError('invalid saved initialization')
        obj = PrototypeGuidedInitialization.__new__(PrototypeGuidedInitialization)
        obj.config = PGCATransferConfig(**init['config'])
        obj.categories, obj.new_categories = self.categories, tuple(init['new_categories'])
        obj.historical_categories, obj.stage_id = tuple(init['historical_categories']), stage.stage_id
        obj.reference_signature, obj.adapter_config = self.model.reference_signature, self.model.adapter_config
        obj._decisions, obj._donor_hashes = copy.deepcopy(init['decisions']), dict(init['historical_adapter_sha256'])
        obj.snapshot_tensor_bytes, obj._applied, obj._snapshots = init['snapshot_tensor_bytes'], True, {}
        self.initialization = obj
        for key in ('completed_epochs', 'iteration_in_epoch', 'optimizer_steps'):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError('invalid training progress')
            setattr(self, key, state[key])
        if (self.completed_epochs > self.config.epochs or self.iteration_in_epoch >= self.config.iterations_per_epoch
                or (self.completed_epochs == self.config.epochs and self.iteration_in_epoch)
                or self.optimizer_steps != self.completed_epochs * self.config.iterations_per_epoch + self.iteration_in_epoch):
            raise ValueError('inconsistent optimizer/epoch cursor')
        self.partial_totals, self.epoch_reports = copy.deepcopy(state['partial_totals']), copy.deepcopy(state['epoch_reports'])
        if len(self.epoch_reports) != self.completed_epochs or set(self.partial_totals) != set(self.categories):
            raise ValueError('inconsistent accumulated reports')
        self.failed = self.finished = False
        self._batch_cache = {}
        self.sampler_contract = self._sampler_contract()
        if self.sampler_contract != state['sampler_contract'] or self.sampling_cursor() != state['sampler_cursor']:
            raise ValueError('sampling configuration/cursor changed')
        self.log_path = Path(log_path) if log_path is not None else None
        # Caller restores global RNG only AFTER all model/loader constructors.
        return self
