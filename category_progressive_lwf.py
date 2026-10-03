"""LwF loss for the current category-progressive ReID stream.

Historical identity heads are retained as frozen output probes. They are never
used for retrieval; only current-stage images enter the training loss.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from reid.adaptation.pgca import FrozenCategoryTeacher
from reid.loss.lwf import LearningWithoutForgettingLoss
from reid.trainer_category_resumable import ResumableCategoryTrainer, model_from_bank


def _validated_heads(heads, categories, feature_dim):
    if not isinstance(heads, dict) or set(heads) != set(categories):
        raise ValueError('historical head categories must match stage history')
    result = {}
    for category, matrices in heads.items():
        if not isinstance(matrices, list) or not matrices:
            raise ValueError('each historical category needs old identity logits')
        result[category] = []
        for matrix in matrices:
            if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 2
                    or matrix.shape[0] < 2 or matrix.shape[1] != feature_dim
                    or not torch.isfinite(matrix).all()):
                raise ValueError('invalid historical classifier weights')
            result[category].append(matrix.detach().to('cpu', dtype=torch.float32).clone())
    return result


class LwFResumableTrainer(ResumableCategoryTrainer):
    """Same stage sampling/optimizer as the main trainer, with old-logit KD."""

    def __init__(self, model, stage, loaders, config, log_path, *, historical_heads,
                 lwf_weight=1.0, lwf_temperature=2.0, **kwargs):
        self.lwf_weight = float(lwf_weight)
        if not math.isfinite(self.lwf_weight) or self.lwf_weight < 0:
            raise ValueError('lwf_weight must be finite and nonnegative')
        self.lwf_loss = LearningWithoutForgettingLoss(lwf_temperature)
        self.old_heads = _validated_heads(historical_heads, stage.seen_before, model.feature_dim)
        recurring = tuple(sorted(stage.recurring_categories))
        self.teacher = FrozenCategoryTeacher(model, recurring) if recurring and self.lwf_weight else None
        self.teacher_metadata = self.teacher.metadata() if self.teacher is not None else None
        super().__init__(model, stage, loaders, config, log_path, **kwargs)
        self._prepare_old_weights()
        self._write_event(dict(event='lwf_stage_setup', stage_id=stage.stage_id,
                               weight=self.lwf_weight, temperature=self.lwf_loss.temperature,
                               old_identity_logits={c: sum(m.shape[0] for m in matrices)
                                                    for c, matrices in self.old_heads.items()},
                               teacher=self.teacher_metadata))

    def _prepare_old_weights(self):
        self.teacher_old_weights = {c: [matrix.to(self.device) for matrix in matrices]
                                    for c, matrices in self.old_heads.items()}
        self.student_old_weights = {c: [nn.Parameter(matrix.to(self.device).clone())
                                        for matrix in self.old_heads[c]]
                                    for c in sorted(self.stage.recurring_categories)}
        self.optimizer.param_groups[1]['params'].extend(
            weight for c in sorted(self.student_old_weights)
            for weight in self.student_old_weights[c])

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
            ce = F.cross_entropy(logits.float(), targets)
            triplet, distances = self.triplet(features, targets)
            lwf = features.float().sum() * 0.0
            if self.teacher is not None and category in self.student_old_weights:
                with torch.no_grad():
                    teacher_features = self.teacher.encode(images, category)
                losses = [self.lwf_loss(F.linear(features.float(), student_weight),
                                        F.linear(teacher_features.float(), teacher_weight))
                          for student_weight, teacher_weight in zip(
                              self.student_old_weights[category], self.teacher_old_weights[category])]
                lwf = torch.stack(losses).mean()
            total = ce + self.config.lambda_tri * triplet + self.lwf_weight * lwf
            if not torch.isfinite(total):
                raise FloatingPointError('non-finite LwF loss: stage={}, category={}'.format(
                    self.stage.stage_id, category))
            self.scaler.scale(total).backward()
            values[category] = dict(ce=ce.detach().item(), triplet=triplet.detach().item(),
                                    weighted_triplet=(self.config.lambda_tri * triplet.detach()).item(),
                                    lwf=lwf.detach().item(), weighted_lwf=(self.lwf_weight * lwf.detach()).item(),
                                    total=total.detach().item(), images=len(images),
                                    accuracy=(logits.argmax(1) == targets).float().mean().item(), **distances)
        return values

    def state_dict(self):
        if self.teacher is not None:
            self.teacher.assert_unchanged()
        state = super().state_dict()
        state['lwf'] = dict(weight=self.lwf_weight, temperature=self.lwf_loss.temperature,
                            old_heads=copy.deepcopy(self.old_heads), teacher=self.teacher_metadata,
                            student_old_heads={c: [p.detach().cpu().clone() for p in weights]
                                               for c, weights in self.student_old_weights.items()},
                            teacher_bank=None if self.teacher is None else self.teacher.model.export_bank())
        return state

    @classmethod
    def from_state_dict(cls, state, reference, stage, loaders, memory, candidate, device, log_path=None):
        lwf = state.get('lwf')
        if not isinstance(lwf, dict):
            raise ValueError('missing LwF trainer state')
        old_heads = _validated_heads(lwf['old_heads'], stage.seen_before,
                                     next(iter(state['heads'].values())).shape[1])
        # Restore the inherited optimizer with just current heads, then append
        # the historical-head parameters in the original stable order.
        base_state = dict(state)
        base_optimizer = copy.deepcopy(state['optimizer'])
        group = base_optimizer['param_groups'][1]['params']
        current_count = len(stage.categories)
        old_count = sum(len(old_heads[c]) for c in stage.recurring_categories)
        if len(group) != current_count + old_count:
            raise ValueError('LwF optimizer parameter count differs from saved heads')
        for parameter_id in group[current_count:]:
            base_optimizer['state'].pop(parameter_id, None)
        base_optimizer['param_groups'][1]['params'] = group[:current_count]
        base_state['optimizer'] = base_optimizer
        self = super().from_state_dict(base_state, reference, stage, loaders, memory, candidate, device, log_path)
        self.lwf_weight = float(lwf['weight'])
        if not math.isfinite(self.lwf_weight) or self.lwf_weight < 0:
            raise ValueError('invalid saved LwF weight')
        self.lwf_loss = LearningWithoutForgettingLoss(lwf['temperature'])
        self.old_heads = _validated_heads(lwf['old_heads'], stage.seen_before, self.model.feature_dim)
        self._prepare_old_weights()
        saved_student = _validated_heads(lwf['student_old_heads'], stage.recurring_categories, self.model.feature_dim)
        if any(len(saved_student[c]) != len(self.student_old_weights[c]) for c in saved_student):
            raise ValueError('LwF student/teacher head history differs')
        for category, matrices in saved_student.items():
            for parameter, matrix in zip(self.student_old_weights[category], matrices):
                parameter.data.copy_(matrix.to(self.device))
        self.optimizer.load_state_dict(state['optimizer'])
        self.teacher_metadata = lwf['teacher']
        self.teacher = None
        recurring = tuple(sorted(stage.recurring_categories))
        expected_teacher = bool(recurring and self.lwf_weight)
        if expected_teacher != (lwf['teacher_bank'] is not None) or expected_teacher != (self.teacher_metadata is not None):
            raise ValueError('missing or unexpected LwF teacher')
        if expected_teacher:
            teacher = FrozenCategoryTeacher.__new__(FrozenCategoryTeacher)
            teacher.model = model_from_bank(reference, lwf['teacher_bank'], device)
            teacher.model.set_trainable_categories([])
            torch.nn.Module.train(teacher.model, False)
            teacher.categories = recurring
            teacher.signature = self.teacher_metadata['state_sha256']
            teacher.reference_signature = self.teacher_metadata['reference_signature']
            teacher.adapter_signatures = dict(self.teacher_metadata['adapter_sha256'])
            if set(teacher.model.categories) != set(stage.seen_before) or teacher.metadata() != self.teacher_metadata:
                raise ValueError('LwF teacher does not match saved stage-start state')
            teacher.assert_unchanged()
            self.teacher = teacher
        return self

    def finish_stage(self):
        if self.teacher is not None:
            self.teacher.assert_unchanged()
        current = {c: self.model.temporary_heads[self.model.category_key(c)].weight.detach().cpu().clone()
                   for c in self.categories}
        result = super().finish_stage()
        self.historical_heads = copy.deepcopy(self.old_heads)
        for category, weights in self.student_old_weights.items():
            self.historical_heads[category] = [p.detach().cpu().clone() for p in weights]
        for category, matrix in current.items():
            self.historical_heads.setdefault(category, []).append(matrix)
        result['lwf'] = dict(weight=self.lwf_weight, temperature=self.lwf_loss.temperature,
                              old_identity_logits={c: sum(m.shape[0] for m in matrices)
                                                   for c, matrices in self.old_heads.items()},
                              teacher=self.teacher_metadata)
        self.teacher = None
        self.teacher_old_weights = {}
        self.student_old_weights = {}
        return result
