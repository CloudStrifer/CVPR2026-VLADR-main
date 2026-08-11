from __future__ import absolute_import, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from reid.loss.lwf import LearningWithoutForgettingLoss
from reid.trainer_stage2 import Stage2Trainer


def _base_model(model):
    wrapped = model.module if hasattr(model, 'module') else model
    return wrapped.base


def previous_classifier_logits(model, images):
    """Return old-class logits without changing the original model API."""

    base = _base_model(model)
    _, image_features, _ = base.image_encoder(images, None)
    feature = image_features[:, 0]
    if getattr(base, 'eval_descriptor_mode', 'raw') == 'bn':
        feature = base.bottleneck(feature)
    return base.classifier(feature)


class LwFTrainer(Stage2Trainer):
    """CLIP-ReID trainer with classic Learning without Forgetting."""

    def __init__(
        self,
        cfg,
        model,
        num_classes,
        teacher_model=None,
        global_loss_weight=1.0,
        anchor_temperature=0.07,
        lwf_weight=1.0,
        lwf_temperature=2.0,
        classifier_scope='current',
    ):
        super().__init__(
            cfg=cfg,
            model=model,
            num_classes=num_classes,
            global_loss_weight=global_loss_weight,
            anchor_temperature=anchor_temperature,
            identity_anchors=None,
            visual_anchor_mode='text',
            teacher_model=teacher_model,
            attr_distill_mode='none',
            classifier_scope=classifier_scope,
        )
        self.lwf_weight = float(lwf_weight)
        if not math.isfinite(self.lwf_weight) or self.lwf_weight < 0.0:
            raise ValueError('LwF weight must be finite and non-negative')
        self.lwf_loss = LearningWithoutForgettingLoss(
            temperature=lwf_temperature,
        )

    def train(
        self,
        clip_stage2_loader,
        optimizer_stage2,
        train_iters,
        add_num=0,
    ):
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()
        base = _base_model(self.model)

        for module in base.modules():
            if isinstance(module, nn.BatchNorm2d):
                affine_is_frozen = (
                    module.weight is not None
                    and module.bias is not None
                    and not module.weight.requires_grad
                    and not module.bias.requires_grad
                )
                if affine_is_frozen:
                    module.eval()

        last_metrics = None
        for iteration in tqdm(range(train_iters), desc='Training-LwF'):
            images, _, targets, _, _ = clip_stage2_loader.next()
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True) + int(add_num)

            cls_score, image_features = self.model(images, targets)
            if self.classifier_scope == 'current':
                current_scores = cls_score[:, int(add_num):self.num_classes]
                current_targets = targets - int(add_num)
                if current_scores.size(1) <= 0:
                    raise ValueError('current-domain classifier slice is empty')
                loss_ce = F.cross_entropy(
                    current_scores,
                    current_targets,
                )
            else:
                loss_ce = F.cross_entropy(cls_score, targets)

            loss_triplet = self.triplet_loss(image_features, targets)[0]

            with torch.no_grad():
                unique_labels, inverse_indices = torch.unique(
                    targets,
                    sorted=True,
                    return_inverse=True,
                )
                text_features = base(label=unique_labels, get_text=True)
                text_features = F.normalize(text_features, dim=1)

            image_features_norm = F.normalize(image_features, dim=1)
            anchor_logits = (
                image_features_norm @ text_features.t()
            ) / self.anchor_temperature
            loss_global = F.cross_entropy(anchor_logits, inverse_indices)

            loss_lwf = cls_score.sum() * 0.0
            if self.teacher_model is not None:
                old_classes = int(add_num)
                if old_classes <= 0:
                    raise ValueError(
                        'LwF teacher is present but there are no old classes'
                    )
                with torch.no_grad():
                    teacher_logits = previous_classifier_logits(
                        self.teacher_model,
                        images,
                    )
                if teacher_logits.size(1) != old_classes:
                    raise ValueError(
                        'LwF teacher has {} classes, expected {}'
                        .format(teacher_logits.size(1), old_classes)
                    )
                loss_lwf = self.lwf_loss(
                    cls_score[:, :old_classes],
                    teacher_logits,
                )

            loss_total = (
                loss_ce
                + loss_triplet
                + self.global_loss_weight * loss_global
                + self.lwf_weight * loss_lwf
            )
            if not torch.isfinite(loss_total):
                raise FloatingPointError(
                    'Non-finite LwF loss: CE={}, Triplet={}, Global={}, '
                    'LwF={}'.format(
                        loss_ce.detach().item(),
                        loss_triplet.detach().item(),
                        loss_global.detach().item(),
                        loss_lwf.detach().item(),
                    )
                )

            optimizer_stage2.zero_grad()
            loss_total.backward()
            optimizer_stage2.step()

            last_metrics = {
                'loss_ce': loss_ce.detach().item(),
                'loss_triplet': loss_triplet.detach().item(),
                'loss_global': loss_global.detach().item(),
                'loss_lwf': loss_lwf.detach().item(),
                'weighted_lwf': self.lwf_weight * loss_lwf.detach().item(),
                'loss_total': loss_total.detach().item(),
            }
            if iteration % 30 == 0 or iteration == train_iters - 1:
                print('\n[lwf] loss_ce: {}'.format(loss_ce.detach()))
                print('[lwf] loss_triplet: {}'.format(loss_triplet.detach()))
                print('[lwf] loss_global: {}'.format(loss_global.detach()))
                print('[lwf] loss_distill: {}'.format(loss_lwf.detach()))
                print('[lwf] loss_total: {}'.format(loss_total.detach()))

        return last_metrics
