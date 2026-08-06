from __future__ import absolute_import, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from reid.loss.triplet_loss_transreid import TripletLoss
from reid.loss.scsd import (
    DISTILLATION_MODES,
    SCSDLoss,
    global_relation_distillation,
)
from reid.models.wrapper import CLIP_Backbone


class Stage2Trainer:
    """Train the global CLIP-ReID continual baseline.

    L_total = L_CE + L_Triplet + lambda_global * L_global
              + lambda_scsd * L_SCSD.

    ``L_global`` uses either the original text-only identity Prompt or a
    fixed cross-modal identity anchor supplied for the current domain.
    """

    def __init__(
        self,
        cfg,
        model: CLIP_Backbone,
        num_classes,
        global_loss_weight=1.0,
        anchor_temperature=0.07,
        identity_anchors=None,
        visual_anchor_mode='text',
        teacher_model=None,
        attr_distill_mode='none',
        scsd_weight=20.0,
        scsd_global_weight=20.0,
        classifier_scope='current',
        old_attribute_text_features=None,
        current_attribute_text_features=None,
        scsd_semantic_threshold=0.70,
        scsd_relation_temperature=0.07,
        scsd_kl_direction='old_to_new',
        scsd_mask_diagonal=True,
    ):
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.global_loss_weight = float(global_loss_weight)
        self.anchor_temperature = float(anchor_temperature)
        if (
            not math.isfinite(self.anchor_temperature)
            or self.anchor_temperature <= 0.0
        ):
            raise ValueError(
                'anchor_temperature must be finite and positive'
            )
        self.visual_anchor_mode = str(visual_anchor_mode)
        self.teacher_model = teacher_model
        self.attr_distill_mode = str(attr_distill_mode)
        self.scsd_weight = float(scsd_weight)
        self.scsd_global_weight = float(scsd_global_weight)
        self.classifier_scope = str(classifier_scope)
        if self.classifier_scope not in ('current', 'all'):
            raise ValueError(
                'classifier scope must be "current" or "all"'
            )
        if self.attr_distill_mode not in DISTILLATION_MODES:
            raise ValueError(
                'Unsupported attribute distillation mode: {}'.format(
                    self.attr_distill_mode
                )
            )
        self.scsd_active = (
            self.attr_distill_mode != 'none'
            and self.teacher_model is not None
        )
        if self.scsd_active and (
            old_attribute_text_features is None
            or current_attribute_text_features is None
        ):
            raise ValueError(
                'SCSD needs old and current attribute text features'
            )
        self.old_attribute_text_features = old_attribute_text_features
        self.current_attribute_text_features = (
            current_attribute_text_features
        )
        self.scsd_loss = SCSDLoss(
            mode=self.attr_distill_mode,
            semantic_threshold=scsd_semantic_threshold,
            relation_temperature=scsd_relation_temperature,
            kl_direction=scsd_kl_direction,
            mask_diagonal=scsd_mask_diagonal,
        )
        if self.visual_anchor_mode not in (
            'text',
            'prototype',
            'centered',
            'ocia',
        ):
            raise ValueError(
                'Unsupported visual anchor mode: {}'.format(
                    self.visual_anchor_mode
                )
            )
        if (
            self.visual_anchor_mode != 'text'
            and identity_anchors is None
        ):
            raise ValueError(
                'identity_anchors are required for visual anchor mode {!r}'
                .format(self.visual_anchor_mode)
            )
        self.identity_anchors = None
        if identity_anchors is not None:
            if identity_anchors.ndim != 2:
                raise ValueError(
                    'identity_anchors must be a 2-D tensor, got {}'.format(
                        tuple(identity_anchors.shape)
                    )
                )
            if not torch.isfinite(identity_anchors).all():
                raise FloatingPointError(
                    'Non-finite values found in identity anchors'
                )
            self.identity_anchors = F.normalize(
                identity_anchors.detach(),
                dim=1,
            )
        margin = None if cfg.MODEL.NO_MARGIN else cfg.SOLVER.MARGIN
        self.triplet_loss = TripletLoss(margin)

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
        base = (
            self.model.module.base
            if hasattr(self.model, 'module')
            else self.model.base
        )

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
        for iteration in tqdm(range(train_iters), desc='Training'):
            images, _, targets, _, _ = clip_stage2_loader.next()
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True) + add_num

            if self.scsd_active:
                (
                    cls_score,
                    image_features,
                    current_attribute_features,
                ) = self.model(
                    images,
                    targets,
                    get_attributes=True,
                )
            else:
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
            loss_triplet = self.triplet_loss(
                image_features,
                targets,
            )[0]

            with torch.no_grad():
                unique_labels, inverse_indices = torch.unique(
                    targets,
                    sorted=True,
                    return_inverse=True,
                )
                if self.identity_anchors is None:
                    text_features = base(
                        label=unique_labels,
                        get_text=True,
                    )
                    text_features = F.normalize(text_features, dim=1)
                else:
                    local_labels = unique_labels - int(add_num)
                    if (
                        int(local_labels.min().item()) < 0
                        or int(local_labels.max().item())
                        >= self.identity_anchors.size(0)
                    ):
                        raise ValueError(
                            'Current-domain label rows {} are outside the '
                            'fixed anchor bank with {} rows'.format(
                                local_labels.tolist(),
                                self.identity_anchors.size(0),
                            )
                        )
                    text_features = self.identity_anchors[
                        local_labels
                    ].to(
                        device=image_features.device,
                        dtype=image_features.dtype,
                    )

            image_features_norm = F.normalize(image_features, dim=1)
            logits = (
                image_features_norm @ text_features.t()
            ) / self.anchor_temperature
            loss_global = F.cross_entropy(logits, inverse_indices)

            loss_scsd = image_features.sum() * 0.0
            loss_global_relation = image_features.sum() * 0.0
            scsd_diagnostics = {
                'effective_attributes': loss_scsd.detach(),
                'mean_confidence': loss_scsd.detach(),
                'mean_similarity': loss_scsd.detach(),
            }
            if self.scsd_active:
                with torch.no_grad():
                    (
                        old_global_features,
                        old_attribute_features,
                    ) = self.teacher_model(
                        images,
                        get_image=True,
                        get_attributes=True,
                        attributes_only=True,
                    )
                loss_scsd, scsd_diagnostics = self.scsd_loss(
                    old_attribute_features,
                    current_attribute_features,
                    self.old_attribute_text_features,
                    self.current_attribute_text_features,
                )
                loss_global_relation = global_relation_distillation(
                    old_global_features,
                    image_features,
                    temperature=self.scsd_loss.relation_temperature,
                    kl_direction=self.scsd_loss.kl_direction,
                    mask_diagonal=self.scsd_loss.mask_diagonal,
                )

            loss_total = (
                loss_ce
                + loss_triplet
                + self.global_loss_weight * loss_global
                + self.scsd_weight * loss_scsd
                + self.scsd_global_weight * loss_global_relation
            )
            if not torch.isfinite(loss_total):
                raise FloatingPointError(
                    'Non-finite Stage2 loss: CE={}, Triplet={}, Global={}, '
                    'SCSD={}, GlobalRelation={}'
                    .format(
                        loss_ce.detach().item(),
                        loss_triplet.detach().item(),
                        loss_global.detach().item(),
                        loss_scsd.detach().item(),
                        loss_global_relation.detach().item(),
                    )
                )

            optimizer_stage2.zero_grad()
            loss_total.backward()
            optimizer_stage2.step()

            last_metrics = {
                'loss_ce': loss_ce.detach().item(),
                'loss_triplet': loss_triplet.detach().item(),
                'loss_global': loss_global.detach().item(),
                'loss_scsd': loss_scsd.detach().item(),
                'loss_global_relation': (
                    loss_global_relation.detach().item()
                ),
                'weighted_scsd': (
                    self.scsd_weight * loss_scsd.detach().item()
                ),
                'weighted_global_relation': (
                    self.scsd_global_weight
                    * loss_global_relation.detach().item()
                ),
                'scsd_effective_attributes': scsd_diagnostics[
                    'effective_attributes'
                ].item(),
                'scsd_mean_confidence': scsd_diagnostics[
                    'mean_confidence'
                ].item(),
                'scsd_mean_similarity': scsd_diagnostics[
                    'mean_similarity'
                ].item(),
                'loss_total': loss_total.detach().item(),
            }
            if iteration % 30 == 0 or iteration == train_iters - 1:
                print('\n[stage2] loss_ce: {}'.format(loss_ce.detach()))
                print(
                    '[stage2] loss_triplet: {}'.format(
                        loss_triplet.detach()
                    )
                )
                print(
                    '[stage2] loss_global: {}'.format(
                        loss_global.detach()
                    )
                )
                print('[stage2] loss_scsd: {}'.format(loss_scsd.detach()))
                print(
                    '[stage2] loss_global_relation: {}'.format(
                        loss_global_relation.detach()
                    )
                )
                if self.scsd_active:
                    print(
                        '[stage2] weighted preservation: attr={:.6f}, '
                        'global={:.6f}'.format(
                            self.scsd_weight * loss_scsd.detach().item(),
                            self.scsd_global_weight
                            * loss_global_relation.detach().item(),
                        )
                    )
                if self.scsd_active:
                    print(
                        '[stage2] SCSD effective attributes: {}, mean '
                        'confidence: {:.4f}'.format(
                            int(
                                scsd_diagnostics[
                                    'effective_attributes'
                                ].item()
                            ),
                            scsd_diagnostics[
                                'mean_confidence'
                            ].item(),
                        )
                    )
                print(
                    '[stage2] loss_total: {}'.format(
                        loss_total.detach()
                    )
                )

        return last_metrics
