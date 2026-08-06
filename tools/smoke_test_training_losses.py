"""Run one real CLIP-ReID optimization step through every retained loss.

The test intentionally uses a tiny identity subset, but it loads real images
and the real pretrained CLIP ViT-B/16. It checks Stage 1 prompt training,
OCIA category-subspace construction, classifier initialization, and Stage 2
updates with identity classification, Triplet, and global image-text losses.
With SCSD enabled, the default two-domain run also checks frozen-teacher
attribute distillation. Pass ``--all-domains`` to exercise every continual
classifier/prompt expansion in sequence.
"""

from __future__ import absolute_import, print_function

import argparse
import gc
import os
import sys
import tempfile
from types import SimpleNamespace

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import cfg as default_cfg
from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from lreid_dataset.datasets.manifest_reid import load_domain_config
from reid.models.CLIP_ReID.loss.supcontrast import SupConLoss
from reid.models.layers import DataParallel
from reid.models.wrapper import make_model
from reid.trainer_stage2 import Stage2Trainer
from reid.utils.feature_tools import (
    category_semantic_subspace,
    initial_classifier,
    visual_identity_statistics,
)
from train_stage1 import (
    _category_generic_prompts,
    _make_prompt_optimizer,
    _set_stage1_mode,
)
from train_stage2 import (
    _add_and_activate_domain_adapter,
    _attribute_entries,
    _base_model,
    _configure_attribute_pooling,
    _freeze_teacher,
    _object_noun,
    _set_domain_adapter_stage2_mode,
    _set_stage2_mode,
    build_domain_identity_anchors,
    expand_model_for_domain,
    make_optimizer_stage2,
    prompt_initialize,
)


def _assert_finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError('{} is not finite'.format(name))


def _make_args(cli_args, domain_config):
    input_size = domain_config.get('input_size') or [224, 224]
    return SimpleNamespace(
        data_dir=os.path.abspath(cli_args.data_dir),
        height=int(input_size[0]),
        width=int(input_size[1]),
        resize_mode=domain_config.get('resize_mode') or 'pad',
        batch_size=cli_args.batch_size,
        workers=cli_args.workers,
        num_instances=cli_args.num_instances,
        max_train_ids=cli_args.max_train_ids,
        stage1_prompts_out_dir=None,
        continual_update_mode=cli_args.continual_update_mode,
        adapter_last_blocks=cli_args.adapter_last_blocks,
        adapter_bottleneck_dim=cli_args.adapter_bottleneck_dim,
        adapter_scale=cli_args.adapter_scale,
        adapter_lr=cli_args.adapter_lr,
    )


def _build_tiny_domains(cli_args):
    domain_config = load_domain_config(cli_args.domain_config)
    configured_domains = domain_config['train_domains']
    if not cli_args.all_domains:
        domain_count = 2 if cli_args.attr_distill_mode != 'none' else 1
        configured_domains = configured_domains[:domain_count]
    domains = []
    for configured_domain in configured_domains:
        domain = dict(configured_domain)
        domain['max_train_ids'] = cli_args.max_train_ids
        domains.append(domain)
    args = _make_args(cli_args, domain_config)
    train_sets, _ = build_data_loaders(args, domains, [])
    return args, domains, train_sets


def _run_stage1_step(args, domain, train_info):
    _, num_classes, prompt_loader, _, _, init_loader, name = train_info
    model = make_model(
        num_class=num_classes,
        camera_num=0,
        view_num=0,
        input_size=(args.height, args.width),
        object_noun=_object_noun(domain),
    )
    model = DataParallel(model.cuda())
    base = _set_stage1_mode(model)
    optimizer = _make_prompt_optimizer(base)
    loss_function = SupConLoss(device='cuda')

    model.train()
    prompt_loader.new_epoch()
    images, _, targets, _, _ = prompt_loader.next()
    images = images.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    optimizer.zero_grad()
    with torch.no_grad():
        image_features = base(x=images, get_image=True)
    text_features = base(label=targets, get_text=True)
    loss_i2t = loss_function(
        image_features,
        text_features,
        targets,
        targets,
    )
    loss_t2i = loss_function(
        text_features,
        image_features,
        targets,
        targets,
    )
    loss = loss_i2t + loss_t2i
    _assert_finite('Stage1 prompt loss', loss)
    loss.backward()

    prompt_gradients = [
        parameter.grad
        for parameter in base.prompt_learner.parameters()
        if parameter.requires_grad
    ]
    if not prompt_gradients or not all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in prompt_gradients
    ):
        raise RuntimeError('Stage1 prompt gradients are missing or invalid')
    optimizer.step()
    visual_prototypes, category_center = visual_identity_statistics(
        model,
        init_loader,
        num_classes,
    )
    category_generic_prompts = _category_generic_prompts(domain)
    with torch.no_grad():
        category_text_features = base.encode_fixed_texts(
            category_generic_prompts
        )
        category_subspace_basis = category_semantic_subspace(
            category_text_features
        ).detach().cpu()

    prompt_state = {
        key: value.detach().cpu().clone()
        for key, value in base.prompt_learner.state_dict().items()
    }
    metrics = {
        'loss_i2t': loss_i2t.detach().item(),
        'loss_t2i': loss_t2i.detach().item(),
        'loss_total': loss.detach().item(),
    }
    print('[smoke][{}][stage1] {}'.format(name, metrics))

    del images, targets, image_features, text_features, loss
    del optimizer, base, model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        'state_dict': prompt_state,
        'num_class': num_classes,
        'dataset_name': name,
        'object_noun': _object_noun(domain),
        'visual_prototypes': visual_prototypes,
        'category_center': category_center,
        'ocia_schema_version': 1,
        'category_semantic_prompts': category_generic_prompts,
        'category_subspace_basis': category_subspace_basis,
        'category_subspace_rank': category_subspace_basis.size(1),
    }


def _check_stage2_gradients(base):
    image_gradients = [
        parameter.grad
        for parameter in base.image_encoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not image_gradients or not any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in image_gradients
    ):
        raise RuntimeError('Stage2 image encoder received no gradient')

    frozen_modules = (
        base.prompt_learner,
        base.text_encoder,
        base.token_embedding,
    )
    if any(
        parameter.requires_grad
        for module in frozen_modules
        for parameter in module.parameters()
    ):
        raise RuntimeError('Stage2 text tower or identity prompts are not frozen')


def _run_stage2_sequence(
    args,
    domains,
    train_infos,
    prompt_checkpoints,
    config_file,
    visual_anchor_mode,
    anchor_residual_weight,
    attr_distill_mode,
    scsd_weight,
    scsd_global_weight,
    scsd_semantic_threshold,
    scsd_relation_temperature,
):
    _, first_classes, _, _, _, _, _ = train_infos[0]
    model = make_model(
        num_class=first_classes,
        camera_num=0,
        view_num=0,
        input_size=(args.height, args.width),
        object_noun=_object_noun(domains[0]),
    )
    model = DataParallel(model.cuda())
    base = _base_model(model)
    prompt_initialize(
        base,
        prompt_checkpoints[0]['state_dict'],
        object_noun=_object_noun(domains[0]),
    )
    _set_stage2_mode(model, trainable_vision_blocks=12)

    loss_cfg = default_cfg.clone()
    loss_cfg.merge_from_file(config_file)
    add_num = 0
    for domain_index, (domain, train_info) in enumerate(
        zip(domains, train_infos)
    ):
        (
            _,
            num_classes,
            _,
            stage2_loader,
            _,
            init_loader,
            name,
        ) = train_info
        teacher_model = None
        old_entries = None
        if domain_index and attr_distill_mode != 'none':
            teacher_model = _freeze_teacher(model)
            old_entries = _attribute_entries(domains[domain_index - 1])
        if domain_index:
            total_classes = expand_model_for_domain(
                model,
                args,
                name,
                domain,
                add_num,
                num_classes,
            )
        else:
            total_classes = num_classes
        base = _base_model(model)

        if args.continual_update_mode == 'domain_adapter':
            _add_and_activate_domain_adapter(model, name, args)

        class_centers = initial_classifier(model, init_loader)
        expected_center_shape = (num_classes, base.classifier.in_features)
        if tuple(class_centers.shape) != expected_center_shape:
            raise RuntimeError(
                'Classifier center shape {} does not match expected {}'
                .format(tuple(class_centers.shape), expected_center_shape)
            )
        base.classifier.weight.data[
            add_num:add_num + num_classes
        ].copy_(class_centers)

        if args.continual_update_mode == 'domain_adapter' and domain_index:
            _set_domain_adapter_stage2_mode(model, name)
        else:
            _set_stage2_mode(
                model,
                trainable_vision_blocks=12 if domain_index == 0 else 2,
            )
        old_attribute_text_features = None
        current_attribute_text_features = None
        if attr_distill_mode != 'none':
            current_entries = _attribute_entries(domain)
            current_attribute_text_features = _configure_attribute_pooling(
                model,
                current_entries,
                temperature=0.07,
            )
            if teacher_model is not None:
                old_attribute_text_features = _configure_attribute_pooling(
                    teacher_model,
                    old_entries,
                    temperature=0.07,
                )
        identity_anchors = None
        if visual_anchor_mode != 'text':
            identity_anchors = build_domain_identity_anchors(
                base,
                domain['prompt_checkpoint'],
                start_idx=add_num,
                num_classes=num_classes,
                mode=visual_anchor_mode,
                residual_weight=anchor_residual_weight,
            )
        optimizer = make_optimizer_stage2(
            base,
            base_lr=5e-6,
            adapter_lr=args.adapter_lr,
        )
        trainer = Stage2Trainer(
            cfg=loss_cfg,
            model=model,
            num_classes=total_classes,
            global_loss_weight=1.0,
            identity_anchors=identity_anchors,
            visual_anchor_mode=visual_anchor_mode,
            teacher_model=teacher_model,
            attr_distill_mode=attr_distill_mode,
            scsd_weight=scsd_weight,
            scsd_global_weight=scsd_global_weight,
            classifier_scope='current',
            old_attribute_text_features=old_attribute_text_features,
            current_attribute_text_features=(
                current_attribute_text_features
            ),
            scsd_semantic_threshold=scsd_semantic_threshold,
            scsd_relation_temperature=scsd_relation_temperature,
            scsd_mask_diagonal=True,
        )
        stage2_loader.new_epoch()
        metrics = trainer.train(
            stage2_loader,
            optimizer,
            train_iters=2 if domain_index else 1,
            add_num=add_num,
        )
        if not metrics or not all(
            torch.isfinite(torch.tensor(value)) for value in metrics.values()
        ):
            raise RuntimeError('Stage2 did not return finite loss metrics')
        _check_stage2_gradients(base)
        print(
            '[smoke][{}][stage2][classes={}] {}'.format(
                name,
                total_classes,
                metrics,
            )
        )
        del trainer, optimizer
        if teacher_model is not None:
            del teacher_model
        gc.collect()
        torch.cuda.empty_cache()
        add_num = total_classes

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print('[smoke] peak CUDA memory: {:.2f} GiB'.format(peak_memory))


def main():
    parser = argparse.ArgumentParser(
        description='Smoke-test the retained CLIP-ReID training losses'
    )
    parser.add_argument('--data-dir', default='data')
    parser.add_argument(
        '--domain-config',
        default='config/cross_category_five_domains.json',
    )
    parser.add_argument('--config-file', default='config/base.yml')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-instances', type=int, default=2)
    parser.add_argument('--max-train-ids', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument(
        '--continual-update-mode',
        choices=['partial_blocks', 'domain_adapter'],
        default='partial_blocks',
    )
    parser.add_argument('--adapter-last-blocks', type=int, default=4)
    parser.add_argument('--adapter-bottleneck-dim', type=int, default=64)
    parser.add_argument('--adapter-scale', type=float, default=1.0)
    parser.add_argument('--adapter-lr', type=float, default=3e-4)
    parser.add_argument(
        '--visual-anchor-mode',
        choices=['text', 'prototype', 'centered', 'ocia'],
        default='ocia',
    )
    parser.add_argument(
        '--anchor-residual-weight',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--attr-distill-mode',
        choices=[
            'none',
            'index',
            'semantic-hard',
            'semantic-soft',
            'scsd',
        ],
        default='scsd',
    )
    parser.add_argument('--scsd-weight', type=float, default=20.0)
    parser.add_argument('--scsd-global-weight', type=float, default=20.0)
    parser.add_argument(
        '--scsd-semantic-threshold',
        type=float,
        default=0.70,
    )
    parser.add_argument(
        '--scsd-relation-temperature',
        type=float,
        default=0.07,
    )
    parser.add_argument(
        '--all-domains',
        action='store_true',
        help='test every configured domain and continual expansion',
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('This CLIP-ReID smoke test requires CUDA')
    if args.batch_size % args.num_instances:
        raise ValueError('batch size must be divisible by num instances')
    if args.batch_size // args.num_instances < 2:
        raise ValueError('Triplet smoke test needs at least two identities')
    if args.max_train_ids < args.batch_size // args.num_instances:
        raise ValueError('max train IDs is too small for one PK batch')

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.cuda.reset_peak_memory_stats()
    loader_args, domains, train_infos = _build_tiny_domains(args)
    prompt_checkpoints = [
        _run_stage1_step(loader_args, domain, train_info)
        for domain, train_info in zip(domains, train_infos)
    ]
    with tempfile.TemporaryDirectory(prefix='vladr_prompt_smoke_') as temp_dir:
        loader_args.stage1_prompts_out_dir = temp_dir
        for domain, train_info, prompt_checkpoint in zip(
            domains,
            train_infos,
            prompt_checkpoints,
        ):
            name = train_info[-1]
            prompt_path = os.path.join(
                temp_dir,
                '{}_clipreid_prompt.pth'.format(name),
            )
            torch.save(prompt_checkpoint, prompt_path)
            domain['prompt_checkpoint'] = prompt_path
        _run_stage2_sequence(
            loader_args,
            domains,
            train_infos,
            prompt_checkpoints,
            os.path.abspath(args.config_file),
            args.visual_anchor_mode,
            args.anchor_residual_weight,
            args.attr_distill_mode,
            args.scsd_weight,
            args.scsd_global_weight,
            args.scsd_semantic_threshold,
            args.scsd_relation_temperature,
        )
    print('All retained training losses passed the real-model smoke test.')


if __name__ == '__main__':
    main()
