from __future__ import absolute_import, print_function

import argparse
import copy
import datetime
import gc
import os.path as osp
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import cfg
from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from lreid_dataset.datasets.manifest_reid import load_domain_config
from reid.evaluation.fast_test import fast_test_p_s
from reid.loss.scsd import DISTILLATION_MODES, semantic_compatibility
from reid.models.CLIP_ReID.model.clip import clip
from reid.models.layers import DataParallel
from reid.models.wrapper import make_model
from reid.trainer_stage2 import Stage2Trainer
from reid.utils.feature_tools import (
    build_cross_modal_identity_anchors,
    initial_classifier,
)
from reid.utils.logging import Logger
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.serialization import (
    copy_state_dict,
    load_checkpoint,
    save_checkpoint,
)
from tools.Logger_results import Logger_res


warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def _legacy_domain_sets(setting):
    orders = {
        1: ['market1501', 'cuhk_sysu', 'lpw', 'msmt17', 'cuhk03'],
        2: ['lpw', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03'],
        51: ['msmt17', 'cuhk_sysu', 'lpw', 'market1501', 'cuhk03'],
        52: ['lpw', 'market1501', 'cuhk03', 'msmt17', 'cuhk_sysu'],
        53: ['cuhk_sysu', 'lpw', 'cuhk03', 'msmt17', 'market1501'],
        54: ['cuhk03', 'msmt17', 'lpw', 'market1501', 'cuhk_sysu'],
        55: ['market1501', 'msmt17', 'lpw', 'cuhk_sysu', 'cuhk03'],
    }
    if setting not in orders:
        raise ValueError('unsupported legacy training order: {}'.format(setting))
    training_set = orders[setting]
    all_sets = [
        'market1501',
        'lpw',
        'msmt17',
        'cuhk_sysu',
        'cuhk03',
        'cuhk01',
        'cuhk02',
        'grid',
        'sense',
        'viper',
        'ilids',
        'prid',
    ]
    return training_set, [
        name for name in all_sets if name not in training_set
    ]


def _resolve_domain_sets(args):
    if not args.domain_config:
        training_set, testing_set = _legacy_domain_sets(args.setting)
        return training_set, testing_set, {}

    domain_config = load_domain_config(args.domain_config)
    if domain_config.get('input_size'):
        args.height, args.width = map(int, domain_config['input_size'])
    if domain_config.get('resize_mode'):
        args.resize_mode = str(domain_config['resize_mode'])

    training_set = domain_config['train_domains']
    testing_set = domain_config['test_domains']
    specs = {
        spec['name']: spec
        for spec in training_set + testing_set
    }
    return training_set, testing_set, specs


def _timestamp():
    return datetime.datetime.now().strftime('%Y-%m%d-%H%M')


def _domain_name(domain):
    return domain['name'] if isinstance(domain, dict) else domain


def _object_noun(spec, category_prompt=True):
    if not category_prompt:
        return 'object'
    if spec is None:
        return 'person'
    return spec.get('object_noun', spec['category'])


def _attribute_entries(spec):
    if spec is None or not spec.get('attributes'):
        raise ValueError(
            'SCSD requires a non-empty attributes list for every training '
            'domain in --domain-config'
        )
    return list(spec['attributes'])


def _freeze_teacher(model):
    teacher = copy.deepcopy(model)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return teacher


def _configure_attribute_pooling(model, entries, temperature):
    base = _base_model(model)
    prompts = [entry['prompt'] for entry in entries]
    with torch.no_grad():
        text_features = base.encode_fixed_texts(prompts).detach()
    base.set_attribute_text_features(text_features)
    base.set_attribute_pooling_temperature(temperature)
    return text_features


def _validate_attribute_model_support(model):
    base = _base_model(model)
    required = (
        'set_attribute_text_features',
        'set_attribute_pooling_temperature',
    )
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise RuntimeError(
            'SCSD model files are out of sync; the active CLIP model is '
            'missing {}. Update reid/models/CLIP_ReID/model/'
            'make_model_clipreid.py and reid/models/attribute_pooling.py '
            'from the same code version as train_stage2.py and wrapper.py.'
            .format(', '.join(missing))
        )


def _print_attribute_matches(
    old_entries,
    current_entries,
    old_text_features,
    current_text_features,
    threshold,
    mode,
):
    if mode == 'index':
        print('[DAlign] Fixed index attribute matches:')
        for index in range(min(len(old_entries), len(current_entries))):
            print(
                '  {} <- {}'.format(
                    current_entries[index]['name'],
                    old_entries[index]['name'],
                )
            )
        return
    similarity = semantic_compatibility(
        old_text_features,
        current_text_features,
    ).detach().cpu()
    print('[SCSD] Best frozen-CLIP attribute matches:')
    for current_index, current_entry in enumerate(current_entries):
        score, old_index = similarity[:, current_index].max(dim=0)
        print(
            '  {} <- {}: {:.4f} ({})'.format(
                current_entry['name'],
                old_entries[int(old_index.item())]['name'],
                float(score.item()),
                'selected' if float(score.item()) > threshold else 'filtered',
            )
        )


def _prompt_path(args, name, spec):
    source = getattr(args, 'prompt_checkpoint_source', 'config')
    if (
        source == 'config'
        and spec is not None
        and spec.get('prompt_checkpoint')
    ):
        return spec['prompt_checkpoint']
    if source not in ('config', 'output-dir'):
        raise ValueError(
            'Unsupported prompt checkpoint source: {}'.format(source)
        )
    return osp.join(
        args.stage1_prompts_out_dir,
        '{}_clipreid_prompt.pth'.format(name),
    )


def set_seed(seed):
    if seed is None:
        return
    print('setting the seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _base_model(model):
    wrapped = model.module if hasattr(model, 'module') else model
    return wrapped.base


def _load_stage1_checkpoint(path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        return state
    return {'state_dict': state}


def load_prompt(path, expected_object_noun=None):
    checkpoint = _load_stage1_checkpoint(path)
    saved_object_noun = checkpoint.get('object_noun')
    if (
        expected_object_noun is not None
        and saved_object_noun is not None
        and saved_object_noun != expected_object_noun
    ):
        raise ValueError(
            'Stage1 prompt {} was trained with object noun {!r}, but '
            'Stage2 expects {!r}. Rerun both stages with the same category-'
            'prompt setting.'.format(
                path,
                saved_object_noun,
                expected_object_noun,
            )
        )
    return checkpoint['state_dict']


def load_visual_identity_statistics(path, expected_num_classes):
    checkpoint = _load_stage1_checkpoint(path)
    if 'visual_prototypes' not in checkpoint:
        raise ValueError(
            'Stage1 checkpoint {} does not contain visual prototypes. '
            'Rerun train_stage1.py with the current code before using '
            '--visual-anchor-mode prototype, centered, or ocia.'.format(path)
        )

    visual_prototypes = checkpoint['visual_prototypes'].float()
    category_center = checkpoint.get('category_center')
    if category_center is None:
        category_center = visual_prototypes.mean(dim=0, keepdim=True)
    else:
        category_center = category_center.float()

    expected_num_classes = int(expected_num_classes)
    if visual_prototypes.ndim != 2:
        raise ValueError(
            'Visual prototypes in {} must be a 2-D tensor, got {}'.format(
                path,
                tuple(visual_prototypes.shape),
            )
        )
    if visual_prototypes.size(0) != expected_num_classes:
        raise ValueError(
            'Stage1 checkpoint {} contains {} visual identities, expected {}'
            .format(
                path,
                visual_prototypes.size(0),
                expected_num_classes,
            )
        )
    expected_center_shape = (1, visual_prototypes.size(1))
    if tuple(category_center.shape) != expected_center_shape:
        raise ValueError(
            'Category center in {} has shape {}, expected {}'.format(
                path,
                tuple(category_center.shape),
                expected_center_shape,
            )
        )
    if not torch.isfinite(visual_prototypes).all():
        raise FloatingPointError(
            'Non-finite visual prototypes found in {}'.format(path)
        )
    if not torch.isfinite(category_center).all():
        raise FloatingPointError(
            'Non-finite category center found in {}'.format(path)
        )

    recomputed_center = visual_prototypes.mean(dim=0, keepdim=True)
    if not torch.allclose(
        category_center,
        recomputed_center,
        atol=1e-6,
        rtol=1e-5,
    ):
        raise ValueError(
            'Stored category center in {} is inconsistent with its visual '
            'prototypes'.format(path)
        )
    return visual_prototypes, category_center


def load_category_subspace_basis(path, expected_feature_dim):
    checkpoint = _load_stage1_checkpoint(path)
    if 'category_subspace_basis' not in checkpoint:
        raise ValueError(
            'Stage1 checkpoint {} does not contain an OCIA category '
            'subspace. Rerun train_stage1.py with the current code before '
            'using --visual-anchor-mode ocia.'.format(path)
        )

    category_basis = checkpoint['category_subspace_basis'].float()
    if category_basis.ndim != 2:
        raise ValueError(
            'Category subspace in {} must be a 2-D tensor, got {}'.format(
                path,
                tuple(category_basis.shape),
            )
        )
    if category_basis.size(0) != int(expected_feature_dim):
        raise ValueError(
            'Category subspace in {} has feature dimension {}, expected {}'
            .format(
                path,
                category_basis.size(0),
                expected_feature_dim,
            )
        )
    if category_basis.size(1) <= 0:
        raise ValueError(
            'Category subspace in {} must have positive rank'.format(path)
        )
    stored_rank = checkpoint.get('category_subspace_rank')
    if (
        stored_rank is not None
        and int(stored_rank) != category_basis.size(1)
    ):
        raise ValueError(
            'Stored OCIA rank in {} is inconsistent with basis shape {}'
            .format(path, tuple(category_basis.shape))
        )
    if not torch.isfinite(category_basis).all():
        raise FloatingPointError(
            'Non-finite category-subspace values found in {}'.format(path)
        )

    gram = category_basis.t() @ category_basis
    identity = torch.eye(category_basis.size(1), dtype=gram.dtype)
    if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
        raise ValueError(
            'Category subspace in {} is not orthonormal'.format(path)
        )
    return category_basis


def build_domain_adapter_routing_entry(
    base,
    prompt_path,
    expected_num_classes,
):
    """Load one source domain's fixed OCIA statistics for OSAF routing."""

    checkpoint = _load_stage1_checkpoint(prompt_path)
    prompts = checkpoint.get('category_semantic_prompts')
    if (
        not isinstance(prompts, (list, tuple))
        or not prompts
        or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
    ):
        raise ValueError(
            'Stage1 checkpoint {} does not contain valid category semantic '
            'prompts required by --adapter-routing osaf. Rerun Stage1 with '
            'the current OCIA code.'.format(prompt_path)
        )

    visual_prototypes, _ = load_visual_identity_statistics(
        prompt_path,
        expected_num_classes=expected_num_classes,
    )
    category_basis = load_category_subspace_basis(
        prompt_path,
        expected_feature_dim=visual_prototypes.size(1),
    )
    with torch.no_grad():
        semantic_features = base.encode_fixed_texts(prompts).float()
        semantic_key = F.normalize(
            F.normalize(semantic_features, dim=1).mean(
                dim=0,
                keepdim=True,
            ),
            dim=1,
        ).squeeze(0)
    if semantic_key.numel() != visual_prototypes.size(1):
        raise ValueError(
            'OSAF semantic-key dimension {} does not match visual '
            'prototype dimension {} in {}'.format(
                semantic_key.numel(),
                visual_prototypes.size(1),
                prompt_path,
            )
        )

    return {
        'semantic_key': semantic_key.detach().cpu().float(),
        'visual_prototypes': F.normalize(
            visual_prototypes.float(),
            dim=1,
        ).cpu(),
        'category_basis': category_basis.detach().cpu().float(),
    }


def _routing_bank_subset(routing_bank, domain_names):
    if routing_bank is None:
        return None
    return {
        name: {
            key: value.detach().cpu()
            for key, value in routing_bank[name].items()
        }
        for name in domain_names
        if name in routing_bank
    }


def build_domain_identity_anchors(
    base,
    prompt_path,
    start_idx,
    num_classes,
    mode,
    residual_weight=1.0,
):
    """Build one fixed Stage 2 anchor row per current-domain identity."""

    if mode not in ('prototype', 'centered', 'ocia'):
        raise ValueError(
            'Cross-modal anchor construction requires prototype, centered, '
            'or ocia mode, got {!r}'.format(mode)
        )
    visual_prototypes, category_center = (
        load_visual_identity_statistics(
            prompt_path,
            expected_num_classes=num_classes,
        )
    )
    category_basis = None
    if mode == 'ocia':
        category_basis = load_category_subspace_basis(
            prompt_path,
            expected_feature_dim=visual_prototypes.size(1),
        )
    labels = torch.arange(
        start_idx,
        start_idx + num_classes,
        device=base.prompt_learner.cls_ctx.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        text_features = base(
            label=labels,
            get_text=True,
        )
        anchors = build_cross_modal_identity_anchors(
            text_features,
            visual_prototypes=visual_prototypes,
            category_center=category_center,
            category_basis=category_basis,
            mode=mode,
            residual_weight=residual_weight,
        )
    return anchors.detach()


def _prompt_template_buffers(base, object_noun, num_classes):
    prompt_learner = base.prompt_learner
    template = 'A photo of a X X X X {}.'.format(object_noun)
    tokenized = clip.tokenize([template] * num_classes).to(
        prompt_learner.cls_ctx.device
    )
    with torch.no_grad():
        embedding = base.token_embedding(tokenized).type(
            prompt_learner.cls_ctx.dtype
        )
    prefix_length = prompt_learner.token_prefix.size(1)
    suffix_start = prefix_length + prompt_learner.n_cls_ctx
    return (
        tokenized,
        embedding[:, :prefix_length, :],
        embedding[:, suffix_start:, :],
    )


def _expanded_rows(tensor, rows):
    if tensor.size(0) == rows:
        return tensor
    if tensor.size(0) == 1:
        return tensor.expand(rows, *tensor.shape[1:]).clone()
    raise ValueError(
        'cannot expand prompt buffer with {} rows to {}'.format(
            tensor.size(0),
            rows,
        )
    )


def prompt_initialize(base, prompt_state, object_noun='person'):
    prompt_learner = base.prompt_learner
    cls_ctx = prompt_state['cls_ctx'].to(prompt_learner.cls_ctx.device)
    if cls_ctx.shape != prompt_learner.cls_ctx.shape:
        raise ValueError(
            'Stage1 prompt shape {} does not match model shape {}'.format(
                tuple(cls_ctx.shape),
                tuple(prompt_learner.cls_ctx.shape),
            )
        )

    prompt_learner.cls_ctx = torch.nn.Parameter(cls_ctx)
    prompt_learner.num_class = cls_ctx.size(0)
    tokenized, prefix, suffix = _prompt_template_buffers(
        base,
        object_noun,
        cls_ctx.size(0),
    )
    prompt_learner.tokenized_prompts = tokenized
    prompt_learner.token_prefix = prefix
    prompt_learner.token_suffix = suffix


def expand_prompt_with_stage1(
    base,
    prompt_state,
    start_idx,
    total_classes,
    object_noun='person',
):
    prompt_learner = base.prompt_learner
    old_ctx = prompt_learner.cls_ctx.data
    domain_ctx = prompt_state['cls_ctx'].to(old_ctx.device)
    expected_new = total_classes - start_idx
    if domain_ctx.size(0) != expected_new:
        raise ValueError(
            'Stage1 prompt contains {} identities, expected {}'.format(
                domain_ctx.size(0),
                expected_new,
            )
        )

    new_ctx = old_ctx.new_empty(
        total_classes,
        old_ctx.size(1),
        old_ctx.size(2),
    )
    new_ctx[:start_idx].copy_(old_ctx[:start_idx])
    new_ctx[start_idx:].copy_(domain_ctx)
    prompt_learner.cls_ctx = torch.nn.Parameter(new_ctx)
    prompt_learner.num_class = total_classes

    old_tokenized = _expanded_rows(
        prompt_learner.tokenized_prompts,
        start_idx,
    )
    old_prefix = _expanded_rows(prompt_learner.token_prefix, start_idx)
    old_suffix = _expanded_rows(prompt_learner.token_suffix, start_idx)
    domain_tokenized, domain_prefix, domain_suffix = (
        _prompt_template_buffers(
            base,
            object_noun,
            domain_ctx.size(0),
        )
    )
    prompt_learner.tokenized_prompts = torch.cat(
        [old_tokenized, domain_tokenized],
        dim=0,
    )
    prompt_learner.token_prefix = torch.cat(
        [old_prefix, domain_prefix],
        dim=0,
    )
    prompt_learner.token_suffix = torch.cat(
        [old_suffix, domain_suffix],
        dim=0,
    )


def expand_linear_head(linear, out_dim):
    old_weight = linear.weight.data.clone()
    new_linear = torch.nn.Linear(
        linear.in_features,
        out_dim,
        bias=False,
    ).to(linear.weight.device)
    torch.nn.init.normal_(new_linear.weight, std=0.01)
    new_linear.weight.data[:old_weight.size(0)].copy_(old_weight)
    return new_linear


def expand_model_for_domain(
    model,
    args,
    name,
    spec,
    add_num,
    num_classes,
):
    base = _base_model(model)
    total_classes = add_num + num_classes
    base.classifier = expand_linear_head(base.classifier, total_classes)
    object_noun = _object_noun(
        spec,
        category_prompt=not getattr(
            args,
            'disable_category_prompt',
            False,
        ),
    )
    prompt_state = load_prompt(
        _prompt_path(args, name, spec),
        expected_object_noun=object_noun,
    )
    expand_prompt_with_stage1(
        base,
        prompt_state,
        add_num,
        total_classes,
        object_noun=object_noun,
    )
    base.num_classes = total_classes
    wrapped = model.module if hasattr(model, 'module') else model
    wrapped.num_classes = total_classes
    return total_classes


def _set_stage2_mode(model, trainable_vision_blocks=None):
    base = _base_model(model)
    for parameter in base.parameters():
        parameter.requires_grad = True
    for module in (
        base.prompt_learner,
        base.text_encoder,
        base.token_embedding,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = False

    if trainable_vision_blocks is None:
        return
    blocks = list(base.image_encoder.transformer.resblocks)
    trainable_vision_blocks = int(trainable_vision_blocks)
    if not 0 <= trainable_vision_blocks <= len(blocks):
        raise ValueError(
            'trainable vision blocks must be in [0, {}], got {}'.format(
                len(blocks),
                trainable_vision_blocks,
            )
        )
    if trainable_vision_blocks == len(blocks):
        return
    for parameter in base.image_encoder.parameters():
        parameter.requires_grad = False
    if trainable_vision_blocks:
        for block in blocks[-trainable_vision_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in base.image_encoder.ln_post.parameters():
            parameter.requires_grad = True
        if base.image_encoder.proj is not None:
            base.image_encoder.proj.requires_grad_(True)


def _add_and_activate_domain_adapter(model, domain_name, args):
    base = _base_model(model)
    base.add_domain_adapter(
        domain_name,
        last_blocks=args.adapter_last_blocks,
        bottleneck_dim=args.adapter_bottleneck_dim,
        scale=args.adapter_scale,
    )
    base.set_active_adapter(domain_name)
    parameter_count = sum(
        parameter.numel()
        for parameter in base.domain_adapter_parameters(domain_name)
    )
    print(
        'Activated domain adapter {!r}: last {}/12 blocks, '
        'bottleneck={}, parameters={:,}.'.format(
            domain_name,
            args.adapter_last_blocks,
            args.adapter_bottleneck_dim,
            parameter_count,
        )
    )
    return parameter_count


def _set_domain_adapter_stage2_mode(model, domain_name):
    """Freeze the shared model and train only the current adapter/head."""

    base = _base_model(model)
    for parameter in base.parameters():
        parameter.requires_grad = False
        parameter.grad = None
    for parameter in base.classifier.parameters():
        parameter.requires_grad = True
    for parameter in base.domain_adapter_parameters(domain_name):
        parameter.requires_grad = True


def make_optimizer_stage2(
    model,
    base_lr,
    classifier_lr_multiplier=10.0,
    adapter_lr=0.0003,
):
    classifier_lr_multiplier = float(classifier_lr_multiplier)
    if (
        not np.isfinite(classifier_lr_multiplier)
        or classifier_lr_multiplier <= 0.0
    ):
        raise ValueError(
            'classifier_lr_multiplier must be finite and positive'
        )
    adapter_lr = float(adapter_lr)
    if not np.isfinite(adapter_lr) or adapter_lr <= 0.0:
        raise ValueError('adapter_lr must be finite and positive')
    parameter_groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if '.domain_adapters.' in name:
            learning_rate = adapter_lr
        elif name.startswith('classifier.'):
            learning_rate = base_lr * classifier_lr_multiplier
        else:
            learning_rate = base_lr * 2 if 'bias' in name else base_lr
        parameter_groups.append(
            {
                'params': [parameter],
                'lr': learning_rate,
                'weight_decay': 1e-4,
            }
        )
    if not parameter_groups:
        raise RuntimeError('No trainable parameters found for Stage2')
    return torch.optim.Adam(parameter_groups)


def prepare_model(args, first_info, specs):
    _, num_classes, _, _, _, _, name = first_info
    spec = specs.get(name)
    object_noun = _object_noun(
        spec,
        category_prompt=not args.disable_category_prompt,
    )
    model = make_model(
        num_class=num_classes,
        camera_num=0,
        view_num=0,
        input_size=(args.height, args.width),
        object_noun=object_noun,
    )
    model = DataParallel(model.cuda())
    _base_model(model).set_eval_descriptor_mode(args.eval_descriptor)
    prompt_state = load_prompt(
        _prompt_path(args, name, spec),
        expected_object_noun=object_noun,
    )
    prompt_initialize(
        _base_model(model),
        prompt_state,
        object_noun=object_noun,
    )
    _set_stage2_mode(
        model,
        args.first_domain_trainable_vision_blocks,
    )
    print('Initialized Stage1 identity prompts for {}'.format(name))
    return model


def evaluate_checkpoint(
    args,
    model,
    all_train_sets,
    all_test_sets,
    training_set,
    specs,
    logger,
    adapter_routing_bank=None,
):
    checkpoint_name = (
        '{}_checkpoint.pth.tar'.format(_domain_name(training_set[-1]))
    )
    checkpoint = load_checkpoint(osp.join(args.testing, checkpoint_name))

    for domain_index in range(1, len(all_train_sets)):
        _, num_classes, _, _, _, _, name = all_train_sets[domain_index]
        add_num = sum(
            all_train_sets[index][1]
            for index in range(domain_index)
        )
        expand_model_for_domain(
            model,
            args,
            name,
            specs.get(name),
            add_num,
            num_classes,
        )

    if checkpoint.get('continual_update_mode') == 'domain_adapter':
        base = _base_model(model)
        adapter_domains = checkpoint.get('adapter_domains', [])
        if not adapter_domains:
            raise ValueError(
                'domain-adapter checkpoint is missing adapter_domains'
            )
        for domain_name in adapter_domains:
            base.add_domain_adapter(
                domain_name,
                last_blocks=checkpoint.get('adapter_last_blocks', 4),
                bottleneck_dim=checkpoint.get(
                    'adapter_bottleneck_dim',
                    64,
                ),
                scale=checkpoint.get('adapter_scale', 1.0),
            )
        base.set_active_adapter(None)
        print(
            'Rebuilt domain adapter bank from checkpoint: {}'.format(
                ', '.join(adapter_domains)
            )
        )

    copy_state_dict(checkpoint['state_dict'], model)
    saved_routing_bank = checkpoint.get('adapter_routing_bank')
    if saved_routing_bank:
        adapter_routing_bank = saved_routing_bank
    elif args.adapter_routing == 'osaf':
        base = _base_model(model)
        adapter_routing_bank = {}
        learned_names = set(base.domain_adapter_names())
        for train_info in all_train_sets:
            _, num_classes, _, _, _, _, name = train_info
            if name not in learned_names:
                continue
            spec = specs.get(name)
            adapter_routing_bank[name] = (
                build_domain_adapter_routing_entry(
                    base,
                    _prompt_path(args, name, spec),
                    expected_num_classes=num_classes,
                )
            )
        print(
            'Rebuilt OSAF routing bank from Stage1 checkpoints: {}'
            .format(', '.join(adapter_routing_bank))
        )
    fast_test_p_s(
        model,
        all_train_sets,
        all_test_sets,
        set_index=len(all_train_sets) - 1,
        args=args,
        logger=logger,
        adapter_routing_bank=adapter_routing_bank,
    )


def main_worker(args):
    timestamp = _timestamp()
    run_kind = 'evaluation' if args.testing else 'stage2'
    stage2_dir = osp.join(args.logs_dir, run_kind, timestamp)
    suffix = (
        '_{}'.format(args.other_details)
        if args.other_details
        else ''
    )
    sys.stdout = Logger(
        osp.join(stage2_dir, 'log_{}{}.txt'.format(timestamp, suffix))
    )
    result_logger = Logger_res(
        osp.join(
            stage2_dir,
            'log_res_{}{}.txt'.format(timestamp, suffix),
        )
    )

    training_set, testing_set, specs = _resolve_domain_sets(args)
    if args.adapter_routing == 'osaf' and not testing_set:
        if args.adapter_routing_scope == 'all':
            routing_note = 'seen domains will still use OSAF routing'
        else:
            routing_note = 'seen domains will use their oracle adapters'
        print(
            '[warning] --adapter-routing osaf is enabled, but the domain '
            'configuration has no test_domains; {}, and UnSeen-Avg will '
            'remain NaN.'.format(routing_note)
        )
    if args.attr_distill_mode != 'none':
        if not args.domain_config:
            raise ValueError(
                'attribute distillation requires --domain-config with '
                'per-domain attributes'
            )
        for domain in training_set:
            _attribute_entries(domain)
    all_train_sets, all_test_sets = build_data_loaders(
        args,
        training_set,
        testing_set,
    )
    model = prepare_model(args, all_train_sets[0], specs)
    adapter_routing_bank = {}
    if args.attr_distill_mode != 'none':
        _validate_attribute_model_support(model)
    print(
        'Category Prompt: {}'.format(
            'disabled (generic "object")'
            if args.disable_category_prompt
            else 'enabled (per-domain object noun)'
        )
    )
    print(
        'Stage2 visual anchor mode: {}'.format(
            args.visual_anchor_mode
        )
    )
    print(
        'OCIA residual weight (lambda_r): {}'.format(
            args.anchor_residual_weight
        )
    )
    print(
        'Anchor alignment temperature: {}'.format(
            args.anchor_temperature
        )
    )
    print(
        'Attribute distillation: {} (lambda_s={})'.format(
            args.attr_distill_mode,
            args.scsd_weight,
        )
    )
    print(
        'Global relation distillation weight: {}'.format(
            args.scsd_global_weight
        )
    )
    print(
        'Classifier scope: {}; eval descriptor: {}'.format(
            args.classifier_scope,
            args.eval_descriptor,
        )
    )
    print(
        'Stage2 base LR: {}; classifier LR multiplier: {} '
        '(classifier LR: {})'.format(
            args.stage2_base_lr,
            args.classifier_lr_multiplier,
            args.stage2_base_lr * args.classifier_lr_multiplier,
        )
    )
    print('Continual update mode: {}'.format(args.continual_update_mode))
    if args.continual_update_mode == 'domain_adapter':
        print(
            'Domain adapters: last {}/12 blocks, bottleneck={}, scale={}, '
                'LR={}, routing={}, routing_scope={}.'.format(
                    args.adapter_last_blocks,
                    args.adapter_bottleneck_dim,
                    args.adapter_scale,
                    args.adapter_lr,
                    args.adapter_routing,
                    args.adapter_routing_scope,
                )
        )
        if args.adapter_routing == 'osaf':
            print(
                'OSAF: Top-K={}, semantic_weight={}, temperature={}, '
                'debias_strength={}, fusion_weight={}.'.format(
                    args.adapter_topk,
                    args.adapter_semantic_weight,
                    args.adapter_routing_temperature,
                    args.adapter_debias_strength,
                    args.adapter_fusion_weight,
                )
            )

    if args.testing:
        evaluate_checkpoint(
            args,
            model,
            all_train_sets,
            all_test_sets,
            training_set,
            specs,
            result_logger,
            adapter_routing_bank=adapter_routing_bank,
        )
        print('Checkpoint evaluation finished.')
        return

    eval_stages = {
        int(value.strip())
        for value in args.eval_stage.split(',')
        if value.strip()
    }

    for domain_index, train_info in enumerate(all_train_sets):
        (
            _,
            num_classes,
            _,
            stage2_loader,
            _,
            init_loader,
            name,
        ) = train_info
        spec = specs.get(name)
        add_num = sum(
            all_train_sets[index][1]
            for index in range(domain_index)
        )
        total_classes = add_num + num_classes

        print('=' * 80)
        print(
            '[Stage2] Training domain {}/{}: {}'.format(
                domain_index + 1,
                len(all_train_sets),
                name,
            )
        )
        print('=' * 80)

        teacher_model = None
        old_entries = None
        if domain_index > 0 and args.attr_distill_mode != 'none':
            teacher_model = _freeze_teacher(model)
            previous_name = all_train_sets[domain_index - 1][-1]
            old_entries = _attribute_entries(specs.get(previous_name))

        if domain_index > 0:
            total_classes = expand_model_for_domain(
                model,
                args,
                name,
                spec,
                add_num,
                num_classes,
            )
            print(
                'Expanded classifier and prompt bank to {} identities.'.format(
                    total_classes
                )
            )

        if args.continual_update_mode == 'domain_adapter':
            _add_and_activate_domain_adapter(model, name, args)

        class_centers = initial_classifier(model, init_loader)
        base = _base_model(model)
        if args.adapter_routing == 'osaf':
            adapter_routing_bank[name] = (
                build_domain_adapter_routing_entry(
                    base,
                    _prompt_path(args, name, spec),
                    expected_num_classes=num_classes,
                )
            )
            print(
                'Added fixed OSAF routing statistics for {}: '
                '{} visual prototypes, semantic rank {}.'.format(
                    name,
                    adapter_routing_bank[name][
                        'visual_prototypes'
                    ].size(0),
                    adapter_routing_bank[name][
                        'category_basis'
                    ].size(1),
                )
            )
        if domain_index == 0:
            base.classifier.weight.data.copy_(class_centers)
        else:
            base.classifier.weight.data[add_num:].copy_(class_centers)

        if (
            args.continual_update_mode == 'domain_adapter'
            and domain_index > 0
        ):
            _set_domain_adapter_stage2_mode(model, name)
            print(
                'Shared visual backbone frozen; training adapter {!r} and '
                'the identity classifier.'.format(name)
            )
        else:
            trainable_vision_blocks = (
                args.first_domain_trainable_vision_blocks
                if domain_index == 0
                else args.continual_trainable_vision_blocks
            )
            _set_stage2_mode(model, trainable_vision_blocks)
            print(
                'Trainable CLIP vision blocks for this domain: {}/12'.format(
                    trainable_vision_blocks
                )
            )
            if (
                args.continual_update_mode == 'domain_adapter'
                and domain_index == 0
            ):
                print(
                    'Training first-domain adapter {!r} jointly with the '
                    'shared ReID initialization; both are frozen before the '
                    'next domain.'.format(name)
                )
        old_attribute_text_features = None
        current_attribute_text_features = None
        current_entries = None
        if args.attr_distill_mode != 'none':
            current_entries = _attribute_entries(spec)
            current_attribute_text_features = _configure_attribute_pooling(
                model,
                current_entries,
                args.attribute_attention_temperature,
            )
            if teacher_model is not None:
                old_attribute_text_features = (
                    _configure_attribute_pooling(
                        teacher_model,
                        old_entries,
                        args.attribute_attention_temperature,
                    )
                )
                _print_attribute_matches(
                    old_entries,
                    current_entries,
                    old_attribute_text_features,
                    current_attribute_text_features,
                    args.scsd_semantic_threshold,
                    args.attr_distill_mode,
                )
        identity_anchors = None
        if args.visual_anchor_mode != 'text':
            identity_anchors = build_domain_identity_anchors(
                base,
                _prompt_path(args, name, spec),
                start_idx=add_num,
                num_classes=num_classes,
                mode=args.visual_anchor_mode,
                residual_weight=args.anchor_residual_weight,
            )
            print(
                'Built {} fixed cross-modal anchors for {} using mode={!r}.'
                .format(
                    identity_anchors.size(0),
                    name,
                    args.visual_anchor_mode,
                )
            )

        optimizer = make_optimizer_stage2(
            base,
            base_lr=args.stage2_base_lr,
            classifier_lr_multiplier=args.classifier_lr_multiplier,
            adapter_lr=args.adapter_lr,
        )
        scheduler = WarmupMultiStepLR(
            optimizer,
            args.milestones,
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=args.warmup_step,
        )
        trainer = Stage2Trainer(
            cfg=cfg,
            model=model,
            num_classes=total_classes,
            global_loss_weight=args.global_loss_weight,
            anchor_temperature=args.anchor_temperature,
            identity_anchors=identity_anchors,
            visual_anchor_mode=args.visual_anchor_mode,
            teacher_model=teacher_model,
            attr_distill_mode=args.attr_distill_mode,
            scsd_weight=args.scsd_weight,
            scsd_global_weight=args.scsd_global_weight,
            classifier_scope=args.classifier_scope,
            old_attribute_text_features=old_attribute_text_features,
            current_attribute_text_features=(
                current_attribute_text_features
            ),
            scsd_semantic_threshold=args.scsd_semantic_threshold,
            scsd_relation_temperature=args.scsd_relation_temperature,
            scsd_kl_direction=args.scsd_kl_direction,
            scsd_mask_diagonal=args.scsd_mask_diagonal,
        )

        epochs = args.epochs0 if domain_index == 0 else args.epochs
        for epoch in tqdm(
            range(epochs),
            desc='[Stage2-{}]'.format(name),
        ):
            stage2_loader.new_epoch()
            trainer.train(
                stage2_loader,
                optimizer,
                train_iters=len(stage2_loader),
                add_num=add_num,
            )
            # Warm-up and milestone values are expressed in epochs, so the
            # scheduler must advance once per completed epoch rather than
            # once per mini-batch.
            scheduler.step()

            should_save = (
                (epoch + 1) % args.eval_epoch == 0
                or epoch + 1 == epochs
            )
            if should_save:
                result_logger.append('epoch: {}'.format(epoch + 1))
                save_checkpoint(
                    {
                        'state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'domain_index': domain_index,
                        'domain_name': name,
                        'category_prompt_enabled': (
                            not args.disable_category_prompt
                        ),
                        'visual_anchor_mode': args.visual_anchor_mode,
                        'anchor_residual_weight': (
                            args.anchor_residual_weight
                        ),
                        'anchor_temperature': args.anchor_temperature,
                        'stage2_base_lr': args.stage2_base_lr,
                        'classifier_lr_multiplier': (
                            args.classifier_lr_multiplier
                        ),
                        'attr_distill_mode': args.attr_distill_mode,
                        'scsd_weight': args.scsd_weight,
                        'scsd_global_weight': args.scsd_global_weight,
                        'classifier_scope': args.classifier_scope,
                        'eval_descriptor': args.eval_descriptor,
                        'first_domain_trainable_vision_blocks': (
                            args.first_domain_trainable_vision_blocks
                        ),
                        'continual_trainable_vision_blocks': (
                            args.continual_trainable_vision_blocks
                        ),
                        'continual_update_mode': (
                            args.continual_update_mode
                        ),
                        'adapter_last_blocks': args.adapter_last_blocks,
                        'adapter_bottleneck_dim': (
                            args.adapter_bottleneck_dim
                        ),
                        'adapter_scale': args.adapter_scale,
                        'adapter_lr': args.adapter_lr,
                        'adapter_routing': args.adapter_routing,
                        'adapter_routing_scope': (
                            args.adapter_routing_scope
                        ),
                        'adapter_first_domain': True,
                        'adapter_topk': args.adapter_topk,
                        'adapter_routing_temperature': (
                            args.adapter_routing_temperature
                        ),
                        'adapter_semantic_weight': (
                            args.adapter_semantic_weight
                        ),
                        'adapter_debias_strength': (
                            args.adapter_debias_strength
                        ),
                        'adapter_fusion_weight': (
                            args.adapter_fusion_weight
                        ),
                        'adapter_domains': list(
                            base.domain_adapter_names()
                        ),
                        'adapter_routing_bank': _routing_bank_subset(
                            adapter_routing_bank,
                            base.domain_adapter_names(),
                        ),
                        'scsd_semantic_threshold': (
                            args.scsd_semantic_threshold
                        ),
                        'scsd_relation_temperature': (
                            args.scsd_relation_temperature
                        ),
                        'scsd_kl_direction': args.scsd_kl_direction,
                        'scsd_mask_diagonal': args.scsd_mask_diagonal,
                        'attribute_attention_temperature': (
                            args.attribute_attention_temperature
                        ),
                        'attributes': current_entries,
                    },
                    True,
                    fpath=osp.join(
                        stage2_dir,
                        '_CKPTS',
                        '{}_checkpoint.pth.tar'.format(name),
                    ),
                )

        del trainer, optimizer, scheduler
        if teacher_model is not None:
            del teacher_model
        gc.collect()
        torch.cuda.empty_cache()

        if domain_index in eval_stages:
            fast_test_p_s(
                model,
                all_train_sets,
                all_test_sets,
                set_index=domain_index,
                args=args,
                logger=result_logger,
                adapter_routing_bank=adapter_routing_bank,
            )

    print('=' * 80)
    print('Stage2 continual training finished.')
    print('=' * 80)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Global CLIP-ReID continual baseline'
    )
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument(
        '--resize-mode',
        choices=['stretch', 'pad'],
        default='stretch',
    )
    parser.add_argument(
        '--max-train-ids',
        type=int,
        default=0,
        help='limit identities per domain; 0 uses all identities',
    )
    parser.add_argument('--num-instances', type=int, default=4)
    parser.add_argument('--stage2-base-lr', type=float, default=0.000005)
    parser.add_argument(
        '--classifier-lr-multiplier',
        type=float,
        default=10.0,
        help='learning-rate multiplier for the identity classifier head',
    )
    parser.add_argument('--global-loss-weight', type=float, default=1.0)
    parser.add_argument(
        '--attr-distill-mode',
        choices=DISTILLATION_MODES,
        default='none',
        help=(
            'attribute relation transfer: disabled, original index-locked '
            'DAlign, hard/soft semantic ablations, or full SCSD'
        ),
    )
    parser.add_argument(
        '--scsd-weight',
        type=float,
        default=20.0,
        help='SCSD loss weight lambda_s',
    )
    parser.add_argument(
        '--scsd-global-weight',
        type=float,
        default=20.0,
        help='global CLS relation-distillation weight',
    )
    parser.add_argument(
        '--classifier-scope',
        choices=['current', 'all'],
        default='current',
        help=(
            'current avoids repelling new-domain identities from every old '
            'identity classifier row; all retains the legacy behavior'
        ),
    )
    parser.add_argument(
        '--eval-descriptor',
        choices=['raw', 'bn'],
        default='raw',
        help='use stable raw CLIP features or legacy BN-neck features',
    )
    parser.add_argument(
        '--first-domain-trainable-vision-blocks',
        type=int,
        default=12,
        help='number of trainable CLIP vision transformer blocks in domain 1',
    )
    parser.add_argument(
        '--continual-trainable-vision-blocks',
        type=int,
        default=2,
        help='number of trainable final CLIP vision blocks after domain 1',
    )
    parser.add_argument(
        '--continual-update-mode',
        choices=['partial_blocks', 'domain_adapter'],
        default='partial_blocks',
        help=(
            'update final shared vision blocks, or freeze the shared '
            'backbone after domain 1 while allocating one private adapter '
            'for every domain, including domain 1'
        ),
    )
    parser.add_argument(
        '--enable-domain-adapter',
        dest='continual_update_mode',
        action='store_const',
        const='domain_adapter',
        help='alias for --continual-update-mode domain_adapter',
    )
    parser.add_argument(
        '--disable-domain-adapter',
        dest='continual_update_mode',
        action='store_const',
        const='partial_blocks',
        help='alias for --continual-update-mode partial_blocks',
    )
    parser.add_argument(
        '--adapter-last-blocks',
        type=int,
        default=4,
        help='number of final vision blocks receiving per-domain adapters',
    )
    parser.add_argument(
        '--adapter-bottleneck-dim',
        type=int,
        default=64,
        help='bottleneck width of each per-domain visual adapter',
    )
    parser.add_argument(
        '--adapter-scale',
        type=float,
        default=1.0,
        help='residual scale applied to each active domain adapter',
    )
    parser.add_argument(
        '--adapter-lr',
        type=float,
        default=0.0003,
        help='learning rate for the newly allocated current-domain adapter',
    )
    parser.add_argument(
        '--adapter-routing',
        choices=['oracle', 'osaf'],
        default='oracle',
        help=(
            'use dataset-name routing for seen domains only, or apply '
            'OCIA-guided semantic-debiased Top-K fusion on unseen domains'
        ),
    )
    parser.add_argument(
        '--adapter-routing-scope',
        choices=['unseen', 'all'],
        default='unseen',
        help=(
            'apply OSAF only to held-out unseen domains, or to all '
            'evaluation domains including learned seen domains'
        ),
    )
    parser.add_argument(
        '--adapter-topk',
        type=int,
        default=2,
        help='number of source adapters fused for each unseen image',
    )
    parser.add_argument(
        '--adapter-routing-temperature',
        type=float,
        default=0.1,
        help='softmax temperature for OSAF Top-K adapter weights',
    )
    parser.add_argument(
        '--adapter-semantic-weight',
        type=float,
        default=0.5,
        help=(
            'OSAF routing balance between object-semantic and visual '
            'prototype similarities'
        ),
    )
    parser.add_argument(
        '--adapter-debias-strength',
        type=float,
        default=1.0,
        help='strength of OCIA subspace removal from adapter residuals',
    )
    parser.add_argument(
        '--adapter-fusion-weight',
        type=float,
        default=1.0,
        help='weight of the fused debiased adapter residual',
    )
    parser.add_argument(
        '--scsd-semantic-threshold',
        type=float,
        default=0.70,
        help='frozen-CLIP semantic compatibility threshold delta',
    )
    parser.add_argument(
        '--scsd-relation-temperature',
        type=float,
        default=0.07,
        help='batch-wise attribute relation temperature tau_r',
    )
    parser.add_argument(
        '--attribute-attention-temperature',
        type=float,
        default=0.07,
        help='text-to-patch attribute pooling temperature',
    )
    parser.add_argument(
        '--scsd-kl-direction',
        choices=['old_to_new', 'paper'],
        default='old_to_new',
        help=(
            'old_to_new uses KL(old||current); paper uses the KL direction '
            'printed by the VLADR paper'
        ),
    )
    parser.add_argument(
        '--scsd-mask-diagonal',
        dest='scsd_mask_diagonal',
        action='store_true',
        help='exclude self-similarity from attribute relation rows',
    )
    parser.add_argument(
        '--scsd-include-diagonal',
        dest='scsd_mask_diagonal',
        action='store_false',
        help='include self-similarity for an original-formula ablation',
    )
    parser.set_defaults(scsd_mask_diagonal=True)
    parser.add_argument(
        '--visual-anchor-mode',
        choices=['text', 'prototype', 'centered', 'ocia'],
        default='text',
        help=(
            'global-alignment target: text Prompt only, Prompt plus visual '
            'prototype, Prompt plus category-centered visual residual, or '
            'full OCIA category-subspace-calibrated anchor'
        ),
    )
    parser.add_argument(
        '--anchor-residual-weight',
        type=float,
        default=1.0,
        help='OCIA visual-residual fusion weight lambda_r',
    )
    parser.add_argument(
        '--anchor-temperature',
        type=float,
        default=0.07,
        help='temperature for text/OCIA anchor classification logits',
    )
    parser.add_argument(
        '--enable-ocia',
        dest='visual_anchor_mode',
        action='store_const',
        const='ocia',
        help='alias for --visual-anchor-mode ocia',
    )
    parser.add_argument(
        '--disable-ocia',
        dest='visual_anchor_mode',
        action='store_const',
        const='text',
        help='alias for --visual-anchor-mode text',
    )
    parser.add_argument(
        '--enable-category-centered-visual-residual',
        dest='visual_anchor_mode',
        action='store_const',
        const='centered',
        help='alias for --visual-anchor-mode centered',
    )
    parser.add_argument(
        '--disable-category-centered-visual-residual',
        dest='visual_anchor_mode',
        action='store_const',
        const='text',
        help='alias for --visual-anchor-mode text',
    )
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument(
        '--milestones',
        nargs='+',
        type=int,
        default=[30],
    )
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval-epoch', type=int, default=100)
    parser.add_argument('--eval-stage', type=str, default='0,1,2,3,4')
    parser.add_argument(
        '--eval-query-chunk-size',
        type=int,
        default=256,
        help=(
            'number of queries per exact ranking chunk when the full '
            'distance matrix exceeds --eval-full-matrix-max-elements'
        ),
    )
    parser.add_argument(
        '--eval-full-matrix-max-elements',
        type=int,
        default=100000000,
        help=(
            'maximum QxG element count for full-matrix ranking; larger '
            'evaluations use exact query chunking, and -1 disables chunking'
        ),
    )
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--logs-dir', type=str, default='./RESULTS')
    parser.add_argument('--config-file', type=str, default='config/base.yml')
    parser.add_argument(
        '--domain-config',
        type=str,
        default=None,
        help='JSON/YAML cross-category domain configuration',
    )
    parser.add_argument(
        '--setting',
        type=int,
        default=1,
        choices=[1, 2, 51, 52, 53, 54, 55],
    )
    parser.add_argument(
        '--stage1-prompts-out-dir',
        dest='stage1_prompts_out_dir',
        type=str,
        default='./STAGE1_PROMPTS_WEIGHT',
    )
    parser.add_argument(
        '--prompt-checkpoint-source',
        choices=['config', 'output-dir'],
        default='config',
        help=(
            'use prompt_checkpoint paths from the domain config, or build '
            'all prompt paths under --stage1-prompts-out-dir'
        ),
    )
    parser.add_argument(
        '--disable-category-prompt',
        action='store_true',
        help=(
            'replace each domain noun with the generic noun "object"; '
            'Stage1 checkpoints must use the same setting'
        ),
    )
    parser.add_argument('--testing', type=str, default=None)
    parser.add_argument('--other-details', type=str, default=None)
    parser.add_argument('--save-evaluation', action='store_true')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    if (
        not np.isfinite(args.anchor_residual_weight)
        or args.anchor_residual_weight < 0.0
    ):
        raise ValueError(
            '--anchor-residual-weight must be finite and non-negative'
        )
    for name, value in (
        ('--stage2-base-lr', args.stage2_base_lr),
        ('--classifier-lr-multiplier', args.classifier_lr_multiplier),
        ('--anchor-temperature', args.anchor_temperature),
        ('--adapter-scale', args.adapter_scale),
        ('--adapter-lr', args.adapter_lr),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError('{} must be finite and positive'.format(name))
    if not np.isfinite(args.scsd_weight) or args.scsd_weight < 0.0:
        raise ValueError('--scsd-weight must be finite and non-negative')
    if (
        not np.isfinite(args.scsd_global_weight)
        or args.scsd_global_weight < 0.0
    ):
        raise ValueError(
            '--scsd-global-weight must be finite and non-negative'
        )
    if not np.isfinite(args.scsd_semantic_threshold):
        raise ValueError('--scsd-semantic-threshold must be finite')
    for name, value in (
        ('--scsd-relation-temperature', args.scsd_relation_temperature),
        (
            '--attribute-attention-temperature',
            args.attribute_attention_temperature,
        ),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError('{} must be finite and positive'.format(name))
    for name, value in (
        (
            '--first-domain-trainable-vision-blocks',
            args.first_domain_trainable_vision_blocks,
        ),
        (
            '--continual-trainable-vision-blocks',
            args.continual_trainable_vision_blocks,
        ),
    ):
        if not 0 <= value <= 12:
            raise ValueError('{} must be in [0, 12]'.format(name))
    if not 1 <= args.adapter_last_blocks <= 12:
        raise ValueError('--adapter-last-blocks must be in [1, 12]')
    if not 1 <= args.adapter_bottleneck_dim <= 768:
        raise ValueError('--adapter-bottleneck-dim must be in [1, 768]')
    if args.adapter_topk <= 0:
        raise ValueError('--adapter-topk must be positive')
    if args.eval_query_chunk_size <= 0:
        raise ValueError('--eval-query-chunk-size must be positive')
    if args.eval_full_matrix_max_elements < -1:
        raise ValueError(
            '--eval-full-matrix-max-elements must be -1 or non-negative'
        )
    if (
        not np.isfinite(args.adapter_routing_temperature)
        or args.adapter_routing_temperature <= 0.0
    ):
        raise ValueError(
            '--adapter-routing-temperature must be finite and positive'
        )
    for name, value in (
        ('--adapter-semantic-weight', args.adapter_semantic_weight),
        ('--adapter-debias-strength', args.adapter_debias_strength),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError('{} must be finite and in [0, 1]'.format(name))
    if (
        not np.isfinite(args.adapter_fusion_weight)
        or args.adapter_fusion_weight < 0.0
    ):
        raise ValueError(
            '--adapter-fusion-weight must be finite and non-negative'
        )
    if (
        args.continual_update_mode == 'domain_adapter'
        and args.eval_descriptor != 'raw'
    ):
        raise ValueError(
            '--continual-update-mode domain_adapter requires '
            '--eval-descriptor raw to avoid shared BN-statistics drift'
        )
    if (
        args.adapter_routing == 'osaf'
        and args.continual_update_mode != 'domain_adapter'
    ):
        raise ValueError(
            '--adapter-routing osaf requires '
            '--continual-update-mode domain_adapter'
        )
    if (
        args.adapter_routing_scope == 'all'
        and args.adapter_routing != 'osaf'
    ):
        raise ValueError(
            '--adapter-routing-scope all requires --adapter-routing osaf'
        )
    if (
        args.adapter_routing == 'osaf'
        and args.visual_anchor_mode != 'ocia'
    ):
        raise ValueError(
            '--adapter-routing osaf requires --visual-anchor-mode ocia'
        )
    if args.adapter_routing == 'osaf' and not args.domain_config:
        raise ValueError(
            '--adapter-routing osaf requires --domain-config with OCIA '
            'source-domain metadata'
        )
    set_seed(args.seed)
    cfg.merge_from_file(args.config_file)
    main_worker(args)
