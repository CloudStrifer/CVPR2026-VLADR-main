from __future__ import absolute_import, print_function

import argparse
import datetime
import os
import os.path as osp
import random
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from lreid_dataset.datasets.manifest_reid import load_domain_config
from reid.models.CLIP_ReID.loss.supcontrast import SupConLoss
from reid.models.layers import DataParallel
from reid.models.wrapper import make_model
from reid.utils.feature_tools import (
    category_semantic_subspace,
    visual_identity_statistics,
)
from reid.utils.logging import Logger
from reid.utils.lr_scheduler import WarmupMultiStepLR


warnings.filterwarnings(action='ignore', category=UserWarning)


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


def _object_noun(spec, category_prompt=True):
    if not category_prompt:
        return 'object'
    if spec is None:
        return 'person'
    return spec.get('object_noun', spec['category'])


def _semantic_object_noun(spec):
    if spec is None:
        return 'person'
    return spec.get('object_noun', spec['category'])


def _category_generic_prompts(spec):
    if spec is not None and spec.get('category_generic_prompts'):
        return list(spec['category_generic_prompts'])

    object_noun = _semantic_object_noun(spec)
    return [
        'A photo showing the overall silhouette of a {}.'.format(
            object_noun
        ),
        'A photo showing the basic part structure of a {}.'.format(
            object_noun
        ),
        'A photo of a {} from a common viewpoint.'.format(object_noun),
    ]


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


def _set_stage1_mode(model):
    base = _base_model(model)
    for parameter in base.parameters():
        parameter.requires_grad = False
    for parameter in base.prompt_learner.parameters():
        parameter.requires_grad = True
    return base


def _make_prompt_optimizer(base):
    parameters = [
        parameter
        for name, parameter in base.named_parameters()
        if 'prompt_learner' in name and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError('No trainable identity-prompt parameters found')
    return torch.optim.Adam(
        parameters,
        lr=0.0035,
        weight_decay=1e-4,
    )


def train_prompt_epoch(
    model,
    base,
    loader,
    optimizer,
    supcon_loss,
    lr_scheduler,
    dataset_name,
    epoch,
):
    model.train()
    loader.new_epoch()
    num_iters = len(loader)

    for _ in tqdm(
        range(num_iters),
        desc='[Prompt][Epoch:{}][{}]'.format(epoch, dataset_name),
    ):
        images, _, targets, _, _ = loader.next()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        with torch.no_grad():
            image_features = base(x=images, get_image=True)
        text_features = base(label=targets, get_text=True)

        loss_i2t = supcon_loss(
            image_features,
            text_features,
            targets,
            targets,
        )
        loss_t2i = supcon_loss(
            text_features,
            image_features,
            targets,
            targets,
        )
        loss = loss_i2t + loss_t2i
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    print('\n[stage1] i2t loss: {}'.format(loss_i2t.detach()))
    print('[stage1] t2i loss: {}'.format(loss_t2i.detach()))
    print('[stage1] total loss: {}'.format(loss.detach()))


def main_worker(args):
    timestamp = _timestamp()
    stage1_dir = osp.join(args.logs_dir, 'stage1', timestamp)
    sys.stdout = Logger(osp.join(stage1_dir, 'log_{}.txt'.format(timestamp)))

    training_set, testing_set, specs = _resolve_domain_sets(args)
    all_train_sets, _ = build_data_loaders(
        args,
        training_set,
        testing_set,
    )
    supcon_loss = SupConLoss(device='cuda')
    print(
        'Category Prompt: {}'.format(
            'disabled (generic "object")'
            if args.disable_category_prompt
            else 'enabled (per-domain object noun)'
        )
    )

    for domain_index, train_info in enumerate(all_train_sets):
        (
            _,
            num_classes,
            prompt_loader,
            _,
            _,
            init_loader,
            name,
        ) = train_info
        spec = specs.get(name)
        object_noun = _object_noun(
            spec,
            category_prompt=not args.disable_category_prompt,
        )
        print('=' * 80)
        print(
            '[Stage1] Training identity prompts for domain {}/{}: {}'.format(
                domain_index + 1,
                len(all_train_sets),
                name,
            )
        )
        print('=' * 80)

        model = make_model(
            num_class=num_classes,
            camera_num=0,
            view_num=0,
            input_size=(args.height, args.width),
            object_noun=object_noun,
        )
        model = DataParallel(model.cuda())
        base = _set_stage1_mode(model)
        optimizer = _make_prompt_optimizer(base)
        scheduler = WarmupMultiStepLR(
            optimizer,
            args.milestones,
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=args.warmup_step,
        )

        for epoch in range(args.prompt_epochs):
            train_prompt_epoch(
                model,
                base,
                prompt_loader,
                optimizer,
                supcon_loss,
                scheduler,
                dataset_name=name,
                epoch=epoch,
            )

        print(
            '[Stage1] Extracting frozen-CLIP visual prototypes for {}...'
            .format(name)
        )
        visual_prototypes, category_center = (
            visual_identity_statistics(
                model,
                init_loader,
                num_classes,
            )
        )
        category_generic_prompts = _category_generic_prompts(spec)
        with torch.no_grad():
            category_text_features = base.encode_fixed_texts(
                category_generic_prompts
            )
            category_subspace_basis = category_semantic_subspace(
                category_text_features
            ).detach().cpu()

        path = _prompt_path(args, name, spec)
        os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
        prompt_learner = base.prompt_learner
        torch.save(
            {
                'state_dict': prompt_learner.state_dict(),
                'num_class': prompt_learner.num_class,
                'n_ctx': prompt_learner.n_cls_ctx,
                'dataset_name': name,
                'object_noun': object_noun,
                'category_prompt_enabled': (
                    not args.disable_category_prompt
                ),
                'visual_prototypes': visual_prototypes,
                'category_center': category_center,
                'visual_feature_dim': visual_prototypes.size(1),
                'visual_prototype_source': (
                    'frozen_clip_vit_b16_image_projection'
                ),
                'ocia_schema_version': 1,
                'category_semantic_object_noun': (
                    _semantic_object_noun(spec)
                ),
                'category_semantic_prompts': category_generic_prompts,
                'category_subspace_basis': category_subspace_basis,
                'category_subspace_rank': (
                    category_subspace_basis.size(1)
                ),
            },
            path,
        )
        print('Identity prompt for {} saved at: {}'.format(name, path))
        print(
            'Saved visual prototypes: {}, category center: {}'.format(
                tuple(visual_prototypes.shape),
                tuple(category_center.shape),
            )
        )
        print(
            'Saved OCIA category subspace: {} prompts, rank {}, shape {}.'
            .format(
                len(category_generic_prompts),
                category_subspace_basis.size(1),
                tuple(category_subspace_basis.shape),
            )
        )

    print('=' * 80)
    print('Stage1 identity-prompt training finished.')
    print('=' * 80)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train global identity prompts for continual ReID'
    )
    parser.add_argument('-b', '--batch-size', type=int, default=16)
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
    parser.add_argument('--prompt-epochs', type=int, default=120)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument(
        '--milestones',
        nargs='+',
        type=int,
        default=[30],
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--logs-dir', type=str, default='./RESULTS')
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
            'use the same setting in Stage 2'
        ),
    )
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    os.makedirs(args.stage1_prompts_out_dir, exist_ok=True)
    set_seed(args.seed)
    main_worker(args)
