from __future__ import absolute_import, print_function

import gc
import math
import os.path as osp
import sys

import torch
from tqdm import tqdm

import train_stage2 as base_train
from config import cfg
from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from reid.evaluation.fast_test import fast_test_p_s
from reid.trainer_lwf import LwFTrainer
from reid.utils.logging import Logger
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.serialization import save_checkpoint
from tools.Logger_results import Logger_res


def build_parser():
    parser = base_train.build_parser()
    parser.description = (
        'Learning without Forgetting baseline for cross-category CLIP-ReID'
    )
    parser.add_argument(
        '--lwf-weight',
        type=float,
        default=1.0,
        help='weight of the old-class LwF distillation loss',
    )
    parser.add_argument(
        '--lwf-temperature',
        type=float,
        default=2.0,
        help='temperature used for old-class LwF distillation',
    )
    parser.set_defaults(
        attr_distill_mode='none',
        classifier_scope='current',
        continual_update_mode='partial_blocks',
        visual_anchor_mode='text',
        adapter_routing='oracle',
        adapter_routing_scope='unseen',
    )
    return parser


def validate_args(args):
    if args.attr_distill_mode != 'none':
        raise ValueError('LwF baseline requires --attr-distill-mode none')
    if args.visual_anchor_mode != 'text':
        raise ValueError('LwF baseline requires --visual-anchor-mode text')
    if args.continual_update_mode != 'partial_blocks':
        raise ValueError(
            'LwF baseline requires --continual-update-mode partial_blocks'
        )
    if args.adapter_routing != 'oracle':
        raise ValueError('LwF baseline does not use OSAF adapter routing')
    if args.classifier_scope != 'current':
        raise ValueError('LwF baseline requires --classifier-scope current')
    for name, value in (
        ('--stage2-base-lr', args.stage2_base_lr),
        ('--classifier-lr-multiplier', args.classifier_lr_multiplier),
        ('--anchor-temperature', args.anchor_temperature),
        ('--lwf-temperature', args.lwf_temperature),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError('{} must be finite and positive'.format(name))
    for name, value in (
        ('--global-loss-weight', args.global_loss_weight),
        ('--lwf-weight', args.lwf_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                '{} must be finite and non-negative'.format(name)
            )
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
    if args.eval_query_chunk_size <= 0:
        raise ValueError('--eval-query-chunk-size must be positive')
    if args.eval_full_matrix_max_elements < -1:
        raise ValueError(
            '--eval-full-matrix-max-elements must be -1 or non-negative'
        )


def main_worker(args):
    timestamp = base_train._timestamp()
    run_kind = 'evaluation_lwf' if args.testing else 'lwf'
    output_dir = osp.join(args.logs_dir, run_kind, timestamp)
    suffix = '_{}'.format(args.other_details) if args.other_details else ''
    sys.stdout = Logger(
        osp.join(output_dir, 'log_{}{}.txt'.format(timestamp, suffix))
    )
    result_logger = Logger_res(
        osp.join(output_dir, 'log_res_{}{}.txt'.format(timestamp, suffix))
    )

    training_set, testing_set, specs = base_train._resolve_domain_sets(args)
    all_train_sets, all_test_sets = build_data_loaders(
        args,
        training_set,
        testing_set,
    )
    model = base_train.prepare_model(args, all_train_sets[0], specs)

    print('Method: Learning without Forgetting (LwF)')
    print(
        'LwF weight: {}; temperature: {}'.format(
            args.lwf_weight,
            args.lwf_temperature,
        )
    )
    print(
        'Update mode: partial shared CLIP blocks; trainable blocks: {}/12 '
        'after domain 1'.format(args.continual_trainable_vision_blocks)
    )

    if args.testing:
        base_train.evaluate_checkpoint(
            args,
            model,
            all_train_sets,
            all_test_sets,
            training_set,
            specs,
            result_logger,
            adapter_routing_bank={},
        )
        print('LwF checkpoint evaluation finished.')
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
            '[LwF] Training domain {}/{}: {}'.format(
                domain_index + 1,
                len(all_train_sets),
                name,
            )
        )
        print('=' * 80)

        teacher_model = None
        if domain_index > 0:
            teacher_model = base_train._freeze_teacher(model)
            total_classes = base_train.expand_model_for_domain(
                model,
                args,
                name,
                spec,
                add_num,
                num_classes,
            )
            print(
                'Frozen previous model and expanded the student to {} '
                'identities.'.format(total_classes)
            )

        class_centers = base_train.initial_classifier(model, init_loader)
        model_base = base_train._base_model(model)
        if domain_index == 0:
            model_base.classifier.weight.data.copy_(class_centers)
        else:
            model_base.classifier.weight.data[add_num:].copy_(class_centers)

        trainable_blocks = (
            args.first_domain_trainable_vision_blocks
            if domain_index == 0
            else args.continual_trainable_vision_blocks
        )
        base_train._set_stage2_mode(model, trainable_blocks)
        print('Trainable CLIP vision blocks: {}/12'.format(trainable_blocks))

        optimizer = base_train.make_optimizer_stage2(
            model_base,
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
        trainer = LwFTrainer(
            cfg=cfg,
            model=model,
            num_classes=total_classes,
            teacher_model=teacher_model,
            global_loss_weight=args.global_loss_weight,
            anchor_temperature=args.anchor_temperature,
            lwf_weight=args.lwf_weight,
            lwf_temperature=args.lwf_temperature,
            classifier_scope=args.classifier_scope,
        )

        epochs = args.epochs0 if domain_index == 0 else args.epochs
        for epoch in tqdm(
            range(epochs),
            desc='[LwF-{}]'.format(name),
        ):
            stage2_loader.new_epoch()
            trainer.train(
                stage2_loader,
                optimizer,
                train_iters=len(stage2_loader),
                add_num=add_num,
            )
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
                        'method': 'lwf',
                        'epoch': epoch + 1,
                        'domain_index': domain_index,
                        'domain_name': name,
                        'lwf_weight': args.lwf_weight,
                        'lwf_temperature': args.lwf_temperature,
                        'global_loss_weight': args.global_loss_weight,
                        'anchor_temperature': args.anchor_temperature,
                        'stage2_base_lr': args.stage2_base_lr,
                        'classifier_lr_multiplier': (
                            args.classifier_lr_multiplier
                        ),
                        'classifier_scope': args.classifier_scope,
                        'eval_descriptor': args.eval_descriptor,
                        'first_domain_trainable_vision_blocks': (
                            args.first_domain_trainable_vision_blocks
                        ),
                        'continual_trainable_vision_blocks': (
                            args.continual_trainable_vision_blocks
                        ),
                        'continual_update_mode': 'partial_blocks',
                        'visual_anchor_mode': 'text',
                        'attr_distill_mode': 'none',
                        'adapter_domains': [],
                        'adapter_routing_bank': {},
                    },
                    True,
                    fpath=osp.join(
                        output_dir,
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
                adapter_routing_bank={},
            )

    print('=' * 80)
    print('LwF continual training finished.')
    print('=' * 80)


if __name__ == '__main__':
    arguments = build_parser().parse_args()
    validate_args(arguments)
    base_train.set_seed(arguments.seed)
    cfg.merge_from_file(arguments.config_file)
    main_worker(arguments)
