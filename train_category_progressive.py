"""Continuous ECPM + PGCA training with transactional optimizer-step resume."""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lreid_dataset.category_stream import load_category_stream
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.adaptation.pgca import PGCATransferConfig
from reid.loss.pgca import PGCAConsistencyConfig
from reid.memory import ECPMMemory, FinchConfig
from reid.models.category_adapter_bank import CategoryAdapterBank, build_category_model
from reid.trainer_category_progressive import CategoryTrainingConfig
from reid.trainer_category_resumable import ResumableCategoryTrainer, model_from_bank
from reid.evaluation.category_progressive import EvaluationConfig, check_evaluation_coverage, evaluate_stage, lifelong_summary
from reid.utils.progressive_checkpoint import (append_event, atomic_save, capture_rng, exact_runtime,
    file_digest, log_positions, restore_logs, restore_rng)


DEFAULTS = dict(device='cpu', reference_checkpoint=None, epochs=10, iterations_per_epoch=100,
    batch_size=32, num_instances=4, prototype_batch_size=128, workers=0, seed=42,
    adapter_lr=.0003, head_lr=.0003, weight_decay=.0001, lambda_tri=1., triplet_margin=.3,
    amp=False, max_grad_norm=None, consistency='drift', lambda_con=1., gamma=1.,
    init_mode='similarity', alpha=.5, delta=.5, finch_chunk_size=256, control_summary='ecpm',
    transfer_source=None, evaluate=False, eval_split='test', eval_gallery='per_dataset',
    routing_summary='ecpm', beta=.5, eval_batch_size=128)


class ProgressiveRun:
    """Only latest.pt is authoritative; logs and stage commits follow its cursor."""

    def __init__(self, output, settings=None, resume=False, model_factory=build_category_model):
        self.output = Path(output).resolve()
        saved = torch.load(self.output / 'latest.pt', map_location='cpu', weights_only=True) if resume else None
        if saved is not None:
            if saved.get('kind') != 'ecpm_pgca_continuous' or saved.get('schema_version') != 1:
                raise ValueError('expected step-8 continuous checkpoint')
            self.settings = dict(saved['settings'])
            for key, value in (settings or {}).items():
                if self.settings.get(key) != value:
                    raise ValueError('resume setting changed: {}'.format(key))
        else:
            self.settings = dict(DEFAULTS, **(settings or {}))
        args = self.settings
        if args['workers'] != 0:
            raise ValueError('exact resume requires --workers 0 (no asynchronous prefetch)')
        for key in ('batch_size', 'num_instances', 'prototype_batch_size', 'finch_chunk_size'):
            if type(args[key]) is not int or args[key] <= 0:
                raise ValueError('{} must be a positive integer'.format(key))
        if args['num_instances'] < 2 or args['batch_size'] % args['num_instances'] or args['batch_size'] // args['num_instances'] < 2:
            raise ValueError('training requires P >= 2 and K >= 2 with batch_size=P*K')
        self.training = CategoryTrainingConfig(**{k: args[k] for k in CategoryTrainingConfig.__dataclass_fields__})
        self.con = PGCAConsistencyConfig(args['consistency'], args['lambda_con'], args['gamma'], args['control_summary'])
        self.init = PGCATransferConfig(args['init_mode'], args['alpha'], args['delta'], args['control_summary'],
                                       args['seed'], args['transfer_source'])
        self.evaluation_config = EvaluationConfig(args['eval_split'], args['eval_gallery'], args['beta'],
                                                   args['routing_summary'], args['eval_batch_size'])
        self.stream = load_category_stream(args['stream_config'])
        if any(len(view.identity_keys) < args['batch_size'] // args['num_instances']
               for stage in self.stream.stages for view in stage.categories):
            raise ValueError('a stage/category has fewer identities than P')
        self.runtime = exact_runtime(args['device'])
        self.log_names = ['progress.jsonl'] + ['stage_{:04d}.jsonl'.format(i) for i in range(len(self.stream.stages))]
        if args['evaluate']:
            # Audit metadata for complete coverage before starting expensive work.
            check_evaluation_coverage(self.stream, self.stream.stages[-1].stage_id, self.evaluation_config)
            self.log_names.append('evaluation.jsonl')
        self.evaluations, self.pending_evaluation = [], None
        self.trainer, self.candidate = None, None
        self.stage_seconds = dict(preparation=0., optimizer_updates=0.)
        if saved is None:
            if self.output.exists() and any(self.output.iterdir()):
                raise FileExistsError('new run needs an empty output directory; use --resume for latest.pt')
            self.output.mkdir(parents=True, exist_ok=True)
            random.seed(args['seed'])
            np.random.seed(args['seed'])
            torch.manual_seed(args['seed'])
            self.model = model_factory(reference_checkpoint=args['reference_checkpoint'], device=args['device'])
            if self.model.categories or self.model.temporary_heads:
                raise ValueError('a new continuous run requires an empty category bank')
            self.reference = self.model.export_checkpoint()
            atomic_save(self.reference, self.output / 'reference.pt')
            self.reference_sha256 = file_digest(self.output / 'reference.pt')
            self.memory = ECPMMemory(self.model, self.stream, FinchConfig(args['finch_chunk_size']))
            self.stage_index, self.total_updates, self.phase, self.reports = 0, 0, 'between_stages', []
            (self.output / 'run_config.json').write_text(json.dumps(dict(settings=args,
                stream_fingerprint=self.stream.fingerprint, runtime=self.runtime), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            self._event('run_start')
            self.save()
        else:
            if saved['stream_fingerprint'] != self.stream.fingerprint:
                raise ValueError('stream manifests/config changed since checkpoint')
            if saved['runtime'] != self.runtime:
                raise ValueError('exact resume requires identical software, code, device and thread settings')
            self.reference_sha256 = file_digest(self.output / 'reference.pt')
            if self.reference_sha256 != saved['reference_sha256']:
                raise ValueError('reference.pt differs from the checkpoint binding')
            self.reference = torch.load(self.output / 'reference.pt', map_location='cpu', weights_only=True)
            self.stage_index, self.total_updates = saved['stage_index'], saved['total_updates']
            self.phase, self.reports = saved['phase'], saved['reports']
            self.evaluations, self.pending_evaluation = saved['evaluations'], saved['pending_evaluation']
            self.stage_seconds = dict(saved['stage_seconds'])
            n = len(self.stream.stages)
            if (type(self.stage_index) is not int or not 0 <= self.stage_index <= n
                    or self.phase not in ('between_stages', 'training', 'complete')
                    or (self.phase == 'complete') != (self.stage_index == n)
                    or len(self.reports) != self.stage_index):
                raise ValueError('invalid stage cursor/phase')
            self.model = model_from_bank(self.reference, saved['bank'], args['device'])
            self.memory = ECPMMemory.from_state_dict(saved['ecpm'], self.model, self.stream)
            if self.memory.processed_stages != tuple(s.stage_id for s in self.stream.stages[:self.stage_index]):
                raise ValueError('ECPM commit cursor differs from training cursor')
            if self.memory.clustering != FinchConfig(args['finch_chunk_size']):
                raise ValueError('clustering config differs from checkpoint')
            if self.phase == 'training':
                self.candidate = saved['candidate']
                self.trainer = ResumableCategoryTrainer.from_state_dict(saved['trainer'], self.reference,
                    self.stage, self._loaders(), self.memory, self.candidate, args['device'], self.stage_log)
                self.model = self.trainer.model
                if (asdict(self.trainer.config) != asdict(self.training)
                        or self.trainer.consistency.config != self.con or self.trainer.initialization.config != self.init):
                    raise ValueError('trainer config differs from run config')
            elif saved['trainer'] is not None or saved['candidate'] is not None:
                raise ValueError('boundary checkpoint contains live stage state')
            if set(self.model.categories) != set(self.memory.snapshot()) | (
                    {c.category for c in self.stage.categories} if self.phase == 'training' else set()):
                raise ValueError('student bank registry differs from stage progress')
            expected = self.stage_index * self.training.epochs * self.training.iterations_per_epoch
            expected += self.trainer.optimizer_steps if self.trainer is not None else 0
            if type(self.total_updates) is not int or self.total_updates != expected:
                raise ValueError('invalid global optimizer cursor')
            if self.pending_evaluation is not None and (
                    not args['evaluate'] or self.phase == 'training' or self.pending_evaluation != self.stage_index - 1):
                raise ValueError('invalid pending evaluation cursor')
            expected_evaluations = self.stage_index - int(self.pending_evaluation is not None) if args['evaluate'] else 0
            if len(self.evaluations) != expected_evaluations:
                raise ValueError('evaluation history disagrees with commit cursor')
            lifelong_summary(self.evaluations)
            archives = restore_logs(self.output, self.log_names, saved['logs'])
            restore_rng(saved['rng'])
            self._event('run_resume', recovered_log_tails=archives)

    @property
    def stage(self):
        return self.stream.stages[self.stage_index]

    @property
    def stage_log(self):
        return self.output / self.log_names[self.stage_index + 1]

    def _event(self, event, **kwargs):
        append_event(self.output / 'progress.jsonl', dict(event=event, phase=self.phase,
            stage_index=self.stage_index, total_updates=self.total_updates, **kwargs))

    def _loaders(self):
        train, reference = self.model.make_transforms()
        return build_stage_loaders(self.stage, batch_size=self.settings['batch_size'],
            num_instances=self.settings['num_instances'], workers=0, seed=self.settings['seed'],
            train_transform=train, reference_transform=reference)

    def save(self):
        trainer_state = self.trainer.state_dict() if self.trainer is not None else None
        bank = trainer_state['bank'] if trainer_state is not None else self.model.export_bank()
        payload = dict(kind='ecpm_pgca_continuous', schema_version=1, settings=self.settings,
            runtime=self.runtime, stream_fingerprint=self.stream.fingerprint, reference_sha256=self.reference_sha256,
            phase=self.phase, stage_index=self.stage_index, total_updates=self.total_updates,
            bank=bank, ecpm=self.memory.state_dict(), candidate=self.candidate, trainer=trainer_state,
            reports=self.reports, rng=capture_rng(), logs=log_positions(self.output, self.log_names))
        payload['stage_seconds'] = dict(self.stage_seconds)
        payload.update(evaluations=self.evaluations, pending_evaluation=self.pending_evaluation)
        atomic_save(payload, self.output / 'latest.pt')

    def prepare_stage(self):
        if self.phase != 'between_stages' or self.pending_evaluation is not None:
            raise RuntimeError('stage already prepared or run complete')
        started = time.perf_counter()
        self.candidate = self.memory.prepare_stage(self.model, self.stage, self.settings['prototype_batch_size'], 0)
        self.trainer = ResumableCategoryTrainer(self.model, self.stage, self._loaders(), self.training,
            self.stage_log, consistency=self.con, initialization=self.init,
            ecpm_memory=self.memory, ecpm_candidate=self.candidate)
        self.phase = 'training'
        self.stage_seconds = dict(preparation=time.perf_counter() - started, optimizer_updates=0.)
        self._event('stage_prepared', stage_id=self.stage.stage_id,
                    new_categories=self.stage.new_categories, recurring_categories=self.stage.recurring_categories,
                    absent_categories=self.stage.absent_categories, seen_before=self.stage.seen_before,
                    preparation_seconds=self.stage_seconds['preparation'])
        self.save()

    def commit_stage(self):
        if self.phase != 'training':
            raise RuntimeError('no active stage')
        report = self.trainer.finish_stage()
        memory_report = self.memory.commit_stage(self.candidate)
        storage = dict(adapter_tensor_bytes=sum(p.numel() * p.element_size()
            for c in self.model.categories for p in self.model.adapter_parameters(c)),
            reference_file_bytes=(self.output / 'reference.pt').stat().st_size)
        self.reports.append(dict(training=report, ecpm=memory_report, compute_seconds=dict(self.stage_seconds), storage=storage))
        stage_id = self.stage.stage_id
        self.stage_index += 1
        self.trainer, self.candidate = None, None
        self.phase = 'complete' if self.stage_index == len(self.stream.stages) else 'between_stages'
        self.pending_evaluation = self.stage_index - 1 if self.settings['evaluate'] else None
        self._event('stage_committed', stage_id=stage_id, ecpm=memory_report,
                    compute_seconds=dict(self.stage_seconds), storage=storage)
        self.stage_seconds = dict(preparation=0., optimizer_updates=0.)
        self.save()

    def evaluate_pending_stage(self):
        if self.pending_evaluation is None:
            return
        stage_id = self.stream.stages[self.pending_evaluation].stage_id
        report = evaluate_stage(self.model, self.memory, self.stream, stage_id, self.evaluation_config)
        self.evaluations.append(report)
        self.pending_evaluation = None
        append_event(self.output / 'evaluation.jsonl', dict(event='stage_evaluation', **report))
        self._event('stage_evaluated', stage_id=stage_id)
        self.save()

    def write_evaluation_summary(self):
        if self.settings['evaluate']:
            from reid.utils.progressive_checkpoint import atomic_json
            atomic_json(dict(stages=self.evaluations, lifelong=lifelong_summary(self.evaluations)),
                        self.output / 'evaluation_summary.json')

    def run(self, max_updates=None, checkpoint_every=1, max_stages=None):
        for name, value in (('checkpoint_every', checkpoint_every), ('max_updates', max_updates), ('max_stages', max_stages)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError('{} must be positive'.format(name))
        start_updates, start_stage = self.total_updates, self.stage_index
        while self.phase != 'complete' or self.pending_evaluation is not None:
            if self.pending_evaluation is not None:
                self.evaluate_pending_stage()
                self.write_evaluation_summary()
                if self.phase == 'complete':
                    break
            if self.phase == 'between_stages':
                self.prepare_stage()
            while self.trainer.completed_epochs < self.training.epochs:
                started = time.perf_counter()
                event = self.trainer.train_step()
                self.stage_seconds['optimizer_updates'] += time.perf_counter() - started
                self.total_updates += 1
                if self.total_updates % checkpoint_every == 0 or self.trainer.completed_epochs == self.training.epochs:
                    self.save()
                if self.trainer.iteration_in_epoch == 0:
                    print(json.dumps(dict(stage_id=self.stage.stage_id, completed_epochs=self.trainer.completed_epochs,
                        total_updates=self.total_updates, summed_loss=event['summed_loss']), ensure_ascii=False), flush=True)
                if max_updates is not None and self.total_updates - start_updates >= max_updates:
                    self._event('run_paused', reason='max_updates')
                    self.save()
                    return self.status()
            # Last optimizer update was durably saved before changing ECPM.
            self.commit_stage()
            self.evaluate_pending_stage()
            self.write_evaluation_summary()
            if max_stages is not None and self.stage_index - start_stage >= max_stages:
                return self.status()
        self.write_evaluation_summary()
        return self.status()

    def status(self):
        return dict(phase=self.phase, stage_index=self.stage_index, total_updates=self.total_updates,
                    pending_evaluation=self.pending_evaluation, evaluated_stages=len(self.evaluations),
                    checkpoint=str(self.output / 'latest.pt'))


def main(argv=None, model_factory=build_category_model):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--max-stages', type=int)
    parser.add_argument('--checkpoint-every', type=int, default=1)
    parser.add_argument('--stream-config', default=argparse.SUPPRESS)
    for key, default in DEFAULTS.items():
        kwargs = dict(default=argparse.SUPPRESS)
        if isinstance(default, bool):
            kwargs['action'] = 'store_true'
        elif key == 'max_grad_norm':
            kwargs['type'] = float
        elif default is not None:
            kwargs['type'] = type(default)
        parser.add_argument('--' + key.replace('_', '-'), **kwargs)
    args = vars(parser.parse_args(argv))
    output, resume = args.pop('output_dir'), args.pop('resume')
    controls = {k: args.pop(k) for k in ('max_updates', 'max_stages', 'checkpoint_every')}
    if 'stream_config' in args:
        args['stream_config'] = str(Path(args['stream_config']).resolve())
    if not resume and 'stream_config' not in args:
        parser.error('a new run requires --stream-config')
    run = ProgressiveRun(output, args, resume, model_factory)
    result = run.run(**controls)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
