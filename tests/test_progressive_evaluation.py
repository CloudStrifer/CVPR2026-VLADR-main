import copy
import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from test_category_stream import StreamFixture
from test_category_training import tiny_model
from test_progressive_resume import assert_same
from reid.evaluation.prototype_router import PrototypeRouter, RoutedEncoder
from reid.evaluation.category_progressive import EvaluationConfig, evaluate_stage, lifelong_summary, routing_diagnostics, _retrieval_report
from reid.memory.prototype_views import identity_mean_summaries, control_view
from reid.utils.progressive_checkpoint import capture_rng
from train_category_progressive import ProgressiveRun
from tools.evaluate_category_progressive import main as evaluate_checkpoint
from tools.run_category_ablations import ABLATIONS, run_suite
from tools.diagnose_pgca_validation import diagnose


class RouterTests(unittest.TestCase):
    def test_formula_modes_max_axis_beta_and_stable_ties(self):
        summaries = dict(b=dict(category_prototype=torch.tensor([0., 1.]), mode_prototypes=torch.tensor([[0., 1.]])),
                         a=dict(category_prototype=torch.tensor([1., 0.]), mode_prototypes=torch.eye(2)))
        vectors = torch.tensor([[0., 2.], [2., 0.], [-1., 0.]])
        scores = PrototypeRouter(summaries, .25).scores(vectors)
        self.assertTrue(torch.allclose(scores, torch.tensor([[.75, 1.], [1., 0.], [-.25, 0.]])))
        self.assertEqual(PrototypeRouter(summaries, 0).predict(vectors[:1])[0], ('a',))
        self.assertEqual(PrototypeRouter(summaries, 1).predict(vectors[:1])[0], ('b',))

    def test_invalid_vectors_empty_bank_and_beta(self):
        summaries = dict(a=dict(category_prototype=torch.tensor([1., 0.]), mode_prototypes=torch.tensor([[1., 0.]])))
        for beta in (-1, 2, float('nan')):
            with self.assertRaises(ValueError):
                PrototypeRouter(summaries, beta)
        with self.assertRaises(ValueError):
            PrototypeRouter({})
        for vector in (torch.zeros(1, 2), torch.tensor([[float('nan'), 1.]])):
            with self.assertRaises(ValueError):
                PrototypeRouter(summaries).predict(vector)

    def test_image_only_grouped_forward_restores_mixed_order(self):
        class Model(torch.nn.Module):
            categories, feature_dim, reference_signature = ('a', 'b'), 2, 'ref'
            def __init__(self):
                super().__init__()
                self.calls = []
            def encode_reference(self, images):
                return images[:, 0, 0, :]
            def encode_category(self, images, category):
                self.calls.append((category, len(images)))
                return images[:, 0, 0, :] + (torch.tensor([0., 2.]) if category == 'a' else torch.tensor([3., 0.]))
        class Memory:
            def summary(self):
                return dict(reference_signature='ref')
            def snapshot(self):
                return {c: dict(category_prototype=v, mode_prototypes=v[None])
                        for c, v in zip(('a', 'b'), torch.eye(2))}
        model = Model()
        encoder = RoutedEncoder(model, Memory())
        images = torch.tensor([[0., 1.], [1., 0.], [0., 2.]]).reshape(3, 1, 1, 2)
        features, prediction, _ = encoder(images)
        self.assertEqual(prediction, ('b', 'a', 'b'))
        self.assertEqual(model.calls, [('a', 1), ('b', 2)])
        self.assertTrue(torch.allclose(features, torch.nn.functional.normalize(torch.tensor([[3., 1.], [1., 2.], [3., 2.]]), dim=1)))
        self.assertEqual(list(inspect.signature(encoder.__call__).parameters), ['images'])
        self.assertTrue(model.training)

    def test_identity_mean_is_identity_weighted_and_does_not_use_cluster_sizes(self):
        rows = [dict(identity_key=('a', 's', str(i)), vector=v) for i, v in enumerate(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]))]
        summary = identity_mean_summaries(rows)['a']
        self.assertTrue(torch.allclose(summary['category_prototype'], torch.tensor([2., 1.]) / 5 ** .5))
        self.assertEqual(summary['mode_prototypes'].shape, (1, 2))


class ProgressiveEvaluationTests(StreamFixture):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        super().setUp()
        self.config['evaluation'] = []
        for category in ('person', 'vehicle', 'panda'):
            for split in ('test', 'validation'):
                name = category + '_' + split
                for subset, camera in (('query', 0), ('gallery', 1)):
                    manifest = name + '_' + subset + '.csv'
                    self.rows[manifest] = [dict(path='{}/{}_{}.png'.format(name, pid, subset),
                        original_pid=split + '_' + pid, source_dataset=category + '_source', camid=str(camera)) for pid in ('x', 'y')]
                self.config['evaluation'].append(dict(name=name, category=category, split=split, protocol='cross_camera',
                    query_manifest=name + '_query.csv', gallery_manifest=name + '_gallery.csv'))
        self.stream = self.load()
        samples = [s for stage in self.stream.stages for view in stage.categories for s in view.samples]
        samples += [s for view in self.stream.evaluations for s in view.query + view.gallery]
        for i, sample in enumerate(samples):
            path = Path(sample.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.arange(13 * 17 * 3).reshape(13, 17, 3)
            Image.fromarray(((pixels * (i % 3 + 1) + i * 7) % 256).astype('uint8')).save(path)
        self.settings = dict(stream_config=str(self.config_path), epochs=1, iterations_per_epoch=2,
            batch_size=4, num_instances=2, prototype_batch_size=4, evaluate=True, eval_gallery='both',
            eval_batch_size=4, delta=-1.)

    def fresh(self, name='run', **settings):
        return ProgressiveRun(self.base / name, dict(self.settings, **settings), model_factory=lambda **kw: tiny_model())

    def test_three_stage_metrics_rng_training_independence_and_checkpoint_evaluation(self):
        run = self.fresh()
        run.run()
        self.assertEqual(len(run.evaluations), 3)
        state = torch.load(run.output / 'latest.pt', weights_only=True)
        baseline = self.fresh('no_eval', evaluate=False)
        baseline.run()
        assert_same(self, baseline.model.export_bank(), run.model.export_bank())
        assert_same(self, state['rng'], capture_rng())
        summary = lifelong_summary(run.evaluations)
        matrix = summary['metrics']['per_dataset/prototype/mAP']['performance']
        self.assertEqual(summary['categories'], ['panda', 'person', 'vehicle'])
        self.assertIsNone(matrix[0][0])
        self.assertEqual(len(matrix[2]), 3)
        report = evaluate_checkpoint(['--run-dir', str(run.output), '--output', str(self.base / 'eval.json'), '--batch-size', '4'])
        assert_same(self, report['retrieval'], run.evaluations[-1]['retrieval'])
        self.assertEqual(report['routing'], run.evaluations[-1]['routing'])
        self.assertEqual(run.evaluations[0]['seen_categories'], ['person', 'vehicle'])
        self.assertEqual(run.evaluations[1]['seen_categories'], ['panda', 'person', 'vehicle'])

    def test_evaluation_upgrade_preserves_original_and_resumes_without_retraining_t1(self):
        from tools.upgrade_evaluation_checkpoint import upgrade
        from reid.utils.progressive_checkpoint import atomic_save, file_digest
        run = self.fresh()
        run.run(max_updates=2)
        run.commit_stage()
        old = torch.load(run.output / 'latest.pt', weights_only=True)
        for name in old['runtime']['source_sha256']:
            if name.replace('\\', '/') == 'reid/evaluation/category_oracle.py':
                old['runtime']['source_sha256'][name] = '0' * 64
        atomic_save(old, run.output / 'latest.pt')
        before = {str(p.relative_to(run.output)): file_digest(p) for p in run.output.rglob('*') if p.is_file()}
        destination = self.base / 'upgraded'
        upgrade(run.output, destination)
        after = {str(p.relative_to(run.output)): file_digest(p) for p in run.output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        new = torch.load(destination / 'latest.pt', weights_only=True)
        for key in ('bank', 'ecpm', 'rng', 'settings', 'reports', 'total_updates', 'pending_evaluation'):
            assert_same(self, old[key], new[key])
        restored = ProgressiveRun(destination, resume=True)
        with patch('reid.trainer_category_resumable.ResumableCategoryTrainer.train_step', side_effect=AssertionError('retrained T1')):
            restored.evaluate_pending_stage()
        restored.run()
        baseline = self.fresh('baseline')
        baseline.run()
        self.assertEqual(restored.total_updates, 6)
        assert_same(self, restored.model.export_bank(), baseline.model.export_bank())
        for actual, expected in zip(restored.evaluations, baseline.evaluations):
            assert_same(self, actual['retrieval'], expected['retrieval'])

    def test_evaluation_upgrade_rejects_other_source_environment_and_non_boundary(self):
        from tools.upgrade_evaluation_checkpoint import validate_upgrade
        run = self.fresh()
        run.run(max_updates=2)
        run.commit_stage()
        state = torch.load(run.output / 'latest.pt', weights_only=True)
        runtime = copy.deepcopy(state['runtime'])
        with self.assertRaisesRegex(ValueError, 'already matches'):
            validate_upgrade(state, runtime)
        for kind in ('training_code', 'environment', 'phase', 'evaluated', 'inventory'):
            old = copy.deepcopy(state)
            if kind == 'training_code':
                old['runtime']['source_sha256']['train_category_progressive.py'] = '0' * 64
            elif kind == 'environment':
                old['runtime']['torch'] = 'different'
            elif kind == 'phase':
                old['phase'] = 'training'
            elif kind == 'evaluated':
                old['pending_evaluation'] = None
            else:
                old['runtime']['source_sha256']['extra.py'] = '0' * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_upgrade(old, runtime)

    def test_evaluation_upgrade_bad_log_does_not_publish_or_modify_original(self):
        from tools.upgrade_evaluation_checkpoint import upgrade
        from reid.utils.progressive_checkpoint import atomic_save, file_digest
        run = self.fresh()
        run.run(max_updates=2)
        run.commit_stage()
        state = torch.load(run.output / 'latest.pt', weights_only=True)
        for name in state['runtime']['source_sha256']:
            if name.replace('\\', '/') == 'reid/evaluation/category_oracle.py':
                state['runtime']['source_sha256'][name] = '0' * 64
        atomic_save(state, run.output / 'latest.pt')
        log = run.output / 'progress.jsonl'
        log.write_bytes(b'X' + log.read_bytes()[1:])
        before = {str(p.relative_to(run.output)): file_digest(p) for p in run.output.rglob('*') if p.is_file()}
        destination = self.base / 'upgraded'
        with self.assertRaisesRegex(ValueError, 'prefix changed'):
            upgrade(run.output, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(before, {str(p.relative_to(run.output)): file_digest(p) for p in run.output.rglob('*') if p.is_file()})

    def test_evaluation_failure_resumes_without_retraining_or_duplicate_commit(self):
        run = self.fresh()
        with patch('train_category_progressive.evaluate_stage', side_effect=RuntimeError('evaluation crashed')):
            with self.assertRaises(RuntimeError):
                run.run(max_stages=1)
        restored = ProgressiveRun(run.output, resume=True)
        self.assertEqual(restored.pending_evaluation, 0)
        self.assertEqual(restored.memory.processed_stages, ('t1',))
        with patch('reid.trainer_category_resumable.ResumableCategoryTrainer.train_step', side_effect=AssertionError('retrained')):
            restored.evaluate_pending_stage()
        self.assertEqual(len(restored.evaluations), 1)
        restored.run()
        self.assertEqual(restored.total_updates, 6)
        self.assertEqual([e['stage_id'] for e in restored.evaluations], ['t1', 't2', 't3'])

    def test_evaluation_write_before_checkpoint_is_rolled_back(self):
        run = self.fresh()
        run.run(max_updates=2)
        run.commit_stage()
        with patch.object(run, 'save', side_effect=OSError('save failed')):
            with self.assertRaises(OSError):
                run.evaluate_pending_stage()
        restored = ProgressiveRun(run.output, resume=True)
        self.assertEqual(restored.evaluations, [])
        self.assertTrue(list(run.output.glob('evaluation.jsonl.recovered-*')))
        restored.evaluate_pending_stage()
        self.assertEqual(len((run.output / 'evaluation.jsonl').read_text(encoding='utf-8').splitlines()), 1)

    def test_missing_coverage_and_future_images_are_not_silently_used(self):
        self.config['evaluation'] = [e for e in self.config['evaluation'] if e['category'] != 'panda']
        self.write()
        with self.assertRaisesRegex(ValueError, 'missing test evaluation'):
            self.fresh()

    def test_only_arrived_train_and_seen_evaluation_images_are_opened(self):
        run = self.fresh()
        real_open = Image.open
        def guarded(path, *args, **kwargs):
            if run.pending_evaluation is not None:
                views = self.stream.evaluations_at(self.stream.stages[run.pending_evaluation].stage_id)
                allowed = {s.path for v in views for s in v.query + v.gallery}
            else:
                allowed = {s.path for v in run.stage.categories for s in v.samples}
            self.assertIn(str(path), allowed)
            return real_open(path, *args, **kwargs)
        with patch('PIL.Image.open', side_effect=guarded):
            run.run()

    def test_forgetting_uses_historical_max_not_only_previous_stage(self):
        run = self.fresh()
        run.run()
        history = copy.deepcopy(run.evaluations)
        for report, value in zip(history, (80., 40., 60.)):
            report['retrieval']['per_dataset']['prototype']['per_category']['person']['mAP'] = value
        summary = lifelong_summary(history)
        column = summary['categories'].index('person')
        self.assertEqual([row[column] for row in summary['metrics']['per_dataset/prototype/mAP']['forgetting']], [0., 40., 20.])
        history[2]['split'] = 'validation'
        with self.assertRaises(ValueError):
            lifelong_summary(history)

    def test_routing_consistency_is_not_accuracy(self):
        samples = self.stream.evaluations[0].query + self.stream.evaluations[0].gallery
        report = routing_diagnostics(samples, ['vehicle'] * len(samples), ['person', 'vehicle'])
        self.assertEqual(report['accuracy'], 0)
        self.assertEqual(report['identity_all_same_percent'], 100)
        self.assertEqual(report['identity_modal_fraction'], 1)

    def test_mixed_gallery_actually_adds_other_category_distractors(self):
        views = self.stream.evaluations_at('t1')
        features = {v.name: dict(query=torch.eye(2), gallery=torch.eye(2)) for v in views}
        local = _retrieval_report(views, features, 'prototype', 'per_dataset')
        mixed = _retrieval_report(views, features, 'prototype', 'mixed')
        self.assertEqual(local['macro']['mAP'], 100)
        self.assertEqual(mixed['per_category']['person']['mAP'], 100)
        self.assertEqual(mixed['per_category']['vehicle']['mAP'], 50)
        self.assertEqual(mixed['per_category']['vehicle']['Rank1'], 0)
        self.assertEqual(mixed['macro']['mAP'], 75)

    def test_all_ablation_recipes_execute_and_resume(self):
        plan, results = run_suite(self.settings, self.base / 'suite', execute=True, model_factory=lambda **kw: tiny_model())
        self.assertEqual(set(results), set(ABLATIONS))
        self.assertEqual(plan['experiments']['global_only_initialization']['alpha'], 1)
        self.assertEqual(plan['experiments']['mean_control_only']['control_summary'], 'identity_mean')
        _, restored = run_suite(self.settings, self.base / 'suite', execute=True, model_factory=lambda **kw: tiny_model())
        self.assertEqual(results, restored)

    def test_mean_control_resume_uses_saved_mean_drift(self):
        run = self.fresh(control_summary='identity_mean', init_mode='random')
        run.run(max_updates=3)
        saved = run.trainer.state_dict()
        restored = ProgressiveRun(run.output, resume=True)
        self.assertEqual(restored.trainer.consistency.metadata(), saved['pgca'])
        self.assertEqual(restored.trainer.initialization.metadata(), saved['initialization'])
        old_candidate = copy.deepcopy(restored.candidate)
        control_view(restored.memory, restored.candidate, 'identity_mean')
        assert_same(self, old_candidate, restored.candidate)
        restored.run()

    def test_source_and_drift_diagnostics_use_one_boundary_validation_only(self):
        run = self.fresh(eval_split='validation')
        run.run(max_stages=1)
        before = (run.output / 'latest.pt').read_bytes()
        for kind in ('sources', 'drift'):
            report = diagnose(run.output, self.base / (kind + '.json'), kind, weights=(0., 1.))
            self.assertTrue(report['completed'])
            self.assertTrue(all(r['split'] == 'validation' for r in report['empirical_rows']))
            if kind == 'sources':
                self.assertEqual(set(report['variants']), {'default', 'similarity', 'source_person', 'source_vehicle'})
            repeated = diagnose(run.output, self.base / (kind + '.json'), kind, weights=(0., 1.))
            self.assertEqual(report, repeated)
        self.assertEqual(before, (run.output / 'latest.pt').read_bytes())


if __name__ == '__main__':
    unittest.main()
