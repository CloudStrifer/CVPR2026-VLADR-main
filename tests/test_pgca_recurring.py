import copy
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from test_category_stream import StreamFixture
from test_category_training import tiny_model, fixed_tensor
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.adaptation.pgca import FrozenCategoryTeacher
from reid.loss.pgca import PGCAConsistencyConfig, FeatureConsistencyLoss, consistency_weight
from reid.memory import ECPMMemory
from reid.memory.ecpm_modes import _same
from reid.models.category_adapter_bank import _tensor_digest
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig
from tools.train_pgca_recurring_stage import main as run_stage


class ConsistencyFormulaTests(unittest.TestCase):
    def test_cosine_hand_values_and_stop_gradient(self):
        student = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], requires_grad=True)
        teacher = torch.tensor([[1., 0.]] * 3, requires_grad=True)
        loss = FeatureConsistencyLoss()(student, teacher)
        self.assertAlmostEqual(loss.item(), 1.)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum(), 0.)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_identical_features_and_positive_scaling(self):
        features = torch.tensor([[1., 2.], [3., 1.]], requires_grad=True)
        loss = FeatureConsistencyLoss()(features, features.detach() * 2)
        self.assertLess(abs(loss.item()), 1e-6)

    def test_weight_formula_monotonicity_and_ablation(self):
        config = PGCAConsistencyConfig('drift', 3., 2.)
        self.assertAlmostEqual(consistency_weight(config, .5), 3 * math.exp(-1))
        self.assertGreater(consistency_weight(config, .1), consistency_weight(config, 1.))
        for d in (0., .5, 2.):
            self.assertEqual(consistency_weight(PGCAConsistencyConfig('drift', 3., 0.), d), 3.)
            self.assertEqual(consistency_weight(PGCAConsistencyConfig('fixed', 3., 2.), d), 3.)
            self.assertEqual(consistency_weight(PGCAConsistencyConfig('drift', 0., 2.), d), 0.)
        self.assertEqual(consistency_weight(config, recurring=False), 0.)
        self.assertEqual(consistency_weight(PGCAConsistencyConfig()), 0.)

    def test_invalid_features_drifts_and_config_fail(self):
        for features in (torch.zeros(2, 2), torch.full((2, 2), float('nan')), torch.full((2, 2), 1e38)):
            with self.assertRaises(FloatingPointError):
                FeatureConsistencyLoss()(features, torch.ones(2, 2))
        with self.assertRaises(ValueError):
            FeatureConsistencyLoss()(torch.ones(2, 2), torch.ones(2, 3))
        for d in (-1., 2.1, float('nan'), None):
            with self.assertRaises(ValueError):
                consistency_weight(PGCAConsistencyConfig('drift'), d)
        for options in (dict(mode='bad'), dict(lambda_con=-1.), dict(gamma=float('inf'))):
            with self.assertRaises(ValueError):
                PGCAConsistencyConfig(**options)


class PGCARecurringTests(StreamFixture):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.stream = self.load()
        self.model = tiny_model()
        self.model.add_category('person')
        self.model.add_category('vehicle')
        # Distinct, already-learned adapters ensure teacher routing is observable.
        with torch.no_grad():
            for category in self.model.categories:
                for p in self.model.adapter_parameters(category):
                    p.add_(torch.randn_like(p) * .04)
        self.memory = ECPMMemory(self.model, self.stream)
        self.memory.commit_stage(self.memory.prepare_identity_stage(self.identity_candidate(0)))
        self.stage = self.stream.stages[1]
        self.candidate = self.memory.prepare_identity_stage(self.identity_candidate(1))

    def identity_candidate(self, index):
        stage = self.stream.stages[index]
        rows = []
        for view in stage.categories:
            vector = torch.nn.functional.normalize(torch.tensor([1., .5 * index, 0., 0.]), dim=0)
            for key in view.identity_keys:
                rows.append(dict(identity_key=key, vector=vector.clone(), first_stage=stage.stage_id,
                                 image_count=sum(s.identity_key == key for s in view.samples)))
        return dict(binding=copy.deepcopy(self.memory._binding), base_stages=self.memory.processed_stages,
                    stage_id=stage.stage_id, rows=rows)

    def loaders(self, stage=None):
        stage = stage or self.stage
        for view in stage.categories:
            for i, sample in enumerate(view.samples):
                path = Path(sample.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (4, 4), (30 + i * 40, 20, 200 - i * 30)).save(path)
        return build_stage_loaders(stage, batch_size=4, num_instances=2, seed=17,
                                   train_transform=fixed_tensor, reference_transform=fixed_tensor)

    def trainer(self, model=None, mode='drift', lambda_con=2., gamma=3., name='train', **options):
        return CategoryProgressiveTrainer(model or self.model, self.stage, self.loaders(),
            CategoryTrainingConfig(epochs=1, iterations_per_epoch=3, adapter_lr=.003, head_lr=.01, **options),
            self.base / (name + '.jsonl'), consistency=PGCAConsistencyConfig(mode, lambda_con, gamma),
            ecpm_memory=self.memory, ecpm_candidate=self.candidate)

    def test_teacher_snapshot_independent_frozen_same_category_and_no_rng_use(self):
        state = torch.get_rng_state().clone()
        teacher = FrozenCategoryTeacher(self.model, ('person',))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        probe = torch.randn(2, 3, 4, 4)
        before = self.model.encode_category(probe, 'person').detach()
        self.assertTrue(torch.equal(before, teacher.encode(probe, 'person')))
        self.assertFalse(torch.allclose(before, self.model.encode_category(probe, 'vehicle')))
        with torch.no_grad():
            next(iter(self.model.adapter_parameters('person'))).add_(1.)
        self.assertTrue(torch.equal(before, teacher.encode(probe, 'person')))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.model.parameters()))
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(self.model.parameters(), teacher.model.parameters())))
        teacher.assert_unchanged()
        with self.assertRaises(ValueError):
            teacher.encode(probe, 'vehicle')

    def test_same_augmented_tensor_used_and_new_category_has_no_teacher(self):
        trainer = self.trainer()
        seen = []
        original_student = self.model.encode_category
        original_teacher = trainer.consistency.teacher.encode
        def student(images, category):
            if category == 'person':
                seen.append(images)
            return original_student(images, category)
        def teacher(images, category):
            self.assertEqual(category, 'person')
            self.assertIs(images, seen[-1])
            return original_teacher(images, category)
        with patch.object(self.model, 'encode_category', side_effect=student), patch.object(
                trainer.consistency.teacher, 'encode', side_effect=teacher) as mocked:
            report = trainer.train_epoch()
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(report['categories']['panda']['weighted_consistency'], 0.)
        self.assertIsNone(report['categories']['panda']['prototype_drift'])

    def test_training_logs_loss_equation_and_preserves_teacher_reference_and_absent(self):
        reference = self.model.reference_signature
        vehicle = self.model.export_adapter('vehicle')
        old_person = self.model.export_adapter('person')
        trainer = self.trainer()
        teacher = trainer.consistency.teacher
        old_memory = self.memory.state_dict()
        report = trainer.train_epoch()
        for category, values in report['categories'].items():
            self.assertAlmostEqual(values['total'], values['ce'] + values['weighted_triplet'] + values['weighted_consistency'], places=6)
            self.assertAlmostEqual(values['weighted_consistency'], values['consistency'] * values['consistency_weight'], places=6)
        self.assertGreater(report['categories']['person']['consistency'], 0.)
        self.assertTrue(_same(vehicle, self.model.export_adapter('vehicle')))
        self.assertFalse(_same(old_person, self.model.export_adapter('person')))
        self.model.assert_reference_unchanged()
        self.assertEqual(reference, self.model.reference_signature)
        teacher.assert_unchanged()
        self.assertTrue(_same(old_memory, self.memory.state_dict()))
        final = trainer.finish_stage()
        self.assertIsNone(trainer.consistency.teacher)
        self.assertEqual(final['pgca']['teacher']['categories'], ['person'])
        events = [json.loads(line) for line in (self.base / 'train.jsonl').read_text().splitlines()]
        self.assertEqual(events[0]['loss_components'], ['ce', 'triplet', 'consistency'])
        self.assertEqual(len([e for e in events if e['event'] == 'train_step']), 3)

    def test_zero_weight_and_off_match_baseline_parameter_updates_exactly(self):
        models = [copy.deepcopy(self.model) for _ in range(3)]
        states = []
        for i, (model, mode, weight) in enumerate(zip(models, ('off', 'drift', 'fixed'), (2., 0., 0.))):
            torch.manual_seed(81)
            trainer = (CategoryProgressiveTrainer(model, self.stage, self.loaders(),
                CategoryTrainingConfig(epochs=1, iterations_per_epoch=3, adapter_lr=.003, head_lr=.01))
                if i == 0 else self.trainer(model, mode=mode, lambda_con=weight, name=str(i)))
            self.assertIsNone(trainer.consistency.teacher)
            trainer.train_epoch()
            trainer.finish_stage()
            states.append(copy.deepcopy(model.state_dict()))
        self.assertTrue(_same(states[0], states[1]))
        self.assertTrue(_same(states[0], states[2]))

    def test_gamma_zero_matches_fixed_parameter_updates(self):
        states = []
        for i, mode in enumerate(('fixed', 'drift')):
            model = copy.deepcopy(self.model)
            torch.manual_seed(81)
            trainer = self.trainer(model, mode=mode, gamma=0., name=str(i))
            trainer.train_epoch()
            trainer.finish_stage()
            states.append(model.state_dict())
        self.assertTrue(_same(states[0], states[1]))

    def test_consistency_alone_has_nonzero_student_adapter_gradient(self):
        trainer = self.trainer()
        probe = torch.randn(4, 3, 4, 4)
        with torch.no_grad():
            for p in self.model.adapter_parameters('person'):
                p.add_(torch.randn_like(p) * .03)
        student = self.model.encode_category(probe, 'person')
        target = trainer.consistency.teacher_features(probe, 'person')
        loss = trainer.consistency_loss(student, target)
        loss.backward()
        gradients = [p.grad for p in self.model.adapter_parameters('person') if p.grad is not None]
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        trainer.consistency.assert_teacher_unchanged()

    def test_candidate_copy_cannot_change_stage_weight(self):
        trainer = self.trainer()
        weight = trainer.consistency.weight('person')
        self.candidate['drifts']['person'] = 2.
        trainer.consistency.metadata()['weights']['person'] = 999.
        self.assertEqual(trainer.consistency.weight('person'), weight)
        self.assertEqual(trainer.train_epoch()['categories']['person']['consistency_weight'], weight)

    def test_invalid_or_committed_candidate_rejected_before_student_mutation(self):
        old = _tensor_digest(self.model.state_dict())
        self.candidate['drifts']['person'] = 2.
        with self.assertRaises(ValueError):
            self.trainer()
        self.assertEqual(old, _tensor_digest(self.model.state_dict()))
        self.assertEqual(len(self.model.temporary_heads), 0)
        self.candidate = self.memory.prepare_identity_stage(self.identity_candidate(1))
        self.memory.commit_stage(self.candidate)
        with self.assertRaises(ValueError):
            self.trainer()

    def test_drift_mode_needs_ecpm_but_fixed_can_run_without_it(self):
        with self.assertRaisesRegex(ValueError, 'requires validated'):
            CategoryProgressiveTrainer(self.model, self.stage, self.loaders(), consistency=PGCAConsistencyConfig('drift'))
        trainer = CategoryProgressiveTrainer(self.model, self.stage, self.loaders(),
            CategoryTrainingConfig(epochs=1, iterations_per_epoch=1), consistency=PGCAConsistencyConfig('fixed'))
        self.assertIsNone(trainer.consistency.drift('person'))
        trainer.train_epoch()
        trainer.finish_stage()

    def test_teacher_corruption_stops_before_any_update(self):
        trainer = self.trainer()
        old = _tensor_digest(self.model.state_dict())
        with torch.no_grad():
            trainer.consistency.teacher.model.visual.proj.add_(1.)
        with self.assertRaisesRegex(RuntimeError, 'teacher changed'):
            trainer.train_epoch()
        self.assertEqual(old, _tensor_digest(self.model.state_dict()))
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertTrue(trainer.failed)

    def test_no_historical_images_read(self):
        trainer = self.trainer()
        current = {s.path for v in self.stage.categories for s in v.samples}
        original = Image.open
        def checked(path, *args, **kwargs):
            self.assertIn(str(path), current)
            return original(path, *args, **kwargs)
        with patch('PIL.Image.open', side_effect=checked):
            trainer.train_epoch()

    def test_invalid_teacher_features_abort_before_optimizer_step(self):
        trainer = self.trainer()
        before = _tensor_digest(self.model.state_dict())
        old_memory = self.memory.state_dict()
        with patch.object(trainer.consistency.teacher, 'encode',
                          return_value=torch.full((4, self.model.feature_dim), float('nan'))):
            with self.assertRaises(FloatingPointError):
                trainer.train_epoch()
        self.assertTrue(trainer.failed)
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertEqual(before, _tensor_digest(self.model.state_dict()))
        self.assertTrue(_same(old_memory, self.memory.state_dict()))

    def test_cli_imports_baseline_plus_ecpm_and_rejects_wrong_memory_stage(self):
        self.loaders()
        baseline, memory_path = self.base / 'baseline.pt', self.base / 'memory.pt'
        torch.save(dict(kind='category_ce_triplet_baseline', schema_version=1, completed=True, stage_id='t1',
                        stream_fingerprint=self.stream.fingerprint, model=self.model.export_checkpoint()), baseline)
        self.memory.save(memory_path)
        args = ['--stream-config', str(self.config_path), '--stage-id', 't2', '--epochs', '1',
                '--iterations-per-epoch', '1', '--batch-size', '4', '--num-instances', '2',
                '--previous-checkpoint', str(baseline), '--previous-memory', str(memory_path)]
        report = run_stage(args + ['--output-dir', str(self.base / 'imported')])
        self.assertEqual(report['training']['pgca']['teacher']['categories'], ['person'])
        self.memory.commit_stage(self.candidate)
        self.memory.save(memory_path)
        with self.assertRaisesRegex(ValueError, 'end exactly'):
            run_stage(args + ['--output-dir', str(self.base / 'wrong')])

    def test_cli_first_and_second_stage_combined_checkpoint(self):
        self.loaders(self.stream.stages[0])
        self.loaders(self.stream.stages[1])
        common = ['--stream-config', str(self.config_path), '--epochs', '1', '--iterations-per-epoch', '2',
                  '--batch-size', '4', '--num-instances', '2', '--consistency', 'drift']
        first, second = self.base / 'run1', self.base / 'run2'
        run_stage(common + ['--stage-id', 't1', '--output-dir', str(first)], model_factory=lambda **kwargs: tiny_model())
        report = run_stage(common + ['--stage-id', 't2', '--output-dir', str(second),
                                    '--previous-checkpoint', str(first / 'completed_stage.pt')])
        checkpoint = torch.load(second / 'completed_stage.pt', weights_only=True)
        self.assertTrue(checkpoint['completed'])
        self.assertEqual(checkpoint['ecpm']['identity_memory']['processed_stages'], ('t1', 't2'))
        self.assertEqual(report['training']['pgca']['teacher']['categories'], ['person'])
        self.assertEqual(report['ecpm']['identities'], 8)


if __name__ == '__main__':
    unittest.main()
