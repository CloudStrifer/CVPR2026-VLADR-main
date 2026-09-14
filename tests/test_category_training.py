import copy
import json
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from test_category_stream import StreamFixture
from lreid_dataset.category_stream import EvaluationView, StreamSample
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.evaluation.category_oracle import evaluate_category_oracle, oracle_retrieval_metrics
from reid.loss.category_triplet import CategoryBatchHardTriplet
from reid.models.category_adapter_bank import CategoryAdapterBank, ReferenceConfig
from reid.models.CLIP_ReID.model.clip.model import VisionTransformer
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig
from tools.train_category_baseline_stage import main as run_stage


def tiny_model():
    return CategoryAdapterBank(VisionTransformer(2, 2, 2, 2, 8, 12, 2, 4),
                               ReferenceConfig('training-test', (4, 4)), bottleneck_dim=2)


def fixed_tensor(image):
    return torch.from_numpy(np.array(image.resize((4, 4)), copy=True)).permute(2, 0, 1).float() / 255


class CategoryTripletTests(unittest.TestCase):
    def test_hand_computed_hardest_positive_and_negative(self):
        features = torch.tensor([[1., 0], [0., 1], [-1., 0], [0., -1]], requires_grad=True)
        loss, stats = CategoryBatchHardTriplet(0.3)(features, torch.tensor([0, 0, 1, 1]))
        self.assertAlmostEqual(loss.item(), 0.3, places=6)
        self.assertAlmostEqual(stats['positive_distance'], math.sqrt(2), places=6)
        self.assertAlmostEqual(stats['negative_distance'], math.sqrt(2), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_normalization_makes_loss_invariant_to_positive_feature_scaling(self):
        features = torch.tensor([[2., 0], [1., 1], [-2., 1], [-1., -1]])
        targets = torch.tensor([0, 0, 1, 1])
        loss = CategoryBatchHardTriplet()
        first = loss(features, targets)[0]
        second = loss(features * torch.tensor([[2.], [3.], [0.2], [7.]]), targets)[0]
        self.assertTrue(torch.allclose(first, second))

    def test_identical_resampled_positive_has_finite_gradient(self):
        features = torch.tensor([[1., 0], [1., 0], [0., 1], [0., 1]], requires_grad=True)
        loss, _ = CategoryBatchHardTriplet(2)(features, torch.tensor([0, 0, 1, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_missing_positive_or_negative_is_error(self):
        for targets in (torch.tensor([0, 1, 2, 3]), torch.zeros(4, dtype=torch.long), torch.tensor([0, 0, 1, 2])):
            with self.subTest(targets=targets.tolist()), self.assertRaises(ValueError):
                CategoryBatchHardTriplet()(torch.randn(4, 3), targets)

    def test_invalid_features_labels_and_margin(self):
        loss = CategoryBatchHardTriplet()
        for features in (torch.zeros(4, 2), torch.full((4, 2), float('nan'))):
            with self.assertRaises(FloatingPointError):
                loss(features, torch.tensor([0, 0, 1, 1]))
        with self.assertRaises(ValueError):
            loss(torch.randn(4, 2), torch.tensor([0., 0, 1, 1]))
        with self.assertRaises(ValueError):
            CategoryBatchHardTriplet(-0.1)


class CategoryTrainingTests(StreamFixture):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.stream = self.load()
        self.model = tiny_model()

    def images_for(self, samples):
        from pathlib import Path

        keys = {key: index for index, key in enumerate(sorted({s.identity_key for s in samples}))}
        for sample in samples:
            path = Path(sample.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            color = (230, 20, 10) if keys[sample.identity_key] % 2 == 0 else (10, 20, 230)
            Image.new('RGB', (4, 4), color).save(path)

    def loaders_for(self, stage):
        for view in stage.categories:
            self.images_for(view.samples)
        return build_stage_loaders(stage, batch_size=4, num_instances=2, seed=8,
                                   train_transform=fixed_tensor, reference_transform=fixed_tensor)

    def trainer(self, stage=None, **options):
        stage = stage or self.stream.stage('t1')
        return CategoryProgressiveTrainer(self.model, stage, self.loaders_for(stage),
                                          CategoryTrainingConfig(**options), self.base / (stage.stage_id + '.jsonl'))

    def test_updates_both_categories_equal_budget_and_logs_only_baseline_losses(self):
        trainer = self.trainer(epochs=1, iterations_per_epoch=3)
        before = {c: self.model.export_adapter(c) for c in trainer.categories}
        reference = self.model.encode_reference(torch.ones(1, 3, 4, 4))
        report = trainer.train_epoch()
        for c in trainer.categories:
            values = report['categories'][c]
            self.assertEqual(values['updates'], 3)
            self.assertEqual(values['images'], 12)
            self.assertEqual(values['loader_restarts'], 2)
            self.assertAlmostEqual(values['total'], values['ce'] + values['weighted_triplet'], places=6)
            self.assertTrue(any(not torch.equal(v, self.model.export_adapter(c)['state'][k])
                                for k, v in before[c]['state'].items()))
        self.assertTrue(torch.equal(reference, self.model.encode_reference(torch.ones(1, 3, 4, 4))))
        result = trainer.finish_stage()
        self.assertEqual(result['category_updates'], {'person': 3, 'vehicle': 3})
        self.assertEqual(len(self.model.temporary_heads), 0)
        self.assertEqual(self.model.trainable_categories, ())
        self.assertEqual(trainer.loaders, {})
        events = [json.loads(line) for line in (self.base / 't1.jsonl').read_text().splitlines()]
        self.assertEqual(events[0]['loss_components'], ['ce', 'triplet'])
        self.assertEqual(sum(e['event'] == 'train_step' for e in events), 3)

    def test_category_gradient_accumulation_matches_explicit_sum(self):
        trainer = self.trainer(epochs=1, iterations_per_epoch=1)
        expected = copy.deepcopy(self.model)
        optimizer = torch.optim.AdamW([
            {'params': [p for c in trainer.categories for p in expected.adapter_parameters(c)], 'lr': trainer.config.adapter_lr},
            {'params': list(expected.temporary_heads.parameters()), 'lr': trainer.config.head_lr},
        ], weight_decay=trainer.config.weight_decay)
        losses = []
        for category in trainer.categories:
            pair = trainer.loaders[category]
            pair.set_epoch(0)
            batch = next(iter(pair.train))
            features = expected.encode_category(batch['images'], category)
            ce = torch.nn.functional.cross_entropy(expected.classify(features, category), batch['targets'])
            tri = trainer.triplet(features, batch['targets'])[0]
            losses.append(ce + trainer.config.lambda_tri * tri)
        sum(losses).backward()
        optimizer.step()
        trainer.train_epoch()
        for p, q in zip(self.model.parameters(), expected.parameters()):
            self.assertTrue(torch.allclose(p, q, atol=1e-7, rtol=1e-6))

    def test_stage_transition_reuses_adapter_replaces_heads_and_freezes_absent(self):
        first = self.trainer(epochs=1, iterations_per_epoch=1)
        first.train_epoch()
        old_head = self.model.temporary_heads[self.model.category_key('person')]
        first.finish_stage()
        vehicle = self.model.export_adapter('vehicle')
        parameter_ids = [id(p) for p in self.model.adapter_parameters('person')]
        second = self.trainer(self.stream.stage('t2'), epochs=1, iterations_per_epoch=2)
        self.assertIsNot(self.model.temporary_heads[self.model.category_key('person')], old_head)
        self.assertEqual(parameter_ids, [id(p) for p in self.model.adapter_parameters('person')])
        self.assertEqual(self.model.categories, ('panda', 'person', 'vehicle'))
        second.train_epoch()
        second.finish_stage()
        for name, value in vehicle['state'].items():
            self.assertTrue(torch.equal(value, self.model.export_adapter('vehicle')['state'][name]))

    def test_head_size_tracks_current_identity_count_not_history(self):
        self.rows['p2.csv'] += [
            {'path': 'person/g_{}.png'.format(cam), 'original_pid': 'g',
             'source_dataset': 'person_source', 'camid': str(cam)} for cam in (0, 1)
        ]
        self.stream = self.load()
        first = self.trainer(epochs=1, iterations_per_epoch=1)
        first.train_epoch()
        first.finish_stage()
        self.trainer(self.stream.stage('t2'), epochs=1, iterations_per_epoch=1)
        self.assertEqual(self.model.temporary_heads[self.model.category_key('person')].out_features, 3)
        self.assertEqual(self.model.temporary_heads[self.model.category_key('panda')].out_features, 2)

    def test_unequal_loader_lengths_get_equal_update_budget(self):
        extra = []
        for row in self.rows['p1.csv']:
            copied = dict(row)
            copied['path'] = row['path'].replace('.png', '_extra.png')
            extra.append(copied)
        self.rows['p1.csv'] += extra
        self.stream = self.load()
        trainer = self.trainer(epochs=1, iterations_per_epoch=5)
        report = trainer.train_epoch()['categories']
        self.assertEqual(report['person']['updates'], report['vehicle']['updates'])
        self.assertEqual(report['person']['images'], 20)
        self.assertEqual(report['vehicle']['images'], 20)
        self.assertEqual(report['person']['loader_restarts'], 2)
        self.assertEqual(report['vehicle']['loader_restarts'], 4)

    def test_small_fixture_identity_learning_improves(self):
        trainer = self.trainer(epochs=2, iterations_per_epoch=25, head_lr=0.03, adapter_lr=0.003)
        first = trainer.train_epoch()
        last = trainer.train_epoch()
        for category in trainer.categories:
            self.assertLess(last['categories'][category]['ce'], first['categories'][category]['ce'])
            self.assertGreaterEqual(last['categories'][category]['accuracy'], 0.95)
        (self.base / 'learning.json').write_text(json.dumps([first, last]))

    def test_batch_guard_rejects_cross_category_stage_split_and_wrong_label(self):
        trainer = self.trainer(epochs=1, iterations_per_epoch=1)
        original = next(iter(trainer.loaders['person'].train))
        for field, value in (('categories', 'vehicle'), ('stage_ids', 't2'), ('splits', 'query'), ('paths', 'unknown.png')):
            batch = copy.deepcopy(original)
            batch[field][0] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                trainer._validate_batch('person', batch)
        batch = copy.deepcopy(original)
        batch['targets'][0] = 100
        with self.assertRaises(ValueError):
            trainer._validate_batch('person', batch)

    def test_zero_triplet_weight_is_exact_ce_objective(self):
        trainer = self.trainer(epochs=1, iterations_per_epoch=1, lambda_tri=0)
        for values in trainer.train_epoch()['categories'].values():
            self.assertEqual(values['weighted_triplet'], 0)
            self.assertEqual(values['total'], values['ce'])

    def test_image_errors_propagate_and_failed_trainer_cannot_continue(self):
        trainer = self.trainer(epochs=1, iterations_per_epoch=1)
        with patch('PIL.Image.open', side_effect=OSError('broken image')):
            with self.assertRaisesRegex(OSError, 'broken image'):
                trainer.train_epoch()
        self.assertTrue(trainer.failed)
        self.assertEqual(trainer.optimizer_steps, 0)
        with self.assertRaises(RuntimeError):
            trainer.train_epoch()
        with self.assertRaises(RuntimeError):
            trainer.finish_stage()

    def test_nonfinite_second_category_loss_does_not_commit_first_category_update(self):
        trainer = self.trainer(epochs=1, iterations_per_epoch=1)
        before = {n: value.clone() for n, value in self.model.state_dict().items()}
        original = self.model.encode_category

        def corrupted(images, category):
            features = original(images, category)
            return features * float('nan') if category == 'vehicle' else features

        with patch.object(self.model, 'encode_category', side_effect=corrupted):
            with self.assertRaises(FloatingPointError):
                trainer.train_epoch()
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertTrue(trainer.failed)
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[n]) for n, value in before.items()))

    def test_incomplete_stage_cannot_be_marked_finished(self):
        trainer = self.trainer(epochs=2, iterations_per_epoch=1)
        with self.assertRaises(RuntimeError):
            trainer.finish_stage()
        trainer.train_epoch()
        with self.assertRaises(RuntimeError):
            trainer.train_epoch(0)

    def test_non_amp_invalid_gradient_still_fails_without_update(self):
        trainer = self.trainer(epochs=1, iterations_per_epoch=1)
        before = {n: v.clone() for n, v in self.model.state_dict().items()}
        parameter = next(self.model.temporary_heads.parameters())
        handle = parameter.register_hook(lambda grad: torch.full_like(grad, float('inf')))
        try:
            with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                trainer.train_epoch()
        finally:
            handle.remove()
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertFalse(trainer.optimizer.state)
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[n]) for n, v in before.items()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP required')
    def test_amp_overflow_retries_same_batch_and_only_updates_once(self):
        from reid.utils.progressive_checkpoint import capture_rng, restore_rng
        self.model.to('cuda')
        trainer = self.trainer(epochs=1, iterations_per_epoch=1, amp=True)
        trainer.scaler = torch.amp.GradScaler('cuda', init_scale=8.)
        batches = {c: next(iter(pair.train)) for c, pair in trainer.loaders.items()}
        before = copy.deepcopy(self.model.state_dict())
        rng = capture_rng()
        calls = []
        parameter = next(self.model.temporary_heads.parameters())

        def overflow_once(grad):
            calls.append(1)
            # Verify stochastic state is replayed as well as the input tensors.
            torch.rand(3, device='cuda')
            return torch.full_like(grad, float('inf')) if len(calls) == 1 else grad

        handle = parameter.register_hook(overflow_once)
        try:
            with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
                values, norm = trainer._train_batches(batches)
                self.assertEqual(step.call_count, 1)
        finally:
            handle.remove()
        actual = copy.deepcopy(self.model.state_dict())
        actual_rng = torch.cuda.get_rng_state()
        self.assertTrue(torch.isfinite(norm))
        self.assertEqual(len(calls), 2)
        self.assertEqual(trainer.scaler.get_scale(), 4.)
        events = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
        retries = [e for e in events if e['event'] == 'amp_overflow_retry']
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]['scale_after'], 4.)
        # Compare with an ordinary update starting at the successful scale.
        self.model.load_state_dict(before)
        trainer.optimizer.state.clear()
        trainer.scaler = torch.amp.GradScaler('cuda', init_scale=4.)
        restore_rng(rng)
        handle = parameter.register_hook(lambda grad: (torch.rand(3, device='cuda'), grad)[1])
        try:
            expected_values, expected_norm = trainer._train_batches(batches)
        finally:
            handle.remove()
        self.assertEqual(values, expected_values)
        self.assertTrue(torch.equal(norm, expected_norm))
        self.assertTrue(torch.equal(actual_rng, torch.cuda.get_rng_state()))
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[n]) for n, v in actual.items()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP required')
    def test_amp_persistent_invalid_gradient_is_bounded_and_never_updates(self):
        self.model.to('cuda')
        trainer = self.trainer(epochs=1, iterations_per_epoch=1, amp=True)
        trainer.scaler = torch.amp.GradScaler('cuda', init_scale=8.)
        before = copy.deepcopy(self.model.state_dict())
        handle = next(self.model.temporary_heads.parameters()).register_hook(
            lambda grad: torch.full_like(grad, float('nan')))
        try:
            with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
                with self.assertRaisesRegex(FloatingPointError, 'remain non-finite'):
                    trainer.train_epoch()
                self.assertEqual(step.call_count, 0)
        finally:
            handle.remove()
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertFalse(trainer.optimizer.state)
        self.assertTrue(trainer.failed)
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[n]) for n, v in before.items()))
        events = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
        self.assertEqual(sum(e['event'] == 'amp_overflow_retry' for e in events), 21)
        self.assertFalse(any(e['event'] == 'train_step' for e in events))

    def test_missing_stage_history_and_stale_heads_rejected(self):
        stage = self.stream.stage('t2')
        with self.assertRaisesRegex(ValueError, 'history'):
            CategoryProgressiveTrainer(self.model, stage, self.loaders_for(stage))
        first = self.trainer(epochs=1, iterations_per_epoch=1)
        with self.assertRaisesRegex(ValueError, 'temporary heads'):
            CategoryProgressiveTrainer(self.model, self.stream.stage('t1'), first.loaders)

    def test_bad_config_rejected(self):
        for kwargs in ({'epochs': 0}, {'iterations_per_epoch': 0}, {'lambda_tri': -1}, {'adapter_lr': 0}, {'amp': 'false'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CategoryTrainingConfig(**kwargs)

    def test_oracle_evaluation_preserves_model_mode(self):
        trainer = self.trainer(epochs=1, iterations_per_epoch=1)
        trainer.train_epoch()
        evaluation = self.stream.evaluations[0]
        self.images_for(evaluation.query + evaluation.gallery)
        result = evaluate_category_oracle(self.model, evaluation, batch_size=1)
        self.assertTrue(self.model.training)
        self.assertEqual(result['routing'], 'oracle')
        self.assertEqual(result['mAP'], 100)

    def test_single_stage_runner_and_next_stage_checkpoint(self):
        for stage in self.stream.stages:
            self.loaders_for(stage)
        config_path = str(self.config_path)
        base_args = ['--stream-config', config_path, '--epochs', '1', '--iterations-per-epoch', '1',
                     '--batch-size', '4', '--num-instances', '2', '--device', 'cpu']
        first_dir, second_dir = self.base / 'run1', self.base / 'run2'
        run_stage(base_args + ['--stage-id', 't1', '--output-dir', str(first_dir)],
                  model_factory=lambda **kwargs: tiny_model())
        run_stage(base_args + ['--stage-id', 't2', '--output-dir', str(second_dir),
                              '--previous-checkpoint', str(first_dir / 'completed_model.pt')])
        checkpoint = torch.load(second_dir / 'completed_model.pt', weights_only=True)
        self.assertTrue(checkpoint['completed'])
        self.assertEqual(checkpoint['stage_id'], 't2')
        restored = CategoryAdapterBank.from_checkpoint(checkpoint['model'])
        self.assertEqual(restored.categories, ('panda', 'person', 'vehicle'))
        with self.assertRaises(FileExistsError):
            run_stage(base_args + ['--stage-id', 't2', '--output-dir', str(second_dir)])
        with self.assertRaisesRegex(ValueError, 'preceding stage'):
            run_stage(base_args + ['--stage-id', 't3', '--output-dir', str(self.base / 'run3'),
                                  '--previous-checkpoint', str(first_dir / 'completed_model.pt')])


class CategoryOracleMetricTests(unittest.TestCase):
    def test_hand_computed_rank_and_ap_with_camera_and_self_filtering(self):
        def sample(path, identity, camid):
            return StreamSample(path, 'person', 'source', identity, camid,
                                ('person', 'source', identity), '', 'gallery')
        query = sample('q', 'a', '0')
        gallery = (query, sample('same_camera', 'a', '0'), sample('negative', 'b', '0'), sample('positive', 'a', '1'))
        evaluation = EvaluationView('test', 'person', 'test', 'cross_camera', (query,), gallery)
        query_features = torch.tensor([[1., 0]])
        gallery_features = torch.tensor([[1., 0], [1., 0], [1., 0], [0., 1]])
        result = oracle_retrieval_metrics(query_features, gallery_features, evaluation)
        self.assertEqual(result['Rank1'], 0)
        self.assertEqual(result['mAP'], 50)
        result = oracle_retrieval_metrics(query_features, gallery_features, replace(evaluation, protocol='exclude_self'))
        self.assertEqual(result['Rank1'], 100)
        self.assertAlmostEqual(result['mAP'], (1 + 2 / 3) / 2 * 100)


if __name__ == '__main__':
    unittest.main()
