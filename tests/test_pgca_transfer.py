import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from test_category_stream import StreamFixture
from test_category_training import tiny_model, fixed_tensor
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.adaptation.pgca import PGCATransferConfig, PrototypeGuidedInitialization, prototype_similarity, select_transfer_source
from reid.loss.pgca import PGCAConsistencyConfig
from reid.memory import ECPMMemory
from reid.memory.ecpm_modes import _same
from reid.models.category_adapter_bank import CategoryAdapterBank, _tensor_digest
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig
from tools.train_pgca_stage import main as run_stage


def category(modes, center):
    return dict(mode_prototypes=torch.tensor(modes, dtype=torch.float32),
                category_prototype=torch.tensor(center, dtype=torch.float32))


class TransferFormulaTests(unittest.TestCase):
    def test_direction_is_new_row_max_then_mean_not_reverse(self):
        new = category([[1, 0], [0, 1]], [2 ** -.5, 2 ** -.5])
        old = category([[1, 0]], [1, 0])
        self.assertAlmostEqual(prototype_similarity(new, old, 0)['score'], .5)
        self.assertAlmostEqual(prototype_similarity(old, new, 0)['score'], 1.)
        self.assertAlmostEqual(prototype_similarity(new, old, .25)['score'], .25 * 2 ** -.5 + .75 * .5, places=6)

    def test_global_and_mode_alpha_endpoints(self):
        new = category([[1, 0], [0, 1]], [2 ** -.5, 2 ** -.5])
        old = category([[1, 0]], [1, 0])
        self.assertAlmostEqual(prototype_similarity(new, old, 1)['score'], 2 ** -.5, places=6)
        self.assertEqual(prototype_similarity(new, old, 0)['score'], .5)

    def test_scoring_ignores_outer_cpu_autocast(self):
        new, old = category([[.6, .8]], [.6, .8]), category([[1, 0]], [1, 0])
        config = PGCATransferConfig('similarity', alpha=0., delta=.6005)
        expected = select_transfer_source(new, {'old': old}, config)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            actual = select_transfer_source(new, {'old': old}, config)
        self.assertEqual(actual, expected)
        self.assertIsNone(actual['selected_source'])

    def test_threshold_inclusive_negative_scores_empty_history_and_ties(self):
        new = category([[1, 0]], [1, 0])
        opposite = category([[-1, 0]], [-1, 0])
        self.assertEqual(select_transfer_source(new, {'z': new, 'a': new}, PGCATransferConfig(delta=1))['selected_source'], 'a')
        self.assertEqual(select_transfer_source(new, {'old': opposite}, PGCATransferConfig(delta=-1))['selected_source'], 'old')
        below = select_transfer_source(new, {'old': opposite}, PGCATransferConfig(delta=0))
        self.assertEqual(below['reason'], 'below_threshold')
        self.assertIsNone(below['selected_source'])
        self.assertEqual(select_transfer_source(new, {}, PGCATransferConfig())['reason'], 'no_history')

    def test_bad_config_and_prototypes_rejected(self):
        for kwargs in (dict(mode='bad'), dict(alpha=-.1), dict(alpha=1.1), dict(delta=-1.1), dict(delta=float('nan'))):
            with self.assertRaises(ValueError):
                PGCATransferConfig(**kwargs)
        good = category([[1, 0]], [1, 0])
        for bad in (category([[0, 0]], [1, 0]), category([[1, 0]], [float('nan'), 0]), category([[1, 0, 0]], [1, 0])):
            with self.assertRaises(ValueError):
                prototype_similarity(good, bad)


class TransferStateTests(StreamFixture):
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
        for c in ('person', 'vehicle'):
            self.model.add_category(c)
            with torch.no_grad():
                for p in self.model.adapter_parameters(c):
                    p.add_(torch.randn_like(p) * .05)
        self.memory, self.candidate = self.memory_for(self.stream)
        self.stage = self.stream.stages[1]

    def identity_candidate(self, memory, stage):
        rows = []
        for view in stage.categories:
            vector = ([1., 0., 0., 0.] if view.category == 'person' else [0., 1., 0., 0.]) if stage.stage_id == 't1' else [.6, .8, 0., 0.]
            for key in view.identity_keys:
                rows.append(dict(identity_key=key, vector=torch.tensor(vector), first_stage=stage.stage_id,
                                 image_count=sum(s.identity_key == key for s in view.samples)))
        return dict(binding=copy.deepcopy(memory._binding), base_stages=memory.processed_stages,
                    stage_id=stage.stage_id, rows=rows)

    def memory_for(self, stream):
        memory = ECPMMemory(self.model, stream)
        memory.commit_stage(memory.prepare_identity_stage(self.identity_candidate(memory, stream.stages[0])))
        return memory, memory.prepare_identity_stage(self.identity_candidate(memory, stream.stages[1]))

    def initializer(self, **config):
        return PrototypeGuidedInitialization(self.model, self.stage, PGCATransferConfig('similarity', **config),
                                             self.memory, self.candidate)

    def loaders(self, stage=None):
        stage = stage or self.stage
        for view in stage.categories:
            for i, sample in enumerate(view.samples):
                path = Path(sample.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (4, 4), (30 + 40 * i, 20, 200 - 30 * i)).save(path)
        return build_stage_loaders(stage, batch_size=4, num_instances=2, seed=12,
                                   train_transform=fixed_tensor, reference_transform=fixed_tensor)

    def test_absent_donor_uses_old_centers_not_current_recurring_update(self):
        init = self.initializer(alpha=1, delta=.7)
        decision = init.metadata()['decisions']['panda']
        self.assertEqual(decision['selected_source'], 'vehicle')
        self.assertEqual(set(decision['candidates']), {'person', 'vehicle'})
        self.assertGreater(prototype_similarity(self.candidate['category_updates']['panda'],
                           self.candidate['category_updates']['person'], 1)['score'], decision['best_score'])
        self.assertNotIn('panda', self.model.categories)

    def test_snapshot_copy_survives_live_source_update_and_has_no_shared_storage(self):
        old = self.model.export_adapter('vehicle')
        recurring = self.model.export_adapter('person')
        init = self.initializer(delta=.7)
        with torch.no_grad():
            for p in self.model.adapter_parameters('vehicle'):
                p.add_(1.)
        init.apply(self.model)
        target = self.model.export_adapter('panda')
        self.assertTrue(_same(old['state'], target['state']))
        self.assertTrue(_same(recurring, self.model.export_adapter('person')))
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(self.model.adapter_parameters('vehicle'), self.model.adapter_parameters('panda'))))
        self.assertEqual(len(self.model.temporary_heads), 0)
        self.assertEqual(init._snapshots, {})
        with self.assertRaises(RuntimeError):
            init.apply(self.model)

    def test_low_similarity_fallback_is_zero_residual(self):
        init = self.initializer(delta=.9)
        init.apply(self.model)
        self.assertEqual(init.metadata()['decisions']['panda']['reason'], 'below_threshold')
        state = self.model.export_adapter('panda')['state']
        self.assertTrue(all(torch.count_nonzero(v) == 0 for k, v in state.items() if '.up.' in k))
        images = torch.randn(2, 3, 4, 4)
        self.assertTrue(torch.equal(self.model.encode_reference(images), self.model.encode_category(images, 'panda')))

    def test_preexisting_target_wrong_candidate_and_tampered_snapshot_rejected(self):
        bad = copy.deepcopy(self.candidate)
        bad['drifts']['person'] = 2.
        with self.assertRaises(ValueError):
            PrototypeGuidedInitialization(self.model, self.stage, PGCATransferConfig('similarity'), self.memory, bad)
        init = self.initializer()
        init._snapshots['vehicle']['state'][next(iter(init._snapshots['vehicle']['state']))].add_(1.)
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            init.apply(self.model)
        self.assertNotIn('panda', self.model.categories)
        self.model.add_category('panda')
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            self.initializer()

    def test_empty_history_matches_original_default_trainer_rng_and_parameters(self):
        models = [tiny_model()]
        models.append(copy.deepcopy(models[0]))
        stage = self.stream.stages[0]
        results = []
        for i, model in enumerate(models):
            memory = ECPMMemory(model, self.stream)
            candidate = memory.prepare_identity_stage(self.identity_candidate(memory, stage))
            torch.manual_seed(99)
            trainer = CategoryProgressiveTrainer(model, stage, self.loaders(stage),
                CategoryTrainingConfig(epochs=1, iterations_per_epoch=1),
                initialization=PGCATransferConfig('default' if i == 0 else 'similarity'),
                ecpm_memory=memory, ecpm_candidate=candidate)
            if i:
                self.assertTrue(all(d['reason'] == 'no_history' for d in trainer.initialization.metadata()['decisions'].values()))
            trainer.train_epoch()
            trainer.finish_stage()
            results.append(model.state_dict())
        self.assertTrue(_same(results[0], results[1]))

    def test_multiple_new_categories_and_reversed_input_order_are_independent(self):
        self.config['stages'][1]['categories'].append(self.entry('boat', 'boat2', ['a', 'b']))
        first_stream = self.load()
        self.config['stages'][1]['categories'].reverse()
        second_stream = self.load()
        states, decisions = [], []
        # Also covers fallback random initialization, not just copied parameters.
        for threshold in (.7, .9):
            states.clear()
            decisions.clear()
            for stream in (first_stream, second_stream):
                memory, candidate = self.memory_for(stream)
                model = copy.deepcopy(self.model)
                torch.manual_seed(32)
                init = PrototypeGuidedInitialization(model, stream.stages[1], PGCATransferConfig('similarity', delta=threshold), memory, candidate)
                init.apply(model)
                states.append(model.state_dict())
                decisions.append(init.metadata()['decisions'])
            self.assertTrue(_same(states[0], states[1]))
            self.assertTrue(_same(decisions[0], decisions[1]))
            self.assertTrue(all(set(d['candidates']) == {'person', 'vehicle'} for d in decisions[0].values()))

    def test_metadata_mutation_cannot_change_prepared_source(self):
        init = self.initializer(delta=.7)
        init.metadata()['decisions']['panda']['selected_source'] = 'person'
        self.candidate['old_categories']['vehicle']['category_prototype'].zero_()
        init.apply(self.model)
        self.assertEqual(init.metadata()['decisions']['panda']['selected_source'], 'vehicle')

    def test_transferred_new_category_trains_without_source_teacher_or_source_updates(self):
        old_source = self.model.export_adapter('vehicle')
        old_memory = self.memory.state_dict()
        trainer = CategoryProgressiveTrainer(self.model, self.stage, self.loaders(),
            CategoryTrainingConfig(epochs=1, iterations_per_epoch=2, adapter_lr=.003), self.base / 'transfer.jsonl',
            consistency=PGCAConsistencyConfig('drift'), initialization=PGCATransferConfig('similarity', delta=.7),
            ecpm_memory=self.memory, ecpm_candidate=self.candidate)
        self.assertTrue(_same(old_source['state'], self.model.export_adapter('panda')['state']))
        self.assertEqual(trainer.consistency.teacher.categories, ('person',))
        teacher = trainer.consistency.teacher.encode
        with patch.object(trainer.consistency.teacher, 'encode', wraps=teacher) as mocked:
            report = trainer.train_epoch()
        self.assertTrue(all(call.args[1] == 'person' for call in mocked.call_args_list))
        self.assertEqual(report['categories']['panda']['weighted_consistency'], 0.)
        self.assertFalse(_same(old_source['state'], self.model.export_adapter('panda')['state']))
        self.assertTrue(_same(old_source, self.model.export_adapter('vehicle')))
        self.assertTrue(_same(old_memory, self.memory.state_dict()))
        final = trainer.finish_stage()
        self.assertEqual(final['initialization']['decisions']['panda']['selected_source'], 'vehicle')
        start = json.loads((self.base / 'transfer.jsonl').read_text().splitlines()[0])
        self.assertEqual(start['initialization']['decisions']['panda']['source_adapter_sha256'], _tensor_digest(old_source['state']))

    def test_full_cli_transfer_checkpoint_restores_and_continues(self):
        self.loaders()
        self.loaders(self.stream.stages[2])
        baseline, memory_path = self.base / 'baseline.pt', self.base / 'ecpm.pt'
        torch.save(dict(kind='category_ce_triplet_baseline', schema_version=1, completed=True, stage_id='t1',
                        stream_fingerprint=self.stream.fingerprint, model=self.model.export_checkpoint()), baseline)
        self.memory.save(memory_path)
        common = ['--stream-config', str(self.config_path), '--epochs', '1', '--iterations-per-epoch', '1',
                  '--batch-size', '4', '--num-instances', '2', '--delta', '-1']
        second, third = self.base / 't2', self.base / 't3'
        report = run_stage(common + ['--stage-id', 't2', '--output-dir', str(second), '--previous-checkpoint', str(baseline),
                                    '--previous-memory', str(memory_path)])
        checkpoint = torch.load(second / 'completed_stage.pt', weights_only=True)
        self.assertEqual(checkpoint['kind'], 'pgca_stage')
        self.assertIsNotNone(report['training']['initialization']['decisions']['panda']['selected_source'])
        model = CategoryAdapterBank.from_checkpoint(checkpoint['model'])
        ECPMMemory.from_state_dict(checkpoint['ecpm'], model, self.stream)
        result = run_stage(common + ['--stage-id', 't3', '--output-dir', str(third),
                                    '--previous-checkpoint', str(second / 'completed_stage.pt')])
        self.assertEqual(result['training']['initialization']['new_categories'], [])
        self.assertEqual(result['ecpm']['identities'], 12)


if __name__ == '__main__':
    unittest.main()
