import copy
import importlib.metadata
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from pathlib import Path

from test_category_stream import StreamFixture
from test_category_training import tiny_model
from reid.memory import ECPMMemory, FinchConfig, IdentityPrototypeMemory
from reid.memory.ecpm_modes import _same
from reid.memory.finch_modes import aggregate_modes, finch_first_partition, prototype_drift
from tools.update_ecpm_memory import main as run_ecpm


class ModeFormulaTests(unittest.TestCase):
    def test_unequal_clusters_have_equal_mode_weight(self):
        vectors = torch.tensor([[1., 0.]] * 3 + [[0., 1.]])
        modes, center, sizes = aggregate_modes(vectors, torch.tensor([0, 0, 0, 1]))
        self.assertTrue(torch.equal(sizes, torch.tensor([3, 1])))
        self.assertTrue(torch.allclose(center, torch.tensor([.5 ** .5, .5 ** .5])))
        self.assertFalse(torch.allclose(center, torch.nn.functional.normalize(vectors.mean(0), dim=0)))

    def test_drift_zero_orthogonal_and_opposite(self):
        first = torch.tensor([1., 0.])
        self.assertAlmostEqual(prototype_drift(first, first), 0., places=6)
        self.assertAlmostEqual(prototype_drift(first, torch.tensor([0., 1.])), 1., places=6)
        self.assertAlmostEqual(prototype_drift(first, -first), 2., places=6)

    def test_zero_mode_or_category_center_is_an_error(self):
        v = torch.tensor([[1., 0.], [-1., 0.]])
        for labels in (torch.tensor([0, 0]), torch.tensor([0, 1])):
            with self.assertRaises(FloatingPointError):
                aggregate_modes(v, labels)

    def test_invalid_vectors_and_labels_rejected(self):
        for v in (torch.zeros(2, 2), torch.full((2, 2), float('nan')), torch.ones(2, 2, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                finch_first_partition(v)
        with self.assertRaises(ValueError):
            aggregate_modes(torch.eye(2), torch.tensor([0, 3]))
        with self.assertRaises(ValueError):
            FinchConfig(0)

    def test_first_partition_matches_official_complete_call(self):
        from finch import FINCH
        rng = np.random.default_rng(42)
        random = torch.nn.functional.normalize(torch.from_numpy(rng.normal(size=(40, 8)).astype('float32')), dim=1)
        duplicates = torch.tensor([[1., 0.]] * 4 + [[0., 1.]] * 2)
        for vectors in (random, duplicates, torch.tensor([[1., 0.], [0., 1.]])):
            official, _, _ = FINCH(vectors.numpy(), distance='cosine', ensure_early_exit=False, verbose=False)
            expected = official[:, 0]
            for chunk in (1, 7, 256):
                labels, details = finch_first_partition(vectors, FinchConfig(chunk))
                actual = labels.numpy()
                self.assertTrue(np.array_equal(actual[:, None] == actual, expected[:, None] == expected))
                self.assertEqual(details['partition'], 0)

    def test_singleton_and_identical_vectors(self):
        for n in (1, 2, 7):
            v = torch.tensor([[1., 0.]] * n)
            labels, _ = finch_first_partition(v, FinchConfig(2))
            modes, center, sizes = aggregate_modes(v, labels)
            self.assertEqual(sizes.tolist(), [n])
            self.assertTrue(torch.equal(center, v[0]))

    def test_missing_dependency_is_explicit_even_for_singleton(self):
        with patch('reid.memory.finch_modes.importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(ImportError, 'finch-clust'):
                finch_first_partition(torch.tensor([[1., 0.]]))

    def test_symmetric_new_modes_can_leave_center_drift_zero(self):
        old = torch.tensor([[1., 0.]] * 2)
        new = torch.tensor([[1., 0.]] * 2 + [[.6, .8]] * 2 + [[.6, -.8]] * 2)
        old_labels, _ = finch_first_partition(old)
        new_labels, _ = finch_first_partition(new)
        old_modes, old_center, _ = aggregate_modes(old, old_labels)
        new_modes, new_center, _ = aggregate_modes(new, new_labels)
        self.assertEqual((len(old_modes), len(new_modes)), (1, 3))
        self.assertAlmostEqual(prototype_drift(old_center, new_center), 0., places=6)


class ECPMStateTests(StreamFixture):
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
        self.memory = ECPMMemory(self.model, self.stream, FinchConfig(2))

    def candidate(self, index, memory=None):
        memory = memory or self.memory
        stage = self.stream.stages[index]
        rows = []
        for view in stage.categories:
            v = {'person': [1., .3 * index, 0., 0.], 'vehicle': [0., 1., 0., 0.],
                 'panda': [0., 0., 1., 0.]}[view.category]
            vector = torch.nn.functional.normalize(torch.tensor(v), dim=0)
            for key in view.identity_keys:
                rows.append(dict(identity_key=key, vector=vector.clone(), first_stage=stage.stage_id,
                                 image_count=sum(s.identity_key == key for s in view.samples)))
        return dict(binding=copy.deepcopy(memory._binding), stage_id=stage.stage_id,
                    base_stages=memory.processed_stages, rows=rows)

    def commit(self, index):
        prepared = self.memory.prepare_identity_stage(self.candidate(index))
        self.memory.commit_stage(prepared)
        return prepared

    def test_new_recurring_absent_and_immutable_old_snapshot(self):
        self.commit(0)
        old = self.memory.state_dict()
        prepared = self.memory.prepare_identity_stage(self.candidate(1))
        self.assertTrue(_same(old, self.memory.state_dict()))
        self.assertEqual(set(prepared['old_categories']), {'person', 'vehicle'})
        self.assertIsNone(prepared['drifts']['panda'])
        self.assertGreater(prepared['drifts']['person'], 0.)
        self.memory.commit_stage(prepared)
        new = self.memory.snapshot()
        self.assertTrue(_same(new['vehicle'], old['category_memory']['vehicle']))
        self.assertEqual(len(new['person']['identity_keys']), 4)
        self.assertEqual(new['person']['last_updated_stage'], 't2')
        prepared['old_categories']['person']['category_prototype'].zero_()
        prepared['category_updates']['person']['mode_prototypes'].zero_()
        self.assertTrue(_same(self.memory.last_transition['old_categories'], old['category_memory']))
        self.assertGreater(self.memory.snapshot()['person']['mode_prototypes'].norm(), .9)
        for before, after in zip(old['identity_memory']['rows'], self.memory.state_dict()['identity_memory']['rows']):
            self.assertTrue(_same(before, after))

    def test_only_current_categories_are_clustered(self):
        self.commit(0)
        from reid.memory.finch_modes import build_category_modes
        with patch('reid.memory.ecpm_modes.build_category_modes', wraps=build_category_modes) as wrapped:
            self.commit(1)
        self.assertEqual(wrapped.call_count, 2)
        self.assertEqual({call.args[0][0]['identity_key'][0] for call in wrapped.call_args_list}, {'person', 'panda'})

    def test_cluster_failure_does_not_partially_update_history(self):
        self.commit(0)
        old = self.memory.state_dict()
        from reid.memory.finch_modes import build_category_modes
        def fail_person(rows, *args):
            if rows[0]['identity_key'][0] == 'person':
                raise RuntimeError('simulated clustering failure after panda')
            return build_category_modes(rows, *args)
        with patch('reid.memory.ecpm_modes.build_category_modes', side_effect=fail_person):
            with self.assertRaises(RuntimeError):
                self.memory.prepare_identity_stage(self.candidate(1))
        self.assertTrue(_same(old, self.memory.state_dict()))

    def test_corrupt_candidate_rejected_without_partial_commit(self):
        self.commit(0)
        old = self.memory.state_dict()
        prepared = self.memory.prepare_identity_stage(self.candidate(1))
        mutations = [lambda p: p['drifts'].update(person=2.),
                     lambda p: p['drifts'].update(panda=0.),
                     lambda p: p['old_categories']['vehicle']['category_prototype'].zero_(),
                     lambda p: p['category_updates']['person']['cluster_sizes'].add_(1),
                     lambda p: p['category_updates']['person']['category_prototype'].zero_()]
        for mutate in mutations:
            bad = copy.deepcopy(prepared)
            mutate(bad)
            with self.assertRaises(ValueError):
                self.memory.commit_stage(bad)
            self.assertTrue(_same(old, self.memory.state_dict()))

    def test_repeated_and_out_of_order_candidate_rejected(self):
        with self.assertRaises(ValueError):
            self.memory.prepare_identity_stage(self.candidate(1))
        prepared = self.commit(0)
        with self.assertRaises(ValueError):
            self.memory.commit_stage(prepared)

    def test_save_restore_then_third_stage(self):
        self.commit(0)
        self.commit(1)
        path = self.base / 'ecpm.pt'
        self.memory.save(path)
        restored = ECPMMemory.load(path, self.model, self.stream)
        self.assertTrue(_same(restored.state_dict(), self.memory.state_dict()))
        self.memory = restored
        self.commit(2)
        self.assertEqual(self.memory.summary()['identities'], 12)
        self.assertEqual(set(self.memory.last_transition['drifts']), {'person', 'vehicle'})
        self.assertEqual(self.memory.snapshot()['panda']['last_updated_stage'], 't2')

    def test_restore_validates_assignments_formulas_snapshot_and_runtime(self):
        self.commit(0)
        self.commit(1)
        state = self.memory.state_dict()
        mutations = [lambda s: s['category_memory']['person']['labels'].fill_(99),
                     lambda s: s['category_memory']['vehicle'].update(last_updated_stage='t2'),
                     lambda s: s['category_memory']['person']['mode_prototypes'].zero_(),
                     lambda s: s['last_transition']['old_categories'].pop('person'),
                     lambda s: s['last_transition']['drifts'].update(person=-1.),
                     lambda s: s['runtime'].update(version='other')]
        for mutate in mutations:
            bad = copy.deepcopy(state)
            mutate(bad)
            with self.assertRaises(ValueError):
                ECPMMemory.from_state_dict(bad, self.model, self.stream)

    def test_upgrade_identity_history_without_images_recovers_old_drift(self):
        identity = IdentityPrototypeMemory(self.model, self.stream)
        for index in (0, 1):
            identity.commit_stage(self.candidate(index, identity))
        reports = self.memory.extend_from_identity_memory(identity)
        self.assertEqual(len(reports), 2)
        self.assertGreater(reports[-1]['drifts']['person'], 0.)
        self.assertEqual(len(self.memory.last_transition['old_categories']['person']['identity_keys']), 2)
        self.assertTrue(_same(identity.state_dict()['rows'], self.memory.state_dict()['identity_memory']['rows']))
        self.assertEqual(self.memory.extend_from_identity_memory(identity), [])

    def test_changed_historical_identity_rejected_on_upgrade(self):
        self.commit(0)
        identity = IdentityPrototypeMemory(self.model, self.stream)
        candidate = self.candidate(0, identity)
        candidate['rows'][0]['vector'] = torch.tensor([0., 0., 0., 1.])
        identity.commit_stage(candidate)
        with self.assertRaisesRegex(ValueError, 'historical'):
            self.memory.extend_from_identity_memory(identity)

    def test_empty_checkpoint_and_copy_isolation(self):
        restored = ECPMMemory.from_state_dict(self.memory.state_dict(), self.model, self.stream)
        self.assertEqual(restored.snapshot(), {})
        self.commit(0)
        snapshot = self.memory.snapshot()
        snapshot['person']['mode_prototypes'].zero_()
        self.assertGreater(self.memory.snapshot()['person']['mode_prototypes'].norm(), .99)

    def test_atomic_save_failure_keeps_previous_complete_ecpm(self):
        self.commit(0)
        path = self.base / 'ecpm.pt'
        self.memory.save(path)
        original = path.read_bytes()
        self.commit(1)
        with patch('reid.memory.ecpm.os.replace', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                self.memory.save(path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(ECPMMemory.load(path, self.model, self.stream).processed_stages, ('t1',))

    def test_cli_upgrade_from_identity_only_and_continue(self):
        identity = IdentityPrototypeMemory(self.model, self.stream)
        common = ['--stream-config', str(self.config_path), '--finch-chunk-size', '2']
        previous = None
        for index in (0, 1):
            identity.commit_stage(self.candidate(index, identity))
            source = self.base / ('identity{}.pt'.format(index))
            output = self.base / ('ecpm{}.pt'.format(index))
            identity.save(source)
            args = common + ['--identity-memory', str(source), '--output-memory', str(output)]
            if previous:
                args += ['--previous-memory', str(previous)]
            report = run_ecpm(args, model_factory=lambda **kwargs: self.model)
            self.assertEqual(len(report['stage_reports']), 1)
            previous = output
        self.assertEqual(report['stage_reports'][0]['identities'], 8)

    def test_cli_direct_image_extraction(self):
        for view in self.stream.stages[0].categories:
            for sample in view.samples:
                path = Path(sample.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (6, 8), (40, 80, 160)).save(path)
        report = run_ecpm(['--stream-config', str(self.config_path), '--stage-id', 't1',
                          '--output-memory', str(self.base / 'ecpm.pt')], model_factory=lambda **kwargs: self.model)
        self.assertEqual(report['stage_reports'][0]['identities'], 4)


if __name__ == '__main__':
    unittest.main()
