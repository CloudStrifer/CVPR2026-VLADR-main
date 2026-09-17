import copy
import json
import unittest
from unittest.mock import patch

import torch

from test_category_stream import StreamFixture
from test_category_training import tiny_model
import test_ecpm_modes as mode_fixture
import test_progressive_evaluation as evaluation_fixture
from test_progressive_resume import assert_same
from reid.memory import ECPMMemory
from reid.memory.ecpm_modes import _same
from reid.memory.prototype_views import control_view, memory_summaries
from train_category_progressive import MODULE_SWITCHES, ProgressiveRun, main
from tools.run_category_ablations import DEFAULT_ABLATIONS, run_suite


class ECPMSwitchTests(StreamFixture):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        super().setUp()
        self.stream, self.model = self.load(), tiny_model()

    def candidate(self, index):
        return mode_fixture.ECPMStateTests.candidate(self, index)

    def test_clustering_off_never_calls_finch_and_computes_identity_mean(self):
        with patch('reid.memory.ecpm_modes.finch_runtime', side_effect=AssertionError('FINCH loaded')), \
             patch('reid.memory.finch_modes.finch_first_partition', side_effect=AssertionError('FINCH called')):
            self.memory = ECPMMemory(self.model, self.stream, category_clustering=False)
            for index in range(2):
                candidate = self.memory.prepare_identity_stage(self.candidate(index))
                self.memory.commit_stage(candidate)
            current = self.memory.snapshot()['person']
            expected = torch.stack([r['vector'] for r in self.memory.category_prototypes('person')]).mean(0)
            expected = torch.nn.functional.normalize(expected, dim=0)
            self.assertTrue(torch.allclose(current['category_prototype'], expected))
            self.assertEqual(current['cluster_sizes'].tolist(), [4])
            self.assertEqual(current['clustering']['distance_block_bytes'], 0)
            restored = ECPMMemory.from_state_dict(self.memory.state_dict(), self.model, self.stream)
            self.assertTrue(_same(restored.state_dict(), self.memory.state_dict()))

    def test_evolution_off_uses_current_ids_and_keeps_absent_categories(self):
        self.memory = ECPMMemory(self.model, self.stream, prototype_evolution=False)
        first = self.memory.prepare_identity_stage(self.candidate(0))
        self.memory.commit_stage(first)
        old = self.memory.snapshot()
        identity = self.candidate(1)
        candidate = self.memory.prepare_identity_stage(identity)
        person = candidate['category_updates']['person']
        self.assertEqual(len(person['identity_keys']), 2)
        self.assertTrue(all(key in self.stream.stage('t2').category('person').identity_keys for key in person['identity_keys']))
        expected = next(r['vector'] for r in identity['rows'] if r['identity_key'][0] == 'person')
        self.assertTrue(torch.allclose(person['category_prototype'], expected))
        self.assertGreater(candidate['drifts']['person'], 0.)
        # Legacy identity_mean control/routing must not secretly reintroduce old IDs.
        alternative = control_view(self.memory, candidate, 'identity_mean')
        self.assertTrue(torch.allclose(alternative['category_updates']['person']['category_prototype'], expected))
        self.memory.commit_stage(candidate)
        self.assertTrue(_same(self.memory.snapshot()['vehicle'], old['vehicle']))
        self.assertTrue(torch.allclose(memory_summaries(self.memory, 'identity_mean')['person']['category_prototype'], expected))
        self.assertEqual(len(self.memory.category_prototypes('person')), 4)
        self.assertEqual(self.memory.summary()['identities'], 8)

    def test_all_ecpm_combinations_roundtrip_and_reject_policy_tampering(self):
        for clustering in (True, False):
            for evolution in (True, False):
                with self.subTest(clustering=clustering, evolution=evolution):
                    self.memory = ECPMMemory(self.model, self.stream, category_clustering=clustering,
                                             prototype_evolution=evolution)
                    for index in range(3):
                        candidate = self.memory.prepare_identity_stage(self.candidate(index))
                        bad = copy.deepcopy(candidate)
                        bad['module_switches']['prototype_evolution'] = not evolution
                        with self.assertRaisesRegex(ValueError, 'module switches'):
                            self.memory.commit_stage(bad)
                        self.memory.commit_stage(candidate)
                        state = self.memory.state_dict()
                        restored = ECPMMemory.from_state_dict(state, self.model, self.stream)
                        self.assertTrue(_same(restored.state_dict(), state))
                    self.assertEqual(len(self.memory.snapshot()['person']['identity_keys']), 6 if evolution else 2)

    def test_legacy_full_memory_without_switch_metadata_loads_as_all_on(self):
        self.memory = ECPMMemory(self.model, self.stream)
        self.memory.commit_stage(self.memory.prepare_identity_stage(self.candidate(0)))
        state = self.memory.state_dict()
        del state['module_switches']
        restored = ECPMMemory.from_state_dict(state, self.model, self.stream)
        self.assertEqual(restored.module_switches, dict(category_clustering=True, prototype_evolution=True))
        self.assertTrue(_same(restored.snapshot(), self.memory.snapshot()))


class TrainingSwitchTests(StreamFixture):
    setUpClass = classmethod(ECPMSwitchTests.setUpClass.__func__)
    tearDownClass = classmethod(ECPMSwitchTests.tearDownClass.__func__)

    def setUp(self):
        # Reuse its data fixture without inheriting and rerunning its tests.
        fixture = evaluation_fixture.ProgressiveEvaluationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.base, self.config_path = fixture.base, fixture.config_path
        self.stream, self.settings = fixture.stream, fixture.settings

    def fresh(self, name, **options):
        return ProgressiveRun(self.base / name, dict(self.settings, **options),
                              model_factory=lambda **kw: tiny_model())

    def test_default_and_explicit_all_on_preserve_model_and_metrics(self):
        baseline = self.fresh('implicit')
        baseline.run()
        explicit = self.fresh('explicit', **dict.fromkeys(MODULE_SWITCHES, 'on'))
        explicit.run()
        assert_same(self, baseline.model.export_bank(), explicit.model.export_bank())
        for left, right in zip(baseline.evaluations, explicit.evaluations):
            assert_same(self, left['retrieval'], right['retrieval'])

    def test_pgca_off_matches_legacy_baseline_without_teacher_or_transfer(self):
        legacy = self.fresh('legacy', consistency='off', init_mode='default')
        legacy.run()
        disabled = self.fresh('disabled', recurring_adaptation='off', emerging_transfer='off')
        disabled.run(max_stages=1)
        with patch('reid.adaptation.pgca.FrozenCategoryTeacher', side_effect=AssertionError('teacher allocated')), \
             patch('reid.adaptation.pgca.select_transfer_source', side_effect=AssertionError('transfer selected')):
            disabled.prepare_stage()
            self.assertIsNone(disabled.trainer.consistency.teacher)
            self.assertEqual(disabled.trainer.consistency.weight('person'), 0.)
            self.assertEqual(disabled.trainer.initialization.config.mode, 'default')
            disabled.run()
        assert_same(self, legacy.model.export_bank(), disabled.model.export_bank())

    def test_each_module_off_and_all_off_survive_mid_stage_resume_and_evaluation(self):
        variants = [{k: 'off'} for k in MODULE_SWITCHES] + [dict.fromkeys(MODULE_SWITCHES, 'off')]
        for index, flags in enumerate(variants):
            with self.subTest(flags=flags):
                whole = self.fresh('whole' + str(index), **flags)
                whole.run()
                split = self.fresh('split' + str(index), **flags)
                split.run(max_updates=3)  # T2 mid-epoch, with recurring and emerging categories.
                restored = ProgressiveRun(split.output, resume=True)
                restored.run()
                assert_same(self, whole.model.export_bank(), restored.model.export_bank())
                self.assertEqual(restored.phase, 'complete')
                for left, right in zip(whole.evaluations, restored.evaluations):
                    assert_same(self, left['retrieval'], right['retrieval'])
                report = json.loads((restored.output / 'stage_results.json').read_text(encoding='utf-8'))
                self.assertEqual(len(report['stages']), 3)
                self.assertEqual(report['stages'][-1]['module_switches'], restored.module_switches)
                for key, value in flags.items():
                    with self.assertRaisesRegex(ValueError, 'resume setting changed'):
                        ProgressiveRun(split.output, {key: 'on'}, resume=True)

    def test_legacy_modes_remain_available_and_off_has_priority(self):
        run = self.fresh('fixed', consistency='fixed', init_mode='random')
        self.assertEqual(run.con.mode, 'fixed')
        self.assertEqual(run.init.mode, 'random')
        disabled = self.fresh('override', consistency='fixed', init_mode='random',
                              recurring_adaptation='off', emerging_transfer='off')
        self.assertEqual(disabled.con.mode, 'off')
        self.assertEqual(disabled.init.mode, 'default')
        with self.assertRaisesRegex(ValueError, 'must be on or off'):
            self.fresh('bad', prototype_evolution='invalid')

    def test_validation_diagnostics_keep_non_target_pgca_branch_disabled(self):
        import tools.diagnose_pgca_validation as diagnostic
        run = self.fresh('diagnostic_boundary', recurring_adaptation='off', emerging_transfer='off')
        run.run(max_stages=1)
        original = diagnostic.ResumableCategoryTrainer
        for kind in ('sources', 'drift'):
            with patch.object(diagnostic, 'ResumableCategoryTrainer', wraps=original) as constructor:
                diagnostic.diagnose(run.output, self.base / (kind + '.json'), kind=kind, weights=(1.,))
            self.assertGreater(constructor.call_count, 0)
            for call in constructor.call_args_list:
                if kind == 'sources':
                    self.assertEqual(call.kwargs['consistency'].mode, 'off')
                else:
                    self.assertEqual(call.kwargs['initialization'].mode, 'default')

    def test_cli_accepts_four_switches_and_named_suite_preserves_old_default(self):
        argv = ['--stream-config', str(self.config_path), '--output-dir', str(self.base/'cli'),
                '--batch-size', '4', '--num-instances', '2', '--max-updates', '1']
        for key in MODULE_SWITCHES:
            argv += ['--' + key.replace('_', '-'), 'off']
        main(argv, model_factory=lambda **kw: tiny_model())
        saved = torch.load(self.base/'cli/latest.pt', weights_only=True)
        self.assertEqual({k: saved['settings'][k] for k in MODULE_SWITCHES}, dict.fromkeys(MODULE_SWITCHES, 'off'))
        names = ['full'] + ['no_' + k for k in MODULE_SWITCHES]
        plan, _ = run_suite(self.settings, self.base/'plan', names=names)
        for key in MODULE_SWITCHES:
            self.assertEqual(plan['experiments']['no_' + key][key], 'off')
        default, _ = run_suite(self.settings, self.base/'default_plan')
        self.assertEqual(tuple(default['experiments']), DEFAULT_ABLATIONS)


if __name__ == '__main__':
    unittest.main()
