import copy
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from test_category_stream import StreamFixture
from test_category_training import tiny_model
from lreid_dataset.category_stream_loaders import build_stage_prototype_loaders
from reid.memory.ecpm import IdentityPrototypeAccumulator, IdentityPrototypeMemory
from tools.extract_identity_prototypes import main as extract_stage


def metadata(samples):
    return dict(paths=[s.path for s in samples], identity_keys=[s.identity_key for s in samples],
                categories=[s.category for s in samples], stage_ids=[s.stage_id for s in samples],
                splits=[s.split for s in samples])


class IdentityMemoryTests(StreamFixture):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        super().setUp()
        torch.manual_seed(23)
        self.stream = self.load()
        self.model = tiny_model()
        self.memory = IdentityPrototypeMemory(self.model, self.stream)
        self.t1, self.t2 = self.stream.stages[:2]

    def images(self, stage):
        for view in stage.categories:
            for i, sample in enumerate(view.samples):
                path = Path(sample.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (7, 5), (30 + 15 * i, 70, 180)).save(path)

    def prepared(self):
        self.images(self.t1)
        return self.memory.prepare_stage(self.model, self.t1, batch_size=3)

    def test_mean_raw_features_then_normalize_not_mean_normalized(self):
        view = self.t1.categories[0]
        key = view.identity_keys[0]
        samples = tuple(s for s in view.samples if s.identity_key == key)
        self.assertEqual(len(samples), 2)
        accumulator = IdentityPrototypeAccumulator(replace(view, samples=samples), 2)
        accumulator.add(torch.tensor([[3., 0.], [0., 4.]], dtype=torch.float64), metadata(samples))
        row = accumulator.finish()[0]
        self.assertTrue(torch.allclose(row['vector'], torch.tensor([.6, .8])))
        self.assertEqual(row['image_count'], 2)
        self.assertEqual(row['vector'].dtype, torch.float32)

    def test_missing_and_repeated_images_are_rejected(self):
        view = self.t1.categories[0]
        accumulator = IdentityPrototypeAccumulator(view, 4)
        batch = metadata(view.samples[:1])
        accumulator.add(torch.ones(1, 4), batch)
        with self.assertRaisesRegex(ValueError, 'every'):
            accumulator.finish()
        with self.assertRaisesRegex(ValueError, 'repeated'):
            accumulator.add(torch.ones(1, 4), batch)

    def test_wrong_metadata_and_unknown_image_rejected(self):
        view = self.t1.categories[0]
        for field, value in [('stage_ids', 'T9'), ('splits', 'test'), ('categories', 'fake'),
                             ('identity_keys', ('person', 'fake', 'a')), ('paths', str(self.base / 'absent.jpg'))]:
            with self.subTest(field=field):
                accumulator = IdentityPrototypeAccumulator(view, 4)
                batch = metadata(view.samples[:1])
                batch[field][0] = value
                with self.assertRaises(ValueError):
                    accumulator.add(torch.ones(1, 4), batch)
                self.assertEqual(accumulator.counts, {})

    def test_zero_mean_nonfinite_and_wrong_shape_rejected(self):
        view = self.t1.categories[0]
        view = replace(view, samples=view.samples[:1])
        for features in [torch.zeros(1, 4), torch.full((1, 4), float('nan')), torch.ones(1, 3)]:
            with self.subTest(features=features), self.assertRaises((ValueError, FloatingPointError)):
                acc = IdentityPrototypeAccumulator(view, 4)
                acc.add(features, metadata(view.samples))
                acc.finish()

    def test_independent_loader_supports_one_identity_one_image(self):
        view = self.t1.categories[0]
        view = replace(view, samples=view.samples[:1])
        stage = replace(self.t1, categories=(view,))
        self.images(stage)
        loader = build_stage_prototype_loaders(stage, self.model.make_transforms()[1], batch_size=32)[view.category]
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        acc = IdentityPrototypeAccumulator(view, 4)
        acc.add(self.model.encode_reference(batches[0]['images']), batches[0])
        self.assertEqual(acc.finish()[0]['image_count'], 1)

    def test_loader_rejects_evaluation_rows_and_bad_options(self):
        view = self.t1.categories[0]
        bad = replace(view, samples=(replace(view.samples[0], split='test'),))
        with self.assertRaises(ValueError):
            build_stage_prototype_loaders(replace(self.t1, categories=(bad,)), self.model.make_transforms()[1])
        with self.assertRaises(ValueError):
            build_stage_prototype_loaders(self.t1, self.model.make_transforms()[1], batch_size=0)

    def test_prepare_does_not_commit_and_commit_copies_candidate(self):
        prepared = self.prepared()
        self.assertEqual(self.memory.processed_stages, ())
        report = self.memory.commit_stage(prepared)
        self.assertEqual(report['identities'], 4)
        self.assertEqual(report['images'], 8)
        prepared['rows'][0]['vector'].zero_()
        self.assertTrue(all(row['vector'].norm() > .99 for row in self.memory.state_dict()['rows']))

    def test_recurrence_appends_and_absent_category_stays_exactly_unchanged(self):
        self.memory.commit_stage(self.prepared())
        old = self.memory.state_dict()['rows']
        self.images(self.t2)
        # Remove every T1 image: processing T2 must not replay previous images.
        for view in self.t1.categories:
            for sample in view.samples:
                Path(sample.path).unlink()
        summary = self.memory.update_stage(self.model, self.t2)
        self.assertEqual(summary['categories'], {'person': 4, 'vehicle': 2, 'panda': 2})
        for before, after in zip(old, self.memory.state_dict()['rows']):
            self.assertEqual(before['identity_key'], after['identity_key'])
            self.assertTrue(torch.equal(before['vector'], after['vector']))

    def test_image_failure_in_second_category_does_not_partially_commit(self):
        self.images(replace(self.t1, categories=self.t1.categories[:1]))
        with self.assertRaises(FileNotFoundError):
            self.memory.update_stage(self.model, self.t1)
        self.assertEqual(self.memory.summary()['identities'], 0)
        self.assertEqual(self.memory.processed_stages, ())

    def test_repeated_and_out_of_order_stages_rejected_before_images(self):
        with self.assertRaisesRegex(ValueError, 'next unprocessed'):
            self.memory.update_stage(self.model, self.t2)
        prepared = self.prepared()
        self.memory.commit_stage(prepared)
        with patch.object(self.model, 'encode_reference', side_effect=AssertionError('must not run')):
            with self.assertRaises(ValueError):
                self.memory.update_stage(self.model, self.t1)
        with self.assertRaises(ValueError):
            self.memory.commit_stage(prepared)

    def test_modified_stage_contract_rejected(self):
        view = self.t1.categories[0]
        altered = replace(self.t1, categories=(replace(view, samples=view.samples[:1]),))
        with self.assertRaisesRegex(ValueError, 'audited'):
            self.memory.prepare_stage(self.model, altered)

    def test_changed_reference_and_preprocessing_rejected(self):
        other = tiny_model()
        with self.assertRaisesRegex(ValueError, 'signature'):
            self.memory.prepare_stage(other, self.t1)
        with torch.no_grad():
            self.model.visual.proj.add_(1)
        with self.assertRaises((ValueError, RuntimeError)):
            self.memory.prepare_stage(self.model, self.t1)

    def test_extraction_independent_of_adapters_and_preserves_policy(self):
        first = self.prepared()
        self.model.add_category('person')
        self.model.set_trainable_categories(['person'])
        self.model.train()
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.add_(torch.randn_like(parameter))
        policy = [(name, p.requires_grad) for name, p in self.model.named_parameters()]
        second = self.memory.prepare_stage(self.model, self.t1, batch_size=3)
        for a, b in zip(first['rows'], second['rows']):
            self.assertTrue(torch.equal(a['vector'], b['vector']))
        self.assertEqual(policy, [(name, p.requires_grad) for name, p in self.model.named_parameters()])
        self.assertTrue(self.model.training)
        self.model.assert_reference_unchanged()

    def test_save_restore_then_continue(self):
        self.memory.commit_stage(self.prepared())
        path = self.base / 'memory.pt'
        self.memory.save(path)
        restored = IdentityPrototypeMemory.load(path, self.model, self.stream)
        self.assertEqual(restored.summary(), self.memory.summary())
        self.images(self.t2)
        restored.update_stage(self.model, self.t2)
        self.assertEqual(restored.summary()['identities'], 8)
        self.assertEqual(self.memory.summary()['identities'], 4)

    def test_restore_rejects_wrong_stream_and_malformed_rows(self):
        self.memory.commit_stage(self.prepared())
        state = self.memory.state_dict()
        with self.assertRaises(ValueError):
            IdentityPrototypeMemory.from_state_dict(state, self.model, replace(self.stream, fingerprint='different'))
        mutations = [lambda s: s['rows'].pop(), lambda s: s['rows'].append(copy.deepcopy(s['rows'][0])),
                     lambda s: s['rows'][0].update(image_count=99),
                     lambda s: s['rows'][0].update(first_stage='T2'),
                     lambda s: s['rows'][0].update(vector=torch.zeros(4)),
                     lambda s: s['rows'][0].update(vector=torch.ones(4, dtype=torch.float64)),
                     lambda s: s.update(processed_stages=('T2',)),
                     lambda s: s['rows'][0].update(path='must not persist')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                bad = copy.deepcopy(state)
                mutate(bad)
                with self.assertRaises(ValueError):
                    IdentityPrototypeMemory.from_state_dict(bad, self.model, self.stream)

    def test_failed_commit_keeps_all_history(self):
        prepared = self.prepared()
        prepared['rows'][-1]['vector'].zero_()
        with self.assertRaises(ValueError):
            self.memory.commit_stage(prepared)
        self.assertEqual(self.memory.processed_stages, ())
        self.assertEqual(self.memory.summary()['identities'], 0)

    def test_atomic_save_failure_preserves_previous_file(self):
        self.memory.commit_stage(self.prepared())
        path = self.base / 'memory.pt'
        self.memory.save(path)
        old = path.read_bytes()
        with patch('reid.memory.ecpm.os.replace', side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError):
                self.memory.save(path)
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(list(self.base.glob('memory.pt.*.tmp')), [])

    def test_snapshots_do_not_expose_mutable_history_or_image_paths(self):
        self.memory.commit_stage(self.prepared())
        snapshot = self.memory.category_prototypes('person')
        snapshot[0]['vector'].zero_()
        self.assertGreater(self.memory.category_prototypes('person')[0]['vector'].norm(), .99)
        state = self.memory.state_dict()
        for view in self.t1.categories:
            for sample in view.samples:
                self.assertNotIn(sample.path, repr(state))
        self.assertEqual(sum(t.numel() for row in state['rows'] for t in row.values() if isinstance(t, torch.Tensor)), 4 * 4)

    def test_cli_two_stages_and_reject_existing_output(self):
        self.images(self.t1)
        self.images(self.t2)
        first, second = self.base / 't1.pt', self.base / 't2.pt'
        common = ['--stream-config', str(self.config_path), '--batch-size', '3']
        factory = lambda **kwargs: self.model
        extract_stage(common + ['--stage-id', self.t1.stage_id, '--output-memory', str(first)], model_factory=factory)
        report = extract_stage(common + ['--stage-id', self.t2.stage_id, '--previous-memory', str(first),
                                        '--output-memory', str(second)], model_factory=factory)
        self.assertEqual(report['identities'], 8)
        with self.assertRaises(FileExistsError):
            extract_stage(common + ['--stage-id', 'T2', '--output-memory', str(second)], model_factory=factory)

    def test_preprocessing_signature_mismatch_rejected(self):
        self.model.reference = replace(self.model.reference, resize_mode='stretch')
        with self.assertRaisesRegex(ValueError, 'signature'):
            self.memory.prepare_stage(self.model, self.t1)

    def test_alias_identity_merges_sources_within_stage(self):
        rows = self.rows['p1.csv']
        rows[1]['source_dataset'] = 'alias_source'
        rows[1]['original_pid'] = 'alias_a'
        self.config['identity_aliases'] = [
            dict(category='person', source_dataset='person_source', original_pid='a', canonical_id='real_a'),
            dict(category='person', source_dataset='alias_source', original_pid='alias_a', canonical_id='real_a')]
        stream = self.load()
        self.images(stream.stages[0])
        memory = IdentityPrototypeMemory(self.model, stream)
        memory.update_stage(self.model, stream.stages[0])
        merged = next(row for row in memory.category_prototypes('person') if row['identity_key'][1] == '__canonical__')
        self.assertEqual(merged['image_count'], 2)
        self.assertEqual(memory.summary()['identities'], 4)

    def test_cli_accepts_completed_baseline_model_checkpoint(self):
        self.images(self.t1)
        path = self.base / 'completed_model.pt'
        torch.save(dict(kind='category_ce_triplet_baseline', schema_version=1, completed=True,
                        stage_id=self.t1.stage_id, stream_fingerprint=self.stream.fingerprint,
                        model=self.model.export_checkpoint()), path)
        report = extract_stage(['--stream-config', str(self.config_path), '--stage-id', self.t1.stage_id,
                                '--model-checkpoint', str(path), '--output-memory', str(self.base / 'memory.pt')])
        self.assertEqual(report['identities'], 4)


if __name__ == '__main__':
    unittest.main()
