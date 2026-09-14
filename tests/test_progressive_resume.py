import copy
import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from test_category_stream import StreamFixture
from test_category_training import tiny_model
from train_category_progressive import ProgressiveRun, main
from reid.utils.progressive_checkpoint import atomic_save, capture_rng, restore_rng


def assert_same(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a.cpu(), b.cpu()), 'tensor differs')
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            assert_same(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            assert_same(test, left, right)
    else:
        test.assertEqual(a, b)


class ProgressiveResumeTests(StreamFixture):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        super().setUp()
        self.stream = self.load()
        for stage in self.stream.stages:
            for view in stage.categories:
                for index, sample in enumerate(view.samples):
                    path = Path(sample.path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pixels = np.arange(13 * 17 * 3, dtype=np.int32).reshape(13, 17, 3)
                    Image.fromarray(((pixels * (index + 1) + 37 * index) % 256).astype('uint8')).save(path)
        self.settings = dict(stream_config=str(self.config_path), epochs=2, iterations_per_epoch=3,
                             batch_size=4, num_instances=2, prototype_batch_size=4, delta=-1., max_grad_norm=.1)

    def fresh(self, name='run', **overrides):
        return ProgressiveRun(self.base / name, dict(self.settings, **overrides),
                              model_factory=lambda **kw: tiny_model().to(kw['device']))

    def resume(self, name='run'):
        return ProgressiveRun(self.base / name, resume=True)

    def checkpoint(self, name='run'):
        return torch.load(self.base / name / 'latest.pt', map_location='cpu', weights_only=True)

    def events(self, name, stage):
        return [json.loads(line) for line in (self.base / name / 'stage_{:04d}.jsonl'.format(stage)).read_text(encoding='utf-8').splitlines()]

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP required')
    def test_amp_retry_then_resume_preserves_scaler_counters_and_rng(self):
        def first_update(name):
            run = self.fresh(name, device='cuda', amp=True)
            run.prepare_stage()
            run.trainer.scaler = torch.amp.GradScaler('cuda', init_scale=8.)
            calls = []
            def overflow_once(grad):
                calls.append(1)
                return torch.full_like(grad, float('inf')) if len(calls) == 1 else grad
            handle = next(run.model.temporary_heads.parameters()).register_hook(overflow_once)
            try:
                run.run(max_updates=1)
            finally:
                handle.remove()
            self.assertEqual(run.total_updates, 1)
            self.assertEqual(run.trainer.optimizer_steps, 1)
            self.assertEqual(run.trainer.iteration_in_epoch, 1)
            self.assertEqual(run.trainer.scaler.get_scale(), 4.)
            return run
        baseline = first_update('baseline')
        baseline.run(max_updates=1)
        first_update('split')
        restored = self.resume('split')
        self.assertEqual(restored.trainer.scaler.get_scale(), 4.)
        restored.run(max_updates=1)
        expected, actual = self.checkpoint('baseline'), self.checkpoint('split')
        assert_same(self, expected['trainer'], actual['trainer'])
        assert_same(self, expected['rng'], actual['rng'])
        self.assertEqual(expected['total_updates'], actual['total_updates'])
        self.assertEqual(self.events('baseline', 0), self.events('split', 0))

    def test_mid_epoch_resume_preserves_teacher_optimizer_rng_and_all_steps(self):
        baseline = self.fresh('baseline')
        baseline.run(max_updates=9)
        expected = self.checkpoint('baseline')
        baseline.run()
        interrupted = self.fresh()
        interrupted.run(max_updates=8)
        self.assertEqual(interrupted.trainer.iteration_in_epoch, 2)
        with (patch('reid.adaptation.pgca.PrototypeGuidedInitialization.apply', side_effect=AssertionError('reinitialized')),
             patch('reid.adaptation.pgca.FrozenCategoryTeacher.__init__', side_effect=AssertionError('new teacher')),
             patch('reid.memory.ecpm_modes.ECPMMemory.prepare_stage', side_effect=AssertionError('reread prototypes'))):
            restored = self.resume()
            restored.run(max_updates=1)
        actual = self.checkpoint()
        assert_same(self, expected['trainer'], actual['trainer'])
        assert_same(self, expected['rng'], actual['rng'])
        restored.run()
        self.assertEqual(restored.phase, 'complete')
        assert_same(self, baseline.model.export_bank(), restored.model.export_bank())
        assert_same(self, capture_rng(), self.checkpoint()['rng'])
        for stage in range(3):
            self.assertEqual(self.events('baseline', stage), self.events('run', stage))
        self.assertEqual(restored.memory.processed_stages, ('t1', 't2', 't3'))
        self.assertEqual(restored.memory.summary()['identities'], 12)

    def test_last_update_pending_commit_resumes_without_duplicate_identities(self):
        run = self.fresh()
        run.run(max_updates=6)
        self.assertEqual(run.memory.processed_stages, ())
        self.assertEqual(run.trainer.completed_epochs, 2)
        restored = self.resume()
        restored.run(max_stages=1)
        self.assertEqual(restored.memory.processed_stages, ('t1',))
        again = self.resume()
        again.run()
        self.assertEqual(again.memory.processed_stages, ('t1', 't2', 't3'))

    def test_preparation_interruption_restarts_from_boundary_and_archives_log_tail(self):
        run = self.fresh()
        with patch.object(run, 'save', side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):
                run.prepare_stage()
        restored = self.resume()
        self.assertEqual(restored.phase, 'between_stages')
        self.assertTrue(list(restored.output.glob('stage_0000.jsonl.recovered-*')))
        restored.run(max_updates=1)
        self.assertEqual(sum(e['event'] == 'stage_start' for e in self.events('run', 0)), 1)

    def test_unsaved_updates_are_replayed_and_log_tail_is_archived(self):
        run = self.fresh()
        run.run(max_updates=2)
        run.trainer.train_step()  # Simulate crash after log write, before latest.pt.
        restored = self.resume()
        self.assertEqual(restored.total_updates, 2)
        self.assertTrue(list(restored.output.glob('stage_0000.jsonl.recovered-*')))
        restored.run(max_updates=1)
        steps = [e['optimizer_step'] for e in self.events('run', 0) if e['event'] == 'train_step']
        self.assertEqual(steps, [1, 2, 3])

    def test_changed_log_prefix_is_rejected_without_truncation(self):
        self.fresh().run(max_updates=1)
        path = self.base / 'run' / 'stage_0000.jsonl'
        data = path.read_bytes()
        path.write_bytes(b'X' + data[1:] + b'extra')
        with self.assertRaisesRegex(ValueError, 'prefix changed'):
            self.resume()
        self.assertEqual(path.read_bytes(), b'X' + data[1:] + b'extra')

    def test_changed_reference_and_config_are_rejected(self):
        self.fresh().run(max_updates=1)
        with self.assertRaisesRegex(ValueError, 'setting changed'):
            ProgressiveRun(self.base / 'run', dict(seed=99), resume=True)
        path = self.base / 'run' / 'reference.pt'
        with path.open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'reference.pt differs'):
            self.resume()

    def test_changed_cursor_or_teacher_is_rejected(self):
        self.fresh().run(max_updates=8)
        original = self.checkpoint()
        for kind in ('cursor', 'teacher'):
            state = copy.deepcopy(original)
            if kind == 'cursor':
                state['trainer']['sampler_cursor']['person']['offset'] += 1
            else:
                state['trainer']['teacher_bank'] = None
            atomic_save(state, self.base / 'run' / 'latest.pt')
            with self.assertRaises(ValueError):
                self.resume()

    def test_no_old_or_future_image_access_and_completed_resume_is_noop(self):
        run = self.fresh()
        real_open = Image.open
        def guarded(path, *args, **kwargs):
            allowed = {s.path for view in run.stage.categories for s in view.samples}
            self.assertIn(str(path), allowed)
            return real_open(path, *args, **kwargs)
        with patch('PIL.Image.open', side_effect=guarded):
            run.run()
        with patch('PIL.Image.open', side_effect=AssertionError('completed run read images')):
            restored = self.resume()
            restored.run()
        self.assertEqual(restored.total_updates, 18)

    def test_cli_resume_uses_saved_configuration(self):
        self.fresh().run(max_updates=1)
        result = main(['--output-dir', str(self.base / 'run'), '--resume', '--max-updates', '1'])
        self.assertEqual(result['total_updates'], 2)
        with self.assertRaisesRegex(ValueError, 'workers 0'):
            self.fresh('bad', workers=1)

    def test_atomic_failure_keeps_previous_checkpoint(self):
        run = self.fresh()
        original = (run.output / 'latest.pt').read_bytes()
        with patch('reid.utils.progressive_checkpoint.os.replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError):
                run.save()
        self.assertEqual((run.output / 'latest.pt').read_bytes(), original)
        self.assertFalse(list(run.output.glob('*.tmp')))

    def test_commit_save_failure_reloads_uncommitted_candidate_once(self):
        run = self.fresh()
        run.run(max_updates=6)
        candidate = copy.deepcopy(run.candidate)
        with patch.object(run, 'save', side_effect=OSError('commit checkpoint failed')):
            with self.assertRaises(OSError):
                run.commit_stage()
        restored = self.resume()
        self.assertEqual(restored.memory.processed_stages, ())
        assert_same(self, candidate, restored.candidate)
        restored.run(max_stages=1)
        self.assertEqual(restored.memory.summary()['identities'], 4)
        self.assertEqual(sum(e['event'] == 'stage_end' for e in self.events('run', 0)), 1)

    def test_variable_pass_lengths_resume_at_correct_batch(self):
        # Three identities with 3/1/1 K-groups: picking the two short IDs
        # first yields one batch, while other pairings yield two batches.
        template = self.rows['p1.csv'][0]
        self.rows['p1.csv'] += [dict(template, path='person/a_extra_{}.png'.format(i)) for i in range(4)]
        self.rows['p1.csv'] += [dict(template, original_pid='z', path='person/z_{}.png'.format(i)) for i in range(2)]
        self.stream = self.load()
        for sample in self.stream.stages[0].category('person').samples:
            if not Path(sample.path).exists():
                Image.new('RGB', (17, 13), (140, 60, 20)).save(sample.path)
        baseline = self.fresh('baseline', iterations_per_epoch=8)
        baseline.prepare_stage()
        sampler = baseline.trainer.loaders['person'].train.batch_sampler
        lengths = set()
        for pass_index in range(20):
            sampler.set_epoch(pass_index)
            lengths.add(len(list(sampler)))
        self.assertEqual(lengths, {1, 2})
        baseline.run(max_updates=7)
        expected = self.checkpoint('baseline')
        self.fresh(iterations_per_epoch=8).run(max_updates=5)
        restored = self.resume()
        restored.run(max_updates=2)
        actual = self.checkpoint()
        assert_same(self, expected['trainer'], actual['trainer'])
        assert_same(self, expected['rng'], actual['rng'])

    def test_rng_roundtrip_including_gaussian_cache(self):
        random.gauss(0, 1)
        np.random.normal()
        saved = capture_rng()
        expected = (random.gauss(0, 1), np.random.normal(), torch.rand(5))
        restore_rng(saved)
        assert_same(self, expected, (random.gauss(0, 1), np.random.normal(), torch.rand(5)))


if __name__ == '__main__':
    unittest.main()
