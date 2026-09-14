import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from lreid_dataset.category_stream import load_category_stream
from tools.build_category_progressive_stream import build_stream


class BuildProgressiveStreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        (self.base / 'config').mkdir()
        self.root = self.base / 'data'
        self.root.mkdir()
        self.source = self.base / 'config/source.json'
        self.recipe = self.base / 'config/recipe.json'
        self.recipe.write_text(json.dumps(dict(seed=42, validation_fraction=.2, min_train_ids_per_stage=2,
            stages=[dict(stage_id='t1', categories=['person']), dict(stage_id='t2', categories=['person'])])))
        self.domain = dict(name='source', category='person', manifest='source.csv', root='../data/source', eval_protocol='market1501')
        self.rows = [dict(path='train/{}_{}.jpg'.format(pid, camera), pid=pid, camid=str(camera), split='train', augmentation='')
                     for pid in ('00', '01', '02', '03', '04', '05') for camera in (0, 1)]
        self.rows += [dict(path='test/{}_{}.jpg'.format(pid, camera), pid=pid, camid=str(camera),
                          split='query' if camera == 0 else 'gallery', augmentation='') for pid in ('test_a', 'test_b') for camera in (0, 1)]
        self.write_source()

    def write_source(self):
        self.source.write_text(json.dumps(dict(train_domains=[self.domain])))
        with (self.source.parent / 'source.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['path', 'pid', 'camid', 'split', 'augmentation'])
            writer.writeheader()
            writer.writerows(self.rows)
        for row in self.rows:
            path = self.root / 'source' / row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'not decoded during metadata audit')

    def build(self, name='generated'):
        return build_stream(self.source, self.recipe, self.base / 'config' / name, self.root, True)

    def test_deterministic_whole_ids_and_original_test_preserved(self):
        stream, report = self.build('a')
        second, _ = self.build('b')
        self.assertEqual(stream.fingerprint, second.fingerprint)
        for path in (self.base / 'config/a/manifests').glob('*.csv'):
            self.assertEqual(path.read_bytes(), (self.base / 'config/b/manifests' / path.name).read_bytes())
        self.assertEqual([len(s.categories[0].identity_keys) for s in stream.stages], [2, 2])
        self.assertEqual(report['sources']['person']['validation_identities'], 2)
        test = next(v for v in stream.evaluations if v.split == 'test')
        for subset in ('query', 'gallery'):
            actual = [(Path(s.path).relative_to(self.root / 'source').as_posix(), s.original_pid, s.camid) for s in getattr(test, subset)]
            expected = [(r['path'], r['pid'], r['camid']) for r in self.rows if r['split'] == subset]
            self.assertEqual(actual, expected)
        with (self.base / 'config/a/identity_assignments.csv').open() as handle:
            assignments = list(csv.DictReader(handle))
        self.assertEqual({r['original_pid'] for r in assignments}, {'00', '01', '02', '03', '04', '05'})
        self.assertEqual(len(assignments), 6)

    def test_relative_bundle_survives_relocation(self):
        self.build()
        relocated = self.base / 'relocated'
        shutil.copytree(self.base / 'config/generated', relocated / 'config/generated')
        shutil.copytree(self.root, relocated / 'data')
        stream = load_category_stream(relocated / 'config/generated/main.json', check_images=True)
        self.assertEqual(stream.audit_report()['total_train_identities'], 4)
        self.assertTrue(all(Path(s.path).is_relative_to(relocated) for stage in stream.stages for view in stage.categories for s in view.samples))

    def test_boat_validation_excludes_augmentations_and_entire_ids_leave_train(self):
        self.domain['eval_protocol'] = 'cross_view'
        for row in self.rows:
            row['augmentation'] = 'original'
        self.rows += [dict(path='train/{}_aug.jpg'.format(pid), pid=pid, camid='0', split='train', augmentation='rotate')
                      for pid in ('00', '01', '02', '03', '04', '05')]
        self.write_source()
        stream, report = self.build()
        val = next(v for v in stream.evaluations if v.split == 'validation')
        self.assertEqual(report['sources']['person']['excluded_heldout_augmented_images'], 2)
        self.assertFalse(any('_aug' in s.path for s in val.query + val.gallery))
        held_out = {s.original_pid for s in val.query}
        self.assertFalse(any(s.original_pid in held_out for stage in stream.stages for view in stage.categories for s in view.samples))

    def test_unknown_camera_validation_uses_exclude_self(self):
        self.domain['eval_protocol'] = 'single_query'
        for row in self.rows:
            if row['split'] == 'train':
                row['camid'] = '0'
        self.write_source()
        stream, _ = self.build()
        val = next(v for v in stream.evaluations if v.split == 'validation')
        self.assertEqual(val.protocol, 'exclude_self')
        self.assertEqual({s.camid for s in val.query + val.gallery}, {'0'})

    def test_rejects_overlap_insufficient_ids_and_existing_output(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        self.rows[-1]['pid'] = '00'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.build('bad_overlap')
        self.rows[-1]['pid'] = 'test_b'
        self.write_source()
        recipe = json.loads(self.recipe.read_text())
        recipe['min_train_ids_per_stage'] = 8
        self.recipe.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'not enough training identities'):
            self.build('bad_count')
        self.assertFalse((self.base / 'config/bad_count').exists())

    def test_failed_existence_audit_does_not_publish_partial_bundle(self):
        (self.root / 'source' / self.rows[0]['path']).unlink()
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.base / 'config/generated').exists())


if __name__ == '__main__':
    unittest.main()
