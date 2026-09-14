import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lreid_dataset.category_stream import StreamProtocolError, load_category_stream


ROOT = Path(__file__).resolve().parents[1]


class StreamFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.config_path = self.base / 'stream.json'
        self.rows = {}
        self.config = {
            'schema_version': 1, 'data_root': 'images',
            'stages': [
                {'stage_id': 't1', 'categories': [self.entry('person', 'p1', ['a', 'b']),
                                                self.entry('vehicle', 'v1', ['a', 'b'])]},
                {'stage_id': 't2', 'categories': [self.entry('person', 'p2', ['c', 'd']),
                                                self.entry('panda', 'a2', ['a', 'b'])]},
                {'stage_id': 't3', 'categories': [self.entry('vehicle', 'v3', ['c', 'd']),
                                                self.entry('person', 'p3', ['e', 'f'])]},
            ],
            'evaluation': [],
        }
        self.add_evaluation('person_test', 'test', 'holdout')

    def entry(self, category, name, pids):
        self.rows[name + '.csv'] = [
            {'path': '{}/{}_{}.png'.format(category, pid, camera), 'original_pid': pid,
             'source_dataset': category + '_source', 'camid': str(camera)}
            for pid in pids for camera in (0, 1)
        ]
        return {'category': category, 'train_manifest': name + '.csv'}

    def add_evaluation(self, name, split, pid, protocol='cross_camera'):
        for subset, camera in (('query', 0), ('gallery', 1)):
            self.rows[name + '_' + subset + '.csv'] = [
                {'path': '{}/{}_{}.png'.format(name, pid, subset), 'original_pid': pid,
                 'source_dataset': 'person_source', 'camid': str(camera)}
            ]
        self.config['evaluation'].append({
            'name': name, 'category': 'person', 'split': split, 'protocol': protocol,
            'query_manifest': name + '_query.csv', 'gallery_manifest': name + '_gallery.csv',
        })

    def write(self):
        self.config_path.write_text(json.dumps(self.config), encoding='utf-8')
        for name, rows in self.rows.items():
            fields = ['path', 'original_pid', 'source_dataset', 'camid']
            fields += sorted(set().union(*(set(row) for row in rows)) - set(fields)) if rows else []
            with (self.base / name).open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        return self.config_path

    def load(self, **kwargs):
        return load_category_stream(self.write(), **kwargs)


class CategoryStreamTests(StreamFixture):
    def test_stage_status_and_identity_counts(self):
        stream = self.load()
        self.assertEqual(stream.stage('t1').new_categories, ('person', 'vehicle'))
        self.assertEqual(stream.stage('t2').recurring_categories, ('person',))
        self.assertEqual(stream.stage('t2').absent_categories, ('vehicle',))
        self.assertEqual(stream.stage('t3').new_categories, ())
        self.assertEqual(stream.stage('t3').absent_categories, ('panda',))
        self.assertEqual(stream.audit_report()['total_train_identities'], 12)
        with self.assertRaises(KeyError):
            stream.stage('missing')
        with self.assertRaises(KeyError):
            stream.stage('t2').category('vehicle')

    def test_rejects_cross_stage_identity_even_with_new_path(self):
        self.rows['p2.csv'][0]['original_pid'] = 'a'
        with self.assertRaisesRegex(StreamProtocolError, 'identity leakage'):
            self.load()

    def test_local_zero_is_not_global_identity(self):
        stream = self.load()
        first = stream.stage('t1').category('person')
        second = stream.stage('t2').category('person')
        self.assertEqual(first.label_map[first.identity_keys[0]], 0)
        self.assertEqual(second.label_map[second.identity_keys[0]], 0)
        self.assertNotEqual(first.identity_keys[0], second.identity_keys[0])

    def test_same_pid_in_different_source_is_distinct(self):
        for row in self.rows['p2.csv']:
            row['source_dataset'] = 'another_person_source'
            row['original_pid'] = {'c': 'a', 'd': 'b'}[row['original_pid']]
        self.assertEqual(self.load().audit_report()['total_train_identities'], 12)

    def test_aliases_detect_same_real_identity_across_sources(self):
        self.config['identity_aliases'] = [
            {'category': 'person', 'source_dataset': 'person_source',
             'original_pid': pid, 'canonical_id': 'real_person'} for pid in ('a', 'c')
        ]
        with self.assertRaisesRegex(StreamProtocolError, 'identity leakage'):
            self.load()

    def test_rejects_unused_alias_to_catch_typos(self):
        self.config['identity_aliases'] = [
            {'category': 'person', 'source_dataset': 'typo', 'original_pid': 'a', 'canonical_id': 'x'}
        ]
        with self.assertRaisesRegex(StreamProtocolError, 'do not match'):
            self.load()

    def test_aliases_detect_leakage_after_source_renaming(self):
        for row in self.rows['p2.csv']:
            row['source_dataset'] = 'other_source'
        self.config['identity_aliases'] = [
            {'category': 'person', 'source_dataset': source,
             'original_pid': pid, 'canonical_id': 'real_a'}
            for source, pid in (('person_source', 'a'), ('other_source', 'c'))
        ]
        with self.assertRaisesRegex(StreamProtocolError, 'identity leakage'):
            self.load()

    def test_alias_does_not_turn_same_camera_into_cross_camera_positive(self):
        self.rows['person_test_gallery.csv'][0].update(source_dataset='other_source', camid='0')
        self.config['identity_aliases'] = [
            {'category': 'person', 'source_dataset': source,
             'original_pid': 'holdout', 'canonical_id': 'real_holdout'}
            for source in ('person_source', 'other_source')
        ]
        with self.assertRaisesRegex(StreamProtocolError, 'no valid cross_camera positive'):
            self.load()

    def test_aliases_merge_views_within_one_stage(self):
        self.config['identity_aliases'] = [
            {'category': 'person', 'source_dataset': 'person_source',
             'original_pid': pid, 'canonical_id': 'same'} for pid in ('a', 'b')
        ]
        self.assertEqual(len(self.load().stage('t1').category('person').identity_keys), 1)

    def test_string_pid_leading_zeros_preserved(self):
        self.rows['p1.csv'][0]['original_pid'] = '001'
        self.rows['p1.csv'][1]['original_pid'] = '1'
        keys = self.load().stage('t1').category('person').identity_keys
        self.assertIn(('person', 'person_source', '001'), keys)
        self.assertIn(('person', 'person_source', '1'), keys)

    def test_cross_stage_same_image_with_changed_pid_rejected(self):
        self.rows['p2.csv'][0]['path'] = self.rows['p1.csv'][0]['path']
        with self.assertRaisesRegex(StreamProtocolError, 'image leakage'):
            self.load()

    def test_duplicate_path_in_manifest_rejected(self):
        self.rows['p1.csv'].append(dict(self.rows['p1.csv'][0]))
        with self.assertRaisesRegex(StreamProtocolError, 'duplicate image'):
            self.load()

    def test_train_test_identity_leakage_rejected(self):
        for subset in ('query', 'gallery'):
            self.rows['person_test_' + subset + '.csv'][0]['original_pid'] = 'a'
        with self.assertRaisesRegex(StreamProtocolError, 'identity leakage'):
            self.load()

    def test_validation_test_identity_leakage_rejected(self):
        self.add_evaluation('person_validation', 'validation', 'holdout')
        with self.assertRaisesRegex(StreamProtocolError, 'identity leakage'):
            self.load()

    def test_empty_training_never_falls_back_to_evaluation(self):
        self.rows['p1.csv'] = []
        with self.assertRaisesRegex(StreamProtocolError, 'empty manifest'):
            self.load()

    def test_optional_category_stage_and_split_must_match_parent(self):
        for field, value in (('category', 'vehicle'), ('stage_id', 't2'), ('split', 'query')):
            with self.subTest(field=field):
                self.rows['p1.csv'][0][field] = value
                with self.assertRaisesRegex(StreamProtocolError, 'does not match parent'):
                    self.load()
                del self.rows['p1.csv'][0][field]

    def test_metadata_only_default_and_explicit_existence_check(self):
        self.load()  # No images exist in this fixture.
        with self.assertRaisesRegex(StreamProtocolError, 'does not exist'):
            self.load(check_images=True)

    def test_relative_paths_resolve_against_config_not_cwd(self):
        sample = self.load().stage('t1').category('person').samples[0]
        self.assertEqual(sample.path, str((self.base / 'images/person/a_0.png').resolve()))

    def test_entry_data_root_override(self):
        self.config['stages'][0]['categories'][0]['data_root'] = 'other_images'
        sample = self.load().stage('t1').category('person').samples[0]
        self.assertEqual(sample.path, str((self.base / 'other_images/person/a_0.png').resolve()))

    def test_no_cross_camera_positive_rejected(self):
        self.rows['person_test_gallery.csv'][0]['camid'] = '0'
        with self.assertRaisesRegex(StreamProtocolError, 'no valid cross_camera positive'):
            self.load()

    def test_exclude_self_allows_same_camera_distinct_images(self):
        self.config['evaluation'][0]['protocol'] = 'exclude_self'
        self.rows['person_test_gallery.csv'][0]['camid'] = '0'
        self.load()

    def test_query_itself_cannot_be_its_only_positive(self):
        self.config['evaluation'][0]['protocol'] = 'exclude_self'
        self.rows['person_test_gallery.csv'] = [dict(self.rows['person_test_query.csv'][0])]
        with self.assertRaisesRegex(StreamProtocolError, 'no valid exclude_self positive'):
            self.load()

    def test_query_can_also_be_in_gallery_if_another_positive_exists(self):
        self.rows['person_test_gallery.csv'].append(dict(self.rows['person_test_query.csv'][0]))
        self.load()

    def test_fixed_eval_is_exposed_only_after_category_arrival(self):
        self.config['evaluation'][0]['category'] = 'panda'
        stream = self.load()
        self.assertEqual(stream.evaluations_at('t1'), ())
        self.assertEqual(len(stream.evaluations_at('t2')), 1)
        self.assertEqual(len(stream.evaluations_at('t3')), 1)

    def test_jsonl_and_tsv_preserve_identity_fields(self):
        self.write()
        for suffix in ('.tsv', '.jsonl'):
            with self.subTest(suffix=suffix):
                name = 'p1' + suffix
                rows = self.rows['p1.csv']
                if suffix == '.jsonl':
                    (self.base / name).write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
                else:
                    with (self.base / name).open('w', newline='', encoding='utf-8') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
                        writer.writeheader()
                        writer.writerows(rows)
                self.config['stages'][0]['categories'][0]['train_manifest'] = name
                self.config_path.write_text(json.dumps(self.config), encoding='utf-8')
                stream = load_category_stream(self.config_path)
                self.assertEqual(len(stream.stage('t1').category('person').samples), 4)

    def test_schema_typos_and_duplicate_stage_category_rejected(self):
        for change in ('typo', 'stage', 'category', 'version'):
            with self.subTest(change=change):
                payload = json.loads(json.dumps(self.config))
                if change == 'typo':
                    payload['stage'] = []
                elif change == 'stage':
                    payload['stages'].append(payload['stages'][0])
                elif change == 'category':
                    payload['stages'][0]['categories'].append(payload['stages'][0]['categories'][0])
                else:
                    payload['schema_version'] = True
                self.write()
                self.config_path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(StreamProtocolError):
                    load_category_stream(self.config_path)

    def test_data_root_is_explicit(self):
        del self.config['data_root']
        with self.assertRaisesRegex(StreamProtocolError, 'data_root'):
            self.load()

    def test_missing_evaluation_coverage_is_reported(self):
        stream = self.load()
        warnings = stream.audit_report()['warnings']
        self.assertTrue(any('validation' in warning for warning in warnings))
        self.assertTrue(any('panda' in warning and 'test' in warning for warning in warnings))
        with self.assertRaises(ValueError):
            stream.evaluations_at('t1', 'typo')

    def test_fingerprint_stable_and_changes_with_metadata(self):
        first = self.load().fingerprint
        self.assertEqual(first, self.load().fingerprint)
        self.rows['p1.csv'][0]['camid'] = '3'
        self.assertNotEqual(first, self.load().fingerprint)

    def test_cli_from_outside_repository_and_structured_report(self):
        self.write()
        report = self.base / 'audit/report.json'
        command = [sys.executable, str(ROOT / 'tools/validate_category_stream.py'),
                   '--stream-config', str(self.config_path), '--output', str(report)]
        result = subprocess.run(command, cwd=str(self.base), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(report.read_text(encoding='utf-8'))
        self.assertEqual(payload['total_train_identities'], 12)
        self.assertFalse(payload['image_contents_read'])
        self.assertFalse(payload['image_existence_checked'])
        failed = subprocess.run(command + ['--check-images'], cwd=str(self.base), capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        self.assertIn('does not exist', failed.stderr)

    def test_parser_import_does_not_import_training_dependencies(self):
        script = (
            'import sys; from lreid_dataset.category_stream import load_category_stream; '
            'load_category_stream(sys.argv[1]); '
            'assert not any(n in sys.modules for n in ("torch", "PIL", "torchvision"))'
        )
        result = subprocess.run([sys.executable, '-c', script, str(self.write())],
                                cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checked_in_example(self):
        stream = load_category_stream(ROOT / 'config/category_progressive_example.json')
        self.assertEqual(len(stream.stages), 3)
        self.assertEqual(len(stream.evaluations), 4)
        self.assertEqual(stream.audit_report()['total_train_images'], 24)


if __name__ == '__main__':
    unittest.main()
