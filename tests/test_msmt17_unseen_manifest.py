import json
import tempfile
import unittest
from pathlib import Path

from tools.build_msmt17_unseen_manifest import (
    MSMT17ManifestError,
    build_evaluation_config,
    build_msmt17_unseen_rows,
    write_manifest,
)


class MSMT17UnseenManifestTests(unittest.TestCase):
    def _make_root(self, base, same_camera=False):
        root = Path(base) / 'MSMT17'
        query = root / 'query'
        gallery = root / 'bounding_box_test'
        query.mkdir(parents=True)
        gallery.mkdir(parents=True)
        (query / '0000_c1_0000.jpg').touch()
        (query / '0001_c2_0000.jpg').touch()
        (gallery / '0000_c{}_0001.jpg'.format(
            1 if same_camera else 2
        )).touch()
        (gallery / '0001_c1_0001.jpg').touch()
        return root

    def test_builds_query_gallery_only_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._make_root(temporary_directory)
            rows, audit = build_msmt17_unseen_rows(root)

        self.assertEqual(len(rows), 4)
        self.assertEqual({row['split'] for row in rows}, {'query', 'gallery'})
        self.assertEqual(audit['training_images_used'], 0)
        self.assertEqual(audit['eval_identities'], 2)
        self.assertEqual(
            audit['queries_without_cross_camera_positive'],
            0,
        )
        self.assertEqual(rows[0]['path'], 'query/0000_c1_0000.jpg')

    def test_rejects_query_without_cross_camera_positive(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._make_root(temporary_directory, same_camera=True)
            with self.assertRaises(MSMT17ManifestError):
                build_msmt17_unseen_rows(root)

    def test_writes_manifest_and_evaluation_only_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            root = self._make_root(temporary_directory)
            rows, _ = build_msmt17_unseen_rows(root)
            config_dir = temporary_directory / 'config'
            manifest_path = config_dir / 'manifests' / 'msmt17_unseen.csv'
            base_config = config_dir / 'base.json'
            output_config = config_dir / 'eval.json'
            write_manifest(manifest_path, rows)
            base_config.parent.mkdir(parents=True, exist_ok=True)
            base_config.write_text(
                json.dumps(
                    {
                        'train_domains': [
                            {
                                'name': 'source',
                                'category': 'person',
                                'manifest': 'source.csv',
                            }
                        ],
                        'test_domains': [],
                    }
                ),
                encoding='utf-8',
            )
            build_evaluation_config(
                base_config,
                manifest_path,
                output_config,
                dataset_root=root,
            )
            payload = json.loads(output_config.read_text(encoding='utf-8'))

        self.assertEqual(len(payload['train_domains']), 1)
        self.assertEqual(len(payload['test_domains']), 1)
        unseen = payload['test_domains'][0]
        self.assertEqual(unseen['name'], 'msmt17_unseen')
        self.assertNotIn('prompt_checkpoint', unseen)


if __name__ == '__main__':
    unittest.main()
