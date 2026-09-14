import random
import unittest
from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from test_category_stream import StreamFixture
from lreid_dataset.category_stream import StreamProtocolError
from lreid_dataset.category_stream_loaders import (
    CategoryIdentityBatchSampler, build_evaluation_loaders, build_stage_loaders,
    make_category_transforms,
)


def image_tensor(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255


class CategoryStreamLoaderTests(StreamFixture):
    def create_images(self, samples):
        from pathlib import Path

        for index, sample in enumerate(samples):
            path = Path(sample.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (8, 8), (index * 17 % 255, 10, 20)).save(path)

    def loaders(self, stage, **kwargs):
        return build_stage_loaders(stage, batch_size=4, num_instances=2,
                                   prototype_batch_size=3, train_transform=image_tensor,
                                   reference_transform=image_tensor, **kwargs)

    def test_only_current_images_opened_and_reference_once(self):
        stream = self.load()
        current = stream.stage('t2')
        expected = {sample.path for view in current.categories for sample in view.samples}
        for view in current.categories:
            self.create_images(view.samples)
        # Historical, future and evaluation images do not exist. Any access fails.
        opened = []
        original = Image.open

        def tracking_open(path, *args, **kwargs):
            opened.append(str(path))
            return original(path, *args, **kwargs)

        with patch('PIL.Image.open', side_effect=tracking_open):
            loaders = self.loaders(current)
            self.assertEqual(opened, [])
            for category, pair in loaders.items():
                batches = list(pair.prototype)
                paths = [path for batch in batches for path in batch['paths']]
                self.assertEqual(paths, [s.path for s in pair.view.samples])
                self.assertEqual(len(paths), len(set(paths)))
                self.assertEqual([len(batch['paths']) for batch in batches], [3, 1])
                self.assertTrue(all(c == category for b in batches for c in b['categories']))
                self.assertTrue(all(s == 't2' for b in batches for s in b['stage_ids']))
            self.assertEqual(set(opened), expected)
            self.assertEqual(len(opened), len(expected))
            for pair in loaders.values():
                first = torch.cat([b['images'] for b in pair.prototype])
                second = torch.cat([b['images'] for b in pair.prototype])
                self.assertTrue(torch.equal(first, second))
                train_batch = next(iter(pair.train))
                self.assertEqual(sorted(Counter(train_batch['targets'].tolist()).values()), [2, 2])
                self.assertEqual(len(set(train_batch['categories'])), 1)
                self.assertEqual(len(train_batch['identity_keys']), 4)
                self.assertIsInstance(train_batch['identity_keys'][0], tuple)

    def test_sampler_is_repeatable_epoch_local_and_does_not_mutate_global_rng(self):
        view = self.load().stage('t1').category('person')
        sampler = CategoryIdentityBatchSampler(view, 4, 2, seed=17)
        before = random.getstate()
        first = list(sampler)
        self.assertEqual(before, random.getstate())
        self.assertEqual(len(sampler), len(first))
        self.assertEqual(list(sampler), first)
        sampler.set_epoch(5)
        expected = list(sampler)
        restored = CategoryIdentityBatchSampler(view, 4, 2, seed=17)
        restored.set_epoch(5)
        self.assertEqual(list(restored), expected)

    def test_sampler_replacement_for_short_identity(self):
        self.rows['p1.csv'] = [self.rows['p1.csv'][0], self.rows['p1.csv'][2]]
        view = self.load().stage('t1').category('person')
        batch = list(CategoryIdentityBatchSampler(view, 4, 2))[0]
        self.assertEqual(Counter(batch), {0: 2, 1: 2})

    def test_insufficient_identities_rejected_without_cross_category_fallback(self):
        view = self.load().stage('t1').category('person')
        with self.assertRaisesRegex(StreamProtocolError, 'P=3'):
            CategoryIdentityBatchSampler(view, 6, 2)
        for batch_size, k in ((2, 2), (4, 1), (5, 2), (0, 0)):
            with self.subTest(batch_size=batch_size, k=k), self.assertRaises(ValueError):
                CategoryIdentityBatchSampler(view, batch_size, k)

    def test_builder_rejects_whole_stream(self):
        with self.assertRaises(TypeError):
            self.loaders(self.load())

    def test_builder_rejects_manually_mixed_stage_or_evaluation_rows(self):
        stage = self.load().stage('t2')
        view = stage.categories[0]
        bad_sample = replace(view.samples[0], split='query')
        bad_view = replace(view, samples=(bad_sample,) + view.samples[1:])
        bad_stage = replace(stage, categories=(bad_view,) + stage.categories[1:])
        with self.assertRaisesRegex(StreamProtocolError, 'non-current or non-training'):
            self.loaders(bad_stage)

    def test_default_reference_preprocessing_deterministic(self):
        train_transform, reference_transform = make_category_transforms(height=12, width=16)
        image = Image.new('RGB', (17, 9), (50, 100, 150))
        first = reference_transform(image)
        self.assertEqual(tuple(first.shape), (3, 12, 16))
        self.assertTrue(torch.equal(first, reference_transform(image)))
        self.assertEqual(tuple(train_transform(image).shape), (3, 12, 16))

    def test_evaluation_query_gallery_share_labels(self):
        evaluation = self.load().evaluations[0]
        self.create_images(evaluation.query + evaluation.gallery)
        loaders = build_evaluation_loaders(evaluation, image_tensor, batch_size=1)
        query = next(iter(loaders['query']))
        gallery = next(iter(loaders['gallery']))
        self.assertTrue(torch.equal(query['targets'], gallery['targets']))
        self.assertEqual(query['identity_keys'], gallery['identity_keys'])
        self.assertEqual(query['stage_ids'], [''])

    def test_worker_process_can_load_current_stage(self):
        stage = self.load().stage('t2')
        for view in stage.categories:
            self.create_images(view.samples)
        loaders = self.loaders(stage, workers=1)
        batch = next(iter(loaders['person'].prototype))
        self.assertEqual(batch['stage_ids'], ['t2'] * 3)


if __name__ == '__main__':
    unittest.main()
