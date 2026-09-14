"""Lazy, category-local image loaders for a single already-audited StageView.

No future/previous stage or evaluation set is accepted by the training builder.
Imports of torch, PIL and torchvision are delayed until their functionality is
requested, so the metadata CLI can run without the training environment.
"""

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass

from .category_stream import CategoryStage, StageView, StreamProtocolError


class CategoryImageDataset:
    def __init__(self, samples, label_map, transform):
        if not samples:
            raise StreamProtocolError('cannot construct an empty image dataset')
        if transform is None:
            raise ValueError('an explicit tensor-producing transform is required')
        self.samples = tuple(samples)
        self.label_map = dict(label_map)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert('RGB'))
        return {
            'images': tensor,
            'targets': self.label_map[sample.identity_key],
            'identity_keys': sample.identity_key,
            'paths': sample.path,
            'categories': sample.category,
            'stage_ids': sample.stage_id,
            'source_datasets': sample.source_dataset,
            'original_pids': sample.original_pid,
            'camids': sample.camid,
            'splits': sample.split,
        }


def collate_category_samples(samples):
    """Keep identity keys row-wise instead of default_collate transposing tuples."""
    import torch

    batch = {key: [sample[key] for sample in samples] for key in samples[0]}
    batch['images'] = torch.stack(batch['images'])
    batch['targets'] = torch.tensor(batch['targets'], dtype=torch.long)
    return batch


class CategoryIdentityBatchSampler:
    """Deterministic epoch-local P x K sampling without changing global RNGs.

    Like the legacy RandomIdentityBatchSampler, short identities are sampled
    with replacement, incomplete K groups are dropped, and sampling stops when
    fewer than P identities have groups left. An epoch may omit some images;
    prototype extraction therefore always uses a separate sequential loader.
    """

    def __init__(self, view, batch_size, num_instances, seed=0):
        if not isinstance(view, CategoryStage):
            raise TypeError('expected CategoryStage')
        if type(batch_size) is not int or type(num_instances) is not int:
            raise ValueError('batch_size and num_instances must be integers')
        if num_instances < 2 or batch_size % num_instances or batch_size // num_instances < 2:
            raise ValueError('Triplet training requires P >= 2 and K >= 2, with batch_size = P*K')
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_identities = batch_size // num_instances
        self.seed = int(seed)
        self.epoch = 0
        self.index_by_id = defaultdict(list)
        for index, sample in enumerate(view.samples):
            self.index_by_id[sample.identity_key].append(index)
        if len(self.index_by_id) < self.num_identities:
            raise StreamProtocolError(
                '{}/{} has {} identities, but P={} are required'.format(
                    view.stage_id, view.category, len(self.index_by_id), self.num_identities
                )
            )

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('epoch must be a non-negative integer')
        self.epoch = epoch

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        groups = {}
        k = self.num_instances
        for key in sorted(self.index_by_id):
            indices = list(self.index_by_id[key])
            if len(indices) < k:
                indices = rng.choices(indices, k=k)
            rng.shuffle(indices)
            groups[key] = [indices[i:i + k] for i in range(0, len(indices) - k + 1, k)]
        available = list(groups)
        while len(available) >= self.num_identities:
            batch = []
            for key in rng.sample(available, self.num_identities):
                batch.extend(groups[key].pop())
                if not groups[key]:
                    available.remove(key)
            yield batch

    def __iter__(self):
        return self._batches()

    def __len__(self):
        return sum(1 for _ in self._batches())


def _seed_worker(worker_id):
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _seed_for(seed, stage_id, category):
    digest = hashlib.sha256('{}\0{}\0{}'.format(seed, stage_id, category).encode('utf-8'))
    return int.from_bytes(digest.digest()[:8], 'big') % (2 ** 63)


def make_category_transforms(height=224, width=224, resize_mode='pad',
                             hflip_prob=0.5, crop_padding=10, erasing_prob=0.5,
                             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Reuse repository preprocessing; return (augmented_train, fixed_reference).

    Normalization is an explicit engineering setting. Keep it identical for
    reference prototype extraction and reference inference in later steps.
    """
    from reid.utils.data import transforms as T

    if height <= 0 or width <= 0 or crop_padding < 0:
        raise ValueError('invalid image size or crop padding')
    if not 0 <= hflip_prob <= 1 or not 0 <= erasing_prob <= 1:
        raise ValueError('augmentation probabilities must be in [0, 1]')
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError('RGB normalization needs three means and positive standard deviations')
    if resize_mode == 'pad':
        resize = T.ResizePad((height, width), interpolation=3)
    elif resize_mode == 'stretch':
        resize = T.Resize((height, width), interpolation=3)
    else:
        raise ValueError('resize_mode must be pad or stretch')
    fixed = T.Compose([resize, T.ToTensor(), T.Normalize(mean=mean, std=std)])
    train = T.Compose([
        resize, T.RandomHorizontalFlip(p=hflip_prob), T.Pad(crop_padding),
        T.RandomCrop((height, width)), T.ToTensor(), T.Normalize(mean=mean, std=std),
        T.RandomErasing(probability=erasing_prob, mean=mean),
    ])
    return train, fixed


@dataclass
class CategoryLoaders:
    view: CategoryStage
    train: object
    prototype: object

    def set_epoch(self, epoch):
        self.train.batch_sampler.set_epoch(epoch)


def build_stage_loaders(stage, batch_size=32, num_instances=4, workers=0,
                        prototype_batch_size=128, seed=0, train_transform=None,
                        reference_transform=None, transform_options=None, pin_memory=False):
    """Build loaders only for `stage`. Caller must release them at stage end.

    The prototype loader is sequential, drop_last=False, and visits each
    current training row once. It never falls back to evaluation data.
    Custom reference_transform must be deterministic; callers own that contract.
    """
    import torch
    from torch.utils.data import DataLoader

    if not isinstance(stage, StageView):
        raise TypeError('build_stage_loaders accepts one StageView, not a whole stream')
    if type(workers) is not int or workers < 0 or type(prototype_batch_size) is not int or prototype_batch_size <= 0:
        raise ValueError('workers must be non-negative; prototype_batch_size must be positive')
    if (train_transform is None) != (reference_transform is None):
        raise ValueError('provide both train_transform and reference_transform, or neither')
    if train_transform is not None and transform_options:
        raise ValueError('transform_options cannot be combined with custom transforms')
    if train_transform is None:
        train_transform, reference_transform = make_category_transforms(**(transform_options or {}))
    loaders = {}
    for view in stage.categories:
        if view.stage_id != stage.stage_id or any(
                s.stage_id != stage.stage_id or s.category != view.category or s.split != 'train'
                for s in view.samples):
            raise StreamProtocolError('stage loader contains non-current or non-training records')
        sampler = CategoryIdentityBatchSampler(view, batch_size, num_instances,
                                               _seed_for(seed, stage.stage_id, view.category))
        labels = view.label_map
        train_dataset = CategoryImageDataset(view.samples, labels, train_transform)
        prototype_dataset = CategoryImageDataset(view.samples, labels, reference_transform)
        common = dict(num_workers=workers, pin_memory=pin_memory,
                      collate_fn=collate_category_samples, worker_init_fn=_seed_worker)
        train_generator = torch.Generator().manual_seed(sampler.seed)
        prototype_generator = torch.Generator().manual_seed(sampler.seed + 1)
        loaders[view.category] = CategoryLoaders(
            view,
            DataLoader(train_dataset, batch_sampler=sampler, generator=train_generator, **common),
            DataLoader(prototype_dataset, batch_size=prototype_batch_size, shuffle=False,
                       drop_last=False, generator=prototype_generator, **common),
        )
    return loaders


def build_stage_prototype_loaders(stage, reference_transform, batch_size=128, workers=0):
    """Sequential current-train-only extraction, including one-image identities.

    Unlike the training builder this never constructs a P x K sampler.
    The caller supplies the frozen encoder's deterministic preprocessing.
    """
    import torch
    from torch.utils.data import DataLoader

    if not isinstance(stage, StageView):
        raise TypeError('expected one StageView')
    if type(batch_size) is not int or batch_size <= 0 or type(workers) is not int or workers < 0:
        raise ValueError('batch_size must be positive and workers non-negative integers')
    loaders = {}
    if not stage.categories:
        raise StreamProtocolError('empty stage')
    for view in stage.categories:
        if view.category in loaders or view.stage_id != stage.stage_id or any(
                s.stage_id != stage.stage_id or s.category != view.category or s.split != 'train'
                for s in view.samples):
            raise StreamProtocolError('prototype loader requires distinct current training categories')
        loaders[view.category] = DataLoader(
            CategoryImageDataset(view.samples, view.label_map, reference_transform),
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers,
            collate_fn=collate_category_samples, worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(_seed_for(0, stage.stage_id, view.category)),
        )
    return loaders


def build_evaluation_loaders(evaluation, reference_transform, batch_size=128, workers=0):
    """Explicit evaluation-only builder with a shared query/gallery PID map.

    Metadata stays available for metric calculation. Inference code must pass
    only batch['images'] to the router/model, never batch['categories'].
    """
    from torch.utils.data import DataLoader
    from .category_stream import EvaluationView

    if not isinstance(evaluation, EvaluationView):
        raise TypeError('expected EvaluationView')
    return {
        subset: DataLoader(
            CategoryImageDataset(getattr(evaluation, subset), evaluation.label_map, reference_transform),
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers,
            collate_fn=collate_category_samples, worker_init_fn=_seed_worker,
        ) for subset in ('query', 'gallery')
    }
