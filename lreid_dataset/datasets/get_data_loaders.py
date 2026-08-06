import math

from torch.utils.data import DataLoader

import lreid_dataset.datasets as datasets
from reid.utils.data import IterLoader, Preprocessor
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomIdentityBatchSampler


def get_data(
    domain,
    data_dir,
    height,
    width,
    batch_size,
    workers,
    num_instances,
    select_num=0,
    resize_mode='stretch',
):
    dataset = datasets.create(domain, data_dir)
    name = domain['name'] if isinstance(domain, dict) else domain
    hflip_prob = 0.5
    random_erasing_prob = 0.5
    crop_padding = 10
    if isinstance(domain, dict):
        select_num = int(domain.get('max_train_ids', select_num))
        hflip_prob = float(domain.get('hflip_prob', hflip_prob))
        random_erasing_prob = float(
            domain.get('random_erasing_prob', random_erasing_prob)
        )
        crop_padding = int(domain.get('crop_padding', crop_padding))
    if not 0.0 <= hflip_prob <= 1.0:
        raise ValueError('hflip_prob must be in [0, 1]')
    if not 0.0 <= random_erasing_prob <= 1.0:
        raise ValueError('random_erasing_prob must be in [0, 1]')
    if crop_padding < 0:
        raise ValueError('crop_padding must be non-negative')

    if select_num > 0:
        train = [
            sample
            for sample in dataset.train
            if sample[1] < select_num
        ]
        dataset.train = train
        dataset.num_train_pids = len({sample[1] for sample in train})
        dataset.num_train_imgs = len(train)

    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if resize_mode == 'pad':
        resize = T.ResizePad((height, width), interpolation=3)
    elif resize_mode == 'stretch':
        resize = T.Resize((height, width), interpolation=3)
    else:
        raise ValueError(
            "resize_mode must be 'stretch' or 'pad', got {!r}".format(
                resize_mode
            )
        )

    stage1_transform = T.Compose([
        resize,
        T.ToTensor(),
        normalizer,
    ])
    stage2_transform = T.Compose([
        resize,
        T.RandomHorizontalFlip(p=hflip_prob),
        T.Pad(crop_padding),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(
            probability=random_erasing_prob,
            mean=[0.485, 0.456, 0.406],
        ),
    ])
    test_transform = T.Compose([
        resize,
        T.ToTensor(),
        normalizer,
    ])

    train_set = sorted(dataset.train)
    loader_train_set = train_set or sorted(
        list(set(dataset.query) | set(dataset.gallery))
    )
    num_classes = dataset.num_train_pids
    if num_classes == 0:
        num_classes = len({sample[1] for sample in loader_train_set})

    use_identity_sampler = num_instances > 0
    sampler = (
        RandomIdentityBatchSampler(
            loader_train_set,
            batch_size,
            num_instances,
        )
        if use_identity_sampler
        else None
    )
    sample_count = len(sampler) if sampler is not None else len(loader_train_set)
    iterations = max(1, int(math.ceil(sample_count / float(batch_size))))
    drop_last = sample_count >= batch_size

    def make_iter_loader(transform):
        data_loader = DataLoader(
            Preprocessor(
                loader_train_set,
                root=dataset.images_dir,
                transform=transform,
            ),
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            shuffle=not use_identity_sampler,
            pin_memory=True,
            drop_last=drop_last,
        )
        return IterLoader(data_loader, length=iterations)

    clip_stage1_loader = make_iter_loader(stage1_transform)
    clip_stage2_loader = make_iter_loader(stage2_transform)
    test_loader = DataLoader(
        Preprocessor(
            list(set(dataset.query) | set(dataset.gallery)),
            root=dataset.images_dir,
            transform=test_transform,
        ),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
    )
    init_loader = DataLoader(
        Preprocessor(
            loader_train_set,
            root=dataset.images_dir,
            transform=test_transform,
        ),
        batch_size=128,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return [
        dataset,
        num_classes,
        clip_stage1_loader,
        clip_stage2_loader,
        test_loader,
        init_loader,
        name,
    ]


def build_data_loaders(cfg, training_set, testing_only_set):
    common = {
        'data_dir': cfg.data_dir,
        'height': cfg.height,
        'width': cfg.width,
        'batch_size': cfg.batch_size,
        'workers': cfg.workers,
        'num_instances': cfg.num_instances,
        'resize_mode': getattr(cfg, 'resize_mode', 'stretch'),
    }
    max_train_ids = getattr(cfg, 'max_train_ids', 0)

    training_loaders = [
        get_data(
            domain,
            select_num=max_train_ids,
            **common,
        )
        for domain in training_set
    ]
    testing_loaders = [
        get_data(
            domain,
            select_num=0,
            **common,
        )
        for domain in testing_only_set
    ]
    return training_loaders, testing_loaders
