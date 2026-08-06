from __future__ import absolute_import

import os.path as osp

from PIL import Image
from torch.utils.data import Dataset


class Preprocessor(Dataset):
    """Load one image and return the standard five-field ReID sample."""

    def __init__(self, dataset, root=None, transform=None):
        super().__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if len(sample) < 4:
            raise ValueError(
                'ReID samples need at least (path, pid, camid, domain)'
            )
        fname, pid, camid, domain = sample[:4]
        fpath = osp.join(self.root, fname) if self.root is not None else fname

        image = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, fpath, pid, camid, domain
