from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomIdentityBatchSampler(Sampler):
    """Build complete PK batches with a fixed number of images per identity.

    This follows the sampler used by the validated PAD object-ReID pipeline.
    In contrast to ``RandomMultipleGallerySampler``, every emitted mini-batch
    contains exactly ``batch_size / num_instances`` identities and
    ``num_instances`` samples for each identity. Identities with too few
    images are sampled with replacement.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError('batch_size must be >= num_instances')
        if batch_size % num_instances != 0:
            raise ValueError(
                'batch_size must be divisible by num_instances, got '
                '{} and {}'.format(batch_size, num_instances)
            )
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, sample in enumerate(self.data_source):
            if len(sample) < 2:
                raise ValueError(
                    'ReID samples need at least (path, pid, ...)'
                )
            self.index_dic[sample[1]].append(index)
        self.pids = list(self.index_dic.keys())
        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError(
                'dataset has {} identities but a batch requires {}'.format(
                    len(self.pids),
                    self.num_pids_per_batch,
                )
            )

        self.length = 0
        for pid in self.pids:
            count = max(len(self.index_dic[pid]), self.num_instances)
            self.length += count - count % self.num_instances
        self.length -= self.length % self.batch_size

    def __iter__(self):
        batch_indices = defaultdict(list)
        for pid in self.pids:
            indices = copy.deepcopy(self.index_dic[pid])
            if len(indices) < self.num_instances:
                indices = np.random.choice(
                    indices,
                    size=self.num_instances,
                    replace=True,
                ).tolist()
            random.shuffle(indices)
            group = []
            for index in indices:
                group.append(index)
                if len(group) == self.num_instances:
                    batch_indices[pid].append(group)
                    group = []

        available_pids = [
            pid for pid in self.pids if batch_indices[pid]
        ]
        result = []
        while len(available_pids) >= self.num_pids_per_batch:
            selected = random.sample(
                available_pids,
                self.num_pids_per_batch,
            )
            for pid in selected:
                result.extend(batch_indices[pid].pop(0))
                if not batch_indices[pid]:
                    available_pids.remove(pid)
        self.length = len(result)
        return iter(result)

    def __len__(self):
        return self.length


class MultiDomainRandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances

        self.domain2pids = defaultdict(list)
        self.pid2index = defaultdict(list)

        for index, (_, pid, _, domain) in enumerate(data_source):
            if pid not in self.domain2pids[domain]:
                self.domain2pids[domain].append(pid)
            self.pid2index[pid].append(index)

        self.pids = list(self.pid2index.keys())
        self.domains = list(sorted(self.domain2pids.keys()))

        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):

        ret = []
        domain2pids = copy.deepcopy(self.domain2pids)
        for _ in range(8):
            for domain in self.domains:
                pids = np.random.choice(domain2pids[domain], size=8, replace=False)
                for pid in pids:
                    idxs = copy.deepcopy(self.pid2index[pid])
                    idxs = np.random.choice(idxs, size=2, replace=False)
                    ret.extend(idxs)
        return iter(ret)

class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        try :
            for index, (_, pid, cam,frame) in enumerate(data_source):
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)
        except:
            for index, (_, pid, cam,_,_) in enumerate(data_source):
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)
        #print(self.pid_cam)
        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam, _ = self.data_source[i]
            #_, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)
