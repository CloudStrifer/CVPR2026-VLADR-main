"""Fixed first-partition FINCH and the ECPM equal-mode aggregation formulas."""

import hashlib
import importlib.metadata
import inspect
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


@dataclass(frozen=True)
class FinchConfig:
    # First partition only. No ANN switch or automatic hierarchy selection.
    chunk_size: int = 256

    def __post_init__(self):
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError('FINCH chunk_size must be a positive integer')


def finch_runtime():
    """Fail explicitly without the pinned official implementation, even for N=1."""
    try:
        version = importlib.metadata.version('finch-clust')
        if version != '0.2.3':
            raise ImportError('ECPM requires finch-clust==0.2.3, found ' + version)
        from finch.finch import clust_rank, get_clust
        versions = {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')}
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise ImportError('Install ECPM dependencies with python -m pip install -r requirements.txt; '
                          'the official finch-clust==0.2.3 is required') from error
    return dict(package='finch-clust', version=version, dependencies=versions,
                source_sha256=hashlib.sha256(Path(inspect.getfile(clust_rank)).read_bytes()).hexdigest())


def unit_rows(vectors):
    if (not isinstance(vectors, torch.Tensor) or vectors.ndim != 2 or min(vectors.shape) < 1
            or vectors.dtype != torch.float32 or vectors.device.type != 'cpu' or vectors.requires_grad):
        raise ValueError('expected nonempty detached CPU FP32 [N,D] prototypes')
    if not torch.isfinite(vectors).all() or not torch.allclose(
            vectors.norm(dim=1), torch.ones(vectors.shape[0]), atol=1e-5, rtol=1e-5):
        raise ValueError('prototypes must be finite unit vectors')


def _unit_mean(vectors, description):
    mean = vectors.mean(dim=0)
    norm = mean.norm()
    if not torch.isfinite(norm) or norm <= 1e-8:
        raise FloatingPointError('near-zero or non-finite mean for ' + description)
    return mean / norm


def aggregate_modes(vectors, labels):
    """g_k = Norm(mean(p in k)); m = Norm(mean(g_k)), NOT size-weighted."""
    unit_rows(vectors)
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
            or labels.device.type != 'cpu' or labels.shape != (len(vectors),)):
        raise ValueError('labels must be CPU int64 [N]')
    unique = torch.unique(labels, sorted=True)
    if not torch.equal(unique, torch.arange(len(unique))):
        raise ValueError('cluster labels must be contiguous from zero')
    sizes = torch.bincount(labels)
    modes = torch.stack([_unit_mean(vectors[labels == k], 'mode {}'.format(k)) for k in range(len(unique))])
    center = _unit_mean(modes, 'equal-mode category center')
    return modes, center, sizes


def prototype_drift(old, new):
    unit_rows(old.unsqueeze(0))
    unit_rows(new.unsqueeze(0))
    if old.shape != new.shape:
        raise ValueError('old/new category dimensions differ')
    # Divide by both norms, including their small FP32 normalization residuals.
    similarity = torch.nn.functional.cosine_similarity(old[None], new[None]).item()
    return 1.0 - min(1.0, max(-1.0, similarity))


def finch_first_partition(vectors, config=FinchConfig()):
    """Official FINCH first partition with exact, non-self 1-NN in cosine space.

    Feed chunked exact ranks to the official clust_rank/get_clust used by FINCH.
    This avoids unused higher partitions and a full N x N distance allocation.
    Row-order ties select the first index, as numpy.argmin does upstream.
    Caller must sort rows by stable identity key. Duplicates remain separate IDs.
    Runtime remains quadratic in N; this is not approximate nearest neighbors.
    """
    unit_rows(vectors)
    runtime = finch_runtime()
    from finch.finch import clust_rank, get_clust
    from sklearn.metrics import pairwise_distances

    start_time = perf_counter()
    n = len(vectors)
    if n == 1:
        labels = torch.zeros(1, dtype=torch.long)
    else:
        data = vectors.numpy()
        ranks = np.empty(n, dtype=np.int32)
        for start in range(0, n, config.chunk_size):
            stop = min(n, start + config.chunk_size)
            distances = pairwise_distances(data[start:stop], data, metric='cosine')
            distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
            ranks[start:stop] = np.argmin(distances, axis=1)
        adjacency, _, _, _ = clust_rank(data, initial_rank=ranks, metric='cosine')
        partition, _ = get_clust(adjacency, [], min_sim=None)
        # Canonical cluster IDs in first-member order, independent of backend labels.
        mapping = {}
        labels = torch.tensor([mapping.setdefault(int(k), len(mapping)) for k in partition], dtype=torch.long)
    return labels, dict(runtime=runtime, algorithm='FINCH', partition=0, distance='cosine',
                        neighbors='exact_chunked', tie_break='first_identity_key',
                        chunk_size=config.chunk_size, singleton=(n == 1),
                        clustering_seconds=perf_counter() - start_time,
                        distance_block_bytes=0 if n == 1 else min(n, config.chunk_size) * n * 4)


def build_category_modes(rows, stage_id, config=FinchConfig()):
    rows = sorted(rows, key=lambda row: row['identity_key'])
    if not rows or len({row['identity_key'][0] for row in rows}) != 1:
        raise ValueError('modes require one nonempty category')
    keys = tuple(row['identity_key'] for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate identity in mode memory')
    vectors = torch.stack([row['vector'] for row in rows])
    labels, details = finch_first_partition(vectors, config)
    try:
        modes, center, sizes = aggregate_modes(vectors, labels)
    except FloatingPointError as error:
        raise FloatingPointError('{} at {}/{}'.format(error, stage_id, keys[0][0])) from error
    return dict(identity_keys=keys, labels=labels, mode_prototypes=modes,
                category_prototype=center, cluster_sizes=sizes, last_updated_stage=stage_id,
                clustering=details)
