from __future__ import print_function, absolute_import

from .classification import accuracy
from .rank import fast_evaluate_rank
from .metric import tensor_euclidean_dist, tensor_cosine_dist
from .distance import compute_distance_matrix


_LEGACY_REID_EXPORTS = {
    'ReIDEvaluator',
    'PrecisionRecall',
    'np_cosine_dist',
    'np_euclidean_dist',
}


def __getattr__(name):
    """Import the sklearn-based legacy evaluators only when requested."""
    if name not in _LEGACY_REID_EXPORTS:
        raise AttributeError(name)
    from . import reid

    return getattr(reid, name)
