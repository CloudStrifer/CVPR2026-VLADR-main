from __future__ import absolute_import

import importlib
import warnings

from .manifest_reid import ManifestReID


_LEGACY_DATASETS = {
    'market1501': ('.market1501', 'IncrementalSamples4market'),
    'dukemtmc': ('.dukemtmcreid', 'IncrementalSamples4duke'),
    'msmt17': ('.msmt17', 'IncrementalSamples4msmt17'),
    'cuhk_sysu': ('.cuhksysu', 'IncrementalSamples4subcuhksysu'),
    'cuhk03': ('.cuhk03', 'IncrementalSamples4cuhk03'),
    'cuhk01': ('.cuhk01', 'IncrementalSamples4cuhk01'),
    'grid': ('.grid', 'IncrementalSamples4grid'),
    'sense': ('.sensereid', 'IncrementalSamples4sensereid'),
    'ilids': ('.ilids', 'IncrementalSamples4ilids'),
    'viper': ('.viper', 'IncrementalSamples4viper'),
    'prid': ('.prid', 'IncrementalSamples4prid'),
    'cuhk02': ('.cuhk02', 'IncrementalSamples4cuhk02'),
    'lpw': ('.lpw', 'IncrementalSamples4lpw'),
}


def names():
    return sorted(_LEGACY_DATASETS)


def _legacy_dataset_class(name):
    module_name, class_name = _LEGACY_DATASETS[name]
    module = importlib.import_module(module_name, package=__name__)
    return getattr(module, class_name)


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if isinstance(name, dict):
        return ManifestReID(root, name, *args, **kwargs)
    if name not in _LEGACY_DATASETS:
        raise KeyError("Unknown dataset:", name)
    return _legacy_dataset_class(name)(root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
