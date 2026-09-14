"""Atomic checkpoints and RNG/log transactions for the single-process runner."""

import hashlib
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def file_digest(path, limit=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        remaining = limit
        while remaining is None or remaining > 0:
            block = handle.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining not in (None, 0):
        raise ValueError('log shorter than checkpoint prefix: {}'.format(path))
    return digest.hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=(name, keys.tolist(), pos, has_gauss, cached),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    name, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])


def exact_runtime(device):
    # Must be called before constructing CUDA models/performing GEMMs.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('exact resume supports one CPU or CUDA device')
    import PIL
    import torchvision
    root = Path(__file__).resolve().parents[2]
    sources = [root / 'train_category_progressive.py']
    for folder in ('reid', 'lreid_dataset'):
        sources.extend(sorted((root / folder).rglob('*.py')))
    return dict(python=platform.python_version(), platform=platform.platform(), processor=platform.processor(), torch=str(torch.__version__),
                numpy=np.__version__, pillow=PIL.__version__, torchvision=str(torchvision.__version__),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=str(device),
                device_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                source_sha256={str(p.relative_to(root)): file_digest(p) for p in sources if p.exists()})


def append_event(path, event):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def log_positions(output, names):
    result = {}
    for name in names:
        path = Path(output) / name
        if path.exists():
            # Checkpoint never becomes durable before its log prefix.
            with path.open('ab') as handle:
                handle.flush()
                os.fsync(handle.fileno())
            result[name] = dict(bytes=path.stat().st_size, sha256=file_digest(path))
    return result


def restore_logs(output, names, positions):
    """Validate all prefixes, archive speculative tails, then truncate them."""
    if set(positions) - set(names):
        raise ValueError('unknown log path in checkpoint')
    output = Path(output)
    empty = hashlib.sha256(b'').hexdigest()
    plans = []
    for name in names:
        path = output / name
        position = positions.get(name, dict(bytes=0, sha256=empty))
        size = position['bytes']
        if type(size) is not int or size < 0:
            raise ValueError('invalid log offset')
        if not path.exists():
            if size:
                raise ValueError('missing checkpoint log: {}'.format(name))
            continue
        if file_digest(path, size) != position['sha256']:
            raise ValueError('checkpoint log prefix changed: {}'.format(name))
        if path.stat().st_size > size:
            plans.append((path, size))
    archives = []
    for path, size in plans:
        with path.open('rb') as source, tempfile.NamedTemporaryFile(
                dir=output, prefix=path.name + '.recovered-', suffix='.jsonl', delete=False) as target:
            source.seek(size)
            while block := source.read(1024 * 1024):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
            archives.append(Path(target.name).name)
        with path.open('r+b') as handle:
            handle.truncate(size)
            handle.flush()
            os.fsync(handle.fileno())
    return archives
