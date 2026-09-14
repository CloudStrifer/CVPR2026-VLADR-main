"""Ablation views derived from validated identity memory, never new images."""

import copy

import torch

from .finch_modes import _unit_mean, prototype_drift


def check_summary(mode):
    if mode not in ('ecpm', 'identity_mean'):
        raise ValueError('summary must be ecpm or identity_mean')


def identity_mean_summaries(rows):
    result = {}
    for category in sorted({r['identity_key'][0] for r in rows}):
        vectors = torch.stack([r['vector'] for r in rows if r['identity_key'][0] == category])
        center = _unit_mean(vectors, 'identity-mean ablation: ' + category)
        result[category] = dict(category_prototype=center, mode_prototypes=center[None])
    return result


def memory_summaries(memory, mode='ecpm'):
    check_summary(mode)
    if mode == 'ecpm':
        return memory.snapshot()
    return identity_mean_summaries(memory.state_dict()['identity_memory']['rows'])


def control_view(memory, candidate, mode='ecpm'):
    """Call only after validate_candidate. Does not mutate or commit ECPM."""
    check_summary(mode)
    if mode == 'ecpm':
        return copy.deepcopy(candidate)
    old_rows = memory.state_dict()['identity_memory']['rows']
    old = identity_mean_summaries(old_rows)
    all_new = identity_mean_summaries(old_rows + candidate['rows'])
    updates = {c: all_new[c] for c in candidate['category_updates']}
    return dict(old_categories=old, category_updates=updates,
                drifts={c: prototype_drift(old[c]['category_prototype'], v['category_prototype'])
                        if c in old else None for c, v in updates.items()})
