"""Label-free hard routing. Public inference accepts tensors only."""

import math

import torch
from torch.nn import functional as F

from reid.memory.finch_modes import unit_rows
from reid.memory.prototype_views import memory_summaries


def normalized(features):
    if (not isinstance(features, torch.Tensor) or features.ndim != 2 or min(features.shape) == 0
            or not torch.isfinite(features).all() or (features.float().norm(dim=1) <= 1e-8).any()):
        raise ValueError('features must be finite, nonzero [N,D] tensors')
    return F.normalize(features.float(), dim=1)


class PrototypeRouter:
    def __init__(self, summaries, beta=.5):
        if type(beta) not in (float, int) or not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('beta must be in [0,1]')
        if not summaries:
            raise ValueError('routing requires at least one seen category')
        self.categories, self.beta = tuple(sorted(summaries)), float(beta)
        self.centers, self.modes = [], []
        for category in self.categories:
            center, modes = summaries[category]['category_prototype'], summaries[category]['mode_prototypes']
            unit_rows(center[None])
            unit_rows(modes)
            if modes.shape[1] != center.numel():
                raise ValueError('mode/center dimension mismatch')
            self.centers.append(center.detach().clone())
            self.modes.append(modes.detach().clone())
        self.centers = torch.stack(self.centers)

    def scores(self, reference_features):
        with torch.autocast(device_type=reference_features.device.type, enabled=False):
            features = normalized(reference_features)
            centers = normalized(self.centers.to(features.device))
            global_scores = (features @ centers.T).clamp(-1, 1)
            mode_scores = torch.stack([(features @ normalized(m.to(features.device)).T).clamp(-1, 1).max(1).values
                                       for m in self.modes], dim=1)
            return self.beta * global_scores + (1 - self.beta) * mode_scores

    def predict(self, reference_features):
        scores = self.scores(reference_features)
        # torch.argmax picks first on ties; categories are sorted canonically.
        indices = scores.argmax(dim=1)
        return tuple(self.categories[i] for i in indices.tolist()), scores


class RoutedEncoder:
    def __init__(self, model, memory, beta=.5, summary='ecpm'):
        if model.reference_signature != memory.summary()['reference_signature']:
            raise ValueError('router memory/reference mismatch')
        self.router = PrototypeRouter(memory_summaries(memory, summary), beta)
        if set(model.categories) != set(self.router.categories):
            raise ValueError('router candidates must equal the committed adapter registry')
        self.model = model

    @torch.no_grad()
    def __call__(self, images):
        previous = self.model.training
        self.model.eval()
        try:
            prediction, scores = self.router.predict(self.model.encode_reference(images))
            descriptors = torch.empty((len(images), self.model.feature_dim), device=images.device, dtype=torch.float32)
            for category in self.router.categories:
                indices = torch.tensor([i for i, c in enumerate(prediction) if c == category], device=images.device, dtype=torch.long)
                if len(indices):
                    descriptors[indices] = normalized(self.model.encode_category(images[indices], category))
            return descriptors, prediction, scores
        finally:
            self.model.train(previous)
