"""Frozen reference ViT plus persistent, independent object-category adapters.

This path returns raw projected CLS features in both train and eval modes.
It deliberately does not include legacy prompts, BN necks, or identity anchors.
"""

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .CLIP_ReID.model.clip.model import VisionTransformer


def _category_name(category):
    if not isinstance(category, str) or not category.strip():
        raise ValueError('category must be a non-empty string')
    return category.strip()


def _cpu_copy(state):
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _tensor_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode('utf-8'))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceConfig:
    weight_id: str
    input_size: tuple = (224, 224)
    resize_mode: str = 'pad'
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self):
        if not isinstance(self.weight_id, str) or not self.weight_id.strip():
            raise ValueError('reference weight_id is required')
        object.__setattr__(self, 'input_size', tuple(self.input_size))
        object.__setattr__(self, 'mean', tuple(self.mean))
        object.__setattr__(self, 'std', tuple(self.std))
        if len(self.input_size) != 2 or any(type(n) is not int or n <= 0 for n in self.input_size):
            raise ValueError('input_size must be two positive integers')
        if self.resize_mode not in ('pad', 'stretch'):
            raise ValueError('resize_mode must be pad or stretch')
        if (len(self.mean) != 3 or len(self.std) != 3
                or any(not math.isfinite(v) for v in self.mean + self.std)
                or any(v <= 0 for v in self.std)):
            raise ValueError('normalization needs finite RGB means and positive standard deviations')


class CategoryAdapterBank(nn.Module):
    """Own one clean reference visual encoder and its category adapter bank.

    `set_trainable_categories` controls gradients, independently of train/eval.
    Creating a category does not make it trainable. Re-registering it preserves
    the exact same Parameter objects. Temporary heads live outside snapshots.
    """

    def __init__(self, visual, reference, last_blocks=4, bottleneck_dim=64, scale=1.0):
        super().__init__()
        if not isinstance(visual, VisionTransformer):
            raise TypeError('expected the repository CLIP VisionTransformer')
        if len(visual.transformer.resblocks) != 12:
            raise ValueError('the current CLIP-ReID forward requires a 12-block visual encoder')
        if visual.proj is None or visual.domain_adapter_names():
            raise ValueError('reference encoder must have a projection and no pre-existing adapters')
        if not isinstance(reference, ReferenceConfig):
            raise TypeError('reference must be ReferenceConfig')
        if type(last_blocks) is not int or not 1 <= last_blocks <= 12:
            raise ValueError('last_blocks must be an integer in [1, 12]')
        if type(bottleneck_dim) is not int or bottleneck_dim <= 0:
            raise ValueError('bottleneck_dim must be a positive integer')
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be finite and positive')
        actual_grid = tuple((n - k) // s + 1 for n, k, s in zip(
            reference.input_size, visual.conv1.kernel_size, visual.conv1.stride))
        if actual_grid != (visual.h_resolution, visual.w_resolution):
            raise ValueError('reference input size and visual positional grid do not match')

        self.visual = visual
        self.reference = reference
        self.feature_dim = int(visual.proj.shape[1])
        self._adapter_config = dict(last_blocks=last_blocks, bottleneck_dim=bottleneck_dim, scale=float(scale))
        self._category_keys = {}
        self._trainable_categories = frozenset()
        self.temporary_heads = nn.ModuleDict()
        self._reference_hash = _tensor_digest(self._reference_state())
        self.visual.set_active_domain_adapter(None)
        self._enforce_policy()

    @property
    def categories(self):
        return tuple(sorted(self._category_keys))

    @property
    def trainable_categories(self):
        return tuple(sorted(self._trainable_categories))

    @property
    def adapter_config(self):
        return dict(self._adapter_config)

    @property
    def reference_signature(self):
        metadata = {'config': asdict(self.reference), 'weights': self._reference_hash,
                    'feature': 'raw_projected_cls', 'feature_dim': self.feature_dim}
        return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest()

    def _reference_state(self):
        return {name: value for name, value in self.visual.state_dict().items()
                if '.domain_adapters.' not in name}

    def assert_reference_unchanged(self):
        if _tensor_digest(self._reference_state()) != self._reference_hash:
            raise RuntimeError('reference weights/buffers changed; existing prototype coordinates are invalid')

    def category_key(self, category):
        category = _category_name(category)
        if category not in self._category_keys:
            raise KeyError('unknown category {!r}; register it explicitly first'.format(category))
        return self._category_keys[category]

    def _adapter_modules(self, key):
        return [(i, block.domain_adapters[key])
                for i, block in enumerate(self.visual.transformer.resblocks)
                if key in block.domain_adapters]

    def add_category(self, category):
        """Return True for a newly created category, False for a recurring one."""
        category = _category_name(category)
        if category in self._category_keys:
            return False
        key = 'cat_' + hashlib.sha256(category.encode('utf-8')).hexdigest()
        if key in self.visual.domain_adapter_names():
            raise ValueError('category key collision or externally registered adapter')
        self.visual.add_domain_adapter(key, **self._adapter_config)
        self._category_keys[category] = key
        self._enforce_policy()
        return True

    def _enforce_policy(self):
        # eval() also fixes shared running buffers and any future dropout.
        self.visual.eval()
        enabled = set()
        for category in self._trainable_categories:
            key = self.category_key(category)
            enabled.update(id(p) for p in self.visual.domain_adapter_parameters(key))
            for _, adapter in self._adapter_modules(key):
                adapter.train(self.training)
        for parameter in self.visual.parameters():
            trainable = id(parameter) in enabled
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None  # stale gradients can otherwise trigger optimizer updates
        for category, key in self._category_keys.items():
            if key in self.temporary_heads:
                head = self.temporary_heads[key]
                trainable = category in self._trainable_categories
                head.train(self.training and trainable)
                for parameter in head.parameters():
                    parameter.requires_grad_(trainable)
                    if not trainable:
                        parameter.grad = None

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError('training mode must be boolean')
        self.training = mode
        if hasattr(self, '_category_keys'):
            self._enforce_policy()
        return self

    def set_trainable_categories(self, categories):
        if isinstance(categories, str):
            raise TypeError('pass an iterable of category names, not one string')
        names = frozenset(_category_name(c) for c in categories)
        for category in names:
            self.category_key(category)
        self._trainable_categories = names
        self._enforce_policy()

    def adapter_parameters(self, category):
        return self.visual.domain_adapter_parameters(self.category_key(category))

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @contextmanager
    def _selected(self, key):
        previous = self.visual.get_active_domain_adapter()
        self._enforce_policy()
        self.visual.set_active_domain_adapter(key)
        try:
            yield
        finally:
            self.visual.set_active_domain_adapter(previous)

    def _projected_cls(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('images must have shape [B, 3, H, W]')
        if tuple(images.shape[-2:]) != self.reference.input_size:
            raise ValueError('image size does not match frozen reference preprocessing')
        if not images.is_floating_point():
            raise TypeError('images must be preprocessed floating-point tensors')
        images = images.to(dtype=self.visual.conv1.weight.dtype)
        _, _, projected = self.visual(images, None)
        return projected[:, 0]  # raw, not normalized, not concatenated, and no BN

    def encode_reference(self, images):
        # Reference extraction must not depend on a surrounding training AMP
        # context; the factory's fixed FP32 reference remains FP32 throughout.
        with self._selected(None), torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            return self._projected_cls(images)

    def encode_category(self, images, category):
        with self._selected(self.category_key(category)):
            # Do not use no_grad: frozen blocks must propagate adapter gradients.
            return self._projected_cls(images)

    def forward(self, images, category=None):
        return self.encode_reference(images) if category is None else self.encode_category(images, category)

    def make_transforms(self, **augmentation):
        from lreid_dataset.category_stream_loaders import make_category_transforms

        if set(augmentation) - {'hflip_prob', 'crop_padding', 'erasing_prob'}:
            raise ValueError('reference preprocessing is fixed; only training augmentation may be configured')
        return make_category_transforms(
            height=self.reference.input_size[0], width=self.reference.input_size[1],
            resize_mode=self.reference.resize_mode, mean=self.reference.mean, std=self.reference.std,
            **augmentation)

    def create_temporary_head(self, category, num_identities):
        key = self.category_key(category)
        if type(num_identities) is not int or num_identities <= 0:
            raise ValueError('num_identities must be a positive integer')
        if key in self.temporary_heads:
            raise ValueError('discard the previous temporary head before creating a new one')
        head = nn.Linear(self.feature_dim, num_identities, bias=False)
        nn.init.normal_(head.weight, std=0.001)
        # Heads and default adapters use FP32; mixed precision is handled by autocast.
        self.temporary_heads[key] = head.to(device=self.visual.proj.device)
        self._enforce_policy()
        return head

    def classify(self, features, category):
        head = self.temporary_heads[self.category_key(category)]
        return head(features.to(dtype=head.weight.dtype))

    def discard_temporary_heads(self):
        self.temporary_heads = nn.ModuleDict()

    def _adapter_state(self, category):
        return {'{}.{}'.format(i, name): value
                for i, module in self._adapter_modules(self.category_key(category))
                for name, value in module.state_dict().items()}

    def export_adapter(self, category):
        """Independent CPU snapshot, suitable for future teacher/transfer use."""
        return {'schema_version': 1, 'source_category': _category_name(category),
                'reference_signature': self.reference_signature,
                'adapter_config': self.adapter_config,
                'state': _cpu_copy(self._adapter_state(category))}

    def _validate_adapter_snapshot(self, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1:
            raise ValueError('invalid adapter snapshot schema')
        if snapshot.get('reference_signature') != self.reference_signature:
            raise ValueError('adapter snapshot belongs to a different reference encoder/preprocessing')
        if snapshot.get('adapter_config') != self._adapter_config:
            raise ValueError('adapter architecture or scale does not match')
        expected = {}
        width = self.visual.proj.shape[0]
        bottleneck = self._adapter_config['bottleneck_dim']
        for i in range(12 - self._adapter_config['last_blocks'], 12):
            expected.update({
                '{}.down.weight'.format(i): (bottleneck, width),
                '{}.down.bias'.format(i): (bottleneck,),
                '{}.up.weight'.format(i): (width, bottleneck),
                '{}.up.bias'.format(i): (width,),
            })
        state = snapshot.get('state')
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError('adapter snapshot parameter names do not match')
        for name, shape in expected.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError('invalid adapter tensor {}'.format(name))

    def load_adapter(self, category, snapshot):
        """Load into a new or explicitly named existing adapter; never share storage.

        Call at stage boundaries, before constructing its optimizer or any graph.
        Validation completes before category creation or parameter mutation.
        """
        self._validate_adapter_snapshot(snapshot)
        self.add_category(category)
        for i, module in self._adapter_modules(self.category_key(category)):
            prefix = '{}.'.format(i)
            state = {name[len(prefix):]: value for name, value in snapshot['state'].items()
                     if name.startswith(prefix)}
            module.load_state_dict(state, strict=True)
        self._enforce_policy()

    def copy_category(self, source, target):
        target = _category_name(target)
        if target in self._category_keys:
            raise ValueError('copy target already exists; recurring adapters must not be reinitialized')
        self.load_adapter(target, self.export_adapter(source))

    def export_bank(self):
        self.assert_reference_unchanged()
        return {'schema_version': 1, 'reference': asdict(self.reference),
                'reference_signature': self.reference_signature,
                'adapter_config': self.adapter_config,
                'category_keys': dict(self._category_keys),
                'adapters': {c: self.export_adapter(c) for c in self.categories}}

    def load_bank(self, snapshot):
        """Restore into an empty bank with the exact same reference; freeze all."""
        if self.categories or self.temporary_heads:
            raise ValueError('restore requires an empty category bank')
        if (not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1
                or snapshot.get('reference_signature') != self.reference_signature
                or snapshot.get('adapter_config') != self._adapter_config):
            raise ValueError('bank schema, reference or architecture mismatch')
        if ReferenceConfig(**snapshot.get('reference', {})) != self.reference:
            raise ValueError('bank reference metadata mismatch')
        adapters = snapshot.get('adapters')
        keys = snapshot.get('category_keys')
        if not isinstance(adapters, dict) or not isinstance(keys, dict) or set(adapters) != set(keys):
            raise ValueError('invalid category registry')
        for category, adapter in adapters.items():
            if _category_name(category) != category:
                raise ValueError('noncanonical category name')
            expected_key = 'cat_' + hashlib.sha256(category.encode('utf-8')).hexdigest()
            if keys[category] != expected_key:
                raise ValueError('category key mapping does not match')
            self._validate_adapter_snapshot(adapter)
            if adapter.get('source_category') != category:
                raise ValueError('bank category and adapter source metadata do not match')
        for category in sorted(adapters):
            self.load_adapter(category, adapters[category])
        self.set_trainable_categories([])

    def export_checkpoint(self):
        """Full model snapshot for offline restoration, excluding temporary heads.

        Training loop/optimizer/ECPM progress is intentionally a later-step concern.
        """
        bank = self.export_bank()
        visual = self.visual
        architecture = dict(
            h_resolution=visual.h_resolution, w_resolution=visual.w_resolution,
            patch_size=visual.conv1.kernel_size[0], stride_size=visual.conv1.stride[0],
            width=visual.proj.shape[0], layers=12,
            heads=visual.transformer.resblocks[0].attn.num_heads, output_dim=self.feature_dim)
        return {'schema_version': 1, 'visual_architecture': architecture,
                'reference_state': _cpu_copy(self._reference_state()), 'bank': bank}

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cpu'):
        if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != 1:
            raise ValueError('invalid category model checkpoint')
        state = checkpoint['reference_state']
        visual = VisionTransformer(**checkpoint['visual_architecture'])
        # The reference digest includes tensor dtype. Build-time mixed dtypes
        # are restored before loading, rather than silently converting to FP32.
        for name, parameter in visual.named_parameters():
            if name in state:
                parameter.data = parameter.data.to(dtype=state[name].dtype)
        visual.load_state_dict(state, strict=True)
        bank_state = checkpoint['bank']
        model = cls(visual, ReferenceConfig(**bank_state['reference']), **bank_state['adapter_config'])
        model.load_bank(bank_state)
        return model.to(device).eval()


def build_category_model(reference_checkpoint=None, input_size=(224, 224),
                         resize_mode='pad', mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225), last_blocks=4,
                         bottleneck_dim=64, scale=1.0, device='cpu'):
    """Build from a local OpenAI CLIP ViT-B/16 file; never download implicitly.

    Accepts TorchScript or a tensor state_dict. Only the visual tower is retained,
    with FP32 reference weights. Use autocast for training, not model.half() after
    creating reference prototypes. This factory does not consume old ReID/prompt
    checkpoints or existing OCIA adapters.
    """
    path = Path(reference_checkpoint or Path.home() / '.cache/clip/ViT-B-16.pt').expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('local CLIP ViT-B/16 checkpoint not found: {}'.format(path))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    reference = ReferenceConfig('sha256:' + digest.hexdigest(), input_size, resize_mode, mean, std)
    try:
        scripted = torch.jit.load(str(path), map_location='cpu').eval()
    except RuntimeError:
        state = torch.load(str(path), map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
    else:
        state = scripted.state_dict()
        del scripted
    if not isinstance(state, dict):
        raise ValueError('expected an OpenAI CLIP tensor state_dict')
    visual_state = {name[len('visual.'):]: value.detach().float()
                    for name, value in state.items()
                    if name.startswith('visual.') and isinstance(value, torch.Tensor)}
    del state
    if 'proj' not in visual_state or 'conv1.weight' not in visual_state:
        raise ValueError('checkpoint has no CLIP ViT visual tower')
    width, channels, patch_h, patch_w = visual_state['conv1.weight'].shape
    layers = sum(name.endswith('.attn.in_proj_weight') for name in visual_state)
    if (width, channels, patch_h, patch_w, layers, visual_state['proj'].shape[1]) != (768, 3, 16, 16, 12, 512):
        raise ValueError('this factory supports only CLIP ViT-B/16')
    height, width_pixels = reference.input_size
    grid_h, grid_w = (height - 16) // 16 + 1, (width_pixels - 16) // 16 + 1
    if min(grid_h, grid_w) < 1:
        raise ValueError('input size must cover at least one 16x16 patch')
    visual = VisionTransformer(grid_h, grid_w, 16, 16, 768, 12, 12, 512)
    position = visual_state['positional_embedding']
    old_grid = math.isqrt(position.shape[0] - 1)
    if old_grid ** 2 != position.shape[0] - 1:
        raise ValueError('source CLIP positional grid must be square')
    if (grid_h, grid_w) != (old_grid, old_grid):
        patch_position = position[1:].reshape(1, old_grid, old_grid, 768).permute(0, 3, 1, 2)
        patch_position = torch.nn.functional.interpolate(
            patch_position, size=(grid_h, grid_w), mode='bilinear', align_corners=False)
        visual_state['positional_embedding'] = torch.cat([
            position[:1], patch_position.permute(0, 2, 3, 1).reshape(grid_h * grid_w, 768)], dim=0)
    visual.load_state_dict(visual_state, strict=True)
    return CategoryAdapterBank(visual, reference, last_blocks, bottleneck_dim, scale).to(device)
