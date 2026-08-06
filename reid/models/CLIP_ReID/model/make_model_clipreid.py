import torch
import torch.nn as nn

from reid.models.attribute_pooling import TextConditionedAttributePooler

from .clip import clip


def weights_init_kaiming(module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif classname.find('BatchNorm') != -1 and module.affine:
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def weights_init_classifier(module):
    if module.__class__.__name__.find('Linear') != -1:
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class TextEncoder(nn.Module):
    """Frozen CLIP text tower used by the identity prompt learner."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        features = prompts + self.positional_embedding.type(self.dtype)
        features = features.permute(1, 0, 2)
        features = self.transformer(features)
        features = features.permute(1, 0, 2)
        features = self.ln_final(features).type(self.dtype)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        batch_indices = torch.arange(
            features.shape[0],
            device=features.device,
        )
        return features[batch_indices, eot_indices] @ self.text_projection


class PromptLearner(nn.Module):
    """One learnable four-token context for every global identity."""

    def __init__(
        self,
        num_class,
        dataset_name,
        dtype,
        token_embedding,
        object_noun='person',
    ):
        super().__init__()
        del dataset_name
        object_noun = object_noun or 'object'
        template = "A photo of a X X X X {}.".format(object_noun)
        n_ctx = 4
        n_cls_ctx = 4
        ctx_dim = 512

        tokenized_prompts = clip.tokenize([template] * num_class).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer('tokenized_prompts', tokenized_prompts)
        self.register_buffer('token_prefix', embedding[:, :n_ctx + 1, :])
        self.register_buffer(
            'token_suffix',
            embedding[:, n_ctx + 1 + n_cls_ctx:, :],
        )

        cls_vectors = torch.empty(
            num_class,
            n_cls_ctx,
            ctx_dim,
            dtype=dtype,
        )
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    @staticmethod
    def _rows_for_labels(buffer, label):
        if buffer.size(0) == 1:
            return buffer.expand(label.shape[0], *buffer.shape[1:])
        return buffer[label]

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        prefix = self._rows_for_labels(self.token_prefix, label)
        suffix = self._rows_for_labels(self.token_suffix, label)
        return torch.cat([prefix, cls_ctx, suffix], dim=1)

    def get_tokenized(self, label):
        return self._rows_for_labels(self.tokenized_prompts, label)


class build_transformer(nn.Module):
    """Global CLIP-ReID model with optional text-conditioned attributes."""

    def __init__(
        self,
        num_classes,
        camera_num,
        view_num,
        cfg,
        object_noun='person',
    ):
        super().__init__()
        del camera_num, view_num
        if cfg.MODEL.NAME != 'ViT-B-16':
            raise ValueError(
                'The simplified baseline only supports CLIP ViT-B/16, got {}'
                .format(cfg.MODEL.NAME)
            )

        self.model_name = cfg.MODEL.NAME
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.in_planes_proj = 512
        self.num_classes = num_classes
        self.eval_descriptor_mode = 'raw'

        self.classifier = nn.Linear(
            self.in_planes,
            self.num_classes,
            bias=False,
        )
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        h_resolution = int(
            (cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1
        )
        w_resolution = int(
            (cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1
        )
        vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(
            self.model_name,
            h_resolution,
            w_resolution,
            vision_stride_size,
        )
        clip_model.to('cuda')

        self.image_encoder = clip_model.visual
        self.token_embedding = clip_model.token_embedding
        self.prompt_learner = PromptLearner(
            num_classes,
            cfg.DATASETS.NAMES,
            clip_model.dtype,
            clip_model.token_embedding,
            object_noun=object_noun,
        )
        self.text_encoder = TextEncoder(clip_model)
        self.attribute_pooler = TextConditionedAttributePooler()
        self.register_buffer(
            '_attribute_text_features',
            torch.empty(0, self.in_planes_proj),
            persistent=False,
        )

    def set_attribute_text_features(self, features):
        if features.ndim != 2:
            raise ValueError(
                'attribute text features must have shape [P, D], got {}'
                .format(tuple(features.shape))
            )
        if features.size(0) == 0 or features.size(1) != self.in_planes_proj:
            raise ValueError(
                'attribute text features must have non-empty shape [P, {}]'
                .format(self.in_planes_proj)
            )
        if not torch.isfinite(features).all():
            raise FloatingPointError('attribute text features are not finite')
        self._attribute_text_features = features.detach().to(
            device=self.image_encoder.proj.device,
            dtype=self.image_encoder.proj.dtype,
        )

    def set_attribute_pooling_temperature(self, temperature):
        self.attribute_pooler.set_temperature(temperature)

    def set_eval_descriptor_mode(self, mode):
        mode = str(mode)
        if mode not in ('raw', 'bn'):
            raise ValueError(
                'eval descriptor mode must be "raw" or "bn", got {!r}'
                .format(mode)
            )
        self.eval_descriptor_mode = mode

    def add_domain_adapter(
        self,
        name,
        last_blocks=4,
        bottleneck_dim=64,
        scale=1.0,
    ):
        return self.image_encoder.add_domain_adapter(
            name,
            last_blocks=last_blocks,
            bottleneck_dim=bottleneck_dim,
            scale=scale,
        )

    def set_active_adapter(self, name=None):
        self.image_encoder.set_active_domain_adapter(name)

    def get_active_adapter(self):
        return self.image_encoder.get_active_domain_adapter()

    def domain_adapter_names(self):
        return self.image_encoder.domain_adapter_names()

    def domain_adapter_parameters(self, name):
        return self.image_encoder.domain_adapter_parameters(name)

    def forward(
        self,
        x=None,
        label=None,
        get_image=False,
        get_text=False,
        get_attributes=False,
        attributes_only=False,
    ):
        if get_text:
            prompts = self.prompt_learner(label)
            tokenized_prompts = self.prompt_learner.get_tokenized(label)
            return self.text_encoder(prompts, tokenized_prompts)

        if attributes_only and not get_attributes:
            raise ValueError('attributes_only requires get_attributes=True')

        if x is None:
            raise ValueError('image input x is required when get_text is False')

        _, image_features, image_features_proj = self.image_encoder(x, None)
        image_feature = image_features[:, 0]
        image_feature_proj = image_features_proj[:, 0]

        attribute_features = None
        if get_attributes:
            if self._attribute_text_features.size(0) == 0:
                raise RuntimeError(
                    'attribute text features must be configured before '
                    'requesting visual attributes'
                )
            attribute_features = self.attribute_pooler(
                image_features_proj[:, 1:],
                self._attribute_text_features,
            )
            if attributes_only:
                if get_image:
                    return image_feature_proj, attribute_features
                return attribute_features

        if get_image:
            return image_feature_proj

        if self.training:
            if self.eval_descriptor_mode == 'raw':
                feature = image_feature
            else:
                feature = self.bottleneck(image_feature)
                self.bottleneck_proj(image_feature_proj)
            cls_score = self.classifier(feature)
            if get_attributes:
                return cls_score, image_feature_proj, attribute_features
            return cls_score, image_feature_proj

        if self.eval_descriptor_mode == 'raw':
            return torch.cat([image_feature, image_feature_proj], dim=1)

        feature = self.bottleneck(image_feature)
        feature_proj = self.bottleneck_proj(image_feature_proj)
        return torch.cat([feature, feature_proj], dim=1)

    def encode_fixed_texts(self, texts):
        """Encode ordinary text strings with the frozen CLIP text tower."""

        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts or not all(
            isinstance(text, str) and text.strip()
            for text in texts
        ):
            raise ValueError('texts must contain non-empty strings')

        device = self.token_embedding.weight.device
        tokenized = clip.tokenize(texts).to(device)
        embedded = self.token_embedding(tokenized).type(
            self.text_encoder.dtype
        )
        return self.text_encoder(embedded, tokenized)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for name in param_dict:
            self.state_dict()[name.replace('module.', '')].copy_(
                param_dict[name]
            )
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for name in param_dict:
            self.state_dict()[name].copy_(param_dict[name])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_clipreid(
    cfg,
    num_class,
    camera_num,
    view_num,
    object_noun='person',
):
    return build_transformer(
        num_class,
        camera_num,
        view_num,
        cfg,
        object_noun=object_noun,
    )


def load_clip_to_cpu(
    backbone_name,
    h_resolution,
    w_resolution,
    vision_stride_size,
):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(
            model_path,
            map_location='cpu',
        ).eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
        model = None

    return clip.build_model(
        state_dict or model.state_dict(),
        h_resolution,
        w_resolution,
        vision_stride_size,
    )
