import torch.nn as nn

from reid.models.CLIP_ReID.model.make_model_clipreid import make_clipreid


class CLIP_Backbone(nn.Module):
    """Thin wrapper around the global CLIP-ReID ViT-B/16 model."""

    def __init__(
        self,
        num_class,
        camera_num,
        view_num=1,
        input_size=(256, 128),
        object_noun='person',
    ):
        super().__init__()
        self.config_file = (
            'reid/models/CLIP_ReID/configs/person/vit_clipreid.yml'
        )
        from reid.models.CLIP_ReID.config import cfg as default_cfg

        model_cfg = default_cfg.clone()
        model_cfg.defrost()
        model_cfg.merge_from_file(self.config_file)
        model_cfg.INPUT.SIZE_TRAIN = list(input_size)
        model_cfg.INPUT.SIZE_TEST = list(input_size)
        model_cfg.freeze()

        self.base = make_clipreid(
            cfg=model_cfg,
            num_class=num_class,
            camera_num=camera_num,
            view_num=view_num,
            object_noun=object_noun,
        )
        self.num_classes = num_class
        print('Using CLIP ViT-B/16 as the global ReID backbone...')

    def add_domain_adapter(
        self,
        name,
        last_blocks=4,
        bottleneck_dim=64,
        scale=1.0,
    ):
        return self.base.add_domain_adapter(
            name,
            last_blocks=last_blocks,
            bottleneck_dim=bottleneck_dim,
            scale=scale,
        )

    def set_active_adapter(self, name=None):
        self.base.set_active_adapter(name)

    def get_active_adapter(self):
        return self.base.get_active_adapter()

    def domain_adapter_names(self):
        return self.base.domain_adapter_names()

    def forward(
        self,
        x=None,
        label=None,
        get_image=False,
        get_text=False,
        get_attributes=False,
        attributes_only=False,
    ):
        forward_kwargs = dict(
            x=x,
            label=label,
            get_image=get_image,
            get_text=get_text,
        )
        if get_attributes or attributes_only:
            forward_kwargs.update(
                get_attributes=get_attributes,
                attributes_only=attributes_only,
            )
        return self.base(**forward_kwargs)


def make_model(
    num_class,
    camera_num,
    view_num=1,
    input_size=(256, 128),
    object_noun='person',
):
    model = CLIP_Backbone(
        num_class=num_class,
        camera_num=camera_num,
        view_num=view_num,
        input_size=input_size,
        object_noun=object_noun,
    )
    print('Global CLIP-ReID baseline built...')
    return model
