# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .im_vit import IM_VIT


def build_model(config):
    model_type = config.MODEL.TYPE
    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm




    if model_type == 'im_vit':
        model = IM_VIT(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.IMVIT.PATCH_SIZE,
                                in_c=config.MODEL.IMVIT.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                stem_channel=config.MODEL.IMVIT.CONV_CHANNEL,
                                embed_dim=config.MODEL.IMVIT.EMBED_DIM,
                                depths=config.MODEL.IMVIT.DEPTHS,
                                num_heads=config.MODEL.IMVIT.NUM_HEADS,
                                mlp_ratio=config.MODEL.IMVIT.MLP_RATIO,
                                qkv_bias=config.MODEL.IMVIT.QKV_BIAS,
                                drop_ratio=config.MODEL.DROP_RATE,
                                attn_drop_ratio=config.MODEL.ATTN_DROP_RATE,
                                drop_path_ratio=config.MODEL.DROP_PATH_RATE,
                                head_dropout = config.MODEL.HEAD_DROPOUT,
                                classifier_head = config.MODEL.IMVIT.HEAD,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.IMVIT.PATCH_NORM,
                                attn_mask_scope = config.MODEL.IMVIT.SCOPE,
                                interaction_mask = config.MODEL.IMVIT.INTERACTION
                                )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
