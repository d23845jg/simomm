# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .decode_heads import BNHead
from .encoder_decoder import SegEncoderDecoder
from ..backbones import _make_dinov2_model
from ..losses import CrossEntropyLoss
from ..utils import CenterPadding


class Weights(Enum):
    NYU = "NYU"
    KITTI = "KITTI"


def _get_num_classes(weights: Weights = Weights.NYU):
    if weights == Weights.NYU:
        return 13

    return None
  

def _make_dinov2_linear_seg_head(
    *,
    embed_dim: int,
    layers: int,
    num_classes: int,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")

    if layers == 1:
        in_index = [0]
    else:
        assert layers == 4
        in_index = [0, 1, 2, 3]

    return BNHead(
        in_channels=[embed_dim] * len(in_index),
        in_index=in_index,
        input_transform="resize_concat",
        channels=embed_dim * len(in_index),
        dropout_ratio=0,
        num_classes=num_classes,
        align_corners=False,
        loss_decode=nn.ModuleList([
            CrossEntropyLoss(
              use_sigmoid=False, 
              loss_weight=1.0,
              loss_name="loss_seg",
            ),
        ]),
        ignore_index=-1,
    )


def _make_dinov2_linear_segmentor(
    *,
    arch_name: str = "vit_large",
    concat_layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.NYU,
    num_classes: Optional[int] = None,
    **kwargs,
):
    if concat_layers not in (1, 4):
        raise AssertionError(f"Unsupported number of concat layers: {concat_layers}")
    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    if num_classes is None:
        num_classes = _get_num_classes(weights)
    
    backbone = _make_dinov2_model(arch_name=arch_name, pretrained=pretrained, **kwargs)

    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size
    # model_name = _make_dinov2_model_name(arch_name, patch_size)
    linear_seg_head = _make_dinov2_linear_seg_head(
        embed_dim=embed_dim,
        concat_layers=concat_layers,
        num_classes=num_classes,
    )

    layer_count = {
        "vit_small": 12,
        "vit_base": 12,
        "vit_large": 24,
        "vit_giant2": 40,
    }[arch_name]

    if concat_layers == 4:
        out_index = {
            "vit_small": [5, 7, 9, 11], # [8, 9, 10, 11]
            "vit_base": [5, 7, 9, 11], # [8, 9, 10, 11]
            "vit_large": [17, 19, 21, 23], # [20, 21, 22, 23]
            "vit_giant2": [33, 35, 37, 39], # [36, 37, 38, 39]
        }[arch_name]
    else:
        assert concat_layers == 1
        out_index = [layer_count - 1]

    model = SegEncoderDecoder(backbone=backbone, decode_head=linear_seg_head)
    model.backbone.forward = partial(
        backbone.get_intermediate_layers,
        n=out_index,
        reshape=True,
        return_class_token=False,
        norm=False,
    )
    model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(patch_size)(x[0]))

    # Avoid init the head
    # if pretrained:
    #     layers_str = str(layers) if layers == 4 else ""
    #     weights_str = weights.value.lower()
    #     url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_{weights_str}_linear{layers_str}_head.pth"
    #     checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    #     if "state_dict" in checkpoint:
    #         state_dict = checkpoint["state_dict"]
    #     model.load_state_dict(state_dict, strict=False)

    return model


def dinov2_vits14_ls(*, concat_layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.NYU, **kwargs):
    return _make_dinov2_linear_segmentor(
        arch_name="vit_small", concat_layers=concat_layers, pretrained=pretrained, weights=weights, **kwargs
    )


def dinov2_vitb14_ls(*, concat_layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.NYU, **kwargs):
    return _make_dinov2_linear_segmentor(
        arch_name="vit_base", concat_layers=concat_layers, pretrained=pretrained, weights=weights, **kwargs
    )


def dinov2_vitl14_ls(*, concat_layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.NYU, **kwargs):
    return _make_dinov2_linear_segmentor(
        arch_name="vit_large", concat_layers=concat_layers, pretrained=pretrained, weights=weights, **kwargs
    )


def dinov2_vitg14_ls(*, concat_layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.NYU, **kwargs):
    return _make_dinov2_linear_segmentor(
        arch_name="vit_giant2", concat_layers=concat_layers, ffn_layer="swiglufused", pretrained=pretrained, weights=weights, **kwargs
    )