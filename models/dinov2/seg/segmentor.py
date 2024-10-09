# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .decode_heads import BNHead
from ..losses import CrossEntropyLoss
  

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