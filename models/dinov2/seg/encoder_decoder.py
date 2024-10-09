# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import resize


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f"{prefix}.{name}"] = value

    return outputs

# from mmseg.models.segmentors.base import BaseSegmentor
# BaseSegmentor
class SegEncoderDecoder(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
    ):
        super(SegEncoderDecoder, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.auxiliary_head = auxiliary_head
        self.num_classes = self.decode_head.num_classes
        self.align_corners = self.decode_head.align_corners
        # if neck is not None:
        #     self.neck = neck


    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas, rescale=True, size=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        
        out = self._decode_head_forward_test(x, img_metas)
        if rescale:
          if size is None:
            if img_metas is not None:
              size = img_metas[0]["ori_shape"][:2]
            else:
              size = img.shape[2:]
          out = resize(input=out, size=size, mode="bilinear", align_corners=self.align_corners)
        return out
    
    def _decode_head_forward_train(self, img, x, img_metas, seg_gt, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(img, x, img_metas, seg_gt, **kwargs)
        losses.update(add_prefix(loss_decode, "decode"))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    # TODO
    def _auxiliary_head_forward_train(self, x, img_metas, seg_gt):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas, seg_gt, self.train_cfg)
                losses.update(add_prefix(loss_aux, f"aux_{idx}"))
        else:
            loss_aux = self.auxiliary_head.forward_train(x, img_metas, seg_gt, self.train_cfg)
            losses.update(add_prefix(loss_aux, "aux"))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, seg_gt, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            seg_gt (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()
        
        loss_decode = self._decode_head_forward_train(img, x, img_metas, seg_gt, **kwargs)
        losses.update(loss_decode)

        if self.auxiliary_head is not None:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, seg_gt)
            losses.update(loss_aux)

        return losses
      
    def forward(self, img, img_metas, return_loss=True, **kwargs):
      """Calls either :func:`forward_train` or :func:`forward_test` depending
      on whether ``return_loss`` is ``True``.

      Note this setting will change the expected inputs. When
      ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
      and List[dict]), and when ``resturn_loss=False``, img and img_meta
      should be double nested (i.e.  List[Tensor], List[List[dict]]), with
      the outer list indicating test time augmentations.
      """
      if return_loss:
          return self.forward_train(img, img_metas, **kwargs)
      else:
          # return self.forward_test(img, img_metas, **kwargs)
          return None
      
      
    def slide_inference(self, img, img_meta, rescale, stride, crop_size):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta, rescale)
                preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]["ori_shape"][:2],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )
        return preds

    def whole_inference(self, img, img_meta, rescale, size=None):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta, rescale, size=size)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]["ori_shape"][:2]
            seg_logit = resize(seg_logit, size=size, mode="bilinear", align_corners=self.align_corners, warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale, size=None, mode="whole"):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert mode in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if mode == "slide":
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, size=size)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                output = output.flip(dims=(3,))
            elif flip_direction == "vertical":
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale) # TODO: i think we need to add size here
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale, size=seg_logit.shape[-2:])
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
