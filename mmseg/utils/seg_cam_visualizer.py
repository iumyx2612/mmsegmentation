# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import copy

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

try:
    from pytorch_grad_cam import (AblationCAM, AblationLayer,
                                  ActivationsAndGradients)
    from pytorch_grad_cam.base_cam import BaseCAM
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
    from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

from mmseg.core import get_classes
from mmseg.datasets import replace_ImageToTensor
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def reshape_transform(feats, max_shape=(20, 20), is_need_grad=False):
    """Reshape and aggregate feature maps when the input is a multi-layer
    feature map.
    Takes these tensors with different sizes, resizes them to a common shape,
    and concatenates them.
    """
    if len(max_shape) == 1:
        max_shape = max_shape * 2

    if isinstance(feats, torch.Tensor):
        feats = [feats]
    else:
        if is_need_grad:
            raise NotImplementedError('The `grad_base` method does not '
                                      'support output multi-activation layers')

    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    if -1 in max_shape:
        max_shape = (max_h, max_w)
    else:
        max_shape = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(feat), max_shape, mode='bilinear'))

    activations = torch.cat(activations, axis=1)
    return activations


class SegCAMModel(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self, cfg, checkpoint, score_thr, device='cuda:0'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.score_thr = score_thr
        self.checkpoint = checkpoint
        self.segmentor = self.build_segmentor()

        self.return_loss = False
        self.input_data = None
        self.img = None

    def build_segmentor(self):
        cfg = copy.deepcopy(self.cfg)

        segmentor = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        if self.checkpoint is not None:
            checkpoint = load_checkpoint(
                segmentor, self.checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint.get('meta', {}):
                segmentor.CLASSES = checkpoint['meta']['CLASSES']
            else:
                import warnings
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use COCO classes by default.')
                segmentor.CLASSES = get_classes('coco')

        segmentor.to(self.device)
        segmentor.eval()
        return segmentor

    def set_return_loss(self, return_loss):
        self.return_loss = return_loss

    def set_input_data(self, img, mask=None):
        self.img = img
        cfg = copy.deepcopy(self.cfg)
        if self.return_loss:
            assert mask is not None
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
            cfg.data.test.pipeline[1].transforms[-1] = dict(
                type='Collect', keys=['img', 'gt_semantic_seg']),
            test_pipeline = Compose(cfg.data.test.pipeline)
            # TODO: support mask
            data = dict(
                img=self.img,
                gt_semantic_seg=mask)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)

            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0][0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]
            data['gt_semantic_seg'] = [
                seg_map.data[0] for seg_map in data['gt_semantic_seg']
            ]
            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]

            data['img'] = data['img'][0]
            data['gt_semantic_seg'] = data['gt_semantic_seg'][0]
        else:
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            data = dict(img=self.img)
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
            test_pipeline = Compose(cfg.data.test.pipeline)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]

            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]
            else:
                for m in self.detector.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'

        self.input_data = data

    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        if self.return_loss:
            loss = self.segmentor(return_loss=True, **self.input_data)
            return [loss]
        else:
            with torch.no_grad():
                results = self.segmentor(
                    return_loss=False, rescale=True, **self.input_data)[0]

                if isinstance(results, tuple):
                    seg_result = results[0]

                bboxes = np.vstack(bbox_result)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)

                segms = None
                if segm_result is not None and len(labels) > 0:  # non empty
                    segms = mmcv.concat_list(segm_result)
                    if isinstance(segms[0], torch.Tensor):
                        segms = torch.stack(
                            segms, dim=0).detach().cpu().numpy()
                    else:
                        segms = np.stack(segms, axis=0)

                if self.score_thr > 0:
                    assert bboxes is not None and bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > self.score_thr
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]
                    if segms is not None:
                        segms = segms[inds, ...]
                return [{'bboxes': bboxes, 'labels': labels, 'segms': segms}]


class DetCAMVisualizer:
    """mmdet cam visualization class.
    Args:
        method:  CAM method. Currently supports
           `ablationcam`,`eigencam` and `featmapam`.
        model (nn.Module): MMDet model.
        target_layers (list[torch.nn.Module]): The target layers
            you want to visualize.
        reshape_transform (Callable, optional): Function of Reshape
            and aggregate feature maps. Defaults to None.
    """

    def __init__(self,
                 method_class,
                 model,
                 target_layers,
                 reshape_transform=None,
                 is_need_grad=False,
                 extra_params=None):
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.is_need_grad = is_need_grad

        if method_class.__name__ == 'AblationCAM':
            batch_size = extra_params.get('batch_size', 1)
            ratio_channels_to_ablate = extra_params.get(
                'ratio_channels_to_ablate', 1.)
            self.cam = AblationCAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=extra_params['ablation_layer'],
                ratio_channels_to_ablate=ratio_channels_to_ablate)
        else:
            self.cam = method_class(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
            )
            if self.is_need_grad:
                self.cam.activations_and_grads.release()

        self.classes = model.detector.CLASSES
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def switch_activations_and_grads(self, model):
        self.cam.model = model

        if self.is_need_grad is True:
            self.cam.activations_and_grads = ActivationsAndGradients(
                model, self.target_layers, self.reshape_transform)
            self.is_need_grad = False
        else:
            self.cam.activations_and_grads.release()
            self.is_need_grad = True

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets, aug_smooth, eigen_smooth)[0, :]

    def show_cam(self,
                 image,
                 boxes,
                 labels,
                 grayscale_cam,
                 with_norm_in_bboxes=False):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        if with_norm_in_bboxes is True:
            boxes = boxes.astype(np.int32)
            renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_cam * 0
                img[y1:y2,
                    x1:x2] = scale_cam_image(grayscale_cam[y1:y2,
                                                           x1:x2].copy())
                images.append(img)

            renormalized_cam = np.max(np.float32(images), axis=0)
            renormalized_cam = scale_cam_image(renormalized_cam)
        else:
            renormalized_cam = grayscale_cam

        cam_image_renormalized = show_cam_on_image(
            image / 255, renormalized_cam, use_rgb=False)

        image_with_bounding_boxes = self._draw_boxes(boxes, labels,
                                                     cam_image_renormalized)
        return image_with_bounding_boxes