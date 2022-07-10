import mmcv
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmseg.apis.inference import init_segmentor, LoadImage
from mmseg.datasets.pipelines import Compose
import torch

import os

import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


config_file = "logs/b1_fpn_sppf_layerAttn_8-1/main.py"
ckpt_file = "logs/b1_fpn_sppf_layerAttn_8-1/iter_19000.pth"
folder = "../Dataset/val"
mask_folder = "../Dataset/val_seg_map"

PALETTE_MAPPING = {
    0: np.asarray([0, 0, 0]),
    1: np.asarray([0, 0, 255]),
    2: np.asarray([255, 0, 0])
}

sem_classes = [
    "background", "neoplastic polyps", "non-neoplastic polyps"
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
category = sem_class_to_idx["non-neoplastic polyps"]


if __name__ == '__main__':
    cfg = Config.fromfile(config_file)

    model = init_segmentor(config_file, ckpt_file)

    target_layers = [
        model.decode_head.convs
    ]

    palette = [0, 1, 2]
    for i in range(len(palette)):
        palette[i] = PALETTE_MAPPING[i]

    for file in os.listdir(folder):
        file = "00fd197cd955fa095f978455cef3593c.jpeg"
        name = file.split('.')[0]
        image_file = os.path.join(folder, file)
        mask_file = os.path.join(mask_folder, f"{name}.png")

        image = mmcv.imread(image_file)
        mask = mmcv.imread(mask_file)[:, :, 0]

        image = mmcv.imresize(image, (512, 512))
        mask = mmcv.imresize(mask, (512, 512))

        device = next(model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=image)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        # forward the model
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)  # binary mask

        #img_w_mask = model.show_result(
        #    image, result, palette=palette, show=False, opacity=0.2)

        mask_uint8 = 255 * np.uint8(result[0] == category)
        mask_float = np.float32(result[0] == category)
        binary_mask = np.repeat(mask_uint8[:, :, None], 3, axis=-1)

        targets = [
            SemanticSegmentationTarget(category, mask_float)
        ]
        model = model.eval()
        with EigenCAM(model=model,
                      target_layers=target_layers,
                      use_cuda=False) as cam:
            grayscale_cam = cam(input_tensor=data,
                                targets=targets)[0]
            cam_img = show_cam_on_image(image / 255., grayscale_cam, use_rgb=True)
            plt.subplot(1, 3, 1)
            plt.imshow(cam_img)
            plt.subplot(1, 3, 2)
            plt.imshow(result[0], cmap=plt.cm.jet)
            plt.subplot(1, 3, 3)
            plt.imshow(mask, cmap=plt.cm.jet)
            plt.show()
