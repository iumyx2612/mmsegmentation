import mmcv
from mmseg.apis.inference import inference_segmentor, init_segmentor

import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gen_mask import mask_to_rgb


config_file = "logs/mitB1_MLPHead/main.py"
ckpt_file = "logs/b1_fpn_sppf_tripleC_8-1/iter_14000.pth"
image_file = "../Dataset/test/test/782707d7c359e27888daefee82519763.jpeg"
folder = "../Dataset/val"
mask_folder = "../Dataset/val_seg_map"

PALETTE_MAPPING = {
    0: np.asarray([0, 0, 0]),
    1: np.asarray([0, 255, 0]),
    2: np.asarray([255, 0, 0])
}


if __name__ == '__main__':
    model = init_segmentor(config_file, ckpt_file)
    for file in tqdm(os.listdir(folder)):
        name = file.split('.')[0]
        #print(name)
        image_file = os.path.join(folder, file)
        mask_file = os.path.join(mask_folder, f"{name}.png")

        image = mmcv.imread(image_file)
        mask = mmcv.imread(mask_file)[:, :, 0]

        rgb_mask = mask_to_rgb(mask, PALETTE_MAPPING)

        result = inference_segmentor(model, image)
        palette = model.PALETTE
        for i in range(len(palette)):
            palette[i] = PALETTE_MAPPING[i]

        img = model.show_result(
            image, result, palette=palette, show=False, opacity=0.2)

        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(name)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title("Pred")
        ax2.set_title("GT")
        im1 = ax1.imshow(img[:, :, ::-1].copy())
        im2 = ax2.imshow(rgb_mask)
        #plt.imshow(result[0].copy(), cmap=plt.cm.jet)
        plt.show()
        #plt.savefig(f'logs/b1_fpn_sppf_tripleC_8-1/gen_masks/{name}.jpg')
        #plt.close(fig)