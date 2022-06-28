from mmcv.runner import HOOKS, Hook

import matplotlib.pyplot as plt
import os


@HOOKS.register_module()
class InspectTrainingImagesHook(Hook):
    def __init__(self, num_inspect=3):
        self.num_inspect = num_inspect
        self.called = False

    def before_train_iter(self, runner):
        if not self.called:
            work_dir = runner.work_dir
            dataloader = runner.data_loader
            num_imgs = 0
            datas = []

            runner.logger.info(f"Saving training images for visualization to "
                               f"{os.path.join(work_dir, 'training_images')}")
            os.makedirs(f"{work_dir}/training_images", exist_ok=True)

            for i in range(self.num_inspect):
                data_batch = next(dataloader)
                batch_size = len(data_batch['img_metas'].data[0])
                num_imgs += batch_size
                # create new sample dict of each image
                for i in range(batch_size):
                    image_name = data_batch['img_metas'].data[0][i]['ori_filename'].split('.')[0]
                    image = data_batch['img'].data[0][i]
                    label = data_batch['gt_semantic_seg'].data[0][i]
                    data_dict = dict(
                        image_name = image_name,
                        image = image,
                        label = label
                    )
                    datas.append(data_dict)
                if num_imgs >= self.num_inspect:
                    break

            for i in range(self.num_inspect):
                data_dict = datas[i]
                image = data_dict["image"].cpu().numpy().transpose(1, 2, 0)
                label = data_dict["label"].cpu().numpy().transpose(1, 2, 0).squeeze()
                image_name = data_dict["image_name"]

                fig = plt.figure(figsize=(10, 10))
                fig.suptitle(f"{data_dict['image_name']}")

                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                ax1.set_title("Image")
                ax2.set_title("Mask")

                im1 = ax1.imshow(image)
                im2 = ax2.imshow(label, cmap=plt.cm.jet)

                save_path = os.path.join(work_dir, f"training_images/{image_name}.jpg")
                plt.savefig(save_path)

            self.called = True
