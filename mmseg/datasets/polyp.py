from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class PolypDataset(CustomDataset):
    CLASSES = ("Background", "neoplastic polyps", "non-neoplastic polyps")
    PALETTE = [0, 1, 2]
    def __init__(self, **kwargs):
        super(PolypDataset, self).__init__(
            img_suffix=".jpeg",
            seg_map_suffix=".png",
            **kwargs)
        assert self.file_client.exists(self.img_dir)