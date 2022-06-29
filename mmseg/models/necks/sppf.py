from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import torch
from torch import nn

from ..builder import NECKS


@NECKS.register_module()
class SPPF(BaseModule):
    """ Spatial Pyramid Pooling Fast from YOLOv5
    This module takes last output feature map from backbone,
    double its channels by concatenating multiple outputs from
    internal max pooling module, then later using 1x1 conv to reduce the channel back
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(SPPF, self).__init__()
        mid_channels = in_channels[-1] // 2
        # 1st channel reduction
        self.conv1 = ConvModule(
            in_channels=in_channels[-1],
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        # 2nd channel reduction after concat
        self.conv2 = ConvModule(
            in_channels=mid_channels * 4,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2
        )

    def forward(self, inputs):
        inputs = list(inputs)       # to list
        x = inputs[-1]              # we only conduct SPPF on the last feature map from backbone
        x = self.conv1(x)           # channel reduction
        x1 = self.max_pool(x)       # 1st pooling
        x2 = self.max_pool(x1)      # 2nd pooling
        x3 = self.max_pool(x2)      # 3rd pooling

        concat = torch.cat([x, x1, x2, x3], dim=1)  # channel concat
        out = self.conv2(concat)    # channel reduction
        inputs[-1] = out
        inputs = tuple(inputs)      # back to tuple

        return inputs
