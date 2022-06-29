from mmcv.cnn import ConvModule
import torch
from torch import nn


class SPPFModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        """ Spatial Pyramid Pooling Fast from YOLOv5 but act as a plug-in module,
        not a neck
        Input feature map first go through a bottleneck module,
        reducing its channels to half. Then sequentially got max pooled,
        concatenate, and go through another bottleneck module to get desired
        out_channels
        Args:
            in_channels: number of channels from the input feature map
            out_channels: desired output channels
            kernel: kernel size of max pooling ops
        """
        super(SPPFModule, self).__init__()
        mid_channels = in_channels // 2
        # 1st channel reduction
        self.conv1 = ConvModule(
            in_channels=in_channels,
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

    def forward(self, x):
        x = self.conv1(x)           # channel reduction
        x1 = self.max_pool(x)       # 1st pooling
        x2 = self.max_pool(x1)      # 2nd pooling
        x3 = self.max_pool(x2)      # 3rd pooling

        concat = torch.cat([x, x1, x2, x3], dim=1)  # channel concat
        out = self.conv2(concat)    # channel reduction

        return out