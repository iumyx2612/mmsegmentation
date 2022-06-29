import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.utils import TripletAttention

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class BAFCNHead(FCNHead):
    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 mid_channels=256,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.mid_channels = mid_channels
        super(BAFCNHead, self).__init__(num_convs,
                                          kernel_size,
                                          concat_input,
                                          dilation,
                                          **kwargs)


        self.attn = TripletAttention()

        conv_padding = (kernel_size // 2) * dilation

        self.bottleneck = ConvModule(
            self.in_channels,
            self.mid_channels,
            kernel_size=1,
            padding=0,
            dilation=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        convs = []
        convs.append(
            ConvModule(
                self.mid_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.mid_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.bottleneck(x)
        x = self.attn(x)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats