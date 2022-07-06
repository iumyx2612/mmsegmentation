import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.utils import TripletAttention, LayerAttention

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class DoubleBranchHead(FCNHead):
    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        groups = len(kwargs["in_channels"])
        super(DoubleBranchHead, self).__init__(num_convs,
                                          kernel_size,
                                          concat_input,
                                          dilation,
                                          **kwargs)

        self.layer_attn = LayerAttention(
            in_channels=self.in_channels,
            groups=groups
        )
        self.attn = TripletAttention()

        conv_padding = (kernel_size // 2) * dilation
        semantic_convs = []
        semantic_convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            semantic_convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        edge_convs = []
        edge_convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            edge_convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.semantic_convs = nn.Sequential(*semantic_convs)
        self.edge_convs = nn.Sequential(*edge_convs)


    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)

        # layer attn branch
        x1 = self.layer_attn(inputs)
        semantic_feats = self.semantic_convs(x1)

        # triplet attn branch
        x2 = self.attn(inputs)
        edge_feats = self.edge_convs(x2)

        feats = semantic_feats + edge_feats

        return feats