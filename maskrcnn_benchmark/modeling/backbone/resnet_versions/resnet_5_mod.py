# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
import copy
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry
from utils_gen import model_utils as utils

import matplotlib.pyplot as plt

# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Numer of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        # This is where the custom module is replaced
        self.stem = stem_module(cfg)
        self.cfg = cfg

        self.number_of_bottleneck_layers = len(stage_specs)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS # 320
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS #256

        stage2_bottleneck_channels = num_groups * width_per_group # 64

        self.stages = []
        self.return_features = {}

        for stage_spec in stage_specs:

            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor # 64*1, 64*2
            out_channels = stage2_out_channels * stage2_relative_factor # 256*1, 256*2,

            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func
    ):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation
        )
        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv2, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func = FrozenBatchNorm2d):
        super(BaseStem, self).__init__()

        self.debugging = True
        self.model_specific_sequence = cfg.MODEL.SPECIFIC_SEQUENCE
        model_specific_sequence = self.model_specific_sequence

        after_cat_expected_number_of_channels = 320
        after_add_expected_number_of_channels = 64

        if "_cat_" in model_specific_sequence:
            new_modules_final_out_channels = after_cat_expected_number_of_channels
        elif "_add_" in model_specific_sequence:
            new_modules_final_out_channels = after_add_expected_number_of_channels

        pure_modules_out = 64
        mixed_modules_out = 64
        final_out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS # Value = 64

        #self.custom_module_combiner = module_combiner(cfg)
        #AG:
        #Implementing a cheap solution until I successfully integrate our custom module into the network

        self.pure_module_ch1_input_range = [0, 3]
        self.pure_module_ch2_input_range = [3, 6]
        self.pure_module_ch3_input_range = [6, 9]

        self.pure_modules_input_size = 3
        self.pure_modules_output_size = 64

        self.mixed_modules_input_size = 6
        self.mixed_modules_output_size = 64

        self.pure_conv_ch1 = Conv2d(
            self.pure_modules_input_size, self.pure_modules_output_size,
            kernel_size = 7, dilation = 1, stride = 2, padding = 3, bias = False
        ) # dilation of 1 = dilation of 0
        self.pure_conv_ch2 = Conv2d(
            self.pure_modules_input_size, self.pure_modules_output_size,
            kernel_size = 7, dilation = 5, stride = 2, padding = 15, bias = False
        )
        self.pure_conv_ch3 = Conv2d(
            self.pure_modules_input_size, self.pure_modules_output_size,
            kernel_size = 7, dilation = 14, stride = 2, padding = 42, bias = False
        )
        self.mixed_conv_ch1_ch2 = Conv2d(
            self.mixed_modules_input_size, self.mixed_modules_output_size,
            kernel_size = 7, dilation = 1, stride = 2, padding = 3, bias = False
        )
        self.mixed_conv_ch2_ch3 = Conv2d(
            self.mixed_modules_input_size, self.mixed_modules_output_size,
            kernel_size = 7, dilation = 5, stride = 2, padding = 15, bias = False
        )

        if utils.network_chunk_activator(self.model_specific_sequence, "bn1"):
            self.pure_ch1_bn1 = norm_func(pure_modules_out)
            self.pure_ch2_bn1 = norm_func(pure_modules_out)
            self.pure_ch3_bn1 = norm_func(pure_modules_out)
            self.mixed_ch1_ch2_bn1 = norm_func(mixed_modules_out)
            self.mixed_ch2_ch3_bn1 = norm_func(mixed_modules_out)
        else: self.smart_print("Model-generation: No bn1 for model " + self.model_specific_sequence)

        if utils.network_chunk_activator(self.model_specific_sequence, "bn2"):
            #bn2 is the normalization after cat/add
            self.bn2 = norm_func(new_modules_final_out_channels)
        else: self.smart_print("Model-generation: No bn2 for model " + self.model_specific_sequence)

        if utils.network_chunk_activator(self.model_specific_sequence, "comp"):
            #There is no if for the add case because the new_modules_final_out_channels parameter
            #is modified during the config preparation to account for this
            self.output_compressor = Conv2d(
               new_modules_final_out_channels, final_out_channels,
               kernel_size = 1, dilation = 1, stride = 1, padding = 0, bias = False
            )

        #bn3 is the batch-norm after the compressor
        self.bn3 = norm_func(final_out_channels)


        for l in [self.pure_conv_ch1, self.pure_conv_ch2, self.pure_conv_ch3,
                  self.mixed_conv_ch1_ch2, self.mixed_conv_ch2_ch3, self.output_compressor,]:
           nn.init.kaiming_uniform_(l.weight, a=1)

    #---------------------NEW-ADDITION---------------------
    def smart_print(self, _str):
    # HARDCORE EXECUTE-IF STATEMENT - SPEEEEED!!
        self.debugging and print(_str)
    # -----------------------------------------------------

    def forward(self, x):
        '''
        changer = x[:, 0:3, :, :]
        refferencer = x[:, 0:3, :, :]
        print("Original value of x: ", refferencer[0, 0, 0, 0])

        changer[0, 0, 0, 0] = 666
        print("New value of x: ", refferencer[0, 0, 0, 0])

        ch1_x = x[:, 12:15, :, :]
        ch2_x = x[:, 15:18, :, :]
        ch3_x = x[:, 18:21, :, :]

        #plt.imshow(ch3_x[0, :, :, :].cpu().permute(1, 2, 0))
        #plt.show()
        #TODO: look at the visualizations from the different channels... they look weird

        ch1_ch2_x = x[:, 12:18, :, :]
        ch2_ch3_x = x[:, 15:21, :, :]
        '''
        ch1_x = x[:, 0:3, :, :]
        ch2_x = x[:, 3:6, :, :]
        ch3_x = x[:, 6:9, :, :]

        ch1_ch2_x = x[:, 0:6, :, :]
        ch2_ch3_x = x[:, 3:9, :, :]

        ch1_x_1 = self.pure_conv_ch1(ch1_x)
        ch2_x_1 = self.pure_conv_ch2(ch2_x)
        ch3_x_1 = self.pure_conv_ch3(ch3_x)
        ch1_ch2_x_1 = self.mixed_conv_ch1_ch2(ch1_ch2_x)
        ch2_ch3_x_1 = self.mixed_conv_ch2_ch3(ch2_ch3_x)

        if utils.network_chunk_activator(self.model_specific_sequence, "bn1"):
            ch1_x_1 = self.pure_ch1_bn1(ch1_x_1)
            ch2_x_1 = self.pure_ch2_bn1(ch2_x_1)
            ch3_x_1 = self.pure_ch3_bn1(ch3_x_1)
            ch1_ch2_x_1 = self.mixed_ch1_ch2_bn1(ch1_ch2_x_1)
            ch2_ch3_x_1 = self.mixed_ch2_ch3_bn1(ch2_ch3_x_1)

        if utils.network_chunk_activator(self.model_specific_sequence, "relu1"):
            ch1_x_1 = F.relu_(ch1_x_1)
            ch2_x_1 = F.relu_(ch2_x_1)
            ch3_x_1 = F.relu_(ch3_x_1)
            ch1_ch2_x_1 = F.relu_(ch1_ch2_x_1)
            ch2_ch3_x_1 = F.relu_(ch2_ch3_x_1)


        if utils.network_chunk_activator(self.model_specific_sequence, "cat"):
            x_1 = torch.cat((ch1_x_1, ch2_x_1, ch3_x_1, ch1_ch2_x_1, ch2_ch3_x_1), dim=1)
        elif utils.network_chunk_activator(self.model_specific_sequence, "add"):
            x_1 = ch1_x_1 + ch2_x_1 + ch3_x_1 + ch1_ch2_x_1 + ch2_ch3_x_1

        if utils.network_chunk_activator(self.model_specific_sequence, "bn2"):
            x_1 = self.bn2(x_1)

        if utils.network_chunk_activator(self.model_specific_sequence, "relu2"):
            x_1 = F.relu_(x_1)


        #Output compression
        if utils.network_chunk_activator(self.model_specific_sequence, "comp"):
            x_1 = self.output_compressor(x_1)

        x_1 = self.bn3(x_1)

        x_1 = F.relu_(x_1)
        x_1 = F.max_pool2d(x_1, kernel_size=3, stride=2, padding=1)

        return x_1

#-----------------------------

class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )
#----------------------

class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)

_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
})

