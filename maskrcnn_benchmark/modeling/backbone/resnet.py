from collections import UserDict
from maskrcnn_benchmark.utils.registry import Registry

import sys
import types
import importlib

from maskrcnn_benchmark.modeling.backbone.resnet_versions import resnet
from maskrcnn_benchmark.modeling.backbone.resnet_versions import resnet_5_mod
from maskrcnn_benchmark.modeling.backbone.resnet_versions import resnet_multi_stacked

def main(cfg):
    assert cfg.MODEL.RESNETS.VERSION in _MODEL_RESNETS_VERSION, \
        "MODEL.RESNETS.VERSION: {} are not registered in registry".format(
            cfg.MODEL.RESNETS.VERSION
        )
    print("MODEL.RESNETS.VERSION: using {} ResNet version for model".format(
        cfg.MODEL.RESNETS.VERSION))

    return _MODEL_RESNETS_VERSION[cfg.MODEL.RESNETS.VERSION].__name__


_MODEL_RESNETS_VERSION = Registry({
    "resnet_version_baseline": resnet,
    "resnet_version_equiconst": resnet,
    "resnet_version_variable": resnet,
    "resnet_version_multi_mixed": resnet,
    "resnet_version_single": resnet,
    "resnet_version_multi_stacked_single": resnet,
    "resnet_version_multi_stacked": resnet_multi_stacked,
    "resnet_version_5_mod": resnet_5_mod,
})