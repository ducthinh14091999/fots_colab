import warnings
from functools import partial
from typing import Any, Optional

from ....models.segmentation.fcn import FCN, _fcn_resnet
from ...transforms.presets import VocEval
from .._api import Weights, WeightEntry
from .._meta import _VOC_CATEGORIES
from ..resnet import ResNet50Weights, ResNet101Weights, resnet50, resnet101


__all__ = ["FCN", "FCNResNet50Weights", "FCNResNet101Weights", "fcn_resnet50", "fcn_resnet101"]


class FCNResNet50Weights(Weights):
    CocoWithVocLabels_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            "categories": _VOC_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet50",
            "mIoU": 60.5,
            "acc": 91.4,
        },
    )


class FCNResNet101Weights(Weights):
    CocoWithVocLabels_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            "categories": _VOC_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet101",
            "mIoU": 63.7,
            "acc": 91.9,
        },
    )


def fcn_resnet50(
    weights: Optional[FCNResNet50Weights] = None,
    weights_backbone: Optional[ResNet50Weights] = None,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> FCN:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = FCNResNet50Weights.CocoWithVocLabels_RefV1 if kwargs.pop("pretrained") else None
    weights = FCNResNet50Weights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The argument pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = ResNet50Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = ResNet50Weights.verify(weights_backbone)

    if weights is not None:
        aux_loss = True
        weights_backbone = None
        num_classes = len(weights.meta["categories"])

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


def fcn_resnet101(
    weights: Optional[FCNResNet101Weights] = None,
    weights_backbone: Optional[ResNet101Weights] = None,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> FCN:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = FCNResNet101Weights.CocoWithVocLabels_RefV1 if kwargs.pop("pretrained") else None
    weights = FCNResNet101Weights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The argument pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = ResNet101Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = ResNet101Weights.verify(weights_backbone)

    if weights is not None:
        aux_loss = True
        weights_backbone = None
        num_classes = len(weights.meta["categories"])

    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model
