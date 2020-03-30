from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class SegmentationModel(nn.Sequential):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        # print(features)
        # result = OrderedDict()
        x = features[0]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result = x

        return result
