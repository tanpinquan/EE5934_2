from deeplab.backbone import resnet, xception, drn
from scene_parsing.model.mobilenet import MobileNetV2
import torch
import torch.nn as nn


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)
        pretrain_dict = torch.load('./scene_parsing/model/mobilenet_VOC.pth')
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
        return model

    else:
        raise NotImplementedError




