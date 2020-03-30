# %%
import torch
import torch.nn as nn
import torchvision
from scene_parsing.model.mobilenet import MobileNetV2
from scene_parsing.model.deeplab import DeepLabHead
from scene_parsing.model.segmentation_model import SegmentationModel
import PIL
import util
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print('Use', device)


image_test = PIL.Image.open('data/CItyscapes/train_img/aachen_000008_000019_leftImg8bit.png')
image_test = util.preprocess(image_test)
label_test = np.array(PIL.Image.open('data/CItyscapes/val_label/aachen_000008_000019_gtFine_color.png'))
id_test = np.array(PIL.Image.open('data/CItyscapes/val_label/aachen_000008_000019_gtFine_labelIds.png'))

plt.imshow(util.deprocess(image_test))
plt.show()
plt.imshow(label_test)
plt.show()
plt.imshow(util.label2color(util.label2train(id_test)))
plt.show()

'''Create mobilenet backbone'''
input = torch.rand(1, 3, 512, 512)
mobilenet_model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)
pretrain_dict = torch.load('./scene_parsing/model/mobilenet_VOC.pth')
model_dict = {}
state_dict = mobilenet_model.state_dict()
for k, v in pretrain_dict.items():
    if k in state_dict:
        model_dict[k] = v
state_dict.update(model_dict)
mobilenet_model.load_state_dict(state_dict)
# output, low_level_feat = mobilenet_model(image_orig)
# print(output.size())
# print(low_level_feat.size())

'''Create deeplab'''
deeplab_model = DeepLabHead(in_channels=320,num_classes=18)


'''Create Deeplab+MobileNET'''
model = SegmentationModel(backbone=mobilenet_model, classifier=deeplab_model)
print(model)

model.eval()
output = model(image_test)
print(output.shape)
output_predictions = output[0,:,:,:].argmax(0)

plt.imshow(util.label2color(output_predictions))
plt.show()

