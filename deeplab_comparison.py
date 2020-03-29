#%%
import torch
import torchvision

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print('Use', device)

# Download and load the pretrained SqueezeNet model.
model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
print(model)




