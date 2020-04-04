from deeplab.deeplab import *
import dataloader
import matplotlib as plt
from util import *
import scene_parsing.compute_iou


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

num_classes = 19
image_size = (224, 224)
hist = 0
model = DeepLab(num_classes=num_classes, backbone='mobilenet', sync_bn=False)
model.load_state_dict(torch.load('model_seg'))
model.to(device)
model.eval()


cityscapes_val_dataset = dataloader.CityscapesValDataset('./data/Cityscapes/train_img', './data/Cityscapes/val_label', image_size=image_size)


# sel_sample = 3
for i, data in enumerate(cityscapes_val_dataset):
    print(i)
    # data = cityscapes_val_dataset[sel_sample]
    target_image = torch.unsqueeze(data['image'], dim=0).to(device)
    target_label = data['label']
    pred_target_label = np.array(model(target_image)[0].argmax(0).cpu())

    hist += scene_parsing.compute_iou.fast_hist(target_label.flatten(), pred_target_label.flatten(), num_classes)

    if i%50==0:
        fig = plt.figure(figsize=[16, 8])

        ax = []

        ax.append(fig.add_subplot(1, 3, 1))
        ax[-1].set_title('Cityscapes image')  # set title
        plt.imshow(deprocess(target_image[0].cpu()))

        ax.append(fig.add_subplot(1, 3, 2))
        ax[-1].set_title('Cityscapes output')  # set title
        plt.imshow(label2color(pred_target_label))

        ax.append(fig.add_subplot(1, 3, 3))
        ax[-1].set_title('Cityscapes label')  # set title
        plt.imshow(label2color(target_label))
        plt.show()

mIoUs = scene_parsing.compute_iou.per_class_iu(hist)


mIoUDict = {}
for i,label in enumerate(LABEL):
    print(label)
    mIoUDict[label] = mIoUs[i]

fig = plt.figure(figsize=[16, 8])

ax = []


ax.append(fig.add_subplot(2, 4, 5))
ax[-1].set_title('Cityscapes image')  # set title
plt.imshow(deprocess(target_image[0].cpu()))

ax.append(fig.add_subplot(2, 4, 6))
ax[-1].set_title('Cityscapes output')  # set title
plt.imshow(label2color(pred_target_label))

ax.append(fig.add_subplot(2, 4, 7))
ax[-1].set_title('Cityscapes label')  # set title
plt.imshow(label2color(target_label))
plt.show()
