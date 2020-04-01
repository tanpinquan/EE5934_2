from deeplab.deeplab import *
from discriminator import FCDiscriminator
import dataloader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from util import *

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# device = 'cpu'
print('Use', device)

'''Create datasets'''
batch_size = 4
shuffle = False
image_size = (224, 224)

gta_datset = dataloader.GtaDataset('./data/GTA_V/train_img', './data/GTA_V/train_label', image_size=image_size)
gta_dataloader = DataLoader(gta_datset, batch_size=batch_size, shuffle=shuffle)

num_gta_images = len(gta_datset.filenames)

cityscapes_dataset = dataloader.CityscapesDataset('./data/Cityscapes/train_img', image_size=image_size)
cityscapes_dataloader = DataLoader(cityscapes_dataset, batch_size=batch_size, shuffle=shuffle)
cityscapes_dataloader_iter = enumerate(gta_dataloader)

num_cityscape_images = len(cityscapes_dataset.filenames)

num_train_batches = int(min(num_gta_images, num_cityscape_images) / batch_size)

'''Plot sample images'''
sample = gta_datset[0]
plt.imshow(deprocess(sample['image']))
plt.title('sample gta image')
plt.show()

plt.imshow(label2color(sample['label']))
plt.title('sample gta label')
plt.show()

sample = cityscapes_dataset[0]
plt.imshow(deprocess(sample['image']))
plt.title('sample cityscapes image')
plt.show()


# a = b

def adjust_learning_rate(optimizer, lr, i_iter, n_iter):
    lr = lr * pow((1 - 1.0 * i_iter / n_iter), 0.9)
    print('learning rate', lr)
    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]['lr'] = lr
    else:
        # enlarge the lr at the head
        optimizer.param_groups[0]['lr'] = lr
        for i in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * 10


''' Define segmentation network (deeplab + mobilenet)'''
lr = 0.01
momentum = 0.9
weight_decay = 5e-4
nesterov = False
ignore_index = 255
num_classes = 19

model = DeepLab(num_classes=num_classes, backbone='mobilenet', sync_bn=False)
model.to(device)
model.train()

train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                {'params': model.get_10x_lr_params(), 'lr': lr * 10}]
optimizer = torch.optim.SGD(train_params, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index, reduction='mean')

''' labels for adversarial training '''
source_label = 0
target_label = 1

'''Define discriminator network'''
lr_D = 2.5e-4
model_D = FCDiscriminator(num_classes=num_classes)
model_D.to(device)
model_D.train()

optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr_D, betas=(0.9, 0.99))
criterion_adv = nn.BCEWithLogitsLoss()

''' Perform training'''
max_iter = 10
for iter_i in range(max_iter):
    optimizer.zero_grad()
    print('epoch', iter_i)
    adjust_learning_rate(optimizer, lr, iter_i, max_iter)
    adjust_learning_rate(optimizer_D, lr_D, iter_i, max_iter)
    gta_dataloader_iter = enumerate(gta_dataloader)
    cityscapes_dataloader_iter = enumerate(cityscapes_dataloader)

    # for i, batch in enumerate(gta_dataloader):
    for i in range(num_train_batches):
        _, source_batch = gta_dataloader_iter.__next__()
        source_images = source_batch['image'].to(device)
        source_labels = source_batch['label'].to(device)

        _, target_batch = cityscapes_dataloader_iter.__next__()
        target_images = target_batch['image'].to(device)

        optimizer.zero_grad()

        pred = model(source_images)
        loss_seg = criterion(pred, source_labels.long())
        # loss_seg = loss_seg / len(gta_datset.filenames)
        loss_seg = loss_seg / batch_size

        pred_target = model(target_images)

        D_out = model_D(F.softmax(pred_target, dim=0))
        adv_target = torch.FloatTensor(pred_target.data.size()).fill_(source_label).to(device)
        loss_adv = criterion_adv(pred_target, adv_target) / batch_size

        loss = loss_seg + loss_adv
        loss.backward()

        optimizer.step()

        print(i, loss_seg)
        if i > 50:
            break
    # optimizer.step()

    pred_label = pred[0].argmax(0)

    plt.imshow(deprocess(source_images[0].cpu()))
    plt.title('epoch ' + str(iter_i) + ': gta image')
    plt.show()

    plt.imshow(label2color(pred_label.cpu()))
    plt.title('epoch ' + str(iter_i) + ': gta label')
    plt.show()

    pred_target = model(target_images)
    pred_target_label = pred_target[0].argmax(0)

    plt.imshow(deprocess(target_images[0].cpu()))
    plt.title('epoch ' + str(iter_i) + ': cityscape image')
    plt.show()
    plt.imshow(label2color(pred_target_label.cpu()))
    plt.title('epoch ' + str(iter_i) + ': cityscape label')
    plt.show()

pred_label = pred[0].argmax(0)

plt.imshow(deprocess(source_images[0].cpu()))
plt.title('image')
plt.show()

plt.imshow(label2color(pred_label.cpu()))
plt.title('label')
plt.show()

# _, batch = gta_dataloader_iter.__next__()
# images = batch['image'].to(device)
# labels = batch['label'].to(device)
#
# pred = model(images)
#
# loss_seg = criterion(pred, labels.long())
# loss_seg = loss_seg / batch_size
# loss_seg.backward()
# optimizer.step()
#
# pred_label = pred[0].argmax(0)
# plt.imshow(label2color(pred_label.cpu()))
# plt.show()
