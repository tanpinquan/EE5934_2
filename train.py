from deeplab.deeplab import *
import dataloader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from util import *

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# device = 'cpu'
print('Use', device)

batch_size = 4
shuffle = False
image_size = (224, 224)
gta_datset = dataloader.GtaDataset('./data/GTA_V/train_img', './data/GTA_V/train_label', image_size=image_size)

gta_dataloader = DataLoader(gta_datset, batch_size=batch_size, shuffle=shuffle)
gta_dataloader_iter = enumerate(gta_dataloader)

'''PLot sample'''
sample = gta_datset[0]
plt.imshow(deprocess(sample['image']))
plt.show()

plt.imshow(label2color(sample['label']))
plt.show()


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


# Define network
lr = 0.01
momentum = 0.9
weight_decay = 5e-4
nesterov = False
ignore_index = 255

model = DeepLab(num_classes=19,
                backbone='mobilenet',
                sync_bn=False)
model.to(device)

train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                {'params': model.get_10x_lr_params(), 'lr': lr * 10}]
optimizer = torch.optim.SGD(train_params, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index, reduction='mean')

model.train()

''' Perform trainind'''
max_iter = 10
for iter_i in range(max_iter):
    optimizer.zero_grad()
    print('epoch', iter_i)
    adjust_learning_rate(optimizer, lr, iter_i, max_iter)
    for i, batch in enumerate(gta_dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()

        pred = model(images)
        loss_seg = criterion(pred, labels.long())
        # loss_seg = loss_seg / len(gta_datset.filenames)

        loss_seg = loss_seg / batch_size
        loss_seg.backward()
        optimizer.step()

        print(i, loss_seg)
        # if i > 50:
        #     break
    # optimizer.step()

    pred_label = pred[0].argmax(0)

    plt.imshow(deprocess(images[0].cpu()))
    plt.title('image')
    plt.show()

    plt.imshow(label2color(pred_label.cpu()))
    plt.title('label')
    plt.show()

pred_label = pred[0].argmax(0)

plt.imshow(deprocess(images[0].cpu()))
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
