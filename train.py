from deeplab.deeplab import *
from discriminator import FCDiscriminator, Discriminator
import dataloader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from util import *
import random

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# device = 'cpu'
print('Use', device)

'''Create datasets'''
batch_size = 4
shuffle = True
image_size = (224, 224)

gta_datset = dataloader.GtaDataset('./data/GTA_V/train_img', './data/GTA_V/train_label', image_size=image_size)
gta_dataloader = DataLoader(gta_datset, batch_size=batch_size, shuffle=shuffle)
num_gta_images = len(gta_datset.filenames)

cityscapes_dataset = dataloader.CityscapesDataset('./data/Cityscapes/train_img', image_size=image_size)
cityscapes_dataloader = DataLoader(cityscapes_dataset, batch_size=batch_size, shuffle=shuffle)
cityscapes_dataloader_iter = enumerate(gta_dataloader)
num_cityscape_images = len(cityscapes_dataset.filenames)

cityscapes_val_dataset = dataloader.CityscapesValDataset('./data/Cityscapes/train_img', './data/Cityscapes/val_label', image_size=image_size)

num_train_batches = int(min(num_gta_images, num_cityscape_images) / batch_size)

'''Plot sample images'''

fig = plt.figure(figsize=[8, 8])

ax = []

sample = gta_datset[0]

ax.append(fig.add_subplot(2, 2, 1))
ax[-1].set_title('GTA image')  # set title
plt.imshow(deprocess(sample['image']))

ax.append(fig.add_subplot(2, 2, 2))
ax[-1].set_title('GTA label')  # set title
plt.imshow(label2color(sample['label']))

sample = cityscapes_val_dataset[0]

ax.append(fig.add_subplot(2, 2, 3))
ax[-1].set_title('Cityscapes image')  # set title
plt.imshow(deprocess(sample['image']))

ax.append(fig.add_subplot(2, 2, 4))
ax[-1].set_title('Cityscapes label')  # set title
plt.imshow(label2color(sample['label']))

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
lr = 0.0025
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
source_label_val = 0
target_label_val = 1

'''Define discriminator network'''
lr_D = 2.4e-4
lambda_adv = 0.01
model_D = FCDiscriminator(num_classes=num_classes)
model_D.to(device)
model_D.train()

optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr_D, betas=(0.9, 0.99))
criterion_adv = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

''' Perform training'''
seg_losses = []
adv_losses = []
G_losses = []
D_losses = []
max_iter = 25
for iter_i in range(max_iter):
    # model.train()
    optimizer.zero_grad()
    optimizer_D.zero_grad()
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

        '''Train segmentation net'''
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        optimizer.zero_grad()

        # Compute Lseg
        pred_source = model(source_images)
        loss_seg = criterion(pred_source, source_labels.long())
        loss_seg.backward()
        # loss_seg = loss_seg / len(gta_datset.filenames)

        # Compute Ladv
        pred_target = model(target_images)

        D_out_target = model_D(F.softmax(pred_target, dim=1))
        loss_adv = criterion_adv(D_out_target,
                                 torch.FloatTensor(D_out_target.data.size()).fill_(source_label_val).to(device))
        loss_adv = lambda_adv * loss_adv
        loss_adv.backward()
        optimizer.step()
        loss = loss_seg + loss_adv

        '''Train discriminator net'''
        # accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = True

        optimizer_D.zero_grad()

        pred_source = pred_source.detach()
        D_out_source = model_D(F.softmax(pred_source, dim=1))

        loss_D_source = criterion_adv(D_out_source,
                                      torch.FloatTensor(D_out_source.data.size()).fill_(source_label_val).to(device))
        loss_D_source.backward()
        pred_target = pred_target.detach()
        D_out_target = model_D(F.softmax(pred_target, dim=1))

        loss_D_target = criterion_adv(D_out_target,
                                      torch.FloatTensor(D_out_target.data.size()).fill_(target_label_val).to(device))
        loss_D_target.backward()
        loss_D = loss_D_source + loss_D_target
        optimizer_D.step()

        seg_losses.append(loss_seg.item())
        adv_losses.append(loss_adv.item() / lambda_adv)
        G_losses.append(loss.item())
        D_losses.append(loss_D.item())
        print(i, 'Lseg:', loss_seg.item(), 'Ladv:', loss_adv.item(), 'loss_d_source:', loss_D_source.item(),
              'loss_d_target:', loss_D_target.item())

        if i > 20:
            break

    # ax enables access to manipulate each of subplots
    # model.eval()
    # sel_sample = random.randint(0, len(cityscapes_val_dataset.filenames_img)-1)
    # data = gta_datset[sel_sample]
    # source_image = torch.unsqueeze(data['image'], dim=0).to(device)
    # source_label = data['label']
    # pred_label = model(source_image)[0].argmax(0)
    #
    # data = cityscapes_val_dataset[sel_sample]
    # target_image = torch.unsqueeze(data['image'], dim=0).to(device)
    # target_label = data['label']
    # pred_target_label = model(target_image)[0].argmax(0)
    # fig = plt.figure(figsize=[16, 8])
    #
    # ax = []
    #
    # ax.append(fig.add_subplot(2, 4, 1))
    # ax[-1].set_title('GTA image')  # set title
    # plt.imshow(deprocess(source_image[0].cpu()))
    #
    # ax.append(fig.add_subplot(2, 4, 2))
    # ax[-1].set_title('GTA output')  # set title
    # plt.imshow(label2color(pred_label.cpu()))
    #
    # ax.append(fig.add_subplot(2, 4, 3))
    # ax[-1].set_title('GTA label')  # set title
    # plt.imshow(label2color(source_label))
    #
    # ax.append(fig.add_subplot(2, 4, 5))
    # ax[-1].set_title('Cityscapes image')  # set title
    # plt.imshow(deprocess(target_image[0].cpu()))
    #
    # ax.append(fig.add_subplot(2, 4, 6))
    # ax[-1].set_title('Cityscapes output')  # set title
    # plt.imshow(label2color(pred_target_label.cpu()))
    #
    # ax.append(fig.add_subplot(2, 4, 7))
    # ax[-1].set_title('Cityscapes label')  # set title
    # plt.imshow(label2color(target_label))
    # plt.show()

    ''''''

    fig = plt.figure()
    fig.suptitle('Epoch ' + str(iter_i))

    ax = []

    pred_label = pred_source[0].argmax(0)
    ax.append(fig.add_subplot(2, 3, 1))
    ax[-1].set_title('GTA image')  # set title
    plt.imshow(deprocess(source_images[0].cpu()))

    ax.append(fig.add_subplot(2, 3, 2))
    ax[-1].set_title('GTA label')  # set title
    plt.imshow(label2color(pred_label.cpu()))

    pred_target_label = pred_target[0].argmax(0)
    ax.append(fig.add_subplot(2, 3, 4))
    ax[-1].set_title('Cityscape image')  # set title
    plt.imshow(deprocess(target_images[0].cpu()))

    ax.append(fig.add_subplot(2, 3, 5))
    ax[-1].set_title('Cityscape label')  # set title
    plt.imshow(label2color(pred_target_label.cpu()))

    # plt.show()

    # fig = plt.figure()
    # ax = []
    ax.append(fig.add_subplot(2, 3, 3))
    ax[-1].set_title('Segmentation Losses')  # set title
    plt.plot(seg_losses)
    plt.plot(adv_losses)
    ax[-1].legend(['Seg loss', 'Adv loss'])

    ax.append(fig.add_subplot(2, 3, 6))
    ax[-1].set_title('Overall losses')  # set title
    plt.plot(G_losses)
    plt.plot(D_losses)
    ax[-1].legend(['L', 'Ld'])
    plt.show()

model.eval()
data = gta_datset[0]
source_image = torch.unsqueeze(data['image'], dim=0).to(device)
source_label = data['label']
pred_label = model(source_image)[0].argmax(0)

data = cityscapes_dataset[8]
target_image = torch.unsqueeze(data['image'], dim=0).to(device)
pred_target_label = model(target_image)[0].argmax(0)
fig = plt.figure()

ax = []

ax.append(fig.add_subplot(2, 3, 1))
ax[-1].set_title('GTA image')  # set title
plt.imshow(deprocess(source_image[0].cpu()))

ax.append(fig.add_subplot(2, 3, 2))
ax[-1].set_title('GTA output')  # set title
plt.imshow(label2color(pred_label.cpu()))

ax.append(fig.add_subplot(2, 3, 3))
ax[-1].set_title('GTA label')  # set title
plt.imshow(label2color(source_label))

ax.append(fig.add_subplot(2, 3, 4))
ax[-1].set_title('Cityscapes image')  # set title
plt.imshow(deprocess(target_image[0].cpu()))

ax.append(fig.add_subplot(2, 3, 5))
ax[-1].set_title('Cityscapes output')  # set title
plt.imshow(label2color(pred_target_label.cpu()))
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
