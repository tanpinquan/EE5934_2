import torchvision.transforms as T
import numpy as np
import PIL
import matplotlib.pyplot as plt
import json

# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
# IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN = np.array([73.158359210711552, 82.908917542625858, 72.392398761941593])
IMAGENET_STD = np.array([47.675755341814678, 48.494214368814916, 47.736546325441594])
LABEL2TRAIN = np.array([
    [0, 255],
    [1, 255],
    [2, 255],
    [3, 255],
    [4, 255],
    [5, 255],
    [6, 255],
    [7, 0],
    [8, 1],
    [9, 255],
    [10, 255],
    [11, 2],
    [12, 3],
    [13, 4],
    [14, 255],
    [15, 255],
    [16, 255],
    [17, 5],
    [18, 255],
    [19, 6],
    [20, 7],
    [21, 8],
    [22, 9],
    [23, 10],
    [24, 11],
    [25, 12],
    [26, 13],
    [27, 14],
    [28, 15],
    [29, 255],
    [30, 255],
    [31, 16],
    [32, 17],
    [33, 18],
    [-1, 255],
    [34, 255]])

PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]]


def preprocess(img, size=(512, 1024), should_resize=True):
    if should_resize:
        transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(),
                        std=IMAGENET_STD.tolist()),
            T.Lambda(lambda x: x[None]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(),
                        std=IMAGENET_STD.tolist()),
            T.Lambda(lambda x: x[None]),
        ])
    return transform(img)


def deprocess(img, should_rescale=False):
    transform = T.Compose([
        T.Lambda(lambda x: x),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / IMAGENET_STD).tolist()),
        T.Normalize(mean=(-IMAGENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def label2train(label_img):
    for val in LABEL2TRAIN:
        # print(val)
        label_img[label_img == val[0]] = val[1]
    return label_img


def label2color(label_ids):
    output_img = np.zeros((label_ids.shape[0], label_ids.shape[1], 3)).astype('uint8')
    for label in LABEL2TRAIN[:, 1]:
        if label != 255:
            output_img[label_ids == label] = PALETTE[label]
    return output_img

# def color2train(color_img):


if __name__ == "__main__":
    ''' Just some code to test the functions'''

    ''' Test preprocess and deprocess'''
    image_orig = PIL.Image.open('data/CItyscapes/train_img/aachen_000008_000019_leftImg8bit.png')
    image_t = preprocess(image_orig)
    image_deproc = deprocess(image_t[0], should_rescale=False)
    image_deproc_array = np.array(image_deproc)

    plt.imshow(image_orig)
    plt.title('original image')
    plt.show()

    plt.imshow(image_deproc)
    plt.title('transformed back image')
    plt.show()





    with open('scene_parsing/cityscapes_info.json') as f:
        cityscapes_info = json.load(f)

    ''' Test converting image label to training labels'''
    cityscapes_info['label2train'] = np.array(cityscapes_info['label2train'])
    label_orig = PIL.Image.open('data/CItyscapes/val_label/aachen_000008_000019_gtFine_labelIds.png')
    label_array = np.array(label_orig)
    train_labels = label2train(label_array)

    '''Test converting training labels to color labels'''
    label_color = PIL.Image.open('data/CItyscapes/val_label/aachen_000008_000019_gtFine_color.png')
    label_color_array = np.array(label_color)
    plt.imshow(label_color_array)
    plt.title('original labels')

    plt.show()
    output_img = label2color(train_labels)
    plt.imshow(output_img)
    plt.title('transformed back labels')
    plt.show()


    '''Test GTAV'''
    image_gta = np.array(PIL.Image.open('data/GTA_V/train_label/00003.png'))
    plt.imshow(label2color(label2train(image_gta)))
    plt.title('transformed back labels for gta')
    plt.show()