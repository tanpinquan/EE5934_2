from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image
from util import *
import glob


class GtaDataset(Dataset):

    def __init__(self, image_dir, label_dir, image_size):
        self.filenames = os.listdir(image_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path)
        image = preprocess(image, size=self.image_size, should_resize=True)
        image = torch.squeeze(image)

        label_path = os.path.join(self.label_dir, self.filenames[idx])
        label = Image.open(label_path)
        label = label.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        label = np.array(label)
        label = label2train(label)

        sample = {'image': image, 'label': label}

        return sample


class CityscapesDataset(Dataset):

    def __init__(self, image_dir, image_size):
        self.filenames = os.listdir(image_dir)
        self.image_dir = image_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path)
        image = preprocess(image, size=self.image_size, should_resize=True)
        image = torch.squeeze(image)

        sample = {'image': image}

        return sample


class CityscapesValDataset(Dataset):

    def __init__(self, image_dir, label_dir, image_size):
        self.filenames_label = [os.path.basename(x) for x in glob.glob(label_dir+'/*_gtFine_labelIds.png')]
        self.filenames_img = [x[:-20] + '_leftImg8bit.png' for x in self.filenames_label]
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.image_size = image_size
        # self.image_number_regex = re.compile(r'\d\d\d\d\d\d_\d\d\d\d\d\d')

    def __len__(self):
        return len(self.filenames_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.image_dir, self.filenames_img[idx])
        image = Image.open(img_path)
        image = preprocess(image, size=self.image_size, should_resize=True)
        image = torch.squeeze(image)

        label_path = os.path.join(self.label_dir, self.filenames_label[idx])
        label = Image.open(label_path)
        label = label.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        label = np.array(label)
        label = label2train(label)

        sample = {'image': image, 'label': label}
        return sample







