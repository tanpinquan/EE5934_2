from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image
from util import *


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




