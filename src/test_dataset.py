import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch


class CUB(Dataset):
    def __init__(self, root):
        """
            Note that CUB has 200 classes, but we only use 180 classes in the training step
            Validation and test are conducted in the remaining 20 classes

            *** Never change the data loader init, len part ***
            *** getitem part can be changed for data augmentation ***
            *** Never include the 20 remaining classes in the training step. It is considered cheating. ***

        """
        self.root = root

        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))

        img_name_list = []

        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        test_img_name_list = img_name_list

        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        test_label_list = label_list

        self._imgs = [plt.imread(os.path.join(self.root, 'images', f))
                           for f in test_img_name_list]
        self._labels = [x for x in test_label_list]

    def __getitem__(self, index):
        """ Data augmentation part

            *** getitem part can be changed for data augmentation ***

        """
        img = self._imgs[index]

        # convert grayscale images into RGB images
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((304, 304), Image.BILINEAR)(img)
        img = transforms.CenterCrop((224, 224))(img)
        #img = transforms.RandomHorizontalFlip()(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.4924, 0.5064, 0.4510], [0.1767, 0.1759, 0.1870])(img)

        return img

    def __len__(self):
        return len(self._labels)


