import os
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import time
import torch
import PIL
import torchvision.utils as tvls
from PIL import Image
import re


def init_dataloader(file_path, img_path, batch_size=64, n_classes=1000, attriID=1, shuffle=False, skiprows=1,
                    allAttri=False, normalization=False):
    tf = time.time()

    data_set = ImageFolder(file_path, img_path, n_classes, attriID, skiprows, allAttri, normalization)
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=0,
                                              pin_memory=True)

    interval = time.time() - tf
    print('Initializing data loader took %ds' % interval)
    return data_set, data_loader


class ImageFolder(data.Dataset):
    def __init__(self, file_path, img_path, n_classes, attriID, skiprows=1, allAttri=False, normalization=False):
        self.img_path = img_path
        self.allAttri = allAttri
        self.trans = self.get_processor() if normalization else transforms.ToTensor()
        self.img_list = os.listdir(self.img_path)
        # self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path, attriID, skiprows)
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = n_classes
        # print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path, attriID, skiprows=1):
        name_list, label_list = [], []
        f = open(file_path, "r", encoding='utf-8-sig')
        for i in range(skiprows):
            f.readline()
        for line in f.readlines():
            item = line.strip()
            item = re.split(r',|\s ', item)
            img_name = item[0]
            if self.allAttri:
                iden = item[1:]
            else:
                iden = item[attriID]
            name_list.append(img_name)
            if self.allAttri:
                label_list.append(list(map(int, iden)))
            else:
                label_list.append(int(iden))

        return name_list, torch.tensor(label_list, dtype=torch.long)

    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            path = self.img_path + "/" + img_name
            img = PIL.Image.open(path)
            img = img.convert('RGB')
            # img = self.trans(img)
            img_list.append(img)
        return img_list

    def get_processor(self):

        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        return transforms.Compose(proc)

    def __getitem__(self, index):
        img = self.trans(self.image_list[index])
        label = self.label_list[index]
        # one_hot = np.zeros(self.n_classes, dtype=np.long)
        # one_hot[label] = 1
        return img, label
        # return img, one_hot, label

    def __len__(self):
        return self.num_img


def save_tensor_images(images, filename, nrow=None, normalize=True):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)


def transform_img_size(fake):
    bs = fake.shape[0]
    # size_original = fake.shape[2]
    fake_img = torch.zeros((bs, 3, 64, 64))
    for i in range(bs):
        img_tmp = transforms.ToPILImage()(fake[i].cpu())
        fake_tmp = transforms.ToTensor()(img_tmp.resize((64, 64)))
        fake_img[i] = fake_tmp
    return fake_img


def find_most_id(identity, bs):
    identity, _ = torch.sort(identity)
    max_score = 0
    index = 0
    for i in range(1001):
        score = 0
        for j in range(bs):
            if identity[j] == i:
                score = score + 1
        if score >= max_score:
            max_score = score
            index = i
    return index


def load_img(image_path):
    img_list = []
    name_list = os.listdir(image_path)
    for img_name in enumerate(name_list):
        path = image_path + "/" + img_name[1]
        img = Image.open(path)
        img = torch.unsqueeze(transforms.ToTensor()(img), dim=0)
        img_list.append(img)
    return img_list, img_name[0] + 1


def get_feature(T, img, index):
    feature = T.module.get_fea(img)
    feature = feature[index]
    feature = feature.view(feature.size(0), -1)
    return feature


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)
