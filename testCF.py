import torch
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

import SegmentVGG16
import train_extractor
import MutualInformation
import decoder
import train_decoder
from loader import *

lam = 0.0
batch_size = 128

attributePath = "/home/lxiang_stu8/LFW/dataset/shuffle_pvt_attri_train.txt"
# attributePath = "/home/lxiang_stu8/LFW/dataset/shuffle_id_train.txt"
figurePath = "/home/lxiang_stu8/LFW/dataset/private"
# from utils.py
_, testLoader = init_dataloader(attributePath, figurePath, batch_size=batch_size, n_classes=2, attriID=2, skiprows=1, normalization=False)

FE = torch.load(f"../params/extractor128/FE_{lam}.pth").cuda().eval()
CF = torch.load(f"../params/extractor128/FE_CF_{lam}.pth").cuda().eval()

hit1 = 0
total = 0

for i ,(images, labels) in enumerate(testLoader):
    images, labels =images.cuda(), labels.cuda()
    features = FE(images)
    output = CF(features)
    hit1 += (labels == output.argmax(axis=1)).sum().item()
    total += images.shape[0]
print("accuracy:", hit1/total)