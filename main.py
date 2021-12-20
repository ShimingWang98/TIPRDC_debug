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
import argparse


torch.manual_seed(0)
device = "cuda"
torch.cuda.current_device()
torch.cuda._initialized = True



transform = transforms.Compose([
    transforms.ToTensor()
])

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--lam", default=0.6, type=float)
opt = parser.parse_args()

total_epoch = 200
batch_size = 128
lr = opt.lr # 0.005 0.001
lam = opt.lam

attributePath = "/home/lxiang_stu8/LFW/dataset/shuffle_pvt_attri_train.txt"
# attributePath = "/home/lxiang_stu8/LFW/dataset/shuffle_id_train.txt"
figurePath = "/home/lxiang_stu8/LFW/dataset/private"
# from utils.py
_, data_train_loader = init_dataloader(attributePath, figurePath, batch_size=batch_size, n_classes=2, attriID=2, skiprows=1, normalization=False)

def adjust_learning_rate(epoch, init_lr=0.0005):
    schedule = [25]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def adjust_learning_rate_classifier(epoch, init_lr=0.0005):
    schedule = [25]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def adjust_learning_rate_decoder(epoch, init_lr=0.0005):
    schedule = [25]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def get_FE():
    FE = torch.load("../params/pre_train128/FE.pth")
    CF = torch.load("../params/pre_train128/CF.pth")
    MI = MutualInformation.MutlInfo()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("parallel")
            FE = torch.nn.DataParallel(FE)
            CF = torch.nn.DataParallel(CF)
            MI = torch.nn.DataParallel(MI)
        FE = FE.cuda()
        CF = CF.cuda()
        MI = MI.cuda()

    os.makedirs('../params/extractor128/', exist_ok=True)
    # lam = 0.1
    for epoch in range(total_epoch):
        print("epoch %d" % epoch)
        current_lr = adjust_learning_rate(epoch, lr)
        FE, CF, MI, tot, loss_p, loss_u = train_extractor.train(FE, CF, MI, data_train_loader, current_lr, lam=lam)
        # smaller loss_p, less privacy; smaller loss_u, better utility
        # train_extractor.test_classifier(FE, CF, data_test_loader)
        print("Epoch {}: tot loss {}, privacy {}; utility {}".format(epoch, tot, loss_p, loss_u))

        if torch.cuda.device_count() > 1: # or no module
            torch.save(FE.module, "../params/extractor128/FE_{}.pth".format(lam))
            torch.save(CF.module, "../params/extractor128/FE_CF_{}.pth".format(lam))
            torch.save(MI.module, "../params/extractor128/FE_MI_{}.pth".format(lam))
        else:
            torch.save(FE, "../params/extractor128/FE_{}.pth".format(lam))
            torch.save(CF, "../params/extractor128/FE_CF_{}.pth".format(lam))
            torch.save(MI, "../params/extractor128/FE_MI_{}.pth".format(lam))

    return FE

'''
def get_ZFE():
    FE = torch.load("Models/mix/pre_train/FE.pth")
    CF = torch.load("Models/mix/pre_train/CF.pth")
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            FE = torch.nn.DataParallel(FE)
            CF = torch.nn.DataParallel(CF)
        FE = FE.cuda()
        CF = CF.cuda()
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate(epoch, lr)
            FE, CF = train_extractor.train_Z(FE, CF, data_train_loader, current_lr, vis)
            train_extractor.test_classifier(FE, CF, data_test_loader)
    except KeyboardInterrupt:
        pass
    if torch.cuda.device_count() > 1:
        torch.save(FE.module, "Models/mix/extractor/FE.pth")
        torch.save(CF.module, "Models/mix/extractor/FE_CF.pth")
    else:
        torch.save(FE, "Models/mix/extractor/FE.pth")
        torch.save(CF, "Models/mix/extractor/FE_CF.pth")

    return FE


def get_zdecoder(FE_path):
    # black box reconstruct image
    FE = torch.load(FE_path)
    for p in FE.parameters():
        p.requires_grad = False
    DC = decoder.ZDecoder()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            FE = torch.nn.DataParallel(FE)
            DC = torch.nn.DataParallel(DC)
        FE = FE.cuda()
        DC = DC.cuda()
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            cur_lr = adjust_learning_rate_decoder(epoch, lr)
            DC = train_decoder.train_zdecoder(FE, DC, data_train_loader, cur_lr, vis)
    except KeyboardInterrupt:
        pass
    if torch.cuda.device_count() > 1:
        torch.save(DC.module, "Models/gender/decoder/decoder.pth")
    else:
        torch.save(DC, "Models/gender/decoder/decoder.pth")

    return DC
'''

def get_classifier(FE_path):
    # utility's learning task: use protected feature to learn downstream utility task
    FE = torch.load(FE_path)
    for p in FE.parameters():
        p.requires_grad = False
    CF = SegmentVGG16.Classifier()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            FE = torch.nn.DataParallel(FE)
            CF = torch.nn.DataParallel(CF)
        FE = FE.cuda()
        CF = CF.cuda()
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate_classifier(epoch, lr)
            CF = train_extractor.train_classifier(FE, CF, data2_train_loader, current_lr, vis)
            train_extractor.test_classifier(FE, CF, data2_test_loader)
    except KeyboardInterrupt:
        pass

    if torch.cuda.device_count() > 1:
        torch.save(CF.module, "Models/mix/smiling/Classifier.pth")
    else:
        torch.save(CF, "Models/mix/smiling/Classifier.pth")

    return CF


if __name__ == '__main__':
    # run get_FE or get_ZFE to get a feature extractor constrained by DIM info
    FE = get_FE()
    # ZD = get_zdecoder("Models/gender/extractor/FE.pth")
    # CF = get_classifier("Models/mix/extractor/FE.pth")
