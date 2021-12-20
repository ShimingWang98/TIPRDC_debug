import torch
from torch import optim
from torch import nn
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
from loader import *
import matplotlib.pyplot as plt

import SegmentVGG16
from MutualInformation import info_loss
from MutualInformation import MutlInfo

torch.manual_seed(0)

device = "cuda"
torch.cuda.current_device()
torch.cuda._initialized = True


transform = transforms.Compose([
    transforms.ToTensor()
])

total_epoch = 1
batch_size = 32
lr = 0.001

attributePath = "/home/lxiang_stu8/LFW/dataset/shuffle_pvt_attri_train.txt"
# attributePath = "/home/lxiang_stu8/LFW/dataset/shuffle_id_train.txt"
figurePath = "/home/lxiang_stu8/LFW/dataset/private"
# from utils.py
_, data_train_loader = init_dataloader(attributePath, figurePath, batch_size=batch_size, n_classes=2, attriID=2, skiprows=1, normalization=False)

criterion = nn.CrossEntropyLoss()
test_loss = []
test_acc = []

save_train = '../params/pre_train128/'
curve_dir = '../curves128/'
os.makedirs(save_train, exist_ok=True)
os.makedirs(curve_dir, exist_ok=True)


def adjust_learning_rate(epoch, init_lr=0.001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def train_FE_CF(FE, CF, data_train_loader, current_lr, vis=None):
    FE.train()
    CF.train()
    FE_optimizer = optim.Adam(FE.parameters(), lr=current_lr, weight_decay=1e-4)
    CF_optimizer = optim.Adam(CF.parameters(), lr=current_lr, weight_decay=1e-4)

    loss = 0
    train_loss = []
    for i, (images, labels) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        z = FE(images)
        u = CF(z)
        loss = criterion(u, labels)

        FE_optimizer.zero_grad()
        CF_optimizer.zero_grad()
        loss.backward()
        CF_optimizer.step()
        FE_optimizer.step()

        loss = loss.detach().cpu().item()
        train_loss.append(loss)

    return FE, CF, torch.mean(torch.Tensor(train_loss))


def test_FE_CF(FE, CF, data_test_loader):
    FE.eval()
    CF.eval()

    avg_loss = 0
    avg_acc = 0
    counter = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            features = FE(images)
            output = CF(features)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            # pred = output.detach().max(1)
            avg_acc += pred.eq(labels.view_as(pred)).sum()
            counter += 1

    avg_loss /= counter
    avg_loss = avg_loss.detach().cpu().item()
    avg_acc = float(avg_acc) / len(data_test_loader)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss, avg_acc))
    test_loss.append(avg_loss)
    test_acc.append(avg_acc)


def plot_loss(loss_list, directory, name, title):
    '''
    name: filename
    title: plot title
    '''
    x = range(len(loss_list))
    plt.plot(x, loss_list)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + name + '.png')  # './curves/gp/'
    plt.close()


def get_FE_CF():
    FE = SegmentVGG16.FeatureExtractor()
    CF = SegmentVGG16.Classifier()
    plot_loss_list = []
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("yes")
            FE = torch.nn.DataParallel(FE)
            CF = torch.nn.DataParallel(CF)
        FE = FE.cuda()
        CF = CF.cuda()

    for epoch in range(total_epoch):
        print("epoch %d" % epoch)
        current_lr = adjust_learning_rate(epoch, lr)
        FE, CF, loss = train_FE_CF(FE, CF, data_train_loader, current_lr)
        # test_FE_CF(FE, CF, data_test_loader)#TODO: 不test啦？
        plot_loss_list.append(loss)
        print("Epoch {}, loss {}".format(epoch, loss))
        '''
        if (epoch + 1) % 5 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(FE.module, save_train + "FE.pth")
                torch.save(CF.module, save_train + "CF.pth")
            else:
                torch.save(FE, save_train + "FE.pth")
                torch.save(CF, save_train + "CF.pth")
        '''
        if torch.cuda.device_count() > 1:
            torch.save(FE.module, save_train + "FE.pth")
            torch.save(CF.module, save_train + "CF.pth")
        else:
            torch.save(FE, save_train + "FE.pth")
            torch.save(CF, save_train + "CF.pth")
    plot_loss(plot_loss_list, curve_dir, 'training_loss', 'train loss')
    return FE, CF


if __name__ == '__main__':
    # run get_FE or get_ZFE to get a feature extractor whether or not constrained by DIM info
    FE, CF = get_FE_CF()
