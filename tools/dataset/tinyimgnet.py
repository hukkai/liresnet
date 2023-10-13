import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image


class simple_dataset(Data.Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index: int):
        X = Image.fromarray(self.X[index])
        if self.transform is not None:
            X = self.transform(X)
        Y = self.Y[index]
        return X, Y

    def __len__(self):
        return self.X.shape[0]


def tinyimagenet_dataset(data_root='./data'):
    data = np.load('%s/tiny200.npz' % data_root)
    trainX = data['trainX']
    trainY = data['trainY']
    valX = data['valX']
    valY = data['valY']

    # save memory from uint8 vs float32, do it on the fly
    # trainX = trainX.float().div_(255.)
    # valX = valX.float().div_(255.)

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandAugment()]),
        transforms.ToTensor()
    ])

    trainset = simple_dataset(trainX, trainY, transform_train)
    testset = simple_dataset(valX, valY, transforms.ToTensor())
    return trainset, testset


def DDPM_dataset(data_root='./data', num_classes=100):
    crop_size, padding = 32, 2

    data = np.load(f'{data_root}/c{num_classes}_ddpm.npz')
    trainX = data['image']
    trainY = data['label']
    if num_classes == 200:
        crop_size, padding = 64, 4

    print(trainX.shape)
    # save memory from uint8 vs float32, do it on the fly
    # trainX = trainX.float().div_(255.)

    transform_train = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trainset = simple_dataset(trainX, trainY, transform_train)
    return trainset
