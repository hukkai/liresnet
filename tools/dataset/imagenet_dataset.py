from typing import Tuple

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

use_ddpm = True


def parse_imagenet(data_root: str,
                   num_classes: int = 1000,
                   seed: int = 2023) -> Tuple[Tuple, Tuple]:
    seed0 = torch.seed()
    torch.manual_seed(seed)
    selected_classes = torch.randperm(1000)[:num_classes]
    torch.manual_seed(seed0)
    selected_classes = set(selected_classes.tolist())
    lookup = {}
    for idx, label in enumerate(selected_classes):
        lookup[label] = idx

    data_lists = []
    for mode in 'train', 'val':
        data_file = f'{data_root}/imagenet/meta/{mode}.txt'
        data_list = []
        with open(data_file) as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.split()
                label = int(label)
                if label not in lookup:
                    continue
                path = f'{data_root}/imagenet/{mode}/{path}'
                path = path.replace('//', '/')
                record = (path, lookup[label])
                data_list.append(record)
        data_lists.append(tuple(data_list))
    if use_ddpm:
        data_list = []
        ddpm = open('data/ddpm1000.txt').readlines()
        for line in ddpm:
            path, label = line.split()
            label = int(label)
            if label not in lookup:
                continue
            path = '/dev/shm/imagenet/' + path
            record = (path, lookup[label])
            data_list.append(record)
        data_lists.append(tuple(data_list))
    else:
        data_lists.append(())
    return tuple(data_lists)


class UniformResize(torch.nn.Module):
    def __init__(self, lower: int, upper: int):
        super(UniformResize, self).__init__()
        self.lower = lower
        self.upper = upper + 1

    def forward(self, x):
        size = torch.randint(self.lower, self.upper, size=[]).item()
        return transforms.Resize(size)(x)


class imagenet_loader(Data.Dataset):
    def __init__(self, datalist: Tuple, data_root: str, mode: bool = True):
        assert mode in ['train', 'val', 'ssl']

        self.datalist = datalist

        if mode == 'train':
            self.transform = transforms.Compose([
                UniformResize(224, 288),
                transforms.RandAugment(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
        elif mode == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])

        self.mode = mode
        self.data_root = data_root

    def __getitem__(self, index):
        path, label = self.datalist[index]
        image = self.read_image(path)
        image = self.transform(image)
        image = image.float().div_(255)
        return image, label

    def __len__(self):
        return len(self.datalist)

    def read_image(self, path):
        # Implement the ead function based on your file system
        pass


def imagenet_dataset(num_classes: int, data_root: str, seed: int = 2023):
    train_list, val_list, ddpm_list = parse_imagenet(data_root, num_classes,
                                                     seed)
    train_dataset = imagenet_loader(train_list, data_root, mode='train')
    val_dataset = imagenet_loader(val_list, data_root, mode='val')
    ddpm_dataset = None
    if use_ddpm:
        ddpm_dataset = imagenet_loader(ddpm_list, data_root, mode='train')
    return train_dataset, val_dataset, ddpm_dataset


def imagenet_ssl_dataset(num_classes: int, data_root: str, seed: int = 2023):
    train_list, val_list = parse_imagenet(data_root, num_classes, seed)
    train_dataset = imagenet_loader(train_list, data_root, mode='ssl')
    return train_dataset
