import torch.utils.data as Data

from .cifar import cifar_dataset
from .imagenet_dataset import imagenet_dataset
from .tinyimgnet import DDPM_dataset, tinyimagenet_dataset


def data_loader(data_name: str = 'cifar',
                num_classes: int = 10,
                batch_size: int = 128,
                distributed: bool = True,
                data_root: str = './data/',
                seed: int = 2023,
                **kargs):

    if 'cifar' in data_name and num_classes in [10, 100]:
        trainset, testset = cifar_dataset(num_classes, data_root=data_root)
    elif 'tiny' in data_name and num_classes == 200:
        trainset, testset = tinyimagenet_dataset(data_root=data_root)
    elif 'ddpm' in data_name:
        trainset = DDPM_dataset(data_root=data_root, num_classes=num_classes)
        testset = None
    elif data_name == 'imagenet' and num_classes <= 1000:
        trainset, testset, _ = imagenet_dataset(data_root=data_root,
                                                num_classes=num_classes,
                                                seed=seed)
    elif data_name == 'imagenet_extra' and num_classes <= 1000:
        _, _, trainset = imagenet_dataset(data_root=data_root,
                                          num_classes=num_classes,
                                          seed=seed)
        testset = None
    else:
        raise ValueError('The given dataset config is not supported!')

    train_sampler = test_sampler = None
    if distributed:
        train_sampler = Data.distributed.DistributedSampler(trainset)
        if testset is not None:
            test_sampler = Data.distributed.DistributedSampler(testset)

    train_loader = Data.DataLoader(trainset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   num_workers=8,
                                   shuffle=(train_sampler is None),
                                   drop_last=True,
                                   pin_memory=True,
                                   persistent_workers=True)

    test_loader = None
    if testset is not None:
        test_loader = Data.DataLoader(testset,
                                      batch_size=batch_size * 2,
                                      sampler=test_sampler,
                                      num_workers=8,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True)
    return train_loader, train_sampler, test_loader, test_sampler
