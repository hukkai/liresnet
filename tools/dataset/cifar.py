import torchvision
import torchvision.transforms as transforms


def cifar_dataset(num_classes, data_root='./data/'):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandAugment()]),
        transforms.ToTensor(),
    ])

    transform_test = transforms.ToTensor()

    dataset = getattr(torchvision.datasets, 'CIFAR%d' % num_classes)

    trainset = dataset(root=data_root,
                       train=True,
                       transform=transform_train,
                       download=True)

    testset = dataset(root=data_root,
                      train=False,
                      transform=transform_test,
                      download=True)
    return trainset, testset
