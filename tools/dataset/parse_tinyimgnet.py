import os

import numpy as np
from PIL import Image

data_root = './tiny-imagenet-200'
save_path = 'tiny200.npz'


def get_array(path):
    image = Image.open(path)
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.repeat(image.reshape(*image.shape, 1), 3, -1)
    return image


class_names = os.listdir('%s/train/' % data_root)
class_names = sorted(class_names)
label_lookup = {}
for class_name in class_names:
    label_lookup[class_name] = len(label_lookup)

trainX, trainY = [], []
for class_name in class_names:
    label = label_lookup[class_name]
    folder_path = '%s/train/%s/images/' % (data_root, class_name)
    for image in os.listdir(folder_path):
        image_path = folder_path + image
        image = get_array(image_path)
        trainX.append(image)
        trainY.append(label)

trainX = np.stack(trainX)
trainY = np.array(trainY, dtype=np.int64)

valX, valY = [], []
anotation = '%s/val/val_annotations.txt' % data_root
anotation = open(anotation).readlines()
for line in anotation:
    image_path, class_name = line.strip().split()[:2]
    image_path = '%s/val/images/%s' % (data_root, image_path)
    image = get_array(image_path)
    label = label_lookup[class_name]
    valX.append(image)
    valY.append(label)

valX = np.stack(valX)
valY = np.array(valY, dtype=np.int64)

np.savez(save_path, trainX=trainX, trainY=trainY, valX=valX, valY=valY)
