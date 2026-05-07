import os
import torch
import natsort
import math
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np


from PIL import Image
from skimage.metrics import mean_squared_error
from skimage import transform, measure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from openpyxl import Workbook
from random import random,  randint
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset


def check_existence(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        remove_list = os.listdir(dir)
        for i in range(len(remove_list)):
            remove_dir = os.path.join(dir, remove_list[i])
            os.remove(remove_dir)


def resize(input, size):
    m1 = np.max(input)
    resized = transform.resize(input, size)
    m2 = np.max(resized)
    output = resized * m1 / m2
    return np.uint16(output)


def to_cpu(input):
    input = Tensor.cpu(input)
    input = input.detach().numpy()
    return input


def getStat(cal_dataset, device):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(cal_dataset))
    cal_loader = DataLoader(
        cal_dataset, batch_size=1, shuffle=False, num_workers=0,
        )
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for X, _ in cal_loader:
        for d in range(1):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(cal_dataset))
    std.div_(len(cal_dataset))
    #return list(mean.numpy()), list(std.numpy())
    return list(mean), list(std)




def write2Yaml(data, save_path="test.yaml"):
    with open(save_path, "w") as f:
        yaml.dump(data, f)