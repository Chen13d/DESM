import os
import torch
import tifffile
import numpy as np
from random import randint, random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def get_crop_params(img_size, output_size):
    h, w = img_size
    th = output_size
    tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w
    i = randint(0, h - th)
    j = randint(0, w - tw)
    return i, j, th, tw


def rand_crop(Input, GT, size):
    i, j, height, width = get_crop_params(img_size=Input.size(), output_size=size)
    Input = Input[i:i+height, j:j+width]
    GT = GT[:, i:i+height, j:j+width]
    return Input, GT


def rand_crop_with_lifetime(Input, GT, lifetime, size):
    i, j, height, width = get_crop_params(img_size=Input.size(), output_size=size)
    Input = Input[i:i+height, j:j+width]
    lifetime = lifetime[i:i+height, j:j+width]
    GT = GT[:, i:i+height, j:j+width]
    return Input, GT, lifetime


class prepared_dataset(Dataset):
    def __init__(self, read_dir, num_file, num_org, org_list, size, device, random_selection_flag=True, crop_flag=True, flip_flag=True, read_lifetime_flag=False):
        super(prepared_dataset, self).__init__()
        self.read_dir = read_dir
        self.num_file = num_file
        self.num_org = num_org
        self.org_list = org_list
        self.device = device
        self.size = size
        self.random_selection_flag = random_selection_flag
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        self.read_lifetime_flag = read_lifetime_flag
        self.generate_read_dir()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

    def generate_read_dir(self):
        self.Input_dir = os.path.join(self.read_dir, "Input")
        self.GT_DS_dir = os.path.join(self.read_dir, "GT_DS")
        self.GT_D_dir = os.path.join(self.read_dir, "GT_D")
        self.GT_S_dir = os.path.join(self.read_dir, "GT_S")
        self.denoised_dir = os.path.join(self.read_dir, "denoised")
        if self.read_lifetime_flag:
            self.lifetime_dir = os.path.join(self.read_dir, "Lifetime")
        self.file_num = self.__get_file_num__()

    def __len__(self):
        self.dataset_length = len(os.listdir(self.Input_dir))
        return self.num_file

    def __get_file_num__(self):
        return len(os.listdir(self.Input_dir))

    def map_values(self, image, new_min=0, new_max=1, min_val=None, max_val=None, percentile=99, index=0):
        if index == 0:
            min_val = torch.quantile(image, (100 - percentile) / 100)
            max_val = torch.quantile(image, percentile / 100)
            if max_val == min_val:
                raise ValueError("最大值和最小值相等，无法进行归一化。")

        scaled = (image - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
        return scaled, min_val, max_val

    def norm_statistic(self, Input, std=None):
        mean = torch.mean(Input).to(self.device)
        mean_zero = torch.zeros_like(mean).to(self.device)
        std = torch.std(Input).to(self.device) if std == None else std
        output = transforms.Normalize(mean_zero, std)(Input)
        return output, mean_zero, std

    def numpy_flip(self, Input):
        if len(Input.shape) == 2:
            if self.v_flip_flag:
                Input = np.flipud(Input)
                if self.h_flip_flag:
                    Input = np.fliplr(Input)
                else:
                    pass
            else:
                pass
        elif len(Input.shape) == 3:
            for i in range(Input.shape[0]):
                if self.v_flip_flag:
                    Input[i, :, :] = np.flipud(Input[i, :, :])
                    if self.h_flip_flag:
                        Input[i, :, :] = np.fliplr(Input[i, :, :])
                    else:
                        pass
                else:
                    pass
        return Input

    def __getitem__(self, index):
        if self.random_selection_flag:
            D_index = randint(0, self.dataset_length - 1)
        else:
            D_index = index

        if self.flip_flag:
            self.h_flip_flag = int(random() > 0.5)
            self.v_flip_flag = int(random() > 0.5)
            Input = torch.tensor(np.float64(self.numpy_flip(tifffile.imread(os.path.join(self.Input_dir, f"{D_index+1}.tif")))), dtype=torch.float, device=self.device)
            GT_DS = torch.tensor(np.float64(self.numpy_flip(tifffile.imread(os.path.join(self.GT_DS_dir, f"{D_index+1}.tif")))), dtype=torch.float, device=self.device)
            if self.read_lifetime_flag:
                simulated_lifetime = torch.tensor(np.float64(self.numpy_flip(tifffile.imread(os.path.join(self.lifetime_dir, f"{D_index+1}.tif")))), dtype=torch.float, device=self.device)
        else:
            Input = torch.tensor(np.float64(tifffile.imread(os.path.join(self.Input_dir, f"{D_index+1}.tif"))), dtype=torch.float, device=self.device)
            GT_DS = torch.tensor(np.float64(tifffile.imread(os.path.join(self.GT_DS_dir, f"{D_index+1}.tif"))), dtype=torch.float, device=self.device)
            if self.read_lifetime_flag:
                simulated_lifetime = torch.tensor(np.float64(tifffile.imread(os.path.join(self.lifetime_dir, f"{D_index+1}.tif"))), dtype=torch.float, device=self.device)

        if len(GT_DS.size()) == 2:
            GT_DS = GT_DS.unsqueeze(0)

        if self.crop_flag:
            if self.read_lifetime_flag:
                Input, GT_DS, simulated_lifetime = rand_crop_with_lifetime(Input=Input, GT=GT_DS, lifetime=simulated_lifetime, size=self.size)
            else:
                Input, GT_DS = rand_crop(Input=Input, GT=GT_DS, size=self.size)
        else:
            Input = Input[:self.size, :self.size]
            GT_DS = GT_DS[:, :self.size, :self.size]
            if self.read_lifetime_flag:
                simulated_lifetime = simulated_lifetime[:self.size, :self.size]

        Input = Input.unsqueeze(0)
        Input, Input_mean, Input_std = self.norm_statistic(Input)

        if self.read_lifetime_flag:
            simulated_lifetime = simulated_lifetime.unsqueeze(0)
            simulated_lifetime, _, _ = self.norm_statistic(simulated_lifetime, Input_std)

        GT_DS, _, _ = self.norm_statistic(GT_DS, Input_std)

        statistic_dict = {
            "Input_mean": Input_mean, "Input_std": Input_std
        }

        if self.read_lifetime_flag:
            return Input, GT_DS, 0, 0, simulated_lifetime, statistic_dict
        else:
            return Input, GT_DS, 0, 0, 0, statistic_dict


def gen_prepared_dataloader(read_dir_train, read_dir_val, num_file_train, num_file_val, num_org, org_list, size, batch_size, device, num_workers=0, read_lifetime_flag=False):
    train_dataset = prepared_dataset(read_dir=read_dir_train, num_file=num_file_train, num_org=num_org, org_list=org_list, size=size,
                                     random_selection_flag=True, crop_flag=True, flip_flag=True, device=device, read_lifetime_flag=read_lifetime_flag)
    val_dataset = prepared_dataset(read_dir=read_dir_val, num_file=num_file_val, num_org=num_org, org_list=org_list, size=size,
                                   random_selection_flag=False, crop_flag=False, flip_flag=False, device=device, read_lifetime_flag=read_lifetime_flag)
    if num_workers > 0:
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1)
    return train_dataloader, val_dataloader


