import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

import glob
from tqdm import tqdm
import torch
import tifffile
import matplotlib.pyplot as plt
import numpy as np


from random import random,  randint
from torchvision import transforms
from torchvision.transforms import Resize 
from skimage.filters import threshold_otsu

from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
sys.path.append('./utils')
from utils import *

try:
    if __name__ == "__main__":
        from degradation_model import *
    else:
        from dataset.degradation_model import *
except:
    from degradation_model import *


# Dataset
class Dataset_degradation(Dataset):
    def __init__(self, GT_dir_list_DS, GT_dir_list_D, device, num_file, up_factor, factor_list, 
                 SR_resolution_list, degradation_method, average, generate_lifetime, 
                 size=512, noise_level=0.5, target_resolution=224, 
                 random_selection_flag=False, crop_flag=True, flip_flag=True,
                 read_LR=False, on_the_fly_flag=True):
        super(Dataset_degradation, self).__init__()
        self.num_file = num_file
        self.up_factor = up_factor
        self.noise_level = noise_level
        self.target_resolution = target_resolution
        self.random_selection_flag = random_selection_flag
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        self.dir_list_DS = []
        self.dir_list_D = []
        self.dir_list_S = []
        self.read_LR = read_LR
        self.on_the_fly = on_the_fly_flag
        self.SR_resolution_list = SR_resolution_list
        self.degradation_method = degradation_method
        self.average = average
        self.generate_lifetiem = generate_lifetime
        for i in range(len(GT_dir_list_DS)):
            self.dir_list_DS.append(natsort.natsorted(glob.glob(GT_dir_list_DS[i]+'/*')))
        for i in range(len(GT_dir_list_D)):
            self.dir_list_D.append(natsort.natsorted(glob.glob(GT_dir_list_D[i]+'/*')))

        self.size = size
        self.device = device 
        self.factor_list = factor_list
        self.plain = np.zeros((self.size, self.size))

        self.deg = Degradation_base_model(target_resolution=target_resolution, noise_level=self.noise_level, average=self.average, 
                                          size=self.size, STED_resolution_list=self.SR_resolution_list, 
                                          factor_list=self.factor_list, 
                                          device=self.device)

        self.psf_list = []
        if not read_LR:
            self.deg.set_image_params()
            if SR_resolution_list:
                for index_resolution in range(len(self.SR_resolution_list)):
                    self.cal_psf = self.deg.generate_cal_psf(resolution_LR=target_resolution, resolution_SR=self.SR_resolution_list[index_resolution])
                    self.psf_list.append(self.cal_psf)

    # set the number of enumerations according to options
    def __len__(self):
        return self.num_file        
    def norm_statistic(self, Input, std=None):
        mean = torch.mean(Input).to(self.device)
        mean_zero = torch.zeros_like(mean).to(self.device)
        std = torch.std(Input).to(self.device) if std == None else std
        output = transforms.Normalize(mean_zero, std)(Input)
        return output, mean_zero, std
    def gen_mask(self, Input, kernel_size=(7,7), iteration=7):
        thresh = threshold_otsu(Input)
        Input[Input<thresh] = 0
        Input[Input>=thresh] = 1
        return Input
    # get the parameters of cropping
    def get_crop_params(self, img_size, output_size):
        h, w = img_size
        th = output_size
        tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = randint(0, h - th)
        j = randint(0, w - tw)
        return i, j, th, tw
    def rand_crop(self, img_1, img_2, image_size, crop_size):
        self.i,self.j,self.height,self.width = self.get_crop_params(img_size=image_size, output_size=crop_size) 
        img_1 = img_1[self.i:self.i+self.height, self.j:self.j+self.width]
        img_2 = img_2[self.i:self.i+self.height, self.j:self.j+self.width]
        return img_1, img_2
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
                    Input[i,:,:] = np.flipud(Input[i,:,:])
                    if self.h_flip_flag:
                        Input[i,:,:] = np.fliplr(Input[i,:,:])
                    else:
                        pass
                else:
                    pass
        return Input
    def __getitem__(self, index):
        # creat a stack for GT_DS and GT_D
        self.deg.create_stack()
        # list for different LR and HR components
        HR_list = []
        confocal_list = []
        LR_list = []
        # read HR images random_selection - select images randomly, or select images base on dataloader
        for it in range(len(self.dir_list_DS)):
            if self.random_selection_flag:
                D_index = randint(0, len(self.dir_list_DS[it])-1)
                #D_index = randint(0, 20)
            else:
                D_index = index % len(self.dir_list_DS[it])
            # STED images
            HR = tifffile.imread(self.dir_list_DS[it][D_index])
            HR_list.append(HR)
            # Confocal images, not necessary
            confocal = tifffile.imread(self.dir_list_D[it][D_index])
            confocal_list.append(confocal)
        # list to acquire smallest size of component images
        min_list = []
        # abandoned
        mask_list = []
        # apply image flip among components
        for it in range(len(self.dir_list_D)):
            if self.flip_flag:
                self.h_flip_flag = int(random()>0.5)
                self.v_flip_flag = int(random()>0.5)
                HR_list[it] = np.float32(self.numpy_flip(HR_list[it]))
                confocal_list[it] = np.float32(self.numpy_flip(confocal_list[it]))
            else:
                HR_list[it] = np.array(HR_list[it])
                confocal_list[it] = np.array(confocal_list[it])
            if not self.read_LR:
                # remap images to a certain range
                HR_list[it] = np.float32(HR_list[it])
                HR_list[it] = self.deg.map_values_numpy(HR_list[it], new_max=255, new_min=0, percentile=99.9)
                HR_list[it][HR_list[it] < 0] = 0

            mask_list.append(self.plain)
            # send min size to the list
            min_list.append(min(HR_list[it].shape))
        # set the crop_size for final output image lateral size
        # in real_time, the crop_size = size in yml file
        if not self.on_the_fly: 
            crop_size = min(min_list) if self.size < min(min_list) else self.size
        else: 
            crop_size = self.size
        # zeros for addition of different LR components
        Input = np.zeros((crop_size//2, crop_size//2, 1)) if self.up_factor != 1\
            else np.zeros((crop_size, crop_size, 1)) 
        self.deg.generate_plain(size=crop_size)
        # enumerate in components
        for it in range(len(self.dir_list_D)):
            # crop the data randomly or not
            if self.crop_flag:
                HR_list[it], confocal_list[it] = self.rand_crop(img_1=HR_list[it], img_2=confocal_list[it], image_size=HR_list[it].shape, crop_size=crop_size)
            else:
                HR_list[it] = HR_list[it][:crop_size,:crop_size]
                confocal_list[it] = confocal_list[it][:crop_size,:crop_size]
            # generate individual LR images h,w,c
            if self.read_LR:
                pass
            else:
                if self.target_resolution != 0: 
                    LR_list.append(self.deg.degrade_resolution_numpy(np.expand_dims(HR_list[it], -1), self.psf_list[it]))
                else:
                    LR_list.append(np.expand_dims(HR_list[it], -1))

            # add images to stack HR_list-h,w Stack-h,w,c
            if not self.read_LR:
                self.deg.add_image(Input_HR=HR_list[it], Input_LR=LR_list[it])
            else:
                self.deg.add_image(Input_HR=HR_list[it], Input_LR=confocal_list[it])
            
            # if degradation is not performed
            if self.read_LR: 
                Input += self.factor_list[it]*np.expand_dims(confocal_list[it], axis=-1)
        
        # concatenation h,w,c
        GT_DS, GT_D = self.deg.images_concatenation()
        # channel degradation - composition h,w,c
        blurred, GT_S = self.deg.composition(factor_list=self.factor_list)

        # do simulated llifetime distribution
        if self.generate_lifetiem:
            thresholded_list = []
            # generate mask for lifetime
            for i in range(len(HR_list)):
                thresholded_list.append(self.deg.make_threshold(HR_list[i]))
                
            simulated_lifetime = self.deg.make_lifetime_distribution(num_org=len(HR_list), thresholded_list=thresholded_list, size=crop_size)
            simulated_lifetime = np.expand_dims(simulated_lifetime, -1)
            simulated_lifetime = np.transpose(simulated_lifetime, (2,0,1))
        else:
            simulated_lifetime = 0
        # add noise for individual LR images stack_LR-h,w,c
        if self.noise_level > 0 and not self.read_LR:
            for j in range(len(HR_list)):                
                self.deg.stack_LR[j] = self.deg.degrade_noise(self.deg.stack_LR[j], version="numpy", noise_scale=self.noise_level, average=self.average)
        # concatenation h,w,c
        _, GT_D = self.deg.images_concatenation()
        
        # resolution degradation Input h,w,c
        if not self.read_LR: 
            if self.target_resolution > 0:
                if self.degradation_method == "composite-blur-noise":
                    Input = np.expand_dims(self.deg.degrade_resolution_numpy(GT_S, self.general_psf), -1)
                elif self.degradation_method == "blur-composite-noise":
                    Input = blurred
            else:
                Input = GT_S
        # noise degradation h,w,c
        if self.noise_level > 0 and not self.read_LR: # and self.degradation_method == "composite-blur-noise":
            Input = self.deg.degrade_noise(Input, version="numpy", noise_scale=self.noise_level, average=self.average)
        
        # make to (b,c,h,w)
        Input = np.transpose(Input, (2,0,1))
        GT_S = np.transpose(GT_S, (2,0,1))
        GT_D = np.transpose(GT_D, (2,0,1))
        GT_DS = np.transpose(GT_DS, (2,0,1))
        
        # do normalization if needed
        statistic_dict = {} 
        if self.on_the_fly:
            # send to tensor
            Input = torch.tensor(np.float32(Input), dtype=torch.float32, device=self.device)
            GT_DS = torch.tensor(np.float32(GT_DS), dtype=torch.float32, device=self.device)
            GT_D = torch.tensor(np.float32(GT_D), dtype=torch.float32, device=self.device)
            GT_S = torch.tensor(np.float32(GT_S), dtype=torch.float32, device=self.device)

            # normalization
            Input, Input_mean, Input_std = self.norm_statistic(Input, std=None)
            GT_DS, GT_DS_mean, GT_DS_std = self.norm_statistic(GT_DS, Input_std)
            GT_D, GT_D_mean, GT_D_std = self.norm_statistic(GT_D, Input_std)
            GT_S, GT_S_mean, GT_S_std = self.norm_statistic(GT_S, Input_std)

            # generate statistic dict for recovering
            statistic_dict = {
                "Input_mean":Input_mean, "Input_std":Input_std                
                }
            if self.generate_lifetiem:
                simulated_lifetime = torch.tensor(np.float32(simulated_lifetime), dtype=torch.float, device=self.device)
                try:
                    simulated_lifetime, _, _ = self.norm_statistic(simulated_lifetime, std=None)
                except:
                    simulated_lifetime, _, _ = self.norm_statistic(simulated_lifetime, std=1)
        
        return Input, GT_DS, GT_D, GT_S, simulated_lifetime, statistic_dict
        


def gen_degradation_dataloader(GT_tag_list, noise_level, SR_resolution_dict, target_resolution, degradation_method, average, generate_lifetime_flag, 
                on_the_fly_flag, num_file_train, num_file_val, size, read_LR_flag, factor_list, num_workers, cwd, device, up_factor=1, batch_size=1, 
                eval_flag=False):
    
    train_dir_LR =  os.path.join(cwd, "data\\train_LR")
    val_dir_LR = os.path.join(cwd, "data\\val_LR")
    train_dir_HR = os.path.join(cwd, "data\\train_HR")
    val_dir_HR = os.path.join(cwd, "data\\val_HR")
    train_dir_GT_HR_list = []
    val_dir_GT_HR_list = []
    train_dir_GT_LR_list = []
    val_dir_GT_LR_list = []
    STED_resolution_list = [SR_resolution_dict[org] for org in GT_tag_list] if SR_resolution_dict else None
    for i in range(len(GT_tag_list)):
        train_dir_GT_HR_list.append(os.path.join(train_dir_HR, GT_tag_list[i]))
        val_dir_GT_HR_list.append(os.path.join(val_dir_HR, GT_tag_list[i]))
        train_dir_GT_LR_list.append(os.path.join(train_dir_LR, GT_tag_list[i]))
        val_dir_GT_LR_list.append(os.path.join(val_dir_LR, GT_tag_list[i]))

    if not eval_flag:
        train_dataset = Dataset_degradation(
            GT_dir_list_DS=train_dir_GT_HR_list, GT_dir_list_D=train_dir_GT_LR_list,
            size=size, device=device, noise_level=noise_level,
            SR_resolution_list=STED_resolution_list, target_resolution=target_resolution, degradation_method=degradation_method, 
            average=average, generate_lifetime=generate_lifetime_flag, on_the_fly_flag=on_the_fly_flag,
            num_file=num_file_train, up_factor=up_factor, factor_list=factor_list, read_LR=read_LR_flag, 
            random_selection_flag=True, crop_flag=True, flip_flag=True
        )
        if num_workers:
            train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
        else:
            train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
    val_dataset = Dataset_degradation(
        GT_dir_list_DS=val_dir_GT_HR_list, GT_dir_list_D=val_dir_GT_LR_list,
        size=size, device=device, noise_level=noise_level, 
        SR_resolution_list=STED_resolution_list, target_resolution=target_resolution, degradation_method=degradation_method, 
        average=average, generate_lifetime=generate_lifetime_flag, on_the_fly_flag=on_the_fly_flag,
        num_file=num_file_val, up_factor=up_factor, factor_list=factor_list, read_LR=read_LR_flag, 
        random_selection_flag=False, crop_flag=False, flip_flag=False
    )
    eval_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1)
    
    if not eval_flag:
        return train_dataloader, eval_dataloader
    else:
        return eval_dataloader


if __name__ == "__main__":
    cwd = os.getcwd()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    if torch.cuda.is_available():        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        print('-----------------------------Using GPU-----------------------------')
    
    # prepare DSCM dataset
    if 1:
        target_resolution = 280
        noise_level = 0.280
        average = 1
        org_list = ['Micro', 'Mito', 'Lyso']
        factor_list = [1, 1, 1]
        #degradation_method = "composite-blur-noise"
        degradation_method = "blur-composite-noise"
        generate_lifetime = True
        
        num_workers = 4

        num_train_image = 1
        num_val_image =  1

        STED_resolution_dict = {
            "Micro": 85.93, "Mito": 87.09, "Lyso": 91.97, "Membrane": 81.24, "NPCs": 85.92, "Mito_inner": 82.88, "Mito_inner_deconv": 81.19
        }
        
        if generate_lifetime:
            combination_name = "_".join(org_list) + "_" + str(target_resolution) + "_" + str(noise_level) + "_" + str(average) + "_lifetime"
        else:
            combination_name = "_".join(org_list) + "_" + str(target_resolution) + "_" + str(noise_level) + "_" + str(average)
        
        cwd = os.getcwd()
        save_dir_train = os.path.join(cwd, "data\\prepared_data\\train")
        save_dir_val = os.path.join(cwd, "data\\prepared_data\\val")

        save_dir_folder = os.path.join(save_dir_train, combination_name)
        Input_dir = os.path.join(save_dir_folder, "Input")
        GT_S_dir = os.path.join(save_dir_folder, "GT_S")
        GT_DS_dir = os.path.join(save_dir_folder, "GT_DS")
        GT_D_dir = os.path.join(save_dir_folder, "GT_D")
        check_existence(Input_dir)                                                                                                                                                                                                                     
        check_existence(GT_S_dir)
        check_existence(GT_D_dir)
        check_existence(GT_DS_dir)
        if generate_lifetime:
            lifetime_dir = os.path.join(save_dir_folder, "lifetime")
            check_existence(lifetime_dir)
        
        train_dataloader, eval_dataloader = gen_degradation_dataloader(GT_tag_list=org_list, noise_level=noise_level,
                factor_list=factor_list, SR_resolution_dict=STED_resolution_dict, target_resolution=target_resolution, generate_lifetime_flag=generate_lifetime, 
                degradation_method=degradation_method, average=average, read_LR_flag=False, num_file_train=num_train_image, num_file_val=num_val_image, size=512, num_workers=num_workers, 
                device=device, cwd=cwd, on_the_fly_flag=False)
        bar = tqdm(total=num_train_image)
        for batch_index, data in enumerate(train_dataloader):
            Input, GT_DS, GT_D, GT_S, simulated_lifetime, sta = data
            
            Input = to_cpu(Input)
            GT_DS = to_cpu(GT_DS)
            GT_D = to_cpu(GT_D)
            GT_S = to_cpu(GT_S)
            # save to folder
            tifffile.imwrite(os.path.join(Input_dir, f"{batch_index+1}.tif"), np.uint16(Input[0,0,:,:]))
            tifffile.imwrite(os.path.join(GT_S_dir, f"{batch_index+1}.tif"), np.uint16(GT_S[0,0,:,:]))
            tifffile.imwrite(os.path.join(GT_D_dir, f'{batch_index+1}.tif'), np.uint16(GT_D[0,:,:,:]), imagej=True)
            tifffile.imwrite(os.path.join(GT_DS_dir, f'{batch_index+1}.tif'), np.uint16(GT_DS[0,:,:,:]), imagej=True)
            
            if generate_lifetime:
                simulated_lifetime = to_cpu(simulated_lifetime)
                tifffile.imwrite(os.path.join(lifetime_dir, f'{batch_index+1}.tif'), np.uint16(simulated_lifetime[0,0,:,:]))

            bar.update(1)
        bar.close()
        save_dir_folder = os.path.join(save_dir_val, combination_name)
        Input_dir = os.path.join(save_dir_folder, "Input")
        GT_S_dir = os.path.join(save_dir_folder, "GT_S")
        GT_DS_dir = os.path.join(save_dir_folder, "GT_DS")
        GT_D_dir = os.path.join(save_dir_folder, "GT_D")
        check_existence(Input_dir)                                                                                                                                                                                                                     
        check_existence(GT_S_dir)
        check_existence(GT_D_dir)
        check_existence(GT_DS_dir)
        if generate_lifetime:
            lifetime_dir = os.path.join(save_dir_folder, "lifetime")
            check_existence(lifetime_dir)

        for batch_index, data in enumerate(eval_dataloader):
            Input, GT_DS, GT_D, GT_S, simulated_lifetime, sta = data

            Input = to_cpu(Input)
            GT_DS = to_cpu(GT_DS)
            GT_D = to_cpu(GT_D)
            GT_S = to_cpu(GT_S)
            # save to folder
            tifffile.imwrite(os.path.join(Input_dir, f"{batch_index+1}.tif"), np.uint16(Input[0,0,:,:]))
            tifffile.imwrite(os.path.join(GT_S_dir, f"{batch_index+1}.tif"), np.uint16(GT_S[0,0,:,:]))
            tifffile.imwrite(os.path.join(GT_D_dir, f'{batch_index+1}.tif'), np.uint16(GT_D[0,:,:,:]), imagej=True)
            tifffile.imwrite(os.path.join(GT_DS_dir, f'{batch_index+1}.tif'), np.uint16(GT_DS[0,:,:,:]), imagej=True)
            if generate_lifetime:
                simulated_lifetime = to_cpu(simulated_lifetime)
                tifffile.imwrite(os.path.join(lifetime_dir, f'{batch_index+1}.tif'), np.uint16(simulated_lifetime[0,0,:,:]))
           