import os
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from openpyxl import Workbook
from utils import to_cpu, resize, write2Yaml


class ToolBox(nn.Module):
    """
    ToolBox for validation
    """
    def __init__(self, opt):
        super(ToolBox, self).__init__()
        self.opt = opt

    def make_folders(self):
        upper_dir = os.path.join(os.getcwd(), self.opt['validation_dir'])
        name_dir = os.path.join(upper_dir, '{}'.format(self.opt['validation_date']))
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        write2Yaml(self.opt, os.path.join(name_dir, 'option.yml'))
        self.save_dir_list = []
        for tag in self.opt['validation_list']:
            tag_dir = os.path.join(name_dir, tag)
            if not os.path.exists(tag_dir):
                os.mkdir(tag_dir)
                target_dir = os.path.join(tag_dir, '{}'.format(1))
                os.mkdir(target_dir)                
            else:
                num_folder = len(os.listdir(tag_dir))            
                target_dir = os.path.join(tag_dir, '{}'.format(num_folder)) if len(os.listdir(os.path.join(tag_dir, '{}'.format(num_folder)))) == 0 else os.path.join(tag_dir, '{}'.format(num_folder+1))           
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir) 
            self.save_dir_list.append(target_dir)
        return self.save_dir_list
    
    def save_model(self, model, name):      
        for i in range(len(name)):
            save_dir = os.path.join(self.save_dir_list[1], f'{name[i]}.pth')
            torch.save(model[i], save_dir)

    def gen_validation_images_IO(self, data_list, epoch=None, batch_index=None):
        self.val_list = []
        Input_list, decouple_list, sta_list = data_list
        for i in range(len(data_list[0])):
            Input = to_cpu((Input_list[i]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))            
            fake_main = to_cpu((decouple_list[i][0]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))
            GT_main = to_cpu((decouple_list[i][1]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))    
            fake_main[fake_main<0] = 0            
            if self.opt['up_factor'] != 1: Input = resize(Input, (self.opt['size'], self.opt['size']))
            plain = np.zeros_like(Input)
            plot = np.vstack((Input, plain))
            for i in range(fake_main.shape[-1]):
                fake_temp_DS = np.hstack((fake_temp_DS, fake_main[:,:,i:i+1])) if i > 0 else fake_main[:,:,0:1]
                GT_temp_DS = np.hstack((GT_temp_DS, GT_main[:,:,i:i+1])) if i > 0 else GT_main[:,:,0:1]
            col_temp = np.vstack((GT_temp_DS, fake_temp_DS))
            plot = np.hstack((plot, col_temp))
            plot[plot < 0] = 0
            self.val_list.append(np.uint16(plot))

    def gen_validation_images_IL(self, data_list, epoch=None, batch_index=None):
        self.val_list = []
        Input_list, decouple_list, lifetime_list, sta_list = data_list
        for i in range(len(data_list[0])):
            Input = to_cpu((Input_list[i]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))            
            fake_main = to_cpu((decouple_list[i][0]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))
            GT_main = to_cpu((decouple_list[i][1]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))           
            lifetime = to_cpu((lifetime_list[i]*sta_list[i]['Input_std']+sta_list[i]['Input_mean']).squeeze(0).permute(1,2,0))
            lifetime = lifetime / np.max(lifetime) * 128
            fake_main[fake_main<0] = 0
            if self.opt['up_factor'] != 1: Input = resize(Input, (self.opt['size'], self.opt['size']))
            plot = np.vstack((Input, lifetime))
            for i in range(fake_main.shape[-1]):
                fake_temp_DS = np.hstack((fake_temp_DS, fake_main[:,:,i:i+1])) if i > 0 else fake_main[:,:,0:1]
                GT_temp_DS = np.hstack((GT_temp_DS, GT_main[:,:,i:i+1])) if i > 0 else GT_main[:,:,0:1]
            col_temp = np.vstack((GT_temp_DS, fake_temp_DS))
            plot = np.hstack((plot, col_temp))
            plot[plot < 0] = 0
            self.val_list.append(np.uint16(plot))
   
    def save_last_train_image(self, Input, GT, Output, epoch):
        Input = to_cpu(Input)
        GT = to_cpu(GT)
        Output = to_cpu(Output)
        plot = np.hstack((Input[0,0,:,:], GT[0,0,:,:]))
        plot = np.hstack((plot, Output[0,0,:,:]))
        if GT.shape[1] > 1:
            for i in range(GT.shape[1] - 1):
                plot = np.hstack((plot, GT[0,i+1,:,:]))
                plot = np.hstack((plot, Output[0,i+1,:,:]))
        tifffile.imwrite(os.path.join(self.save_dir_list[4], '{}.tif'.format(epoch)), plot)

    def save_val_list(self, name):
        for i in range(len(self.val_list)):
            val_data = np.expand_dims(self.val_list[i], axis=0)
            val_stack = val_data if i == 0 else np.concatenate((val_stack, val_data), 0)
        save_dir_val_list = os.path.join(self.save_dir_list[3], '{}.tif'.format(name))
        tifffile.imwrite(save_dir_val_list, np.array(val_stack))

    def make_loss_plots(self, epoch_list_train, epoch_list_val, train_list, val_list, 
                        model_name_list, loss_name_list, lr_list, pearson_list):     
        for i in range(len(train_list)):            
            for j in range(len(train_list[0])-2):
                plt.figure()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(epoch_list_train, train_list[i][j], label=r'train')                
                plt.plot(epoch_list_val, val_list[i][j], label=r'val')
                plt.legend()
                plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_{loss_name_list[j]}.png'))
                plt.close()
            if np.mean(train_list[i][-2]) != 0:
                plt.figure()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(epoch_list_train, train_list[i][-2], label=r'G')                
                plt.plot(epoch_list_train, train_list[i][-1], label=r'D')
                plt.legend()
                plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_GAN.png'))
                plt.close()
            else:
                plt.figure()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(epoch_list_train, train_list[i][-2], label=r'train')                
                plt.plot(epoch_list_val, val_list[i][-1], label=r'val')
                plt.legend()
                plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_degen.png'))
                plt.close()
            plt.figure()
            plt.xlabel('epoch')
            plt.ylabel('lr')
            plt.plot(epoch_list_train, lr_list, label=r'train')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_lr.png'))
            plt.close()

            plt.figure()
            plt.xlabel('epoch')
            plt.ylabel('pearson_coef')
            plt.plot(epoch_list_val, pearson_list, label=r'val')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir_list[0], f'{model_name_list[i]}_pearson_coef.png'))
            plt.close()
                
        

