import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torchvision import transforms

from net.generate_utils import *
from net.model_tools import *
from net.unet import *
from net.SSFIN import *
from net.DFCAN import *
from net.Unet_FLIM import *

from loss.SSIM_loss import *
#from loss.decouple_loss import *
from loss.gradient_loss import *
from loss.pearson_loss import *
from loss.FFT_loss import *
from loss.GAN_loss import *

from utils import *
from dataset.degradation_model import *


class Basemodel():
    """
    Basemodel including functions for network structures, network forward and backward process
    """
    def __init__(self):
        super(Basemodel).__init__()
    def generate_G(self, model_name, in_channels, num_classes, upscale_factor=1):        
        if model_name == 'Unet':
            net = Unet(input_dim=in_channels, num_classes=num_classes)
        elif model_name == 'DFCAN' or model_name == 'DFGAN':
            net = DFCAN(in_channels=in_channels, out_channels=num_classes, n_channels=64, scale=upscale_factor)
        elif model_name == 'SSFIN':
            net = SpatialSpectralSRNet(in_channels=in_channels, out_channels=num_classes, upscale_factor=upscale_factor)
        elif model_name == 'Unet_FLIM':
            net = Unet(input_dim=in_channels+1, num_classes=num_classes)
        elif model_name == 'Unet_FLIM_att':
            net = Unet_FLIM_att(input_dim=in_channels, num_classes=num_classes)
        elif model_name == "Unet_SR":
            net = Unet(input_dim=in_channels, num_classes=1)
        return net
    
    def generate_D(self, model_name, in_channels):
        if model_name == 'UnetD':
            net_D = UnetD(in_channels=in_channels)
        return net_D
    
    def generate_optim(self, model, lr_G, lr_D):
        if self.optimizer_name == 'Adam':
            self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=lr_G)
            if self.weight_list[5] > 0: self.optim_D = torch.optim.Adam(self.net_D.parameters(), lr=lr_D)
    
    # calculate GAN loss for generator
    def cal_GAN_loss_G(self, fake):
        for p in self.net_D.parameters():
            p.requires_grad = False
        e_F, d_F, _, _ = self.net_D(fake) 
        T_eF = torch.ones_like(e_F, device=self.device)
        T_dF = torch.ones_like(d_F, device=self.device)
        T_eF = True
        T_dF = True
        GAN_loss_G = self.GAN_criterion(e_F, T_eF) + self.GAN_criterion(d_F, T_dF)
        #GAN_loss_G = self.GAN_criterion(e_F, T_eF)
        return GAN_loss_G    
    
    # calculate GAN loss for discriminator
    def cal_GAN_loss_D(self, batch_index, fake, GT):
        if (batch_index+1) % self.index_per_D == 0:            
            for p in self.net_D.parameters():
                p.requires_grad = True
            e_F, d_F, _, _ = self.net_D(fake.detach())
            e_T, d_T, _, _ = self.net_D(GT.detach())
            T_eT = torch.ones_like(e_T, device=self.device)
            T_dT = torch.ones_like(d_T, device=self.device)
            F_eF = torch.zeros_like(e_F, device=self.device)
            F_dF = torch.zeros_like(d_F, device=self.device)

            T_eT = True
            T_dT = True
            F_eF = False
            F_dF = False
            GAN_loss_D = self.GAN_criterion(e_F, F_eF) + self.GAN_criterion(d_F, F_dF) + self.GAN_criterion(e_T, T_eT) + self.GAN_criterion(d_T, T_dT)
            #GAN_loss_D = self.GAN_criterion(e_F, F_eF) + self.GAN_criterion(e_T, T_eT)
        return GAN_loss_D
    
    # initialize the network (normal)
    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
        print('initialize network with %s' % init_type)
        net.apply(init_func)

    # initialize the network
    def init_net(self, net, init_type='normal', init_gain=0.02):
        self.init_weights(net, init_type, gain=init_gain)
        return net


class DSCM_with_dataset(Basemodel):
    def __init__(
            self, 
            opt, 
            model_name_G, 
            model_name_D, 
            in_channels, 
            num_classes, 
            device, 
            loss_factor_list, 
            initialize=False, 
            upscale_factor=1, 
            optimizer_name='Adam', 
            lr_G=0.0001, 
            lr_D=0.00001, 
            index_per_D=1, 
            scheduler_name='None'
    ):
        super(DSCM_with_dataset, self).__init__()
        self.opt = opt
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = device
        self.loss_factor_list = loss_factor_list
        self.upscale_factor = upscale_factor        
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        #generate model
        print(f"Model name: {model_name_G}")
        self.net_G = self.generate_G(model_name=model_name_G, in_channels=in_channels, num_classes=num_classes, upscale_factor=upscale_factor).to(device)
        # initialize the model before training 
        if initialize:
            self.net_G = self.init_net(self.net_G)
        # generate and initialize the discriminator when using GAN loss
        if self.loss_factor_list[5] > 0: 
            self.net_D = self.generate_D(model_name=model_name_D, in_channels=num_classes).to(device)
            self.net_D.train()
            if initialize:
                self.net_D = self.init_net(self.net_D)
        # load pretrain models, currently the denoising model is not applied
        if self.opt['net_G']['pretrain_dir'] != "None":
            self.net_G = torch.load(r'{}'.format(opt['net_G']['pretrain_dir']), weights_only=False)
            for param in self.net_G.parameters():
                param.requires_grad = True
            self.net_G.train()
        if self.opt['net_D']['pretrain_dir'] != "None":
            self.net_D = torch.load(r'{}'.format(opt['net_D']['pretrain_dir']), weights_only=False)
            for param in self.net_D.parameters():
                param.requires_grad = True
            self.net_D.train()

        # generate loss criterion
        self.pixel_criterion = nn.MSELoss().to(self.device)            
        self.vgg = VGGFeatureExtractor().to(self.device)
        self.feature_criterion = nn.MSELoss().to(self.device)           
        self.freq_criterion = FFTLoss().to(self.device)
        self.SSIM_criterion = SSIM().to(self.device)
        self.gradient_criterion = nn.MSELoss().to(self.device)        
        self.get_grad = Get_grad_std(device=self.device, num_classes=1, kernel_size=3, blur_kernel_size=7, blur_kernel_std=3)
        self.corr_criterion = nn.MSELoss().to(self.device)#Pearson_loss()
        self.pearson_criterion = Pearson_loss().to(self.device)
        self.GAN_criterion = GANLoss('gan', 1.0, 0.0).to(self.device)
        self.degen_criterion = nn.MSELoss().to(self.device)
        # generate optimizer
        if self.optimizer_name == 'Adam':
            self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(0.9, 0.999))
            if self.loss_factor_list[5] > 0: 
                self.optim_D = torch.optim.Adam(self.net_D.parameters(), lr=lr_D)
        self.index_per_D = index_per_D

        # generate scheduler
        if self.scheduler_name == 'OneCycleLR':
            self.scheduler_G = OneCycleLR(self.optim_G, max_lr=lr_G, 
                total_steps=(self.opt['train']['epoches']), pct_start=0.1)        
            if self.loss_factor_list[5] > 0: 
                self.scheduler_D = OneCycleLR(self.optim_D, max_lr=lr_D, total_steps=(self.opt['train']['epoches']), pct_start=0.1)   
        elif self.scheduler_name == "CosineAnnealingLR":
            self.scheduler_G = CosineAnnealingLR(self.optim_G, T_max=self.opt['train']['epoches'], eta_min=self.opt['train']['lr_G']/100)
            if self.loss_factor_list[5] > 0: 
                self.scheduler_D = CosineAnnealingLR(self.optim_D, T_max=(self.opt['train']['epoches']), eta_min=self.opt['train']['lr_D']/100)

    def feed_data(self, Input, GT, epoch=0):
        self.Input = Input.to(self.device)
        self.GT_main = GT.to(self.device)
        self.fake_main = self.net_G(self.Input).to(self.device)
        return self.fake_main
    
    def calculate_loss(self, batch_index=0, stage="train", mask=None):
        pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, gen_adv_loss = [0, 0, 0, 0, 0, 0]
        # MSE for the pixel loss
        pixel_loss = self.pixel_criterion(self.fake_main, self.GT_main)
        # Calculate PCC while validation
        pearson_coef = 0 if stage == "train" else self.pearson_criterion(self.fake_main, self.GT_main)
        # VGG-19 based feature loss
        if self.loss_factor_list[1] > 0: 
            for i in range(self.fake_main.size()[1]):
                fea_fake = self.vgg(self.fake_main[:,i:i+1,:,:])
                fea_GT = self.vgg(self.GT_main[:,i:i+1,:,:])              
                feature_loss += self.feature_criterion(fea_fake, fea_GT)
        # SSIM loss
        if self.loss_factor_list[2] > 0: 
            for i in range(self.fake_main.size()[1]):
                temp_fake = self.fake_main[:,i:i+1,:,:].detach()
                temp_GT = self.GT_main[:,i:i+1,:,:].detach()
                temp_fake = temp_fake / torch.max(temp_fake)
                temp_GT = temp_GT / torch.max(temp_GT)
                SSIM_value = self.SSIM_criterion(temp_fake, temp_GT)
                value_one = torch.ones_like(SSIM_value)
                SSIM_loss = SSIM_loss + (value_one - SSIM_value)
        # Gradient loss, abondoned in current version
        if self.loss_factor_list[3] > 0:
            grad_fake = self.get_grad(self.fake_main[:,0:1,:,:])
            grad_real = self.get_grad(self.GT_main[:,0:1,:,:])
            grad_loss = self.gradient_criterion(grad_fake, grad_real)
        # Correlation loss, abondoned in current version
        if self.loss_factor_list[4] > 0:
            masked_GT = self.GT_main * mask
            masked_fake = self.fake_main * mask
            corr_loss = self.corr_criterion(masked_fake, masked_GT)
        # GAN loss
        if self.loss_factor_list[5] > 0:
            if self.loss_factor_list[4] > 0:
                mask = mask.clone()
                mask[mask == 0] = 1
                self.fake_main = self.fake_main * mask
                self.GT_main = self.GT_main * mask
            else:
                pass
            
            gen_adv_loss = self.cal_GAN_loss_G(fake=self.fake_main)
            dis_adv_loss = self.cal_GAN_loss_D(batch_index=batch_index, fake=self.fake_main, GT=self.GT_main)

            self.loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, gen_adv_loss, dis_adv_loss]
        else:
            self.loss_list = [pixel_loss, feature_loss, SSIM_loss, grad_loss, corr_loss, 0, 0]
        if stage == "train":
            return self.loss_list
        elif stage == "validation":
            return self.loss_list, pearson_coef

    # update the network after calculating loss
    def update_net(self, loss_list):
        self.loss_list = loss_list
        self.total_loss_G = 0
        self.total_loss_D = 0
        for i in range(len(self.loss_factor_list)):
            self.total_loss_G += self.loss_factor_list[i] * self.loss_list[i]
        self.optim_G.zero_grad()
        self.total_loss_G.backward()
        self.optim_G.step()
        if self.loss_factor_list[5] > 0:
            self.total_loss_D = self.loss_list[-1]
            self.optim_D.zero_grad()
            self.total_loss_D.backward()
            self.optim_D.step()

    def update_scheduler(self):
        if self.scheduler_name != None:
            self.scheduler_G.step()
            if self.loss_factor_list[5] > 0:
                self.scheduler_D.step()     
                   
    def validation(self, mask=None, save_image=False):        
        self.total_loss_G, pearson_coef = self.calculate_loss(stage="validation", mask=mask)
        if save_image:
            image_list = [self.Input, self.fake_main, self.GT_main]
            return self.total_loss_G, image_list, pearson_coef
        else:
            return self.total_loss_G, pearson_coef
        