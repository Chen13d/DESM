import os
cwd = os.getcwd()
from tqdm import tqdm
from options.options import parse

from dataset.read_prepared_data import *
from dataset.gen_datasets import *
from net.make_model import *

#options of .yml format in "options" folder
opt_path = 'options/train_DESM_with_dataset.yml'

# read options
opt = parse(opt_path=os.path.join(cwd, opt_path))

import torch
if torch.cuda.is_available():
    # set rank of GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_rank']
    device = torch.device("cuda:0")
    print('-----------------------------Using GPU-----------------------------')
else:
    device = "cpu"
    print('-----------------------------Using CPU-----------------------------')


def train(opt):
    toolbox = ToolBox(opt=opt)    
    net_main = DSCM_with_dataset(opt, in_channels=1, num_classes=len(opt['category']), model_name_G=opt['net_G']['model_decouple_name'], 
                          model_name_D=opt['net_D']['model_name'], initialize=opt['net_G']['initialize'], 
                          scheduler_name=opt['train']['scheduler'], device=device, loss_factor_list=opt['net_G']['weight_decouple'], 
                          lr_G=opt['train']['lr_G'], lr_D=opt['train']['lr_D'])
    if opt['net_G']['weight_decouple'][5] > 0:
        net_main.net_D.train()
    # "old" = read data pairs, "new" = generate pseudo data pairs
    if opt['read_version'] == "real-time":
        train_loader, val_loader = gen_degradation_dataloader(
            GT_tag_list=opt['category'], 
            noise_level=opt['noise_level'], 
            factor_list=opt['factor_list'], 
            SR_resolution_dict=opt['raw_data_resolution'], 
            target_resolution=opt['degradation_resolution'], 
            generate_lifetime_flag=opt['lifetime_flag'], 
            degradation_method=opt['degradation_method'], 
            average=opt['average'], 
            read_LR_flag=opt['read_LR_flag'], 
            num_file_train=opt['num_file_train'], 
            num_file_val=opt['num_file_val'], 
            size=opt['size'], 
            num_workers=opt['num_workers'], 
            cwd=cwd, 
            device=device, 
            on_the_fly_flag=True, 
            batch_size=opt['train']['batch_size']
        )
    elif opt['read_version'] == "prepared":
        combination_name = "_".join(opt['category']) + f"_{opt['degradation_resolution']}" + f"_{opt['noise_level']}" + f"_{opt['average']}"
        if opt['lifetime_flag']: combination_name += "_lifetime"
        read_dir_train = os.path.join(r'data\prepared_data\train', combination_name)
        read_dir_val = os.path.join(r'data\prepared_data\val', combination_name)
        if not opt['lifetime_flag']:
            train_loader, val_loader = gen_prepared_dataloader(read_dir_train=read_dir_train, read_dir_val=read_dir_val, num_file_train=opt['num_file_train'], 
                                                        num_file_val=opt['num_file_val'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], 
                                                        batch_size=opt['train']['batch_size'], device=opt['device'], num_workers=opt['num_workers'])
        else:
            train_loader, val_loader = gen_prepared_dataloader(read_dir_train=read_dir_train, read_dir_val=read_dir_val, num_file_train=opt['num_file_train'], 
                                                        num_file_val=opt['num_file_val'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], read_lifetime_flag=True,
                                                        batch_size=opt['train']['batch_size'], device=opt['device'], num_workers=opt['num_workers'])
    
    # generate folders for validation
    toolbox.make_folders()
    # list for training and validation loss
    epoch_list_train = []
    epoch_list_val = []
    cols = 7
    loss_list_train = [[] for _ in range(cols)]
    loss_list_val = [[] for _ in range(cols)]
    lr_list = []
    pearson_list = []
    run_time_list = []
    num_org = len(opt['category'])
    num_train_image = opt['num_file_train']
    for epoch in range(opt['train']['epoches']):
        net_main.net_G.train()
        print('======================== training epoch %d ========================'%(epoch+1))    
        print(f'Epoch {epoch+1}, Learning Rate: {net_main.optim_G.param_groups[0]["lr"]}')
        bar = tqdm(total=num_train_image//opt['train']['batch_size']*opt['train']['num_iter'])
        # list for training loss
        curr_list = [[] for _ in range(cols)]
        for iter in range(opt['train']['num_iter']):
            # enumerate in train Dataloader
            for batch_index, data in enumerate(train_loader):  
                Input, GT_DS, GT_D, GT_S, simulated_lifetime, sta = data
                if opt['lifetime_flag']: 
                    fake_main = net_main.feed_data(Input=torch.concatenate([Input, simulated_lifetime], dim=1), GT=GT_DS)
                else:
                    fake_main = net_main.feed_data(Input=Input, GT=GT_DS)
                loss_main = net_main.calculate_loss(batch_index=batch_index, mask=None, stage="train")
                net_main.update_net(loss_list=loss_main)
                for col in range(cols):
                    curr_list[col].append(loss_main[col].item()) if loss_main[col] != 0 else curr_list[col].append(0)
                bar.set_description_str(
                    f'=== pixel: {np.mean(curr_list[0]):.4f}, fea: {np.mean(curr_list[1]):.4f}, SSIM: {1 - (np.mean(curr_list[2])/num_org):.4f}, grad: {np.mean(curr_list[3]):.4f}, corr: {np.mean(curr_list[4]):.4f}, G: {np.mean(curr_list[5]):.4f}, D: {np.mean(curr_list[6]):.4f}'
                )
                bar.update(1)
        # save for every last image
        #if opt['train']['epoches_per_val']%(epoch+1) == 0: toolbox.save_last_train_image(Input, GT_DS, fake_main, epoch+1)
        # update scheduler
        lr_list.append(round(net_main.optim_G.param_groups[0]["lr"], 10))
        if opt['train']['scheduler'] != "None":
            net_main.update_scheduler()
        # append loss to the train list
        epoch_list_train.append(epoch+1)
        for col in range(cols):
            loss_list_train[col].append(np.mean(curr_list[col]))
        # validate per "epoches_per_test"
        if (epoch+1) % opt['train']['epoches_per_val'] == 0:
            with torch.no_grad():
                Input_list = []
                GT_list = []
                simulated_lifetime_list = []
                sta_list = []
                # list for validation loss
                curr_list = [[] for _ in range(cols)]
                pearson_coef_list = []
                # enumerate in test Dataloader 
                for batch_index, data in enumerate(val_loader):
                    Input, GT_DS, GT_D, GT_S, simulated_lifetime, sta = data
                    if opt['lifetime_flag']: 
                        fake_main = net_main.feed_data(Input=torch.concatenate([Input, simulated_lifetime], dim=1), GT=GT_DS)
                    else:
                        fake_main = net_main.feed_data(Input=Input, GT=GT_DS)
                    loss_main, pearson_coef = net_main.validation(mask=None)
                    # append to list for epoches to save
                    if (epoch+1) % opt['train']['epoches_per_save'] == 0:
                        Input_list.append(Input)
                        GT_list.append([fake_main, GT_DS])
                        simulated_lifetime_list.append(simulated_lifetime)
                        sta_list.append(sta)
                    for col in range(7):
                        curr_list[col].append(loss_main[col].item()) if loss_main[col] != 0 else curr_list[col].append(0)
                    # PCC loss
                    pearson_coef_list.append(pearson_coef.item())
                # append loss to the val list
                epoch_list_val.append(epoch+1)
                for col in range(cols):
                    loss_list_val[col].append(np.mean(curr_list[col]))
                pearson_aver = np.mean(pearson_coef_list)
                pearson_list.append(pearson_aver)
                # save val stack and model
                if (epoch+1) % opt['train']['epoches_per_save'] == 0:
                    if opt['lifetime_flag']:
                        toolbox.gen_validation_images_IL(data_list=[Input_list, GT_list, simulated_lifetime_list, sta_list])
                    else:
                        toolbox.gen_validation_images_IO(data_list=[Input_list, GT_list, sta_list])
                    toolbox.save_val_list(name="main")
                    toolbox.save_model(model=[net_main.net_G], name=["main_G"])
                    if opt['net_G']['weight_decouple'][5] > 0:
                        toolbox.save_model(model=[net_main.net_D], name=["main_D"])
                    pixel_aver = np.mean(curr_list[0])
                    if epoch+1 == opt['train']['epoches_per_save']: 
                        best_score_pixel = pixel_aver
                        best_score_pear = pearson_aver
                    else:
                        if pixel_aver < best_score_pixel:
                            best_score_pixel = pixel_aver
                            toolbox.save_model(model=[net_main.net_G], name=["best_G_pixel"])
                            toolbox.save_val_list(name="best_pixel")
                        if pearson_aver > best_score_pear:
                            best_score_pear = pearson_aver
                            toolbox.save_model(model=[net_main.net_G], name=["best_G_pear"])
                            toolbox.save_val_list(name="best_pear")
                    
        bar.close()
        train_list = [loss_list_train]
        val_list = [loss_list_val]
        model_name_list = ['main']
        loss_name_list = ['pixel', 'fea', 'freq', 'grad', 'corr']
        toolbox.make_loss_plots(epoch_list_train=epoch_list_train, epoch_list_val=epoch_list_val, 
                                train_list=train_list, val_list=val_list, model_name_list=model_name_list, 
                                loss_name_list=loss_name_list, lr_list=lr_list, pearson_list=pearson_list)
    return 0

if __name__ == "__main__":
    train(opt=opt)
