import numpy as np
from skimage.filters import threshold_otsu

import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import threshold_otsu

from options.options import parse
from dataset.read_prepared_data import *

cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from loss.pearson_loss import Pearson_loss
from loss.SSIM_loss import SSIM
from loss.NRMAE import nrmae
from loss.PSNR import calculate_psnr
from loss.cal_those_overlaps import *

from net.make_model import *
from dataset.gen_datasets import *


# options of .yml format in "options" folder
opt_path = 'options/Synthetic_eval_DESM_IO.yml'

# read options
opt = parse(opt_path=os.path.join(cwd, opt_path))

# set rank of GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_rank']

import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('-----------------------------Using GPU-----------------------------')
else:
    print('-----------------------------Using CPU-----------------------------')


def safe_nanmean(x):
    if len(x) == 0:
        return np.nan
    return float(np.nanmean(x))


def build_region_masks_from_gt(GT_DS, device):
    """
    从 GT_DS 构造四种 mask:
      1) full_mask          : 整张图
      2) foreground_mask    : 所有通道前景并集
      3) overlap_mask       : 至少两个通道同时存在的区域
      4) non_overlap_mask   : 前景中的非重叠区域
    """
    overlap_count = torch.zeros_like(GT_DS[0, 0], dtype=torch.uint8, device=device)
    foreground_union = torch.zeros_like(GT_DS[0, 0], dtype=torch.uint8, device=device)

    for c in range(GT_DS.size(1)):
        temp_GT = GT_DS[:, c:c+1, :, :].detach()                 # [1,1,H,W]
        temp_GT_np = to_cpu(temp_GT.squeeze(0).squeeze(0))       # [H,W]

        org_thresh = threshold_otsu(temp_GT_np)
        temp_mask_np = (temp_GT_np >= org_thresh).astype(np.uint8)
        temp_mask_torch = torch.from_numpy(temp_mask_np).to(device=device, dtype=torch.uint8)

        overlap_count += temp_mask_torch
        foreground_union = torch.maximum(foreground_union, temp_mask_torch)

    overlap_mask = (overlap_count >= 2).to(torch.uint8)

    overlap_mask_np = to_cpu(overlap_mask).astype(np.uint8)
    foreground_mask_np = to_cpu(foreground_union).astype(np.uint8)
    non_overlap_mask_np = ((foreground_mask_np > 0) & (overlap_mask_np == 0)).astype(np.uint8)
    full_mask_np = np.ones_like(overlap_mask_np, dtype=np.uint8)

    return full_mask_np, foreground_mask_np, overlap_mask_np, non_overlap_mask_np


def compute_region_metrics(fake_main, GT_DS, mask_np, SSIM_criterion):
    """
    在给定 mask 上，统一计算:
      - NRMAE
      - SSIM
      - PCC
      - PSNR

    fake_main, GT_DS: torch.Tensor, [1,C,H,W]
    mask_np: np.ndarray, [H,W]
    """
    fake_np = to_cpu(fake_main.squeeze(0).permute(1, 2, 0))   # [H,W,C]
    gt_np = to_cpu(GT_DS.squeeze(0).permute(1, 2, 0))         # [H,W,C]

    # ---------- NRMAE ----------
    nrmae_vals = []
    for c in range(fake_np.shape[-1]):
        val = masked_nrmae(
            pred=fake_np[:, :, c],
            gt=gt_np[:, :, c],
            mask=mask_np,
            norm_mode="range"
        )
        if not np.isnan(val):
            nrmae_vals.append(val)
    nrmae_value = safe_nanmean(nrmae_vals)

    # ---------- SSIM ----------
    ssim_value = multichannel_overlap_ssim(
        fake_main=fake_main,
        GT_DS=GT_DS,
        overlap_mask=mask_np,
        ssim_criterion=SSIM_criterion,
        min_area=20,
        connectivity=8
    )

    # ---------- PCC ----------
    pcc_vals = []
    for c in range(fake_np.shape[-1]):
        val = masked_pearson_numpy(
            pred=fake_np[:, :, c],
            gt=gt_np[:, :, c],
            mask=mask_np
        )
        if not np.isnan(val):
            pcc_vals.append(val)
    pcc_value = safe_nanmean(pcc_vals)

    # ---------- PSNR ----------
    psnr_vals = []
    for c in range(fake_np.shape[-1]):
        val = masked_psnr(
            pred=fake_np[:, :, c],
            gt=gt_np[:, :, c],
            mask=mask_np,
            data_range=None
        )
        if not np.isnan(val):
            psnr_vals.append(val)
    psnr_value = safe_nanmean(psnr_vals)

    return {
        "NRMAE": nrmae_value,
        "SSIM": ssim_value,
        "PCC": pcc_value,
        "PSNR": psnr_value
    }


def evaluate_IO_synthetic(opt):
    toolbox = ToolBox(opt=opt)
    net_main = DSCM_with_dataset(
        opt,
        in_channels=1,
        num_classes=len(opt['category']),
        model_name_G=opt['net_G']['model_decouple_name'],
        model_name_D=opt['net_D']['model_name'],
        initialize=opt['net_G']['initialize'],
        scheduler_name=opt['train']['scheduler'],
        device=device,
        loss_factor_list=opt['net_G']['weight_decouple'],
        lr_G=opt['train']['lr_G'],
        lr_D=opt['train']['lr_D']
    )
    net_main.net_G = torch.load(opt['net_G']['pretrain_dir'], weights_only=False)

    # "old" = read data pairs, "new" = generate pseudo data pairs
    if opt['read_version'] == "real-time":
        val_loader = gen_degradation_dataloader(
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
            eval_flag=True
        )
    elif opt['read_version'] == "prepared":
        combination_name = "_".join(opt['category']) + f"_{opt['degradation_resolution']}" + f"_{opt['noise_level']}" + f"_{opt['average']}"
        read_dir_train = os.path.join(r'data\prepared_data\train', combination_name)
        read_dir_val = os.path.join(r'data\prepared_data\val', combination_name)

        train_loader, val_loader = gen_prepared_dataloader(
            read_dir_train=read_dir_train,
            read_dir_val=read_dir_val,
            num_file_train=opt['num_file_train'],
            num_file_val=opt['num_file_val'],
            num_org=len(opt['category']),
            org_list=opt['category'],
            size=opt['size'],
            batch_size=opt['train']['batch_size'],
            device=opt['device'],
            num_workers=opt['num_workers']
        )

    Input_list = []
    GT_list = []
    GT_D_list = []
    sta_list = []

    # full-image metrics
    nrmae_full_list = []
    ssim_full_list = []
    pcc_full_list = []
    psnr_full_list = []

    # foreground-only metrics
    nrmae_foreground_list = []
    ssim_foreground_list = []
    pcc_foreground_list = []
    psnr_foreground_list = []

    # overlap metrics
    nrmae_overlap_list = []
    ssim_overlap_list = []
    pcc_overlap_list = []
    psnr_overlap_list = []

    # non-overlap metrics
    nrmae_nonoverlap_list = []
    ssim_nonoverlap_list = []
    pcc_nonoverlap_list = []
    psnr_nonoverlap_list = []

    SSIM_criterion = SSIM().to(device)

    num_val_image = opt['num_file_val']
    print('======================== evaluating ========================')
    bar = tqdm(total=num_val_image)

    with torch.no_grad():
        for batch_index, data in enumerate(val_loader):
            Input, GT_DS, GT_D, _, _, sta = data

            fake_main = net_main.feed_data(Input=Input, GT=GT_DS)
            _ = net_main.validation(mask=None)

            Input_list.append(Input)
            GT_list.append([fake_main, GT_DS])
            sta_list.append(sta)
            GT_D_list.append(to_cpu((GT_D * sta["Input_std"]).squeeze(0).permute(1, 2, 0)))

            # build masks once
            full_mask_np, foreground_mask_np, overlap_mask_np, non_overlap_mask_np = build_region_masks_from_gt(
                GT_DS=GT_DS,
                device=device
            )


            # full-image metrics
            metrics_full = compute_region_metrics(
                fake_main=fake_main,
                GT_DS=GT_DS,
                mask_np=full_mask_np,
                SSIM_criterion=SSIM_criterion
            )
            nrmae_full_list.append(metrics_full["NRMAE"])
            ssim_full_list.append(metrics_full["SSIM"])
            pcc_full_list.append(metrics_full["PCC"])
            psnr_full_list.append(metrics_full["PSNR"])

            # foreground-only metrics
            metrics_foreground = compute_region_metrics(
                fake_main=fake_main,
                GT_DS=GT_DS,
                mask_np=foreground_mask_np,
                SSIM_criterion=SSIM_criterion
            )
            nrmae_foreground_list.append(metrics_foreground["NRMAE"])
            ssim_foreground_list.append(metrics_foreground["SSIM"])
            pcc_foreground_list.append(metrics_foreground["PCC"])
            psnr_foreground_list.append(metrics_foreground["PSNR"])

            # overlap metrics
            metrics_overlap = compute_region_metrics(
                fake_main=fake_main,
                GT_DS=GT_DS,
                mask_np=overlap_mask_np,
                SSIM_criterion=SSIM_criterion
            )
            nrmae_overlap_list.append(metrics_overlap["NRMAE"])
            ssim_overlap_list.append(metrics_overlap["SSIM"])
            pcc_overlap_list.append(metrics_overlap["PCC"])
            psnr_overlap_list.append(metrics_overlap["PSNR"])

            # non-overlap metrics
            metrics_nonoverlap = compute_region_metrics(
                fake_main=fake_main,
                GT_DS=GT_DS,
                mask_np=non_overlap_mask_np,
                SSIM_criterion=SSIM_criterion
            )
            nrmae_nonoverlap_list.append(metrics_nonoverlap["NRMAE"])
            ssim_nonoverlap_list.append(metrics_nonoverlap["SSIM"])
            pcc_nonoverlap_list.append(metrics_nonoverlap["PCC"])
            psnr_nonoverlap_list.append(metrics_nonoverlap["PSNR"])

            bar.update(1)

    # save val stacks in the list
    toolbox.gen_validation_images_IO(data_list=[Input_list, GT_list, sta_list])

    # save the evaluation results of synthetic data
    save_dir = opt['save_eval_dir']
    if save_dir:
        check_existence(save_dir)
        save_list = toolbox.val_list
        size = opt['size']

        for index in range(len(save_list)):
            temp_img = save_list[index]
            Input_img = temp_img[:size, :size]

            for org_index in range(len(opt['category'])):
                GT_img = temp_img[:size, (org_index + 1) * size:(org_index + 2) * size]
                pred_img = temp_img[size:2 * size, (org_index + 1) * size:(org_index + 2) * size]

                tifffile.imwrite(os.path.join(save_dir, f"{index}_GT_{org_index}.tif"), np.uint8(GT_img))
                tifffile.imwrite(os.path.join(save_dir, f"{index}_pred_{org_index}.tif"), np.uint8(pred_img))

                temp_GT_D = GT_D_list[index][:, :, org_index]
                tifffile.imwrite(os.path.join(save_dir, f"{index}_GT_D_{org_index}.tif"), np.uint16(temp_GT_D))

            tifffile.imwrite(os.path.join(save_dir, f"{index}_Input.tif"), Input_img)

        df = pd.DataFrame({
            "NRMAE_full": nrmae_full_list,
            "SSIM_full": ssim_full_list,
            "PCC_full": pcc_full_list,
            "PSNR_full": psnr_full_list,

            "NRMAE_foreground": nrmae_foreground_list,
            "SSIM_foreground": ssim_foreground_list,
            "PCC_foreground": pcc_foreground_list,
            "PSNR_foreground": psnr_foreground_list,

            "NRMAE_overlap": nrmae_overlap_list,
            "SSIM_overlap": ssim_overlap_list,
            "PCC_overlap": pcc_overlap_list,
            "PSNR_overlap": psnr_overlap_list,

            "NRMAE_nonoverlap": nrmae_nonoverlap_list,
            "SSIM_nonoverlap": ssim_nonoverlap_list,
            "PCC_nonoverlap": pcc_nonoverlap_list,
            "PSNR_nonoverlap": psnr_nonoverlap_list
        })
        df.insert(0, "Index", np.arange(1, len(df) + 1))

        excel_path = os.path.join(save_dir, "metrics.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"已保存到: {excel_path}")

    bar.close()

    print(f"Average NRMAE (full image): {np.nanmean(nrmae_full_list):.4f}")
    print(f"Average SSIM (full image): {np.nanmean(ssim_full_list):.4f}")
    print(f"Average PCC (full image): {np.nanmean(pcc_full_list):.4f}")
    print(f"Average PSNR (full image): {np.nanmean(psnr_full_list):.4f}")

    print(f"Average NRMAE (foreground): {np.nanmean(nrmae_foreground_list):.4f}")
    print(f"Average SSIM (foreground): {np.nanmean(ssim_foreground_list):.4f}")
    print(f"Average PCC (foreground): {np.nanmean(pcc_foreground_list):.4f}")
    print(f"Average PSNR (foreground): {np.nanmean(psnr_foreground_list):.4f}")

    print(f"Average NRMAE (overlap): {np.nanmean(nrmae_overlap_list):.4f}")
    print(f"Average SSIM (overlap): {np.nanmean(ssim_overlap_list):.4f}")
    print(f"Average PCC (overlap): {np.nanmean(pcc_overlap_list):.4f}")
    print(f"Average PSNR (overlap): {np.nanmean(psnr_overlap_list):.4f}")

    print(f"Average NRMAE (non-overlap): {np.nanmean(nrmae_nonoverlap_list):.4f}")
    print(f"Average SSIM (non-overlap): {np.nanmean(ssim_nonoverlap_list):.4f}")
    print(f"Average PCC (non-overlap): {np.nanmean(pcc_nonoverlap_list):.4f}")
    print(f"Average PSNR (non-overlap): {np.nanmean(psnr_nonoverlap_list):.4f}")

    return {
        "NRMAE_full": nrmae_full_list,
        "SSIM_full": ssim_full_list,
        "PCC_full": pcc_full_list,
        "PSNR_full": psnr_full_list,

        "NRMAE_foreground": nrmae_foreground_list,
        "SSIM_foreground": ssim_foreground_list,
        "PCC_foreground": pcc_foreground_list,
        "PSNR_foreground": psnr_foreground_list,

        "NRMAE_overlap": nrmae_overlap_list,
        "SSIM_overlap": ssim_overlap_list,
        "PCC_overlap": pcc_overlap_list,
        "PSNR_overlap": psnr_overlap_list,

        "NRMAE_nonoverlap": nrmae_nonoverlap_list,
        "SSIM_nonoverlap": ssim_nonoverlap_list,
        "PCC_nonoverlap": pcc_nonoverlap_list,
        "PSNR_nonoverlap": psnr_nonoverlap_list
    }


if __name__ == "__main__":
    opt_path = 'options/Synthetic_eval_DESM_IO.yml'
    opt = parse(opt_path=os.path.join(cwd, opt_path))
    evaluate_IO_synthetic(opt=opt)