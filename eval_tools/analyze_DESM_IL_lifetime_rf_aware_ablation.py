import os
import sys
import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from skimage.filters import threshold_otsu
from tqdm import tqdm


DESM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if DESM_ROOT not in sys.path:
    sys.path.append(DESM_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from net.make_model import DSCM_with_dataset
from dataset.gen_datasets import gen_degradation_dataloader
from dataset.degradation_model import all_components_cv2
from loss.SSIM_loss import SSIM
from loss.NRMAE import nrmae
from loss.pearson_loss import Pearson_loss


RAW_DATA_RESOLUTION = {
    "Micro": 85.93,
    "Mito": 87.09,
    "Lyso": 91.97,
    "Membrane": 81.24,
    "NPCs": 85.92,
    "Mito_inner": 82.88,
    "Mito_inner_deconv": 81.19,
}


def check_existence_local(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def to_numpy_image(tensor):
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 4:
        array = array[0]
    return array


def lifetime_numpy_to_tensor(lifetime_np, ref_tensor):
    lifetime_tensor = torch.tensor(
        lifetime_np[np.newaxis, np.newaxis, :, :],
        dtype=ref_tensor.dtype,
        device=ref_tensor.device,
    )
    return lifetime_tensor


def mask_numpy_to_tensor(mask_np, ref_tensor):
    mask_tensor = torch.tensor(
        mask_np[np.newaxis, np.newaxis, :, :],
        dtype=ref_tensor.dtype,
        device=ref_tensor.device,
    )
    return mask_tensor


def masked_mean_np(img, mask):
    img_np = np.asarray(img, dtype=np.float32)
    mask_np = np.asarray(mask) > 0
    if not np.any(mask_np):
        return np.nan
    return float(np.mean(img_np[mask_np]))


def mean_np(img):
    img_np = np.asarray(img, dtype=np.float32)
    return float(np.mean(img_np))


def evaluate_prediction_metrics(pred, gt, ssim_criterion, pearson_criterion):
    nrmae_value = float(nrmae(pred, gt).item())
    ssim_values = []
    for channel_index in range(pred.size(1)):
        pred_channel = pred[:, channel_index:channel_index + 1].detach()
        gt_channel = gt[:, channel_index:channel_index + 1].detach()
        pred_channel = pred_channel / max(torch.max(pred_channel).item(), 1e-6)
        gt_channel = gt_channel / max(torch.max(gt_channel).item(), 1e-6)
        ssim_values.append(float(ssim_criterion(pred_channel, gt_channel).item()))
    ssim_value = float(np.mean(ssim_values)) if ssim_values else np.nan
    pcc_value = float(pearson_criterion(pred, gt).item())
    return {
        "nrmae": nrmae_value,
        "ssim": ssim_value,
        "pcc": pcc_value,
    }


def evaluate_prediction_region_nrmae(pred, gt, region_mask, eps=1e-8):
    mask = mask_numpy_to_tensor(region_mask, pred)
    mask = mask.expand(-1, pred.size(1), -1, -1)
    valid = mask > 0
    if not torch.any(valid):
        return np.nan
    masked_abs_error = torch.abs(pred - gt)[valid]
    masked_target = gt[valid]
    mae_value = torch.mean(masked_abs_error)
    target_range = torch.max(masked_target) - torch.min(masked_target)
    return float((mae_value / (target_range + eps)).item())


def collect_metric_values(summary_rows, key):
    values = np.asarray([row[key] for row in summary_rows], dtype=np.float32)
    return values[np.isfinite(values)]


def plot_single_metric_change(
        summary_rows,
        metric_title,
        overlap_key,
        non_overlap_key,
        save_path=None,
        show_plot=True,
        ymin=None,
        ymax=None):
    if not summary_rows:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.2, 3), facecolor="white", constrained_layout=True)
    overlap_color = "#ff4d4f"
    non_overlap_color = "#1f6feb"
    accent_color = "#222222"

    overlap_values = collect_metric_values(summary_rows, overlap_key)
    non_overlap_values = collect_metric_values(summary_rows, non_overlap_key)
    box = ax.boxplot(
        [overlap_values, non_overlap_values],
        patch_artist=True,
        labels=["Overlap\nperturbation", "Non-overlap\nperturbation"],
        widths=0.48,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "s",
            "markerfacecolor": "white",
            "markeredgecolor": "#7a7a7a",
            "markersize": 5,
            "markeredgewidth": 1.3,
        },
        medianprops={"color": "#505050", "linewidth": 1.9},
        whiskerprops={"color": "#6b6b6b", "linewidth": 1.6},
        capprops={"color": "#6b6b6b", "linewidth": 1.6},
        boxprops={"linewidth": 1.8},
    )
    for patch, color in zip(box["boxes"], [overlap_color, non_overlap_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
        patch.set_edgecolor(color)
    ax.axhline(0.0, linestyle="--", color="#909090", linewidth=1.1)
    ax.set_title(metric_title, fontsize=20, pad=10, color=accent_color, fontweight="bold")
    ax.set_ylabel(metric_title, fontsize=20, color=accent_color)
    ax.tick_params(axis="x", labelsize=20, colors=accent_color)
    ax.tick_params(axis="y", labelsize=20, colors=accent_color, width=1.4, length=7)

    if ymin is not None or ymax is not None:
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(
            ymin if ymin is not None else current_ymin,
            ymax if ymax is not None else current_ymax,
        )
    else:
        value_max = max(np.max(overlap_values), np.max(non_overlap_values))
        value_min = min(np.min(overlap_values), np.min(non_overlap_values))
        pad = (value_max - value_min) * 0.16 if value_max > value_min else 0.05
        ax.set_ylim(value_min - pad, value_max + pad * 0.30)

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.6)
    ax.grid(axis="y", color="#e6e6e6", linewidth=1.0)
    ax.grid(axis="x", visible=False)
    if save_path:
        check_existence_local(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def save_sample_outputs(
        save_dir,
        sample_name,
        Input,
        lifetime,
        pred_baseline,
        gt,
        wrong_class_overlap_lifetime=None,
        wrong_class_non_overlap_lifetime=None,
        overlap_mask=None,
        non_overlap_mask=None,
        pred_wrong_class_overlap=None,
        pred_wrong_class_non_overlap=None,
        delta_overlap=None,
        delta_non_overlap=None):
    sample_dir = os.path.join(save_dir, "samples", sample_name)
    check_existence_local(sample_dir)
    remove_if_exists(os.path.join(sample_dir, "prediction.tif"))
    tifffile.imwrite(os.path.join(sample_dir, "input.tif"), to_numpy_image(Input).astype(np.float32))
    tifffile.imwrite(os.path.join(sample_dir, "lifetime.tif"), to_numpy_image(lifetime).astype(np.float32))
    if wrong_class_overlap_lifetime is not None:
        tifffile.imwrite(
            os.path.join(sample_dir, "wrong_class_overlap_lifetime.tif"),
            np.asarray(wrong_class_overlap_lifetime, dtype=np.float32),
        )
    if wrong_class_non_overlap_lifetime is not None:
        tifffile.imwrite(
            os.path.join(sample_dir, "wrong_class_non_overlap_lifetime.tif"),
            np.asarray(wrong_class_non_overlap_lifetime, dtype=np.float32),
        )
    if overlap_mask is not None:
        tifffile.imwrite(
            os.path.join(sample_dir, "overlap_mask.tif"),
            np.asarray(overlap_mask, dtype=np.uint8),
        )
    if non_overlap_mask is not None:
        tifffile.imwrite(
            os.path.join(sample_dir, "non_overlap_mask.tif"),
            np.asarray(non_overlap_mask, dtype=np.uint8),
        )
    tifffile.imwrite(
        os.path.join(sample_dir, "prediction_baseline.tif"),
        to_numpy_image(pred_baseline).astype(np.float32),
    )
    if pred_wrong_class_overlap is not None:
        tifffile.imwrite(
            os.path.join(sample_dir, "prediction_wrong_class_overlap.tif"),
            to_numpy_image(pred_wrong_class_overlap).astype(np.float32),
        )
    if pred_wrong_class_non_overlap is not None:
        tifffile.imwrite(
            os.path.join(sample_dir, "prediction_wrong_class_non_overlap.tif"),
            to_numpy_image(pred_wrong_class_non_overlap).astype(np.float32),
        )
    if delta_overlap is not None:
        tifffile.imwrite(
            os.path.join(sample_dir, "delta_overlap.tif"),
            np.asarray(delta_overlap, dtype=np.float32),
        )
    if delta_non_overlap is not None:
        tifffile.imwrite(
            os.path.join(sample_dir, "delta_non_overlap.tif"),
            np.asarray(delta_non_overlap, dtype=np.float32),
        )
    tifffile.imwrite(os.path.join(sample_dir, "gt.tif"), to_numpy_image(gt).astype(np.float32))


def make_threshold_local(Input):
    thresh = threshold_otsu(Input)
    Output = copy.deepcopy(Input)
    Output[Output < thresh] = 0
    Output[Output >= thresh] = 1
    kernel_size = (5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    Output = cv2.dilate(
        Output,
        kernel,
        anchor=None,
        iterations=3,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return Output


def find_overlap_background_local(thresholded_list):
    masks = np.stack(thresholded_list, axis=0)
    mask = masks.sum(axis=0)
    overlap = mask >= 2
    background = mask == 0
    return overlap, background


def fill_overlapped_components_from_neighbors_local(
        simulated_lifetime,
        overlapped,
        cc_labels,
        comp_value,
        background_value,
        connectivity,
        max_radius,
        seed,
        fallback_global,
        overlap_mode="baseline",
        baseline_choice_by_tid=None):
    sl = np.asarray(simulated_lifetime).copy()
    ov = (np.asarray(overlapped) != 0)

    height, width = sl.shape
    if max_radius is None:
        max_radius = max(height, width)

    rng = np.random.default_rng(seed)
    target_ids = np.unique(cc_labels[ov])
    donor_pixel_mask = (~ov) & (sl != background_value)
    donor_labels = np.where(donor_pixel_mask, cc_labels, 0)

    if connectivity == 4:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    elif connectivity == 8:
        kernel = np.ones((3, 3), dtype=np.uint8)
    else:
        raise ValueError("connectivity must be 4 or 8")

    global_donor_ids = None
    if fallback_global:
        global_donor_ids = np.unique(donor_labels)
        global_donor_ids = global_donor_ids[global_donor_ids != 0]

    chosen_value_by_tid = {}
    for tid in target_ids:
        if tid == 0:
            continue

        target_value = comp_value[tid]
        comp_mask = (cc_labels == tid) & ov
        prev = comp_mask.copy().astype(np.uint8)
        cur = prev.copy()
        neighbor_ids = np.array([], dtype=np.int32)

        for _ in range(max_radius):
            cur = cv2.dilate(cur, kernel, iterations=1)
            ring = (cur.astype(bool) & (~prev.astype(bool)))
            prev = cur

            neighbor_ids = np.unique(donor_labels[ring])
            neighbor_ids = neighbor_ids[neighbor_ids != 0]
            if neighbor_ids.size > 0:
                neighbor_ids = neighbor_ids[comp_value[neighbor_ids] != target_value]
            if neighbor_ids.size > 0:
                break

        if neighbor_ids.size == 0 and fallback_global:
            neighbor_ids = global_donor_ids[comp_value[global_donor_ids] != target_value]

        if neighbor_ids.size == 0:
            continue

        candidate_values = np.unique(comp_value[neighbor_ids].astype(np.float32))
        if overlap_mode == "wrong-class-overlap" and baseline_choice_by_tid is not None:
            baseline_value = baseline_choice_by_tid.get(int(tid), None)
            alt_values = candidate_values
            if baseline_value is not None:
                alt_values = alt_values[np.abs(alt_values - baseline_value) > 1e-8]
            if alt_values.size == 0 and global_donor_ids is not None:
                global_candidate_values = np.unique(
                    comp_value[global_donor_ids][comp_value[global_donor_ids] != target_value].astype(np.float32)
                )
                if baseline_value is not None:
                    alt_values = global_candidate_values[np.abs(global_candidate_values - baseline_value) > 1e-8]
                else:
                    alt_values = global_candidate_values
            chosen_value = float(rng.choice(alt_values if alt_values.size > 0 else candidate_values))
        else:
            chosen_donor_id = int(rng.choice(neighbor_ids))
            chosen_value = float(comp_value[chosen_donor_id])

        sl[comp_mask] = chosen_value
        chosen_value_by_tid[int(tid)] = chosen_value

    return sl, chosen_value_by_tid


def make_lifetime_distribution_local(num_org, thresholded_list, size, seed=123):
    fac_list = [1, 2, 3, 4, 5, 6, 7, 8, 9][:num_org]
    simulated_lifetime = np.zeros((size, size), dtype=np.float32)
    overlapped, _ = find_overlap_background_local(thresholded_list)

    for index_thresh in range(len(thresholded_list)):
        temp_ones = np.ones((size, size), dtype=np.float32)
        organelle_mask = np.asarray(thresholded_list[index_thresh]) != 0
        temp_ones[~organelle_mask] = 0
        temp_ones *= fac_list[index_thresh]
        simulated_lifetime += temp_ones

    baseline_cc_labels, baseline_comp_value, _ = all_components_cv2(simulated_lifetime, connectivity=8)
    baseline_filled, baseline_choice_by_tid = fill_overlapped_components_from_neighbors_local(
        simulated_lifetime,
        overlapped,
        baseline_cc_labels,
        baseline_comp_value,
        background_value=0,
        connectivity=8,
        max_radius=None,
        seed=seed,
        fallback_global=True,
        overlap_mode="baseline",
    )
    non_overlap = ((simulated_lifetime > 0) & (~overlapped)).astype(np.uint8)
    return (
        np.asarray(baseline_filled, dtype=np.float32).copy(),
        simulated_lifetime,
        overlapped.astype(np.uint8),
        non_overlap,
        baseline_cc_labels,
        baseline_comp_value,
        baseline_choice_by_tid,
    )


def resolve_device(gpu_rank="0", device_name="cuda:0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_rank)
    if torch.cuda.is_available():
        print("-----------------------------Using GPU-----------------------------")
        return torch.device(device_name)
    print("-----------------------------Using CPU-----------------------------")
    return torch.device("cpu")


def make_runtime_opt(model_dir, size, num_workers, gpu_rank, device_name, num_file_eval, category, factor_list):
    return {
        "gpu_rank": str(gpu_rank),
        "device": device_name,
        "size": int(size),
        "num_workers": int(num_workers),
        "num_file_train": int(num_file_eval),
        "num_file_val": int(num_file_eval),
        "category": list(category),
        "factor_list": list(factor_list),
        "raw_data_resolution": {name: RAW_DATA_RESOLUTION[name] for name in category},
        "degradation_method": "blur-composite-noise",
        "lifetime_flag": True,
        "read_LR_flag": False,
        "net_G": {
            "initialize": True,
            "model_decouple_name": "Unet_FLIM_att",
            "weight_decouple": [0, 0, 0, 0, 0, 0],
            "pretrain_dir": model_dir,
        },
        "net_D": {
            "model_name": "UnetD",
            "pretrain_dir": "None",
        },
        "train": {
            "scheduler": "CosineAnnealingLR",
            "lr_G": 0.00001,
            "lr_D": 0.0000001,
            "epoches": 1000,
        },
    }


def build_model(opt, device):
    if not os.path.exists(opt["net_G"]["pretrain_dir"]):
        raise FileNotFoundError(f"Model file not found: {opt['net_G']['pretrain_dir']}")
    net_main = DSCM_with_dataset(
        opt,
        in_channels=1,
        num_classes=len(opt["category"]),
        model_name_G=opt["net_G"]["model_decouple_name"],
        model_name_D=opt["net_D"]["model_name"],
        initialize=opt["net_G"]["initialize"],
        scheduler_name=opt["train"]["scheduler"],
        device=device,
        loss_factor_list=opt["net_G"]["weight_decouple"],
        lr_G=opt["train"]["lr_G"],
        lr_D=opt["train"]["lr_D"],
    )
    net_main.net_G.eval()
    return net_main


def build_wrong_class_overlap_lifetime(GT_DS, size, factor_list, device):
    gt_np = GT_DS[0].detach().cpu().float().numpy()

    thresholded_list = []
    for channel_index in range(gt_np.shape[0]):
        thresholded_list.append(make_threshold_local(gt_np[channel_index]))

    _, simulated_lifetime, overlapped, non_overlap, cc_labels, comp_value, baseline_choice_by_tid = make_lifetime_distribution_local(
        num_org=gt_np.shape[0],
        thresholded_list=thresholded_list,
        size=size,
    )
    lifetime_np, _ = fill_overlapped_components_from_neighbors_local(
        simulated_lifetime=simulated_lifetime,
        overlapped=overlapped,
        cc_labels=cc_labels,
        comp_value=comp_value,
        background_value=0,
        connectivity=8,
        max_radius=None,
        seed=123,
        fallback_global=True,
        overlap_mode="wrong-class-overlap",
        baseline_choice_by_tid=baseline_choice_by_tid,
    )
    lifetime_np = np.asarray(lifetime_np, dtype=np.float32)

    lifetime_std = float(np.std(lifetime_np))
    if lifetime_std > 0:
        lifetime_np = lifetime_np / lifetime_std
    return lifetime_np, overlapped, non_overlap


def build_wrong_class_non_overlap_lifetime(GT_DS, size, factor_list, device):
    gt_np = GT_DS[0].detach().cpu().float().numpy()

    thresholded_list = []
    for channel_index in range(gt_np.shape[0]):
        thresholded_list.append(make_threshold_local(gt_np[channel_index]))

    baseline_lifetime, _, overlapped, non_overlap, _, _, _ = make_lifetime_distribution_local(
        num_org=gt_np.shape[0],
        thresholded_list=thresholded_list,
        size=size,
    )

    baseline_lifetime_np = np.asarray(baseline_lifetime, dtype=np.float32).copy()
    lifetime_np = baseline_lifetime_np.copy()
    class_values = np.unique(baseline_lifetime_np[baseline_lifetime_np > 0]).astype(np.float32)
    rng = np.random.default_rng(123)
    donor_map = {}

    for class_value in class_values:
        candidate_values = class_values[np.abs(class_values - class_value) > 1e-8]
        if candidate_values.size == 0:
            continue
        donor_map[float(class_value)] = float(rng.choice(candidate_values))

    for class_value in class_values:
        if float(class_value) not in donor_map:
            continue
        class_mask = (np.abs(baseline_lifetime_np - class_value) < 1e-8) & (non_overlap > 0)
        if np.any(class_mask):
            lifetime_np[class_mask] = donor_map[float(class_value)]

    lifetime_std = float(np.std(lifetime_np))
    if lifetime_std > 0:
        lifetime_np = lifetime_np / lifetime_std
    return lifetime_np, overlapped, non_overlap


def run_synthetic_baseline(
        model_dir,
        size,
        num_workers,
        gpu_rank="0",
        device_name="cuda:0",
        degradation_resolution=280,
        noise_level=0.280,
        average=1,
        num_file_eval=10,
        category=["Mito", "Lyso"],
        factor_list=[1, 1],
        show_sample_plots=True,
        show_summary_plot=True,
        save_summary_plot=True,
        save_dir=None):
    if save_dir is None:
        group_name = "_".join(category)
        save_dir = os.path.join(
            SCRIPT_DIR,
            "DESM_IL_lifetime_rf_aware_ablation",
            f"synthetic_{group_name}",
        )
    else:
        group_name = "_".join(category)
    check_existence_local(save_dir)

    device = resolve_device(gpu_rank=gpu_rank, device_name=device_name)
    opt = make_runtime_opt(
        model_dir=model_dir,
        size=size,
        num_workers=num_workers,
        gpu_rank=gpu_rank,
        device_name=device_name,
        num_file_eval=num_file_eval,
        category=category,
        factor_list=factor_list,
    )
    net_main = build_model(opt, device)
    ssim_criterion = SSIM().to(device)
    pearson_criterion = Pearson_loss().to(device)
    val_loader = gen_degradation_dataloader(
        GT_tag_list=category,
        noise_level=noise_level,
        factor_list=factor_list,
        SR_resolution_dict=opt["raw_data_resolution"],
        target_resolution=degradation_resolution,
        degradation_method="blur-composite-noise",
        average=average,
        generate_lifetime_flag=True,
        read_LR_flag=False,
        num_file_train=num_file_eval,
        num_file_val=num_file_eval,
        size=size,
        num_workers=num_workers,
        cwd=DESM_ROOT,
        device=device,
        on_the_fly_flag=True,
        eval_flag=True,
    )

    print("======================== synthetic baseline inference ========================")
    summary_rows = []

    with torch.no_grad():
        bar = tqdm(total=num_file_eval)
        for batch_index, data in enumerate(val_loader):
            if batch_index >= num_file_eval:
                break

            Input, GT_DS, _, _, lifetime, _ = data
            fake_main = net_main.feed_data(Input=torch.cat([Input, lifetime], dim=1), GT=GT_DS)
            wrong_class_overlap_lifetime, overlap_mask, non_overlap_mask = build_wrong_class_overlap_lifetime(
                GT_DS=GT_DS,
                size=size,
                factor_list=factor_list,
                device=device,
            )
            wrong_class_non_overlap_lifetime, _, _ = build_wrong_class_non_overlap_lifetime(
                GT_DS=GT_DS,
                size=size,
                factor_list=factor_list,
                device=device,
            )
            wrong_class_overlap_lifetime_tensor = lifetime_numpy_to_tensor(wrong_class_overlap_lifetime, lifetime)
            wrong_class_non_overlap_lifetime_tensor = lifetime_numpy_to_tensor(wrong_class_non_overlap_lifetime, lifetime)
            pred_wrong_class_overlap = net_main.net_G(torch.cat([Input, wrong_class_overlap_lifetime_tensor], dim=1))
            pred_wrong_class_non_overlap = net_main.net_G(torch.cat([Input, wrong_class_non_overlap_lifetime_tensor], dim=1))

            baseline_metrics = evaluate_prediction_metrics(fake_main, GT_DS, ssim_criterion, pearson_criterion)
            overlap_metrics = evaluate_prediction_metrics(pred_wrong_class_overlap, GT_DS, ssim_criterion, pearson_criterion)
            non_overlap_metrics = evaluate_prediction_metrics(pred_wrong_class_non_overlap, GT_DS, ssim_criterion, pearson_criterion)
            baseline_overlap_nrmae = evaluate_prediction_region_nrmae(fake_main, GT_DS, overlap_mask)
            overlap_overlap_nrmae = evaluate_prediction_region_nrmae(pred_wrong_class_overlap, GT_DS, overlap_mask)
            non_overlap_overlap_nrmae = evaluate_prediction_region_nrmae(pred_wrong_class_non_overlap, GT_DS, overlap_mask)
            baseline_non_overlap_nrmae = evaluate_prediction_region_nrmae(fake_main, GT_DS, non_overlap_mask)
            overlap_non_overlap_nrmae = evaluate_prediction_region_nrmae(pred_wrong_class_overlap, GT_DS, non_overlap_mask)
            non_overlap_non_overlap_nrmae = evaluate_prediction_region_nrmae(pred_wrong_class_non_overlap, GT_DS, non_overlap_mask)

            input_img = Input[0, 0].detach().cpu().numpy()
            baseline_lifetime_img = lifetime[0, 0].detach().cpu().numpy()
            fake_img = fake_main[0].detach().cpu().numpy().sum(axis=0)
            pred_wrong_class_overlap_img = pred_wrong_class_overlap[0].detach().cpu().numpy().sum(axis=0)
            pred_wrong_class_non_overlap_img = pred_wrong_class_non_overlap[0].detach().cpu().numpy().sum(axis=0)
            delta_overlap_img = np.abs(pred_wrong_class_overlap_img - fake_img)
            delta_non_overlap_img = np.abs(pred_wrong_class_non_overlap_img - fake_img)
            gt_img = GT_DS[0].detach().cpu().numpy().sum(axis=0)

            sample_name = f"synthetic_{batch_index + 1:04d}"
            save_sample_outputs(
                save_dir=save_dir,
                sample_name=sample_name,
                Input=Input,
                lifetime=lifetime,
                pred_baseline=fake_main,
                gt=GT_DS,
                wrong_class_overlap_lifetime=wrong_class_overlap_lifetime,
                wrong_class_non_overlap_lifetime=wrong_class_non_overlap_lifetime,
                overlap_mask=overlap_mask,
                non_overlap_mask=non_overlap_mask,
                pred_wrong_class_overlap=pred_wrong_class_overlap,
                pred_wrong_class_non_overlap=pred_wrong_class_non_overlap,
                delta_overlap=delta_overlap_img,
                delta_non_overlap=delta_non_overlap_img,
            )

            overlap_delta_global = mean_np(delta_overlap_img)
            non_overlap_delta_global = mean_np(delta_non_overlap_img)
            overlap_delta_in_overlap = masked_mean_np(delta_overlap_img, overlap_mask)
            overlap_delta_in_non_overlap = masked_mean_np(delta_overlap_img, non_overlap_mask)
            non_overlap_delta_in_overlap = masked_mean_np(delta_non_overlap_img, overlap_mask)
            non_overlap_delta_in_non_overlap = masked_mean_np(delta_non_overlap_img, non_overlap_mask)
            
            print(
                f"{sample_name}: "
                f"overlap-ablation global={overlap_delta_global:.6f}, "
                f"target(overlap)={overlap_delta_in_overlap:.6f}; "
                f"non-overlap-ablation global={non_overlap_delta_global:.6f}, "
                f"target(non-overlap)={non_overlap_delta_in_non_overlap:.6f}; "
                f"baseline NRMAE={baseline_metrics['nrmae']:.6f}, overlap NRMAE={overlap_metrics['nrmae']:.6f}, non-overlap NRMAE={non_overlap_metrics['nrmae']:.6f}; "
                f"overlap-region NRMAE baseline={baseline_overlap_nrmae:.6f}, overlap abl.={overlap_overlap_nrmae:.6f}; "
                f"non-overlap-region NRMAE baseline={baseline_non_overlap_nrmae:.6f}, non-overlap abl.={non_overlap_non_overlap_nrmae:.6f}"
            )
            summary_rows.append({
                "sample": sample_name,
                "overlap_delta_global": overlap_delta_global,
                "non_overlap_delta_global": non_overlap_delta_global,
                "overlap_delta_in_overlap": overlap_delta_in_overlap,
                "overlap_delta_in_non_overlap": overlap_delta_in_non_overlap,
                "non_overlap_delta_in_overlap": non_overlap_delta_in_overlap,
                "non_overlap_delta_in_non_overlap": non_overlap_delta_in_non_overlap,
                "baseline_nrmae": baseline_metrics["nrmae"],
                "overlap_nrmae": overlap_metrics["nrmae"],
                "non_overlap_nrmae": non_overlap_metrics["nrmae"],
                "overlap_nrmae_change": overlap_metrics["nrmae"] - baseline_metrics["nrmae"],
                "non_overlap_nrmae_change": non_overlap_metrics["nrmae"] - baseline_metrics["nrmae"],
                "baseline_ssim": baseline_metrics["ssim"],
                "overlap_ssim": overlap_metrics["ssim"],
                "non_overlap_ssim": non_overlap_metrics["ssim"],
                "overlap_ssim_change": overlap_metrics["ssim"] - baseline_metrics["ssim"],
                "non_overlap_ssim_change": non_overlap_metrics["ssim"] - baseline_metrics["ssim"],
                "baseline_pcc": baseline_metrics["pcc"],
                "overlap_pcc": overlap_metrics["pcc"],
                "non_overlap_pcc": non_overlap_metrics["pcc"],
                "overlap_pcc_change": overlap_metrics["pcc"] - baseline_metrics["pcc"],
                "non_overlap_pcc_change": non_overlap_metrics["pcc"] - baseline_metrics["pcc"],
                "baseline_overlap_nrmae": baseline_overlap_nrmae,
                "overlap_overlap_nrmae": overlap_overlap_nrmae,
                "non_overlap_overlap_nrmae": non_overlap_overlap_nrmae,
                "baseline_non_overlap_nrmae": baseline_non_overlap_nrmae,
                "overlap_non_overlap_nrmae": overlap_non_overlap_nrmae,
                "non_overlap_non_overlap_nrmae": non_overlap_non_overlap_nrmae,
                "overlap_overlap_nrmae_change": overlap_overlap_nrmae - baseline_overlap_nrmae,
                "non_overlap_overlap_nrmae_change": non_overlap_overlap_nrmae - baseline_overlap_nrmae,
                "overlap_non_overlap_nrmae_change": overlap_non_overlap_nrmae - baseline_non_overlap_nrmae,
                "non_overlap_non_overlap_nrmae_change": non_overlap_non_overlap_nrmae - baseline_non_overlap_nrmae,
            })

            if show_sample_plots:
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                axes[0, 0].imshow(input_img, cmap="gray")
                axes[0, 0].set_title("Input")
                axes[0, 1].imshow(baseline_lifetime_img, cmap="viridis")
                axes[0, 1].set_title("Lifetime baseline")
                axes[0, 2].imshow(wrong_class_overlap_lifetime, cmap="viridis")
                axes[0, 2].set_title("Wrong-class overlap")
                axes[0, 3].imshow(overlap_mask, cmap="gray")
                axes[0, 3].set_title("Overlap mask")
                axes[1, 0].imshow(fake_img, cmap="magma")
                axes[1, 0].set_title("Prediction baseline")
                axes[1, 1].imshow(gt_img, cmap="magma")
                axes[1, 1].set_title("GT")
                axes[1, 2].imshow(wrong_class_non_overlap_lifetime, cmap="viridis")
                axes[1, 2].set_title("Wrong-class non-overlap")
                axes[1, 3].imshow(non_overlap_mask, cmap="gray")
                axes[1, 3].set_title("Non-overlap mask")
                axes[2, 0].imshow(pred_wrong_class_overlap_img, cmap="magma")
                axes[2, 0].set_title("Pred wrong-class overlap")
                axes[2, 1].imshow(delta_overlap_img, cmap="hot")
                axes[2, 1].set_title(
                    "|overlap ablation - baseline|\n"
                    f"global={overlap_delta_global:.4f}, target={overlap_delta_in_overlap:.4f}"
                )
                axes[2, 2].imshow(pred_wrong_class_non_overlap_img, cmap="magma")
                axes[2, 2].set_title("Pred wrong-class non-overlap")
                axes[2, 3].imshow(delta_non_overlap_img, cmap="hot")
                axes[2, 3].set_title(
                    "|non-overlap ablation - baseline|\n"
                    f"global={non_overlap_delta_global:.4f}, target={non_overlap_delta_in_non_overlap:.4f}"
                )
                for ax in axes.flat:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

            bar.update(1)
        bar.close()

    if summary_rows:
        print("======================== summary statistics ========================")
        for key in [
            "overlap_delta_global",
            "non_overlap_delta_global",
            "overlap_delta_in_overlap",
            "non_overlap_delta_in_non_overlap",
            "overlap_nrmae_change",
            "non_overlap_nrmae_change",
            "overlap_ssim_change",
            "non_overlap_ssim_change",
            "overlap_pcc_change",
            "non_overlap_pcc_change",
            "overlap_overlap_nrmae_change",
            "non_overlap_non_overlap_nrmae_change",
        ]:
            values = np.asarray([row[key] for row in summary_rows], dtype=np.float32)
            values = values[np.isfinite(values)]
            if values.size > 0:
                print(f"{key}: mean={np.mean(values):.6f}, std={np.std(values):.6f}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "ablation_summary_metrics.csv"), index=False)
    for legacy_plot_name in [
        "prediction_delta_summary.png",
        "metric_change_overlap_vs_non_overlap.png",
        "nrmae_change_overlap_vs_non_overlap.png",
        "ssim_change_overlap_vs_non_overlap.png",
        "pcc_change_overlap_vs_non_overlap.png",
        "whole_image_metric_vs_gt.png",
        "whole_image_metric_change_vs_baseline.png",
        "region_wise_nrmae_vs_gt.png",
        "region_wise_nrmae_change_vs_baseline.png",
        "metric_summary.png",
        "region_nrmae_summary.png",
        "summary_statistics.png",
    ]:
        remove_if_exists(os.path.join(save_dir, legacy_plot_name))
    if show_summary_plot or save_summary_plot:
        plot_single_metric_change(
            summary_rows,
            "MAE change",
            "overlap_nrmae_change",
            "non_overlap_nrmae_change",
            save_path=os.path.join(save_dir, "nrmae_change_overlap_vs_non_overlap.png") if save_summary_plot else None,
            show_plot=show_summary_plot,
        )
        plot_single_metric_change(
            summary_rows,
            "SSIM change",
            "overlap_ssim_change",
            "non_overlap_ssim_change",
            save_path=os.path.join(save_dir, "ssim_change_overlap_vs_non_overlap.png") if save_summary_plot else None,
            show_plot=show_summary_plot,
        )
        plot_single_metric_change(
            summary_rows,
            "PCC change",
            "overlap_pcc_change",
            "non_overlap_pcc_change",
            save_path=os.path.join(save_dir, "pcc_change_overlap_vs_non_overlap.png") if save_summary_plot else None,
            show_plot=show_summary_plot,
        )

    return summary_rows


if __name__ == "__main__":
    if 0:
        run_synthetic_baseline(
            model_dir=os.path.join(
                DESM_ROOT,
                "Trained_models",
                "DESM_IL",
                "DESM_IL_synthetic_Mito_Lyso.pth",
            ),
            size=512,
            num_workers=0,
            gpu_rank="0",
            device_name="cuda:0",
            degradation_resolution=280,
            noise_level=0.280,
            average=1,
            num_file_eval=50,
            category=["Mito", "Lyso"],
            factor_list=[1, 1],
            show_sample_plots=False,
            show_summary_plot=False,
            save_summary_plot=True,
            save_dir=os.path.join(
                SCRIPT_DIR,
                "DESM_IL_lifetime_rf_aware_ablation",
                "synthetic_Mito_Lyso",
            ),
        )

    if 1:
        run_synthetic_baseline(
            model_dir=os.path.join(
                DESM_ROOT,
                "validation",
                "YOUR_MODEL_DIR",
                "weights",
                "1",
                "main_G.pth",
            ),
            size=512,
            num_workers=0,
            gpu_rank="0",
            device_name="cuda:0",
            degradation_resolution=225,
            noise_level=0,
            average=1,
            num_file_eval=50,
            category=["Micro", "Mito", "Lyso"],
            factor_list=[1, 1, 1],
            show_sample_plots=False,
            show_summary_plot=False,
            save_summary_plot=True,
            save_dir=os.path.join(
                SCRIPT_DIR,
                "DESM_IL_lifetime_rf_aware_ablation",
                "synthetic_Micro_Mito_Lyso",
            ),
        )
