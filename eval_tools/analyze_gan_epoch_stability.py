import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from tqdm import tqdm


DESM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if DESM_ROOT not in sys.path:
    sys.path.append(DESM_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from dataset.gen_datasets import gen_degradation_dataloader
from loss.PSNR import calculate_psnr
from loss.SSIM_loss import SSIM
from loss.pearson_loss import Pearson_loss
from net.make_model import DSCM_with_dataset
from utils import load_pretrained_weights


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


def resolve_device(gpu_rank="0", device_name="cuda:0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_rank)
    if torch.cuda.is_available():
        print("-----------------------------Using GPU-----------------------------")
        return torch.device(device_name)
    print("-----------------------------Using CPU-----------------------------")
    return torch.device("cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_runtime_opt(size, num_workers, gpu_rank, device_name, num_file_eval, category, factor_list, total_epoches):
    return {
        "gpu_rank": str(gpu_rank),
        "device": device_name,
        "size": int(size),
        "num_workers": int(num_workers),
        "num_file_train": int(num_file_eval),
        "num_file_val": int(num_file_eval),
        "category": list(category),
        "factor_list": list(factor_list),
        "raw_data_resolution": dict(RAW_DATA_RESOLUTION),
        "degradation_method": "blur-composite-noise",
        "lifetime_flag": False,
        "read_LR_flag": False,
        "net_G": {
            "initialize": True,
            "model_decouple_name": "Unet",
            "weight_decouple": [1, 0.01, 1, 0, 0, 1],
            "pretrain_dir": "None",
        },
        "net_D": {
            "model_name": "UnetD",
            "pretrain_dir": "None",
        },
        "train": {
            "scheduler": "CosineAnnealingLR",
            "lr_G": 0.00001,
            "lr_D": 0.0000001,
            "epoches": int(total_epoches),
        },
    }


def build_model(opt, device):
    return DSCM_with_dataset(
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


def normalize_to_unit(image):
    image = np.asarray(image, dtype=np.float32)
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max <= image_min:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image_min) / (image_max - image_min)


def percentile_normalize(image, percentile=99.7):
    image = np.asarray(image, dtype=np.float32)
    high = float(np.percentile(image, percentile))
    low = float(np.min(image))
    if high <= low:
        return normalize_to_unit(image)
    image = np.clip(image, low, high)
    return (image - low) / (high - low)


def to_chw_numpy(tensor):
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 4:
        array = array[0]
    return array


def make_rgb_composite(chw_image):
    chw_image = np.asarray(chw_image, dtype=np.float32)
    if chw_image.ndim == 2:
        gray = percentile_normalize(chw_image)
        return np.stack([gray, gray, gray], axis=-1)

    colors = np.array(
        [
            [0.00, 1.00, 0.00],
            [1.00, 0.00, 1.00],
            [0.10, 0.85, 1.00],
        ],
        dtype=np.float32,
    )
    rgb = np.zeros((chw_image.shape[1], chw_image.shape[2], 3), dtype=np.float32)
    for channel_index in range(chw_image.shape[0]):
        color = colors[channel_index % len(colors)]
        channel = percentile_normalize(chw_image[channel_index])
        rgb += channel[:, :, None] * color[None, None, :]
    return np.clip(rgb, 0.0, 1.0)


def make_error_map(pred_chw, gt_chw):
    pred_chw = np.asarray(pred_chw, dtype=np.float32)
    gt_chw = np.asarray(gt_chw, dtype=np.float32)
    error_map = np.mean(np.abs(pred_chw - gt_chw), axis=0)
    return normalize_to_unit(error_map)


def resolve_epoch_list(checkpoint_dir, requested_epochs=None):
    checkpoint_map = {}
    for filename in os.listdir(checkpoint_dir):
        if not (filename.startswith("ckpt_") and filename.endswith(".pt")):
            continue
        epoch_text = filename[len("ckpt_"):-len(".pt")]
        if not epoch_text.isdigit():
            continue
        checkpoint_map[int(epoch_text)] = os.path.join(checkpoint_dir, filename)

    if not checkpoint_map:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")

    available_epochs = sorted(checkpoint_map.keys())
    if requested_epochs is None:
        return available_epochs, checkpoint_map

    epoch_list = []
    for epoch in requested_epochs:
        epoch_int = int(epoch)
        if epoch_int in checkpoint_map:
            epoch_list.append(epoch_int)
        else:
            print(f"[Skip] checkpoint for epoch {epoch_int} not found.")
    if not epoch_list:
        raise ValueError("None of the requested epochs exist in the checkpoint directory.")
    return epoch_list, checkpoint_map


def cache_validation_samples(
        category,
        factor_list,
        noise_level,
        degradation_resolution,
        average,
        size,
        num_workers,
        num_file_eval,
        device,
        cwd,
        seed):
    set_random_seed(seed)
    val_loader = gen_degradation_dataloader(
        GT_tag_list=category,
        noise_level=noise_level,
        factor_list=factor_list,
        SR_resolution_dict={name: RAW_DATA_RESOLUTION[name] for name in category},
        target_resolution=degradation_resolution,
        degradation_method="blur-composite-noise",
        average=average,
        generate_lifetime_flag=False,
        read_LR_flag=False,
        num_file_train=num_file_eval,
        num_file_val=num_file_eval,
        size=size,
        num_workers=num_workers,
        cwd=cwd,
        device=device,
        on_the_fly_flag=True,
        eval_flag=True,
    )

    cached_samples = []
    print("======================== caching synthetic validation set ========================")
    for sample_index, data in enumerate(tqdm(val_loader, total=num_file_eval)):
        Input, GT_DS, GT_D, GT_S, simulated_lifetime, statistic_dict = data
        cached_samples.append(
            {
                "sample_index": sample_index,
                "Input": Input.detach().cpu(),
                "GT_DS": GT_DS.detach().cpu(),
                "GT_D": GT_D.detach().cpu(),
                "GT_S": GT_S.detach().cpu(),
                "simulated_lifetime": simulated_lifetime.detach().cpu() if torch.is_tensor(simulated_lifetime) else simulated_lifetime,
                "statistic_dict": statistic_dict,
            }
        )
    return cached_samples


def compute_metrics(pred, gt, ssim_criterion, pearson_criterion):
    mae_value = float(torch.mean(torch.abs(pred - gt)).item())

    ssim_values = []
    pred_np = to_chw_numpy(pred)
    gt_np = to_chw_numpy(gt)
    for channel_index in range(pred_np.shape[0]):
        pred_channel = pred[:, channel_index:channel_index + 1].detach()
        gt_channel = gt[:, channel_index:channel_index + 1].detach()
        pred_channel = pred_channel / max(float(torch.max(pred_channel).item()), 1e-6)
        gt_channel = gt_channel / max(float(torch.max(gt_channel).item()), 1e-6)
        ssim_values.append(float(ssim_criterion(pred_channel, gt_channel).item()))
    ssim_value = float(np.mean(ssim_values)) if ssim_values else np.nan

    pcc_value = float(pearson_criterion(pred, gt).item())

    psnr_values = []
    for channel_index in range(pred_np.shape[0]):
        psnr_values.append(
            calculate_psnr(
                pred_np[channel_index],
                gt_np[channel_index],
                data_range=None,
            )
        )
    psnr_value = float(np.mean(psnr_values)) if psnr_values else np.nan

    return {
        "mae": mae_value,
        "ssim": ssim_value,
        "pcc": pcc_value,
        "psnr": psnr_value,
    }


def build_crop_bbox(gt_chw, crop_size):
    gt_sum = np.sum(np.asarray(gt_chw, dtype=np.float32), axis=0)
    height, width = gt_sum.shape
    peak_y, peak_x = np.unravel_index(int(np.argmax(gt_sum)), gt_sum.shape)
    half = int(crop_size // 2)
    y0 = max(0, peak_y - half)
    x0 = max(0, peak_x - half)
    y1 = min(height, y0 + crop_size)
    x1 = min(width, x0 + crop_size)
    y0 = max(0, y1 - crop_size)
    x0 = max(0, x1 - crop_size)
    return int(y0), int(y1), int(x0), int(x1)


def crop_image(image, crop_bbox):
    y0, y1, x0, x1 = crop_bbox
    if image.ndim == 2:
        return image[y0:y1, x0:x1]
    if image.ndim == 3 and image.shape[0] <= 4:
        return image[:, y0:y1, x0:x1]
    return image[y0:y1, x0:x1, :]


def build_sample_dir(save_dir, sample_index):
    sample_dir = os.path.join(save_dir, f"sample_{int(sample_index):04d}")
    check_existence_local(sample_dir)
    return sample_dir


def save_sample_epoch_arrays(sample_dir, epoch, input_chw, pred_chw, gt_chw, error_map):
    tifffile.imwrite(os.path.join(sample_dir, "input.tif"), np.asarray(input_chw, dtype=np.float32))
    tifffile.imwrite(os.path.join(sample_dir, "gt.tif"), np.asarray(gt_chw, dtype=np.float32))
    tifffile.imwrite(
        os.path.join(sample_dir, f"prediction_epoch_{epoch:04d}.tif"),
        np.asarray(pred_chw, dtype=np.float32),
    )
    tifffile.imwrite(
        os.path.join(sample_dir, f"abs_error_epoch_{epoch:04d}.tif"),
        np.asarray(error_map, dtype=np.float32),
    )


def save_representative_arrays(sample_dir, epoch, payload):
    rep_dir = os.path.join(sample_dir, "representative")
    check_existence_local(rep_dir)
    tifffile.imwrite(
        os.path.join(rep_dir, f"pred_epoch_{epoch:04d}_crop.tif"),
        np.asarray(payload["pred_crop_chw"], dtype=np.float32),
    )
    tifffile.imwrite(
        os.path.join(rep_dir, f"gt_crop_epoch_{epoch:04d}.tif"),
        np.asarray(payload["gt_crop_chw"], dtype=np.float32),
    )
    tifffile.imwrite(
        os.path.join(rep_dir, f"error_crop_epoch_{epoch:04d}.tif"),
        np.asarray(payload["error_crop"], dtype=np.float32),
    )


def plot_metric_curves(summary_df, save_path):
    metrics = [
        ("mae_mean", "MAE", "#c92a2a"),
        ("ssim_mean", "SSIM", "#1c7ed6"),
        ("pcc_mean", "PCC", "#2b8a3e"),
        ("psnr_mean", "PSNR", "#9c36b5"),
    ]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5), constrained_layout=True, facecolor="white")
    axes = axes.flatten()

    for ax, (column_name, title, color) in zip(axes, metrics):
        ax.plot(summary_df["epoch"], summary_df[column_name], color=color, linewidth=2.2, marker="o", markersize=4.5)
        if f"{column_name[:-5]}_std" in summary_df.columns:
            std_values = summary_df[f"{column_name[:-5]}_std"].to_numpy(dtype=np.float32)
            mean_values = summary_df[column_name].to_numpy(dtype=np.float32)
            epoch_values = summary_df["epoch"].to_numpy(dtype=np.int32)
            ax.fill_between(epoch_values, mean_values - std_values, mean_values + std_values, color=color, alpha=0.14)
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel(title, fontsize=16)
        ax.tick_params(labelsize=16)
        ax.grid(True, color="#e9ecef", linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_color("#343a40")
            spine.set_linewidth(1.1)
        ax.set_facecolor("white")

    #fig.suptitle("GAN Epoch Stability on Synthetic Micro-Mito", fontsize=16, fontweight="bold")
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_representative_panel(payloads, save_path, title_suffix="full field"):
    if not payloads:
        return

    fig, axes = plt.subplots(
        len(payloads),
        4,
        figsize=(12.5, 3.0 * len(payloads)),
        constrained_layout=True,
        facecolor="white",
    )
    if len(payloads) == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Input", "Prediction", "Ground truth", "Absolute error"]
    for col_index, title in enumerate(column_titles):
        axes[0, col_index].set_title(title, fontsize=24, fontweight="bold")

    for row_index, payload in enumerate(payloads):
        axes[row_index, 0].imshow(payload["input_display"], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 1].imshow(payload["pred_display"])
        axes[row_index, 2].imshow(payload["gt_display"])
        axes[row_index, 3].imshow(payload["error_display"], cmap="inferno", vmin=0.0, vmax=1.0)

        axes[row_index, 0].set_ylabel(f"Epoch {payload['epoch']}", fontsize=24, fontweight="bold")
        for col_index in range(4):
            axes[row_index, col_index].set_xticks([])
            axes[row_index, col_index].set_yticks([])
            for spine in axes[row_index, col_index].spines.values():
                spine.set_visible(False)

    fig.suptitle(f"Representative sample across epochs ({title_suffix})", fontsize=24, fontweight="bold")
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(metric_df):
    summary_df = (
        metric_df.groupby("epoch", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            pcc_mean=("pcc", "mean"),
            pcc_std=("pcc", "std"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
        )
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    summary_df = summary_df.fillna(0.0)
    return summary_df


def evaluate_checkpoint_epoch(
        net_main,
        checkpoint_path,
        epoch,
        cached_samples,
        device,
        ssim_criterion,
        pearson_criterion,
        visualization_epoch_set,
        crop_size,
        save_dir,
        save_sample_visualizations,
        sample_indices_to_save):
    load_pretrained_weights(
        path=checkpoint_path,
        model=net_main.net_G,
        device=device,
        preferred_key="net_G",
        strict=True,
        name=f"net_G epoch {epoch}",
    )
    net_main.net_G.eval()

    metric_rows = []
    visual_payloads = {}

    with torch.no_grad():
        for sample in cached_samples:
            sample_index = int(sample["sample_index"])
            input_tensor = sample["Input"].to(device)
            gt_tensor = sample["GT_DS"].to(device)
            pred_tensor = net_main.feed_data(Input=input_tensor, GT=gt_tensor)

            metrics = compute_metrics(
                pred=pred_tensor,
                gt=gt_tensor,
                ssim_criterion=ssim_criterion,
                pearson_criterion=pearson_criterion,
            )
            metric_rows.append(
                {
                    "epoch": int(epoch),
                    "sample_index": sample_index,
                    **metrics,
                }
            )

            if (
                    save_sample_visualizations
                    and int(epoch) in visualization_epoch_set
                    and (sample_indices_to_save is None or sample_index in sample_indices_to_save)
            ):
                input_chw = to_chw_numpy(input_tensor)
                pred_chw = to_chw_numpy(pred_tensor)
                gt_chw = to_chw_numpy(gt_tensor)
                error_map = make_error_map(pred_chw=pred_chw, gt_chw=gt_chw)
                crop_bbox = build_crop_bbox(gt_chw=gt_chw, crop_size=crop_size)
                sample_save_dir = build_sample_dir(save_dir=save_dir, sample_index=sample_index)

                save_sample_epoch_arrays(
                    sample_dir=sample_save_dir,
                    epoch=epoch,
                    input_chw=input_chw,
                    pred_chw=pred_chw,
                    gt_chw=gt_chw,
                    error_map=error_map,
                )

                visual_payload = {
                    "epoch": int(epoch),
                    "input_display": percentile_normalize(input_chw[0]),
                    "pred_display": make_rgb_composite(pred_chw),
                    "gt_display": make_rgb_composite(gt_chw),
                    "error_display": error_map,
                    "pred_crop_display": make_rgb_composite(crop_image(pred_chw, crop_bbox)),
                    "gt_crop_display": make_rgb_composite(crop_image(gt_chw, crop_bbox)),
                    "input_crop_display": percentile_normalize(crop_image(input_chw[0], crop_bbox)),
                    "error_crop": crop_image(error_map, crop_bbox),
                    "pred_crop_chw": crop_image(pred_chw, crop_bbox),
                    "gt_crop_chw": crop_image(gt_chw, crop_bbox),
                }
                save_representative_arrays(sample_dir=sample_save_dir, epoch=epoch, payload=visual_payload)
                visual_payloads[sample_index] = visual_payload

    return metric_rows, visual_payloads


def run_gan_epoch_stability_analysis(
        checkpoint_dir,
        save_dir,
        size=384,
        num_workers=8,
        gpu_rank="0",
        device_name="cuda:0",
        degradation_resolution=225,
        noise_level=0.0,
        average=1,
        num_file_eval=40,
        category=("Micro", "Mito"),
        factor_list=(1, 1),
        epoch_list=(20, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000),
        visualization_epochs=(20, 200, 1000, 3000),
        sample_index_for_visualization=0,
        sample_indices_to_save=None,
        crop_size=128,
        save_sample_visualizations=True,
        seed=123):
    check_existence_local(save_dir)
    sample_indices_to_save = (
        {int(x) for x in sample_indices_to_save}
        if sample_indices_to_save is not None else None
    )
    if save_sample_visualizations:
        build_sample_dir(save_dir=save_dir, sample_index=sample_index_for_visualization)

    device = resolve_device(gpu_rank=gpu_rank, device_name=device_name)
    epoch_list, checkpoint_map = resolve_epoch_list(checkpoint_dir=checkpoint_dir, requested_epochs=epoch_list)

    opt = make_runtime_opt(
        size=size,
        num_workers=num_workers,
        gpu_rank=gpu_rank,
        device_name=device_name,
        num_file_eval=num_file_eval,
        category=category,
        factor_list=factor_list,
        total_epoches=max(epoch_list),
    )
    net_main = build_model(opt=opt, device=device)
    ssim_criterion = SSIM().to(device)
    pearson_criterion = Pearson_loss().to(device)

    cached_samples = cache_validation_samples(
        category=list(category),
        factor_list=list(factor_list),
        noise_level=noise_level,
        degradation_resolution=degradation_resolution,
        average=average,
        size=size,
        num_workers=num_workers,
        num_file_eval=num_file_eval,
        device=device,
        cwd=DESM_ROOT,
        seed=seed,
    )

    metric_rows = []
    visualization_epoch_set = {int(x) for x in visualization_epochs}
    visual_payloads_by_sample = {
        int(sample["sample_index"]): []
        for sample in cached_samples
    }

    print("======================== evaluating epoch checkpoints ========================")
    for epoch in tqdm(epoch_list):
        checkpoint_path = checkpoint_map[int(epoch)]
        epoch_metric_rows, epoch_visual_payloads = evaluate_checkpoint_epoch(
            net_main=net_main,
            checkpoint_path=checkpoint_path,
            epoch=int(epoch),
            cached_samples=cached_samples,
            device=device,
            ssim_criterion=ssim_criterion,
            pearson_criterion=pearson_criterion,
            visualization_epoch_set=visualization_epoch_set,
            crop_size=int(crop_size),
            save_dir=save_dir,
            save_sample_visualizations=save_sample_visualizations,
            sample_indices_to_save=sample_indices_to_save,
        )
        metric_rows.extend(epoch_metric_rows)

        for sample_index, visual_payload in epoch_visual_payloads.items():
            visual_payloads_by_sample[int(sample_index)].append(visual_payload)

    metric_df = pd.DataFrame(metric_rows)
    summary_df = summarize_metrics(metric_df=metric_df)

    metric_df.to_csv(os.path.join(save_dir, "epoch_metrics_per_sample.csv"), index=False)
    summary_df.to_csv(os.path.join(save_dir, "epoch_metrics_summary.csv"), index=False)

    plot_metric_curves(
        summary_df=summary_df,
        save_path=os.path.join(save_dir, "epoch_metric_curves.png"),
    )

    if save_sample_visualizations:
        default_full_payloads = []
        default_crop_payloads = []
        for sample_index, payloads in visual_payloads_by_sample.items():
            if sample_indices_to_save is not None and int(sample_index) not in sample_indices_to_save:
                continue
            payloads = sorted(payloads, key=lambda item: item["epoch"])
            sample_save_dir = build_sample_dir(save_dir=save_dir, sample_index=sample_index)

            full_payloads = [
                {
                    "epoch": payload["epoch"],
                    "input_display": payload["input_display"],
                    "pred_display": payload["pred_display"],
                    "gt_display": payload["gt_display"],
                    "error_display": payload["error_display"],
                }
                for payload in payloads
            ]
            crop_payloads = [
                {
                    "epoch": payload["epoch"],
                    "input_display": payload["input_crop_display"],
                    "pred_display": payload["pred_crop_display"],
                    "gt_display": payload["gt_crop_display"],
                    "error_display": payload["error_crop"],
                }
                for payload in payloads
            ]

            plot_representative_panel(
                payloads=full_payloads,
                save_path=os.path.join(sample_save_dir, "representative_epochs_full_field.png"),
                title_suffix=f"full field, sample {sample_index:04d}",
            )
            plot_representative_panel(
                payloads=crop_payloads,
                save_path=os.path.join(sample_save_dir, "representative_epochs_zoom.png"),
                title_suffix=f"zoomed region, sample {sample_index:04d}",
            )

            if int(sample_index) == int(sample_index_for_visualization):
                default_full_payloads = full_payloads
                default_crop_payloads = crop_payloads

        plot_representative_panel(
            payloads=default_full_payloads,
            save_path=os.path.join(save_dir, "representative_epochs_full_field.png"),
            title_suffix=f"full field, sample {int(sample_index_for_visualization):04d}",
        )
        plot_representative_panel(
            payloads=default_crop_payloads,
            save_path=os.path.join(save_dir, "representative_epochs_zoom.png"),
            title_suffix=f"zoomed region, sample {int(sample_index_for_visualization):04d}",
        )

    print("======================== epoch metric summary ========================")
    print(summary_df.to_string(index=False))
    return metric_df, summary_df


if __name__ == "__main__":
    checkpoint_dir = (
        os.path.join(
            DESM_ROOT,
            "validation",
            "YOUR_CHECKPOINT_DIR",
            "checkpoints",
            "1",
        )
    )
    save_dir = (
        os.path.join(
            SCRIPT_DIR,
            "GAN_epoch_stability",
            "YOUR_RUN_NAME",
        )
    )

    run_gan_epoch_stability_analysis(
        checkpoint_dir=checkpoint_dir,
        save_dir=save_dir,
        size=384,
        num_workers=8,
        gpu_rank="0",
        device_name="cuda:0",
        degradation_resolution=225,
        noise_level=0.0,
        average=1,
        num_file_eval=50,
        category=("Micro", "Mito"),
        factor_list=(1, 1),
        epoch_list=(20, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000),
        visualization_epochs=(20, 200, 1000, 3000),
        sample_index_for_visualization=13,
        sample_indices_to_save=None,
        save_sample_visualizations=True,
        crop_size=128,
        seed=123,
    )
