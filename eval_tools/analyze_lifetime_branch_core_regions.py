import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import tifffile
from tqdm import tqdm


DESM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if DESM_ROOT not in sys.path:
    sys.path.append(DESM_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import analyze_DESM_IL_lifetime_region_dependency as dep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "experimental", "all"])
    parser.add_argument("--gpu-rank", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--save-dir",
        default=os.path.join(SCRIPT_DIR, "DESM_IL_lifetime_branch_core_region_analysis"),
    )

    parser.add_argument("--erosion-radius", type=int, default=3)
    parser.add_argument("--near-radius", type=float, default=8.0)
    parser.add_argument("--far-radius", type=float, default=16.0)
    parser.add_argument("--min-region-pixels", type=int, default=20)
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--num-panels", type=int, default=8)
    parser.add_argument("--save-masks", action="store_true")

    # synthetic data settings
    parser.add_argument("--synthetic-model", default=dep.SYNTHETIC_IL_MODEL)
    parser.add_argument("--synthetic-save-subdir", default="synthetic")
    parser.add_argument("--synthetic-num-classes", type=int, default=2)
    parser.add_argument("--synthetic-categories", nargs="+", default=["Mito", "Lyso"])
    parser.add_argument("--synthetic-degradation-resolution", type=float, default=280)
    parser.add_argument("--synthetic-noise-level", type=float, default=0.280)
    parser.add_argument("--synthetic-average", type=int, default=1)
    parser.add_argument("--synthetic-degradation-method", default="blur-composite-noise")
    parser.add_argument("--synthetic-factor-list", nargs="+", type=float, default=[1, 1])
    parser.add_argument("--synthetic-read-lr-flag", action="store_true")

    # experimental data settings
    parser.add_argument("--experimental-model", default=dep.EXPERIMENTAL_IL_MODEL)
    parser.add_argument("--experimental-prior-model", default=dep.EXPERIMENTAL_IO_PRIOR_MODEL)
    parser.add_argument("--experimental-intensity-dir", default=None)
    parser.add_argument("--experimental-lifetime-dir", default=None)
    parser.add_argument("--experimental-save-subdir", default="experimental")
    parser.add_argument("--experimental-num-classes", type=int, default=3)
    parser.add_argument("--experimental-org-list", nargs="+", default=["Micro", "Mito", "Lyso"])
    parser.add_argument("--tm-min", type=float, default=0)
    parser.add_argument("--tm-max", type=float, default=5000)
    parser.add_argument("--kmeans-clusters", type=int, default=3)
    parser.add_argument("--io-prior-dilate-iter", type=int, default=0)
    parser.add_argument("--smooth-iter", type=int, default=2)
    parser.add_argument("--tile-upper-limit", type=int, default=1440)
    parser.add_argument("--save-maps", action="store_true")
    return parser.parse_args()


def make_disk_kernel(radius):
    radius = int(radius)
    if radius <= 0:
        return None
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def erode_mask(mask, radius):
    mask = mask.astype(np.uint8)
    kernel = make_disk_kernel(radius)
    if kernel is None:
        return mask
    return cv2.erode(mask, kernel, iterations=1).astype(np.uint8)


def build_core_region_masks(masks, erosion_radius=3, near_radius=8.0, far_radius=16.0):
    overlap = masks["overlap"].astype(np.uint8)
    foreground = masks["foreground"].astype(np.uint8)
    non_overlap = masks["non_overlap"].astype(np.uint8)

    overlap_core = erode_mask(overlap, erosion_radius)
    non_overlap_eroded = erode_mask(non_overlap, erosion_radius)

    outside_overlap = (1 - overlap).astype(np.uint8)
    distance_to_overlap = cv2.distanceTransform(outside_overlap, cv2.DIST_L2, 5)

    near_non_overlap = (
        (non_overlap > 0)
        & (distance_to_overlap > 0)
        & (distance_to_overlap <= near_radius)
    ).astype(np.uint8)
    far_non_overlap_core = (
        (non_overlap_eroded > 0)
        & (distance_to_overlap >= far_radius)
    ).astype(np.uint8)
    all_non_overlap_far = (
        (non_overlap > 0)
        & (distance_to_overlap >= far_radius)
    ).astype(np.uint8)

    return {
        "foreground": foreground,
        "overlap_all": overlap,
        "overlap_core": overlap_core,
        "non_overlap_all": non_overlap,
        "near_non_overlap": near_non_overlap,
        "far_non_overlap": all_non_overlap_far,
        "far_non_overlap_core": far_non_overlap_core,
        "full": np.ones_like(overlap, dtype=np.uint8),
        "distance_to_overlap": distance_to_overlap.astype(np.float32),
    }


def masked_mean_np(value, mask):
    mask = mask.astype(bool)
    if np.sum(mask) <= 0:
        return np.nan
    return float(np.mean(value[mask]))


def safe_ratio(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return np.nan
    return a / b


def tensor_abs_mean_channels(tensor):
    value = dep.to_cpu(tensor)
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3:
        value = np.mean(np.abs(value), axis=0)
    return value.astype(np.float32)


def compute_lifetime_branch_ablation(model, Input, lifetime):
    with torch.no_grad():
        normal_pred = model(torch.concatenate([Input, lifetime], dim=1))
        with dep.FLIMLifetimeBranchNeutralizer(model):
            ablated_pred = model(torch.concatenate([Input, lifetime], dim=1))
        diff = torch.abs(ablated_pred - normal_pred)
    return normal_pred, ablated_pred, diff


def core_region_row(sample_name, diff, core_masks, min_region_pixels):
    diff_map = tensor_abs_mean_channels(diff)
    regions = [
        "overlap_all",
        "overlap_core",
        "non_overlap_all",
        "near_non_overlap",
        "far_non_overlap",
        "far_non_overlap_core",
        "foreground",
        "full",
    ]
    row = {
        "sample": sample_name,
        "mask_source": "DESM_IL_output",
        "ablation": "set_all_FLIM_attention_W_g_outputs_to_zero",
    }
    for region in regions:
        mask = core_masks[region]
        pixels = int(np.sum(mask))
        row[f"{region}_pixels"] = pixels
        row[f"{region}_mean_delta"] = masked_mean_np(diff_map, mask) if pixels >= min_region_pixels else np.nan

    row["overlap_core_to_non_overlap_all"] = safe_ratio(
        row["overlap_core_mean_delta"],
        row["non_overlap_all_mean_delta"],
    )
    row["overlap_core_to_far_non_overlap_core"] = safe_ratio(
        row["overlap_core_mean_delta"],
        row["far_non_overlap_core_mean_delta"],
    )
    row["overlap_core_to_near_non_overlap"] = safe_ratio(
        row["overlap_core_mean_delta"],
        row["near_non_overlap_mean_delta"],
    )
    row["near_to_far_non_overlap_core"] = safe_ratio(
        row["near_non_overlap_mean_delta"],
        row["far_non_overlap_core_mean_delta"],
    )
    row["overlap_all_to_non_overlap_all"] = safe_ratio(
        row["overlap_all_mean_delta"],
        row["non_overlap_all_mean_delta"],
    )
    return row


def save_core_masks(save_dir, sample_name, core_masks):
    mask_dir = os.path.join(save_dir, "core_region_masks")
    dep.check_existence_local(mask_dir)
    for key, value in core_masks.items():
        if key == "distance_to_overlap":
            tifffile.imwrite(os.path.join(mask_dir, f"{sample_name}_{key}.tif"), value.astype(np.float32))
        elif key != "full":
            tifffile.imwrite(os.path.join(mask_dir, f"{sample_name}_{key}.tif"), value.astype(np.uint8))


def save_core_panel(save_dir, sample_name, normal_pred, ablated_pred, diff, core_masks):
    panel_dir = os.path.join(save_dir, "core_region_panels")
    dep.check_existence_local(panel_dir)

    normal_img = dep.tensor_to_2d_image(normal_pred)
    ablated_img = dep.tensor_to_2d_image(ablated_pred)
    diff_img = dep.tensor_to_2d_image(diff)
    overlap_core = core_masks["overlap_core"]
    near = core_masks["near_non_overlap"]
    far = core_masks["far_non_overlap_core"]
    distance = core_masks["distance_to_overlap"]

    vmin, vmax = dep.shared_display_limits(normal_img, ablated_img)
    fig, axes = dep.plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    panels = [
        ("Before ablation", dep.normalize_for_display_with_limits(normal_img, vmin, vmax), "magma"),
        ("After lifetime-branch ablation", dep.normalize_for_display_with_limits(ablated_img, vmin, vmax), "magma"),
        ("|Before - after|", dep.normalize_for_display(diff_img), "hot"),
        ("Distance to overlap", dep.normalize_for_display(distance), "viridis"),
        ("Overlap core", overlap_core, "gray"),
        ("Near non-overlap", near, "gray"),
        ("Far non-overlap core", far, "gray"),
        ("Core regions overlay", dep.normalize_for_display(normal_img), "magma"),
    ]
    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap=cmap)
        if title in ["Before ablation", "After lifetime-branch ablation", "|Before - after|"]:
            dep.add_overlap_contour(ax, core_masks["overlap_all"], image.shape)
        if title == "Core regions overlay":
            dep.add_overlap_contour(ax, overlap_core, image.shape)
            near_resized = dep.resize_mask_for_display(near, image.shape)
            far_resized = dep.resize_mask_for_display(far, image.shape)
            if np.sum(near_resized) > 0:
                ax.contour(near_resized, levels=[0.5], colors="yellow", linewidths=0.7)
            if np.sum(far_resized) > 0:
                ax.contour(far_resized, levels=[0.5], colors="lime", linewidths=0.7)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(sample_name, fontsize=11)
    fig.savefig(os.path.join(panel_dir, f"{sample_name}_core_region_panel.png"), dpi=200)
    dep.plt.close(fig)


def append_rows(path, rows):
    if not rows:
        return
    dep.check_existence_local(os.path.dirname(path))
    df_new = pd.DataFrame(rows)
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


def write_summary(csv_path, summary_path):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    summary = df[numeric_cols].agg(["mean", "median", "std", "count"]).transpose().reset_index()
    summary = summary.rename(columns={"index": "metric"})
    summary.to_csv(summary_path, index=False)


def analyze_one_sample(model, sample_name, Input, lifetime, save_dir, args, panel_index):
    normal_pred, ablated_pred, diff = compute_lifetime_branch_ablation(model, Input, lifetime)
    base_masks = dep.build_masks_from_DESM_IL_output(normal_pred)
    core_masks = build_core_region_masks(
        base_masks,
        erosion_radius=args.erosion_radius,
        near_radius=args.near_radius,
        far_radius=args.far_radius,
    )
    row = core_region_row(sample_name, diff, core_masks, args.min_region_pixels)

    if args.save_masks:
        save_core_masks(save_dir, sample_name, core_masks)
    if args.save_panels and panel_index < args.num_panels:
        save_core_panel(save_dir, sample_name, normal_pred, ablated_pred, diff, core_masks)
    return row


def analyze_synthetic(args, device):
    print("======================== synthetic core-region analysis ========================")
    save_dir = os.path.join(args.save_dir, args.synthetic_save_subdir)
    dep.check_existence_local(save_dir)

    if args.synthetic_num_classes != len(args.synthetic_categories):
        raise ValueError("--synthetic-num-classes must match --synthetic-categories.")
    if len(args.synthetic_factor_list) != len(args.synthetic_categories):
        raise ValueError("--synthetic-factor-list must match --synthetic-categories.")

    model = dep.load_model(
        model_path=args.synthetic_model,
        device=device,
        model_name="Unet_FLIM_att",
        num_classes=args.synthetic_num_classes,
    )
    sr_resolution_dict = {name: dep.RAW_DATA_RESOLUTION[name] for name in args.synthetic_categories}
    val_loader = dep.gen_degradation_dataloader(
        GT_tag_list=args.synthetic_categories,
        noise_level=args.synthetic_noise_level,
        factor_list=args.synthetic_factor_list,
        SR_resolution_dict=sr_resolution_dict,
        target_resolution=args.synthetic_degradation_resolution,
        generate_lifetime_flag=True,
        degradation_method=args.synthetic_degradation_method,
        average=args.synthetic_average,
        read_LR_flag=args.synthetic_read_lr_flag,
        num_file_train=args.num_samples,
        num_file_val=args.num_samples,
        size=args.size,
        num_workers=0,
        cwd=DESM_ROOT,
        device=device,
        on_the_fly_flag=True,
        eval_flag=True,
    )

    rows = []
    bar = tqdm(total=args.num_samples)
    for batch_index, data in enumerate(val_loader):
        if batch_index >= args.num_samples:
            break
        Input, GT_DS, _, _, lifetime, sta = data
        sample_name = f"synthetic_{batch_index+1:04d}"
        rows.append(analyze_one_sample(model, sample_name, Input, lifetime, save_dir, args, batch_index))
        bar.update(1)
    bar.close()

    csv_path = os.path.join(save_dir, "lifetime_branch_core_region_ablation.csv")
    append_rows(csv_path, rows)
    write_summary(csv_path, os.path.join(save_dir, "lifetime_branch_core_region_ablation_summary.csv"))
    print(f"Synthetic core-region analysis saved to: {save_dir}")


def analyze_experimental(args, device):
    print("======================== experimental core-region analysis ========================")
    if args.experimental_intensity_dir is None or args.experimental_lifetime_dir is None:
        raise ValueError("Experimental mode requires --experimental-intensity-dir and --experimental-lifetime-dir.")

    save_dir = os.path.join(args.save_dir, args.experimental_save_subdir)
    dep.check_existence_local(save_dir)

    model = dep.load_model(
        model_path=args.experimental_model,
        device=device,
        model_name="Unet_FLIM_att",
        num_classes=args.experimental_num_classes,
    )
    prior_model = dep.load_model(
        model_path=args.experimental_prior_model,
        device=device,
        model_name="Unet",
        num_classes=args.experimental_num_classes,
    )

    intensity_list = dep.natsort.natsorted([
        f for f in os.listdir(args.experimental_intensity_dir)
        if f.lower().endswith((".tif", ".tiff", ".png"))
    ])
    lifetime_list = dep.natsort.natsorted([
        f for f in os.listdir(args.experimental_lifetime_dir)
        if f.lower().endswith((".asc", ".txt", ".tif", ".tiff"))
    ])
    num_file = min(len(intensity_list), len(lifetime_list), args.num_samples)

    rows = []
    panel_index = 0
    bar = tqdm(total=num_file)
    for file_index in range(num_file):
        intensity_file = os.path.join(args.experimental_intensity_dir, intensity_list[file_index])
        lifetime_file = os.path.join(args.experimental_lifetime_dir, lifetime_list[file_index])
        img, tm_data = dep.read_experimental_image(intensity_file, lifetime_file, tm_min=args.tm_min, tm_max=args.tm_max)
        img, tm_data = dep.resize_by_zoom_if_needed(img, tm_data, intensity_list[file_index])

        h, w = img.shape
        row_cood_list, col_cood_list = dep.get_tile_coordinates(h, w, args.tile_upper_limit)
        for row_index, row_cood in enumerate(row_cood_list):
            for col_index, col_cood in enumerate(col_cood_list):
                intensity_input = img[row_cood[0]:row_cood[1], col_cood[0]:col_cood[1]]
                tm_input = tm_data[row_cood[0]:row_cood[1], col_cood[0]:col_cood[1]]
                sample_name = f"experimental_{file_index+1:04d}_r{row_index}_c{col_index}"
                Input, lifetime, prior_masks = dep.make_experimental_lifetime_prior(
                    intensity_input=intensity_input,
                    tm_input=tm_input,
                    prior_model=prior_model,
                    args=args,
                    device=device,
                    save_dir=save_dir,
                    sample_name=sample_name,
                )
                rows.append(analyze_one_sample(model, sample_name, Input, lifetime, save_dir, args, panel_index))
                panel_index += 1
        bar.update(1)
    bar.close()

    csv_path = os.path.join(save_dir, "lifetime_branch_core_region_ablation.csv")
    append_rows(csv_path, rows)
    write_summary(csv_path, os.path.join(save_dir, "lifetime_branch_core_region_ablation_summary.csv"))
    print(f"Experimental core-region analysis saved to: {save_dir}")


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_rank
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available() and args.device.find("cuda") != -1:
        device = torch.device(args.device)
        print("-----------------------------Using GPU-----------------------------")
    else:
        device = torch.device("cpu")
        print("-----------------------------Using CPU-----------------------------")

    if args.mode in ["synthetic", "all"]:
        analyze_synthetic(args, device)
    if args.mode in ["experimental", "all"]:
        analyze_experimental(args, device)


if __name__ == "__main__":
    RUN_FROM_IDE = True

    if RUN_FROM_IDE:
        sys.argv = [
            __file__,
            "--mode", "synthetic",
            "--num-samples", "50",
            "--synthetic-model",
            os.path.join(DESM_ROOT, "validation", "YOUR_MODEL_DIR", "weights", "1", "main_G.pth"),
            "--synthetic-num-classes", "3",
            "--synthetic-categories", "Micro", "Mito", "Lyso",
            "--synthetic-factor-list", "1", "1", "1",
            "--save-dir", os.path.join(SCRIPT_DIR, "DESM_IL_lifetime_branch_core_region_analysis_Micro_Mito_Lyso"),
            "--erosion-radius", "3",
            "--near-radius", "8",
            "--far-radius", "16",
            "--save-panels",
            "--num-panels", "8",
        ]

    main()
