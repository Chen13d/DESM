import os
import sys
import argparse
from copy import deepcopy

import cv2
import matplotlib


def configure_matplotlib_backend():
    if os.environ.get("MPLBACKEND"):
        return
    for backend in ["Qt5Agg", "QtAgg", "TkAgg"]:
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue
    matplotlib.use("Agg", force=True)


configure_matplotlib_backend()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu


cwd = os.getcwd()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DESM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(DESM_ROOT)

from utils import *
from dataset.gen_datasets import gen_degradation_dataloader
from visualization_DESM_IL import (
    norm_statistic,
    make_threshold_otsu,
    relabel_background_to_zero,
    relabel_foreground_by_priors,
    clean_labels_fill_bg_then_smooth,
    size_list,
    zoom_list,
    convert_table,
)


SYNTHETIC_IL_MODEL = os.path.join(
    DESM_ROOT,
    "Trained_models",
    "DESM_IL",
    "DESM_IL_synthetic_Mito_Lyso.pth",
)
EXPERIMENTAL_IL_MODEL = os.path.join(
    DESM_ROOT,
    "Trained_models",
    "DESM_IL",
    "DESM_IL_Micro_Mito_Lyso_Unet_FLIM_att.pth",
)
EXPERIMENTAL_IO_PRIOR_MODEL = os.path.join(
    DESM_ROOT,
    "Trained_models",
    "DESM_IL",
    "DESM_IO_Prior_Micro_Mito_Lyso.pth",
)

RAW_DATA_RESOLUTION = {
    "Micro": 85.93,
    "Mito": 87.09,
    "Lyso": 91.97,
    "Membrane": 81.24,
    "NPCs": 85.92,
    "Mito_inner": 82.88,
    "Mito_inner_deconv": 81.19,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "experimental", "all"])
    parser.add_argument("--gpu-rank", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--perturbation", default="shuffle", choices=["shuffle", "zero", "mean"])
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--save-network-panels", dest="save_network_panels", action="store_true", default=True)
    parser.add_argument("--no-save-network-panels", dest="save_network_panels", action="store_false")
    parser.add_argument("--num-network-panels", type=int, default=8)
    parser.add_argument("--network-panel-gate", default="att_1_FLIM")
    parser.add_argument("--show-overlap-mask", action="store_true")
    parser.add_argument("--save-ablation-comparison-panels", dest="save_ablation_comparison_panels", action="store_true", default=True)
    parser.add_argument("--no-save-ablation-comparison-panels", dest="save_ablation_comparison_panels", action="store_false")
    parser.add_argument("--skip-gate-ablation", action="store_true")
    parser.add_argument("--skip-lifetime-branch-ablation", action="store_true")
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append rows to existing CSV files instead of starting this run with fresh result tables.",
    )

    # synthetic data settings
    parser.add_argument("--synthetic-model", default=SYNTHETIC_IL_MODEL)
    parser.add_argument("--synthetic-save-dir", default=r"eval_tools\DESM_IL_lifetime_region_dependency\synthetic")
    parser.add_argument("--synthetic-num-classes", type=int, default=2)
    parser.add_argument("--synthetic-categories", nargs="+", default=["Mito", "Lyso"])
    parser.add_argument("--synthetic-degradation-resolution", type=float, default=280)
    parser.add_argument("--synthetic-noise-level", type=float, default=0.280)
    parser.add_argument("--synthetic-average", type=int, default=1)
    parser.add_argument("--synthetic-degradation-method", default="blur-composite-noise")
    parser.add_argument("--synthetic-factor-list", nargs="+", type=float, default=[1, 1])
    parser.add_argument("--synthetic-read-lr-flag", action="store_true")

    # experimental data settings
    parser.add_argument("--experimental-model", default=EXPERIMENTAL_IL_MODEL)
    parser.add_argument("--experimental-prior-model", default=EXPERIMENTAL_IO_PRIOR_MODEL)
    parser.add_argument("--experimental-intensity-dir", default=None)
    parser.add_argument("--experimental-lifetime-dir", default=None)
    parser.add_argument("--experimental-save-dir", default=r"eval_tools\DESM_IL_lifetime_region_dependency\experimental")
    parser.add_argument("--experimental-num-classes", type=int, default=3)
    parser.add_argument("--experimental-org-list", nargs="+", default=["Micro", "Mito", "Lyso"])
    parser.add_argument("--tm-min", type=float, default=0)
    parser.add_argument("--tm-max", type=float, default=5000)
    parser.add_argument("--kmeans-clusters", type=int, default=3)
    parser.add_argument("--io-prior-dilate-iter", type=int, default=0)
    parser.add_argument("--smooth-iter", type=int, default=2)
    parser.add_argument("--tile-upper-limit", type=int, default=1440)
    return parser.parse_args()


def load_model(model_path, device, model_name="Unet_FLIM_att", num_classes=2):
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"{model_path} was not saved as a complete torch.nn.Module. "
            "Please provide a full model file instead of a state_dict/checkpoint."
        )
    model = model.to(device)
    return model


class AttentionRecorder:
    def __init__(self, model):
        self.model = model
        self.maps = {}
        self.feature_modulation = {}
        self.handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name.endswith("_FLIM.psi"):
                self.handles.append(module.register_forward_hook(self.make_psi_hook(name)))
            elif name.endswith("_FLIM"):
                self.handles.append(module.register_forward_hook(self.make_feature_hook(name), with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def make_psi_hook(self, name):
        def hook(module, inputs, output):
            self.maps[name] = output.detach()
        return hook

    def make_feature_hook(self, name):
        def hook(module, inputs, kwargs, output):
            if "x" in kwargs:
                feature_before_gate = kwargs["x"].detach()
            else:
                feature_before_gate = inputs[1].detach()
            feature_after_gate = output.detach()
            before_energy = torch.mean(torch.abs(feature_before_gate), dim=1, keepdim=True)
            after_energy = torch.mean(torch.abs(feature_after_gate), dim=1, keepdim=True)
            effect_energy = torch.mean(torch.abs(feature_before_gate - feature_after_gate), dim=1, keepdim=True)
            self.feature_modulation[name] = {
                "feature_before_gate_abs": before_energy,
                "feature_after_gate_abs": after_energy,
                "gate_effect_abs": effect_energy,
                "gate_effect_relative": effect_energy / (before_energy + 1e-6),
                "gate_retention": after_energy / (before_energy + 1e-6),
            }
        return hook

    def clear(self):
        self.maps = {}
        self.feature_modulation = {}


class FLIMGateNeutralizer:
    def __init__(self, model):
        self.model = model
        self.handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name.endswith("_FLIM.psi"):
                self.handles.append(module.register_forward_hook(self.make_neutral_hook()))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def make_neutral_hook(self):
        def hook(module, inputs, output):
            return torch.ones_like(output)
        return hook


class FLIMLifetimeBranchNeutralizer:
    def __init__(self, model):
        self.model = model
        self.handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name.endswith("_FLIM.W_g"):
                self.handles.append(module.register_forward_hook(self.make_zero_hook()))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def make_zero_hook(self):
        def hook(module, inputs, output):
            return torch.zeros_like(output)
        return hook


def check_existence_local(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_ratio(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return np.nan
    return a / b


def append_table(save_path, rows):
    if not rows:
        return
    check_existence_local(os.path.dirname(save_path))
    df_new = pd.DataFrame(rows)
    if os.path.exists(save_path):
        df_old = pd.read_csv(save_path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(save_path, index=False)


RESULT_TABLE_FILES = [
    "attention_gate_region_weights.csv",
    "attention_gate_region_weights_summary.csv",
    "network_gate_feature_modulation.csv",
    "network_gate_feature_modulation_summary.csv",
    "flim_gate_ablation.csv",
    "flim_gate_ablation_summary.csv",
    "flim_lifetime_branch_ablation.csv",
    "flim_lifetime_branch_ablation_summary.csv",
    "lifetime_gradient_region_sensitivity.csv",
    "lifetime_gradient_region_sensitivity_summary.csv",
    "lifetime_region_perturbation.csv",
    "lifetime_region_perturbation_summary.csv",
]


def reset_result_tables(save_dir):
    for file_name in RESULT_TABLE_FILES:
        path = os.path.join(save_dir, file_name)
        if os.path.exists(path):
            os.remove(path)


def summarize_table(save_path, summary_path, group_key=None):
    if not os.path.exists(save_path):
        return
    df = pd.read_csv(save_path)
    numeric_cols = [col for col in df.columns if col != group_key and pd.api.types.is_numeric_dtype(df[col])]
    if group_key is not None and group_key in df.columns:
        summary = df.groupby(group_key)[numeric_cols].agg(["mean", "std", "count"])
        summary.columns = ["_".join(col) for col in summary.columns]
        summary = summary.reset_index()
    else:
        summary = df[numeric_cols].agg(["mean", "std", "count"]).transpose().reset_index()
        summary = summary.rename(columns={"index": "metric"})
    summary.to_csv(summary_path, index=False)


def make_binary_mask(img, mode="otsu"):
    img_np = to_cpu(img) if torch.is_tensor(img) else np.asarray(img)
    if mode == "positive":
        mask = img_np > 0
    else:
        if np.max(img_np) == np.min(img_np):
            mask = img_np > 0
        else:
            mask = img_np >= threshold_otsu(img_np)
    return mask.astype(np.uint8)


def build_masks_from_GT_DS(GT_DS):
    mask_list = []
    for channel_index in range(GT_DS.shape[1]):
        temp_GT = GT_DS[:, channel_index:channel_index+1, :, :]
        temp_mask = make_binary_mask(temp_GT.squeeze(0).squeeze(0), mode="otsu")
        mask_list.append(temp_mask)

    mask_stack = np.stack(mask_list, axis=0)
    count = np.sum(mask_stack, axis=0)
    foreground = (count >= 1).astype(np.uint8)
    overlap = (count >= 2).astype(np.uint8)
    non_overlap = ((foreground > 0) & (overlap == 0)).astype(np.uint8)
    return {
        "foreground": foreground,
        "overlap": overlap,
        "non_overlap": non_overlap,
        "full": np.ones_like(overlap, dtype=np.uint8),
    }


def build_masks_from_IO_prior(IO_prior_list):
    mask_stack = np.stack([(m > 0).astype(np.uint8) for m in IO_prior_list], axis=0)
    count = np.sum(mask_stack, axis=0)
    foreground = (count >= 1).astype(np.uint8)
    overlap = (count >= 2).astype(np.uint8)
    non_overlap = ((foreground > 0) & (overlap == 0)).astype(np.uint8)
    return {
        "foreground": foreground,
        "overlap": overlap,
        "non_overlap": non_overlap,
        "full": np.ones_like(overlap, dtype=np.uint8),
    }


def build_masks_from_DESM_IL_output(pred):
    mask_list = []
    pred_np = to_cpu(pred[0])
    for channel_index in range(pred_np.shape[0]):
        temp_mask = make_binary_mask(pred_np[channel_index], mode="otsu")
        mask_list.append(temp_mask)
    
    mask_stack = np.stack(mask_list, axis=0)
    count = np.sum(mask_stack, axis=0)
    '''plt.figure()
    plt.subplot(131)
    plt.imshow(mask_list[0])
    plt.subplot(132)
    plt.imshow(mask_list[1])
    plt.subplot(133)
    plt.imshow(mask_list[2])

    plt.figure()
    plt.imshow(count)
    plt.show()'''
    foreground = (count >= 1).astype(np.uint8)
    overlap = (count >= 2).astype(np.uint8)
    non_overlap = ((foreground > 0) & (overlap == 0)).astype(np.uint8)
    return {
        "foreground": foreground,
        "overlap": overlap,
        "non_overlap": non_overlap,
        "full": np.ones_like(overlap, dtype=np.uint8),
    }


def mask_to_torch(mask_np, ref_tensor):
    mask = torch.tensor(mask_np, dtype=ref_tensor.dtype, device=ref_tensor.device).unsqueeze(0).unsqueeze(0)
    return mask


def masked_mean(value, mask_np):
    mask = mask_to_torch(mask_np, value)
    if value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    denom = torch.sum(mask)
    if denom.item() <= 0:
        return np.nan
    return ((value * mask).sum() / denom).detach().cpu().item()


def masked_mean_all_channels(value, mask_np):
    mask = mask_to_torch(mask_np, value)
    if mask.shape[-2:] != value.shape[-2:]:
        mask = F.interpolate(mask, size=value.shape[-2:], mode="nearest")
    if value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    denom = torch.sum(mask)
    if denom.item() <= 0:
        return np.nan
    return ((value * mask).sum() / denom).detach().cpu().item()


def mask_at_tensor_resolution(mask_np, ref_tensor):
    mask = mask_to_torch(mask_np, ref_tensor)
    if mask.shape[-2:] != ref_tensor.shape[-2:]:
        mask = F.interpolate(mask, size=ref_tensor.shape[-2:], mode="nearest")
    return to_cpu(mask.squeeze(0).squeeze(0)).astype(np.uint8)


def network_gate_feature_rows(sample_name, attention_maps, feature_modulation, masks):
    rows = []
    for gate_name in sorted(feature_modulation.keys()):
        record = feature_modulation[gate_name]
        psi_name = f"{gate_name}.psi"
        row = {
            "sample": sample_name,
            "gate": gate_name,
            "overlap_pixels_at_gate": float(np.sum(mask_at_tensor_resolution(masks["overlap"], record["gate_effect_abs"]))),
            "non_overlap_pixels_at_gate": float(np.sum(mask_at_tensor_resolution(masks["non_overlap"], record["gate_effect_abs"]))),
        }

        if psi_name in attention_maps:
            psi = attention_maps[psi_name]
            row["psi_overlap_mean"] = masked_mean_all_channels(psi, masks["overlap"])
            row["psi_non_overlap_mean"] = masked_mean_all_channels(psi, masks["non_overlap"])
            row["psi_overlap_minus_non_overlap"] = row["psi_overlap_mean"] - row["psi_non_overlap_mean"]
            row["psi_overlap_to_non_overlap"] = safe_ratio(row["psi_overlap_mean"], row["psi_non_overlap_mean"])

        for key, value in record.items():
            overlap_mean = masked_mean_all_channels(value, masks["overlap"])
            non_overlap_mean = masked_mean_all_channels(value, masks["non_overlap"])
            foreground_mean = masked_mean_all_channels(value, masks["foreground"])
            row[f"{key}_overlap_mean"] = overlap_mean
            row[f"{key}_non_overlap_mean"] = non_overlap_mean
            row[f"{key}_foreground_mean"] = foreground_mean
            row[f"{key}_overlap_minus_non_overlap"] = overlap_mean - non_overlap_mean
            row[f"{key}_overlap_to_non_overlap"] = safe_ratio(overlap_mean, non_overlap_mean)
        rows.append(row)
    return rows


def attention_rows(sample_name, attention_maps, masks):
    rows = []
    for gate_name in sorted(attention_maps.keys()):
        att = attention_maps[gate_name]
        overlap = F.interpolate(mask_to_torch(masks["overlap"], att), size=att.shape[-2:], mode="nearest")
        non_overlap = F.interpolate(mask_to_torch(masks["non_overlap"], att), size=att.shape[-2:], mode="nearest")
        foreground = F.interpolate(mask_to_torch(masks["foreground"], att), size=att.shape[-2:], mode="nearest")

        overlap_mean = masked_mean(att, to_cpu(overlap.squeeze(0).squeeze(0)))
        non_overlap_mean = masked_mean(att, to_cpu(non_overlap.squeeze(0).squeeze(0)))
        foreground_mean = masked_mean(att, to_cpu(foreground.squeeze(0).squeeze(0)))

        rows.append({
            "sample": sample_name,
            "gate": gate_name,
            "attention_overlap_mean": overlap_mean,
            "attention_non_overlap_mean": non_overlap_mean,
            "attention_foreground_mean": foreground_mean,
            "attention_overlap_minus_non_overlap": overlap_mean - non_overlap_mean,
            "attention_overlap_to_non_overlap": safe_ratio(overlap_mean, non_overlap_mean),
            "overlap_pixels_at_gate": float(torch.sum(overlap).detach().cpu().item()),
            "non_overlap_pixels_at_gate": float(torch.sum(non_overlap).detach().cpu().item()),
        })
    return rows


def output_score(pred, masks):
    mask = mask_to_torch(masks["foreground"], pred)
    mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = torch.sum(mask).clamp_min(1.0)
    return torch.sum(pred * mask) / denom


def lifetime_gradient_row(model, sample_name, Input, lifetime, masks):
    model.zero_grad(set_to_none=True)
    Input_for_grad = Input.detach()
    lifetime_for_grad = lifetime.detach().clone().requires_grad_(True)
    pred = model(torch.concatenate([Input_for_grad, lifetime_for_grad], dim=1))
    score = output_score(pred, masks)
    score.backward()
    grad_abs = lifetime_for_grad.grad.detach().abs()

    overlap_mean = masked_mean(grad_abs, masks["overlap"])
    non_overlap_mean = masked_mean(grad_abs, masks["non_overlap"])
    foreground_mean = masked_mean(grad_abs, masks["foreground"])
    full_mean = masked_mean(grad_abs, masks["full"])

    row = {
        "sample": sample_name,
        "lifetime_grad_overlap_mean": overlap_mean,
        "lifetime_grad_non_overlap_mean": non_overlap_mean,
        "lifetime_grad_foreground_mean": foreground_mean,
        "lifetime_grad_full_mean": full_mean,
        "lifetime_grad_overlap_minus_non_overlap": overlap_mean - non_overlap_mean,
        "lifetime_grad_overlap_to_non_overlap": safe_ratio(overlap_mean, non_overlap_mean),
        "overlap_pixels": int(np.sum(masks["overlap"])),
        "non_overlap_pixels": int(np.sum(masks["non_overlap"])),
    }
    return row, grad_abs


def perturb_lifetime(lifetime, mask_np, mode="shuffle"):
    output = lifetime.detach().clone()
    mask = mask_to_torch(mask_np, output).bool()
    if torch.sum(mask).item() == 0:
        return output
    values = output[mask]
    if mode == "shuffle":
        perm = torch.randperm(values.numel(), device=values.device)
        output[mask] = values[perm]
    elif mode == "zero":
        output[mask] = 0
    elif mode == "mean":
        output[mask] = torch.mean(values)
    return output


def perturbation_row(model, sample_name, Input, lifetime, masks, perturbation="shuffle"):
    with torch.no_grad():
        pred = model(torch.concatenate([Input, lifetime], dim=1))
        lifetime_overlap = perturb_lifetime(lifetime, masks["overlap"], mode=perturbation)
        lifetime_non_overlap = perturb_lifetime(lifetime, masks["non_overlap"], mode=perturbation)
        pred_overlap_perturb = model(torch.concatenate([Input, lifetime_overlap], dim=1))
        pred_non_overlap_perturb = model(torch.concatenate([Input, lifetime_non_overlap], dim=1))

        delta_overlap = torch.abs(pred_overlap_perturb - pred)
        delta_non_overlap = torch.abs(pred_non_overlap_perturb - pred)

        overlap_self = masked_mean(delta_overlap, masks["overlap"])
        non_overlap_self = masked_mean(delta_non_overlap, masks["non_overlap"])

        return {
            "sample": sample_name,
            "perturbation": perturbation,
            "delta_when_overlap_lifetime_perturbed_in_overlap": overlap_self,
            "delta_when_overlap_lifetime_perturbed_in_non_overlap": masked_mean(delta_overlap, masks["non_overlap"]),
            "delta_when_non_overlap_lifetime_perturbed_in_overlap": masked_mean(delta_non_overlap, masks["overlap"]),
            "delta_when_non_overlap_lifetime_perturbed_in_non_overlap": non_overlap_self,
            "self_region_delta_overlap_to_non_overlap": safe_ratio(overlap_self, non_overlap_self),
        }


def save_analysis_maps(save_dir, sample_name, lifetime, masks, grad_abs):
    map_dir = os.path.join(save_dir, "maps")
    check_existence_local(map_dir)
    tifffile.imwrite(os.path.join(map_dir, f"{sample_name}_lifetime.tif"), to_cpu(lifetime[0, 0]).astype(np.float32))
    tifffile.imwrite(os.path.join(map_dir, f"{sample_name}_lifetime_grad_abs.tif"), to_cpu(grad_abs[0, 0]).astype(np.float32))
    tifffile.imwrite(os.path.join(map_dir, f"{sample_name}_overlap_mask.tif"), np.uint8(masks["overlap"]))
    tifffile.imwrite(os.path.join(map_dir, f"{sample_name}_non_overlap_mask.tif"), np.uint8(masks["non_overlap"]))


def tensor_to_2d_image(value):
    value_np = to_cpu(value)
    if value_np.ndim == 4:
        value_np = value_np[0]
    if value_np.ndim == 3:
        if value_np.shape[0] == 1:
            value_np = value_np[0]
        else:
            value_np = np.mean(value_np, axis=0)
    return np.asarray(value_np, dtype=np.float32)


def tensor_to_channel_images(value):
    value_np = to_cpu(value)
    if value_np.ndim == 4:
        value_np = value_np[0]
    if value_np.ndim == 2:
        value_np = value_np[np.newaxis, :, :]
    return np.asarray(value_np, dtype=np.float32)


def normalize_for_display(value):
    value = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(value)
    if not np.any(finite):
        return np.zeros_like(value, dtype=np.float32)
    vmin, vmax = np.percentile(value[finite], [1, 99])
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - vmin) / (vmax - vmin), 0, 1)


def normalize_for_display_with_limits(value, vmin, vmax):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - vmin) / (vmax - vmin), 0, 1)


def shared_display_limits(*images):
    values = []
    for image in images:
        image = np.asarray(image, dtype=np.float32)
        finite = image[np.isfinite(image)]
        if finite.size > 0:
            values.append(finite)
    if not values:
        return 0, 1
    values = np.concatenate(values)
    vmin, vmax = np.percentile(values, [1, 99])
    return vmin, vmax


def resize_mask_for_display(mask_np, target_shape):
    if mask_np.shape == target_shape:
        return mask_np.astype(np.uint8)
    return cv2.resize(mask_np.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def add_overlap_contour(ax, mask_np, target_shape):
    mask = resize_mask_for_display(mask_np, target_shape)
    if np.sum(mask) > 0:
        ax.contour(mask, levels=[0.5], colors="cyan", linewidths=1.6)


def display_image_to_uint8_rgb(image_show, cmap_name):
    image_show = np.clip(np.asarray(image_show, dtype=np.float32), 0, 1)
    if cmap_name == "gray":
        image_u8 = np.uint8(np.round(image_show * 255))
        return np.stack([image_u8, image_u8, image_u8], axis=-1)
    cmap = plt.get_cmap(cmap_name)
    return np.uint8(np.round(cmap(image_show)[..., :3] * 255))


def add_overlap_contour_to_rgb(image_rgb, mask_np):
    mask = resize_mask_for_display(mask_np, image_rgb.shape[:2]).astype(np.uint8)
    if np.sum(mask) == 0:
        return image_rgb
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_rgb = np.ascontiguousarray(image_rgb.copy())
    cv2.drawContours(image_rgb, contours, -1, color=(0, 255, 255), thickness=2)
    return image_rgb


def save_ablation_comparison_tifs(
        save_dir,
        sample_name,
        ablation_tag,
        panel_items_by_channel,
        masks,
        add_overlap_contour_flag,
        Input=None,
        lifetime=None):
    folder_name = f"{ablation_tag}_with_overlap" if add_overlap_contour_flag else ablation_tag
    tif_dir = os.path.join(save_dir, "ablation_comparison_tifs", folder_name, sample_name)
    check_existence_local(tif_dir)

    if Input is not None and lifetime is not None:
        input_img = normalize_for_display(tensor_to_2d_image(Input))
        lifetime_img = normalize_for_display(tensor_to_2d_image(lifetime))
        context_items = [
            ("input_intensity", input_img, "gray"),
            ("lifetime_map", lifetime_img, "viridis"),
        ]
        for item_name, image_show, cmap_name in context_items:
            image_rgb = display_image_to_uint8_rgb(image_show, cmap_name)
            if add_overlap_contour_flag:
                image_rgb = add_overlap_contour_to_rgb(image_rgb, masks["overlap"])
            tifffile.imwrite(
                os.path.join(tif_dir, f"{sample_name}_{ablation_tag}_{item_name}.tif"),
                image_rgb,
            )

    suffix_items = [
        "before_ablation",
        "after_ablation",
        "abs_before_minus_after",
    ]
    for channel_index, panel_items in enumerate(panel_items_by_channel):
        for item_index, (_, image_show, cmap_name) in enumerate(panel_items):
            image_rgb = display_image_to_uint8_rgb(image_show, cmap_name)
            if add_overlap_contour_flag:
                image_rgb = add_overlap_contour_to_rgb(image_rgb, masks["overlap"])
            tifffile.imwrite(
                os.path.join(
                    tif_dir,
                    f"{sample_name}_{ablation_tag}_ch{channel_index + 1:02d}_{suffix_items[item_index]}.tif",
                ),
                image_rgb,
            )


def safe_filename_text(text):
    keep_chars = []
    for char in str(text):
        if char.isalnum() or char in ["-", "_"]:
            keep_chars.append(char)
        elif char.isspace():
            keep_chars.append("_")
    safe_text = "".join(keep_chars).strip("_")
    return safe_text or "image"


def save_network_panel_tifs(save_dir, sample_name, panels, masks):
    tif_dir = os.path.join(save_dir, "network_panel_tifs", sample_name)
    check_existence_local(tif_dir)
    for panel_index, (title, image, cmap_name, do_normalize) in enumerate(panels):
        image_show = normalize_for_display(image) if do_normalize else np.asarray(image, dtype=np.float32)
        image_rgb = display_image_to_uint8_rgb(image_show, cmap_name)
        if title not in ["Overlap mask"]:
            image_rgb = add_overlap_contour_to_rgb(image_rgb, masks["overlap"])
        tifffile.imwrite(
            os.path.join(
                tif_dir,
                f"{sample_name}_panel{panel_index + 1:02d}_{safe_filename_text(title)}.tif",
            ),
            image_rgb,
        )


def save_input_lifetime_tifs(save_dir, sample_name, Input, lifetime):
    tif_dir = os.path.join(save_dir, "input_lifetime_tifs", sample_name)
    check_existence_local(tif_dir)

    input_img = tensor_to_2d_image(Input)
    lifetime_img = tensor_to_2d_image(lifetime)

    tifffile.imwrite(
        os.path.join(tif_dir, f"{sample_name}_input_intensity_raw.tif"),
        input_img.astype(np.float32),
    )
    tifffile.imwrite(
        os.path.join(tif_dir, f"{sample_name}_lifetime_map_raw.tif"),
        lifetime_img.astype(np.float32),
    )
    tifffile.imwrite(
        os.path.join(tif_dir, f"{sample_name}_input_intensity_display.tif"),
        display_image_to_uint8_rgb(normalize_for_display(input_img), "gray"),
    )
    tifffile.imwrite(
        os.path.join(tif_dir, f"{sample_name}_lifetime_map_display.tif"),
        display_image_to_uint8_rgb(normalize_for_display(lifetime_img), "viridis"),
    )


def show_image(image):
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def save_ablation_comparison_panel(
        save_dir,
        sample_name,
        Input,
        lifetime,
        normal_pred,
        ablation_pred,
        ablation_diff,
        masks,
        ablation_tag,
        ablation_title,
        add_overlap_contour_flag=False):
    if ablation_pred is None or ablation_diff is None:
        return

    folder_name = f"{ablation_tag}_with_overlap" if add_overlap_contour_flag else ablation_tag
    panel_dir = os.path.join(save_dir, "ablation_comparison_panels", folder_name)
    check_existence_local(panel_dir)

    normal_channels = tensor_to_channel_images(normal_pred)
    ablation_channels = tensor_to_channel_images(ablation_pred)
    diff_channels = tensor_to_channel_images(ablation_diff)
    num_channels = normal_channels.shape[0]

    fig, axes = plt.subplots(
        num_channels,
        3,
        figsize=(9.5, max(2.8, 2.6 * num_channels)),
        squeeze=False,
        constrained_layout=True,
    )
    panel_items_by_channel = []
    for channel_index in range(num_channels):
        before_img = normal_channels[channel_index]
        after_img = ablation_channels[channel_index]
        diff_img = diff_channels[channel_index]
        vmin, vmax = shared_display_limits(before_img, after_img)
        panel_items = [
            ("Before ablation", normalize_for_display_with_limits(before_img, vmin, vmax), "gray"),
            (ablation_title, normalize_for_display_with_limits(after_img, vmin, vmax), "gray"),
            ("|Before - after|", normalize_for_display(diff_img), "hot"),
        ]
        panel_items_by_channel.append(panel_items)
        for col_index, (title, image_show, cmap) in enumerate(panel_items):
            ax = axes[channel_index, col_index]
            ax.imshow(image_show, cmap=cmap)
            if add_overlap_contour_flag:
                add_overlap_contour(ax, masks["overlap"], image_show.shape)
            if channel_index == 0:
                ax.set_title(title, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"Channel {channel_index + 1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    overlap_note = " with DESM-IL output overlap" if add_overlap_contour_flag else ""
    fig.suptitle(f"{sample_name} - {ablation_tag}{overlap_note}", fontsize=11)
    suffix = "with_overlap" if add_overlap_contour_flag else "plain"
    fig.savefig(os.path.join(panel_dir, f"{sample_name}_{ablation_tag}_{suffix}.png"), dpi=200)
    plt.close(fig)
    save_ablation_comparison_tifs(
        save_dir=save_dir,
        sample_name=sample_name,
        ablation_tag=ablation_tag,
        panel_items_by_channel=panel_items_by_channel,
        masks=masks,
        add_overlap_contour_flag=add_overlap_contour_flag,
        Input=Input,
        lifetime=lifetime,
    )


def select_gate_for_panel(attention_maps, feature_modulation, gate_hint):
    if gate_hint in feature_modulation:
        gate_name = gate_hint
    elif feature_modulation:
        gate_name = sorted(feature_modulation.keys())[0]
    else:
        gate_name = None
    psi_name = f"{gate_name}.psi" if gate_name is not None else None
    if psi_name not in attention_maps and attention_maps:
        psi_name = sorted(attention_maps.keys())[0]
    return gate_name, psi_name


def save_network_panel(
        save_dir,
        sample_name,
        Input,
        lifetime,
        masks,
        normal_pred,
        ablation_pred,
        ablation_diff,
        attention_maps,
        feature_modulation,
        gate_hint,
        ablation_title="Prediction, no lifetime branch",
        diff_title="|normal - no lifetime branch|"):
    panel_dir = os.path.join(save_dir, "network_panels")
    check_existence_local(panel_dir)

    gate_name, psi_name = select_gate_for_panel(attention_maps, feature_modulation, gate_hint)
    input_img = tensor_to_2d_image(Input)
    lifetime_img = tensor_to_2d_image(lifetime)
    normal_img = tensor_to_2d_image(normal_pred)
    ablation_img = tensor_to_2d_image(ablation_pred) if ablation_pred is not None else np.zeros_like(normal_img)
    diff_img = tensor_to_2d_image(ablation_diff) if ablation_diff is not None else np.zeros_like(normal_img)

    if psi_name is not None:
        psi_img = tensor_to_2d_image(attention_maps[psi_name])
    else:
        psi_img = np.zeros_like(input_img)
    if gate_name is not None:
        effect_img = tensor_to_2d_image(feature_modulation[gate_name]["gate_effect_relative"])
    else:
        effect_img = np.zeros_like(input_img)

    panels = [
        ("Intensity input", input_img, "gray", True),
        ("Lifetime prior", lifetime_img, "viridis", True),
        ("Overlap mask", masks["overlap"], "gray", False),
        ("Normal prediction", normal_img, "magma", True),
        (f"{psi_name or 'FLIM gate'}", psi_img, "viridis", True),
        (f"{gate_name or 'FLIM gate'} relative effect", effect_img, "inferno", True),
        (ablation_title, ablation_img, "magma", True),
        (diff_title, diff_img, "hot", True),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for ax, (title, image, cmap, do_normalize) in zip(axes.ravel(), panels):
        image_show = normalize_for_display(image) if do_normalize else image
        ax.imshow(image_show, cmap=cmap)
        if title not in ["Overlap mask"]:
            add_overlap_contour(ax, masks["overlap"], image_show.shape)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(sample_name, fontsize=11)
    fig.savefig(os.path.join(panel_dir, f"{sample_name}_network_gate_panel.png"), dpi=200)
    plt.close(fig)
    save_network_panel_tifs(save_dir, sample_name, panels, masks)


def gate_ablation_row(model, sample_name, Input, lifetime, masks, normal_pred, mask_source="DESM_IL_output"):
    with torch.no_grad():
        with FLIMGateNeutralizer(model):
            neutral_pred = model(torch.concatenate([Input, lifetime], dim=1))
        ablation_diff = torch.abs(neutral_pred - normal_pred)

    overlap_mean = masked_mean_all_channels(ablation_diff, masks["overlap"])
    non_overlap_mean = masked_mean_all_channels(ablation_diff, masks["non_overlap"])
    foreground_mean = masked_mean_all_channels(ablation_diff, masks["foreground"])
    full_mean = masked_mean_all_channels(ablation_diff, masks["full"])
    row = {
        "sample": sample_name,
        "ablation_mask_source": mask_source,
        "gate_ablation": "set_all_FLIM_attention_psi_to_one",
        "delta_when_FLIM_gates_neutralized_in_overlap": overlap_mean,
        "delta_when_FLIM_gates_neutralized_in_non_overlap": non_overlap_mean,
        "delta_when_FLIM_gates_neutralized_in_foreground": foreground_mean,
        "delta_when_FLIM_gates_neutralized_in_full": full_mean,
        "gate_ablation_delta_overlap_minus_non_overlap": overlap_mean - non_overlap_mean,
        "gate_ablation_delta_overlap_to_non_overlap": safe_ratio(overlap_mean, non_overlap_mean),
        "overlap_pixels": int(np.sum(masks["overlap"])),
        "non_overlap_pixels": int(np.sum(masks["non_overlap"])),
    }
    return row, neutral_pred, ablation_diff


def lifetime_branch_ablation_row(model, sample_name, Input, lifetime, masks, normal_pred, mask_source="DESM_IL_output"):
    with torch.no_grad():
        with FLIMLifetimeBranchNeutralizer(model):
            no_lifetime_branch_pred = model(torch.concatenate([Input, lifetime], dim=1))
        ablation_diff = torch.abs(no_lifetime_branch_pred - normal_pred)

    overlap_mean = masked_mean_all_channels(ablation_diff, masks["overlap"])
    non_overlap_mean = masked_mean_all_channels(ablation_diff, masks["non_overlap"])
    foreground_mean = masked_mean_all_channels(ablation_diff, masks["foreground"])
    full_mean = masked_mean_all_channels(ablation_diff, masks["full"])
    row = {
        "sample": sample_name,
        "ablation_mask_source": mask_source,
        "lifetime_branch_ablation": "set_all_FLIM_attention_W_g_outputs_to_zero",
        "delta_when_lifetime_branch_neutralized_in_overlap": overlap_mean,
        "delta_when_lifetime_branch_neutralized_in_non_overlap": non_overlap_mean,
        "delta_when_lifetime_branch_neutralized_in_foreground": foreground_mean,
        "delta_when_lifetime_branch_neutralized_in_full": full_mean,
        "lifetime_branch_delta_overlap_minus_non_overlap": overlap_mean - non_overlap_mean,
        "lifetime_branch_delta_overlap_to_non_overlap": safe_ratio(overlap_mean, non_overlap_mean),
        "overlap_pixels": int(np.sum(masks["overlap"])),
        "non_overlap_pixels": int(np.sum(masks["non_overlap"])),
    }
    return row, no_lifetime_branch_pred, ablation_diff


def analyze_one_sample(
        model,
        recorder,
        sample_name,
        Input,
        lifetime,
        masks,
        save_dir,
        perturbation="shuffle",
        save_maps=False,
        save_network_panel_flag=False,
        network_panel_gate="att_1_FLIM",
        save_ablation_comparison_panels=True,
        skip_gate_ablation=False,
        skip_lifetime_branch_ablation=False,
        show_overlap_mask_flag=False):
    recorder.clear()
    with torch.no_grad():
        normal_pred = model(torch.concatenate([Input, lifetime], dim=1))
    ablation_masks = build_masks_from_DESM_IL_output(normal_pred)
    attention_maps = dict(recorder.maps)
    feature_modulation = {key: dict(value) for key, value in recorder.feature_modulation.items()}

    att_rows = attention_rows(sample_name, attention_maps, masks)
    gate_feature_rows = network_gate_feature_rows(sample_name, attention_maps, feature_modulation, masks)
    grad_row, grad_abs = lifetime_gradient_row(model, sample_name, Input, lifetime, masks)
    pert_row = perturbation_row(model, sample_name, Input, lifetime, masks, perturbation=perturbation)

    append_table(os.path.join(save_dir, "attention_gate_region_weights.csv"), att_rows)
    append_table(os.path.join(save_dir, "network_gate_feature_modulation.csv"), gate_feature_rows)
    append_table(os.path.join(save_dir, "lifetime_gradient_region_sensitivity.csv"), [grad_row])
    append_table(os.path.join(save_dir, "lifetime_region_perturbation.csv"), [pert_row])

    gate_neutral_pred = None
    gate_ablation_diff = None
    if not skip_gate_ablation:
        ablation_row, gate_neutral_pred, gate_ablation_diff = gate_ablation_row(
            model=model,
            sample_name=sample_name,
            Input=Input,
            lifetime=lifetime,
            masks=ablation_masks,
            normal_pred=normal_pred,
            mask_source="DESM_IL_output",
        )
        append_table(os.path.join(save_dir, "flim_gate_ablation.csv"), [ablation_row])
        if save_ablation_comparison_panels:
            save_ablation_comparison_panel(
                save_dir=save_dir,
                sample_name=sample_name,
                Input=Input,
                lifetime=lifetime,
                normal_pred=normal_pred,
                ablation_pred=gate_neutral_pred,
                ablation_diff=gate_ablation_diff,
                masks=ablation_masks,
                ablation_tag="whole_gate",
                ablation_title="After gate neutralization",
                add_overlap_contour_flag=False,
            )
            save_ablation_comparison_panel(
                save_dir=save_dir,
                sample_name=sample_name,
                Input=Input,
                lifetime=lifetime,
                normal_pred=normal_pred,
                ablation_pred=gate_neutral_pred,
                ablation_diff=gate_ablation_diff,
                masks=ablation_masks,
                ablation_tag="whole_gate",
                ablation_title="After gate neutralization",
                add_overlap_contour_flag=True,
            )

    lifetime_branch_pred = None
    lifetime_branch_diff = None
    if not skip_lifetime_branch_ablation:
        lifetime_branch_row, lifetime_branch_pred, lifetime_branch_diff = lifetime_branch_ablation_row(
            model=model,
            sample_name=sample_name,
            Input=Input,
            lifetime=lifetime,
            masks=ablation_masks,
            normal_pred=normal_pred,
            mask_source="DESM_IL_output",
        )
        append_table(os.path.join(save_dir, "flim_lifetime_branch_ablation.csv"), [lifetime_branch_row])
        if save_ablation_comparison_panels:
            save_ablation_comparison_panel(
                save_dir=save_dir,
                sample_name=sample_name,
                Input=Input,
                lifetime=lifetime,
                normal_pred=normal_pred,
                ablation_pred=lifetime_branch_pred,
                ablation_diff=lifetime_branch_diff,
                masks=ablation_masks,
                ablation_tag="lifetime_branch",
                ablation_title="After lifetime-branch ablation",
                add_overlap_contour_flag=False,
            )
            save_ablation_comparison_panel(
                save_dir=save_dir,
                sample_name=sample_name,
                Input=Input,
                lifetime=lifetime,
                normal_pred=normal_pred,
                ablation_pred=lifetime_branch_pred,
                ablation_diff=lifetime_branch_diff,
                masks=ablation_masks,
                ablation_tag="lifetime_branch",
                ablation_title="After lifetime-branch ablation",
                add_overlap_contour_flag=True,
            )

    if save_maps:
        save_analysis_maps(save_dir, sample_name, lifetime, ablation_masks, grad_abs)

    if save_network_panel_flag:
        panel_pred = lifetime_branch_pred if lifetime_branch_pred is not None else gate_neutral_pred
        panel_diff = lifetime_branch_diff if lifetime_branch_diff is not None else gate_ablation_diff
        panel_title = "Prediction, no lifetime branch" if lifetime_branch_pred is not None else "Prediction, neutral FLIM gates"
        panel_diff_title = "|normal - no lifetime branch|" if lifetime_branch_diff is not None else "|normal - neutral gates|"
        save_network_panel(
            save_dir=save_dir,
            sample_name=sample_name,
            Input=Input,
            lifetime=lifetime,
            masks=ablation_masks,
            normal_pred=normal_pred,
            ablation_pred=panel_pred,
            ablation_diff=panel_diff,
            attention_maps=attention_maps,
            feature_modulation=feature_modulation,
            gate_hint=network_panel_gate,
            ablation_title=panel_title,
            diff_title=panel_diff_title,
        )


def write_summary(save_dir):
    summarize_table(
        os.path.join(save_dir, "attention_gate_region_weights.csv"),
        os.path.join(save_dir, "attention_gate_region_weights_summary.csv"),
        group_key="gate",
    )
    summarize_table(
        os.path.join(save_dir, "network_gate_feature_modulation.csv"),
        os.path.join(save_dir, "network_gate_feature_modulation_summary.csv"),
        group_key="gate",
    )
    summarize_table(
        os.path.join(save_dir, "flim_gate_ablation.csv"),
        os.path.join(save_dir, "flim_gate_ablation_summary.csv"),
    )
    summarize_table(
        os.path.join(save_dir, "flim_lifetime_branch_ablation.csv"),
        os.path.join(save_dir, "flim_lifetime_branch_ablation_summary.csv"),
    )
    summarize_table(
        os.path.join(save_dir, "lifetime_gradient_region_sensitivity.csv"),
        os.path.join(save_dir, "lifetime_gradient_region_sensitivity_summary.csv"),
    )
    summarize_table(
        os.path.join(save_dir, "lifetime_region_perturbation.csv"),
        os.path.join(save_dir, "lifetime_region_perturbation_summary.csv"),
    )


def analyze_synthetic_data(args, device):
    print("======================== synthetic data analysis ========================")
    save_dir = args.synthetic_save_dir
    check_existence_local(save_dir)
    if not args.append_output:
        reset_result_tables(save_dir)
    if args.synthetic_num_classes != len(args.synthetic_categories):
        raise ValueError("--synthetic-num-classes must match the number of --synthetic-categories.")
    if len(args.synthetic_factor_list) != len(args.synthetic_categories):
        raise ValueError("--synthetic-factor-list must match the number of --synthetic-categories.")
    for category in args.synthetic_categories:
        if category not in RAW_DATA_RESOLUTION:
            raise KeyError(f"No raw-data resolution is defined for category: {category}")

    model = load_model(
        model_path=args.synthetic_model,
        device=device,
        model_name="Unet_FLIM_att",
        num_classes=args.synthetic_num_classes,
    )
    model.eval()

    SR_resolution_dict = {name: RAW_DATA_RESOLUTION[name] for name in args.synthetic_categories}
    val_loader = gen_degradation_dataloader(
        GT_tag_list=args.synthetic_categories,
        noise_level=args.synthetic_noise_level,
        factor_list=args.synthetic_factor_list,
        SR_resolution_dict=SR_resolution_dict,
        target_resolution=args.synthetic_degradation_resolution,
        generate_lifetime_flag=True,
        degradation_method=args.synthetic_degradation_method,
        average=args.synthetic_average,
        read_LR_flag=args.synthetic_read_lr_flag,
        num_file_train=args.num_samples,
        num_file_val=args.num_samples,
        size=args.size,
        num_workers=0,
        cwd=cwd,
        device=device,
        on_the_fly_flag=True,
        eval_flag=True,
    )

    with AttentionRecorder(model) as recorder:
        bar = tqdm(total=args.num_samples)
        for batch_index, data in enumerate(val_loader):
            if batch_index >= args.num_samples:
                break
            Input, GT_DS, _, _, lifetime, sta = data
            masks = build_masks_from_GT_DS(GT_DS)

            sample_name = f"synthetic_{batch_index+1:04d}"

            analyze_one_sample(
                model=model,
                recorder=recorder,
                sample_name=sample_name,
                Input=Input,
                lifetime=lifetime,
                masks=masks,
                save_dir=save_dir,
                perturbation=args.perturbation,
                save_maps=args.save_maps,
                save_network_panel_flag=args.save_network_panels and batch_index < args.num_network_panels,
                network_panel_gate=args.network_panel_gate,
                save_ablation_comparison_panels=args.save_ablation_comparison_panels,
                skip_gate_ablation=args.skip_gate_ablation,
                skip_lifetime_branch_ablation=args.skip_lifetime_branch_ablation,
                show_overlap_mask_flag=args.show_overlap_mask,
            )
            bar.update(1)
        bar.close()

    write_summary(save_dir)
    print(f"Synthetic analysis saved to: {save_dir}")


def read_experimental_image(intensity_path, lifetime_path, tm_min=0, tm_max=5000):
    img = tifffile.imread(intensity_path)
    if len(img.shape) == 3:
        img = img[0]
    if str(lifetime_path).lower().endswith(".asc") or str(lifetime_path).lower().endswith(".txt"):
        tm_data = np.loadtxt(lifetime_path)
    else:
        tm_data = tifffile.imread(lifetime_path)
    img = np.asarray(img, dtype=np.float32)
    tm_data = np.asarray(tm_data, dtype=np.float32)
    tm_data[tm_data < tm_min] = tm_min
    tm_data[tm_data > tm_max] = tm_max
    return img, tm_data


def resize_by_zoom_if_needed(img, tm_data, file_name):
    zoom = 0
    for zoom_name in zoom_list:
        if file_name.find(zoom_name) != -1:
            zoom = zoom_name
    if zoom == 0:
        return img, tm_data
    raw_h, raw_w = img.shape
    for zoom_index in range(len(zoom_list)):
        if zoom == zoom_list[zoom_index]:
            for size_index in range(len(size_list)):
                if raw_h == size_list[size_index]:
                    convert_ratio = convert_table[zoom_index][size_index] / 20
                    img = resize(img, (int(raw_h * convert_ratio), int(raw_w * convert_ratio)))
                    tm_data = resize(tm_data, (int(raw_h * convert_ratio), int(raw_w * convert_ratio)))
                    y0 = int(0.05 * img.shape[0])
                    y1 = int(0.95 * img.shape[0])
                    x0 = int(0.05 * img.shape[1])
                    x1 = int(0.95 * img.shape[1])
                    return img[y0:y1, x0:x1], tm_data[y0:y1, x0:x1]
    return img, tm_data


def get_tile_coordinates(h, w, upper_limit):
    size = min(h, w)
    if size > upper_limit:
        num_row = (h // upper_limit) + 1
        num_col = (w // upper_limit) + 1
        size = upper_limit
    else:
        upper_limit = (size // 16) * 16
        num_row = max((h // upper_limit), 1)
        num_col = max((w // upper_limit), 1)
        size = upper_limit

    row_cood_list = []
    col_cood_list = []
    for row in range(num_row):
        if (row + 1) * size < h:
            row_cood_list.append([row * size, (row + 1) * size])
        else:
            row_cood_list.append([h - size, h])
    for col in range(num_col):
        if (col + 1) * size < w:
            col_cood_list.append([col * size, (col + 1) * size])
        else:
            col_cood_list.append([w - size, w])
    return row_cood_list, col_cood_list


def make_experimental_lifetime_prior(intensity_input, tm_input, prior_model, args, device, save_dir, sample_name):
    vector = tm_input.reshape(-1, 1)
    kmeans = KMeans(n_clusters=args.kmeans_clusters, random_state=42, n_init=10)
    kmeans.fit(vector)
    clustered_input = kmeans.labels_.reshape(tm_input.shape).astype(np.uint8)

    intensity_tensor = torch.tensor(np.float64(intensity_input), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
    intensity_tensor, mean, std = norm_statistic(intensity_tensor, device)

    IO_output = prior_model(intensity_tensor)
    IO_prior_list = []
    IO_prior = np.zeros_like(clustered_input, dtype=np.uint8)
    for org_index in range(IO_output.shape[1]):
        IO_per_c = IO_output[:, org_index:org_index+1, :, :].detach().squeeze(0).squeeze(0).cpu().numpy()
        IO_per_c_thresholded = make_threshold_otsu(IO_per_c)
        if args.io_prior_dilate_iter > 0:
            IO_per_c_thresholded = cv2.dilate(
                IO_per_c_thresholded.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=args.io_prior_dilate_iter,
            )
        IO_prior[IO_per_c_thresholded != 0] = 1
        IO_prior_list.append(IO_per_c_thresholded.astype(np.uint8))

    masks = build_masks_from_IO_prior(IO_prior_list)

    IO_prior_background = np.zeros_like(IO_prior)
    IO_prior_background[IO_prior == 0] = 1

    # Keep the same relabeling style used in visualization_DESM_IL.py.
    IO_prior_list_for_label = [x.copy() for x in IO_prior_list]
    if len(args.experimental_org_list) == 3 and len(IO_prior_list_for_label) >= 3:
        # micro lyso the same
        #IO_prior_list_for_label = [IO_prior_list_for_label[0] + IO_prior_list_for_label[2], IO_prior_list_for_label[1]]
        # micro lyso different
        IO_prior_list_for_label = [IO_prior_list_for_label[0], IO_prior_list_for_label[1], IO_prior_list_for_label[2]]
    elif len(args.experimental_org_list) == 2 and len(IO_prior_list_for_label) >= 2:
        IO_prior_list_for_label = [IO_prior_list_for_label[0], IO_prior_list_for_label[1]]

    clustered_bg0, bg_label, bg_mapping, bg_scores = relabel_background_to_zero(
        clustered_input=clustered_input,
        IO_prior_background=IO_prior_background,
    )
    permuted_clustered_input, fg_mapping, fg_scores = relabel_foreground_by_priors(
        clustered_input_bg0=clustered_bg0,
        IO_prior_list=IO_prior_list_for_label,
        keep_unmatched=True,
    )
    permuted_clustered_input = IO_prior * permuted_clustered_input
    if args.smooth_iter > 0:
        permuted_clustered_input = clean_labels_fill_bg_then_smooth(
            permuted_clustered_input,
            ksize=5,
            open_iter=args.smooth_iter,
            close_iter=args.smooth_iter,
        )

    clustered_tensor = torch.tensor(np.float64(permuted_clustered_input), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
    clustered_tensor, _, _ = norm_statistic(clustered_tensor, device=device)

    if args.save_maps:
        check_existence_local(os.path.join(save_dir, "experimental_prior_maps"))
        tifffile.imwrite(os.path.join(save_dir, "experimental_prior_maps", f"{sample_name}_IO_prior.tif"), np.uint8(IO_prior))
        tifffile.imwrite(os.path.join(save_dir, "experimental_prior_maps", f"{sample_name}_clustered_prior.tif"), np.uint8(permuted_clustered_input))
        tifffile.imwrite(os.path.join(save_dir, "experimental_prior_maps", f"{sample_name}_overlap_mask.tif"), np.uint8(masks["overlap"]))

    return intensity_tensor, clustered_tensor, masks


def analyze_experimental_data(args, device):
    print("======================== experimental data analysis ========================")
    if args.experimental_intensity_dir is None or args.experimental_lifetime_dir is None:
        raise ValueError("Experimental analysis requires --experimental-intensity-dir and --experimental-lifetime-dir.")

    save_dir = args.experimental_save_dir
    check_existence_local(save_dir)
    if not args.append_output:
        reset_result_tables(save_dir)

    model = load_model(
        model_path=args.experimental_model,
        device=device,
        model_name="Unet_FLIM_att",
        num_classes=args.experimental_num_classes,
    )
    prior_model = load_model(
        model_path=args.experimental_prior_model,
        device=device,
        model_name="Unet",
        num_classes=args.experimental_num_classes,
    )

    intensity_list = natsort.natsorted([f for f in os.listdir(args.experimental_intensity_dir) if f.lower().endswith((".tif", ".tiff", ".png"))])
    lifetime_list = natsort.natsorted([f for f in os.listdir(args.experimental_lifetime_dir) if f.lower().endswith((".asc", ".txt", ".tif", ".tiff"))])
    num_file = min(len(intensity_list), len(lifetime_list), args.num_samples)

    with AttentionRecorder(model) as recorder:
        bar = tqdm(total=num_file)
        panel_count = 0
        for file_index in range(num_file):
            intensity_file = os.path.join(args.experimental_intensity_dir, intensity_list[file_index])
            lifetime_file = os.path.join(args.experimental_lifetime_dir, lifetime_list[file_index])
            img, tm_data = read_experimental_image(intensity_file, lifetime_file, tm_min=args.tm_min, tm_max=args.tm_max)
            img, tm_data = resize_by_zoom_if_needed(img, tm_data, intensity_list[file_index])

            h, w = img.shape
            row_cood_list, col_cood_list = get_tile_coordinates(h, w, args.tile_upper_limit)
            for row_index, row_cood in enumerate(row_cood_list):
                for col_index, col_cood in enumerate(col_cood_list):
                    intensity_input = img[row_cood[0]:row_cood[1], col_cood[0]:col_cood[1]]
                    tm_input = tm_data[row_cood[0]:row_cood[1], col_cood[0]:col_cood[1]]
                    sample_name = f"experimental_{file_index+1:04d}_r{row_index}_c{col_index}"

                    Input, lifetime, masks = make_experimental_lifetime_prior(
                        intensity_input=intensity_input,
                        tm_input=tm_input,
                        prior_model=prior_model,
                        args=args,
                        device=device,
                        save_dir=save_dir,
                        sample_name=sample_name,
                    )
                    analyze_one_sample(
                        model=model,
                        recorder=recorder,
                        sample_name=sample_name,
                        Input=Input,
                        lifetime=lifetime,
                        masks=masks,
                        save_dir=save_dir,
                        perturbation=args.perturbation,
                        save_maps=args.save_maps,
                        save_network_panel_flag=args.save_network_panels and panel_count < args.num_network_panels,
                        network_panel_gate=args.network_panel_gate,
                        save_ablation_comparison_panels=args.save_ablation_comparison_panels,
                        skip_gate_ablation=args.skip_gate_ablation,
                        skip_lifetime_branch_ablation=args.skip_lifetime_branch_ablation,
                        show_overlap_mask_flag=args.show_overlap_mask,
                    )
                    if args.save_network_panels and panel_count < args.num_network_panels:
                        panel_count += 1
            bar.update(1)
        bar.close()

    write_summary(save_dir)
    print(f"Experimental analysis saved to: {save_dir}")


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
        analyze_synthetic_data(args, device)
    if args.mode in ["experimental", "all"]:
        analyze_experimental_data(args, device)


if __name__ == "__main__":
    RUN_FROM_IDE = True

    # semi-synthetic mito lyso
    if 1:
        sys.argv = [
            __file__,
            "--mode", "synthetic",
            "--num-samples", "50",
            "--synthetic-save-dir",
            os.path.join(SCRIPT_DIR, "DESM_IL_lifetime_branch_synthetic_Mito_Lyso_ide_run"),
            "--num-network-panels", "8",
            "--show-overlap-mask",
        ]

        main()

    # semi-synthetic micro mito lyso
    if 1:
        sys.argv = [
            __file__,
            "--mode", "synthetic",
            "--num-samples", "50",
            "--synthetic-model",
            os.path.join(DESM_ROOT, "validation", "YOUR_MODEL_DIR", "weights", "1", "main_G.pth"),
            "--synthetic-degradation-resolution", "225",
            "--synthetic-noise-level", "0",
            "--synthetic-num-classes", "3",
            "--synthetic-categories", "Micro", "Mito", "Lyso",
            "--synthetic-factor-list", "1", "1", "1",
            "--synthetic-save-dir",
            os.path.join(SCRIPT_DIR, "DESM_IL_lifetime_branch_synthetic_Micro_Mito_Lyso_ide_run"),
            "--num-network-panels", "8",
            "--show-overlap-mask",
        ]


        main()

    if 0:
        sys.argv = [
            __file__,
            "--mode", "experimental",
            "--num-samples", "1",
            "--experimental-intensity-dir",
            os.path.join(
                DESM_ROOT,
                "visualization",
                "Experimental data",
                "DESM_IL_Microtubules_Mitochondria_Lysosomes",
                "Intensity",
            ),
            "--experimental-lifetime-dir",
            os.path.join(
                DESM_ROOT,
                "visualization",
                "Experimental data",
                "DESM_IL_Microtubules_Mitochondria_Lysosomes",
                "ASC",
            ),
            "--experimental-save-dir",
            os.path.join(SCRIPT_DIR, "DESM_IL_lifetime_branch_experimental_ide_run"),
            "--num-network-panels", "8",
            "--show-overlap-mask",
        ]

        main()
