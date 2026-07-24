import glob
import os
import subprocess
import sys
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from tqdm import tqdm


DESM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if DESM_ROOT not in sys.path:
    sys.path.append(DESM_ROOT)

DEFAULT_FIJI_EXECUTABLE = os.environ.get("FIJI_EXECUTABLE", "")
if not DEFAULT_FIJI_EXECUTABLE and os.environ.get("FIJI_HOME"):
    DEFAULT_FIJI_EXECUTABLE = os.path.join(os.environ["FIJI_HOME"], "ImageJ-win64.exe")

from dataset.gen_datasets import gen_degradation_dataloader
from net.make_model import DSCM_with_dataset


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


def save_array(path, array):
    tifffile.imwrite(path, np.asarray(array))


def apply_binary_mask(image, binary_mask):
    image = np.asarray(image, dtype=np.float32)
    binary_mask = (np.asarray(binary_mask) > 0).astype(np.float32)
    return image * binary_mask


def make_binary_mask(image):
    image_norm = normalize_to_unit(image)
    if float(np.max(image_norm)) <= 0:
        return np.zeros_like(image_norm, dtype=np.uint8)
    thresh = threshold_otsu(image_norm)
    return (image_norm >= thresh).astype(np.uint8)


def connected_components_filtered(mask, min_area=4, connectivity=8):
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=connectivity)
    filtered = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        filtered[labels == label_id] = next_id
        next_id += 1
    return filtered


def estimate_nucleus_mask(reference_image, gaussian_sigma=12.0, threshold_scale=0.9, dilation_radius=8):
    image_norm = normalize_to_unit(reference_image)
    if float(np.max(image_norm)) <= 0:
        return np.ones_like(image_norm, dtype=np.uint8)

    blurred = cv2.GaussianBlur(image_norm, (0, 0), gaussian_sigma)
    positive = blurred[blurred > 0]
    if positive.size == 0:
        return np.ones_like(image_norm, dtype=np.uint8)

    base_threshold = threshold_otsu(positive)
    nucleus_binary = (blurred >= float(base_threshold) * float(threshold_scale)).astype(np.uint8)

    if np.count_nonzero(nucleus_binary) == 0:
        return np.ones_like(image_norm, dtype=np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    nucleus_binary = cv2.morphologyEx(nucleus_binary, cv2.MORPH_CLOSE, kernel_close)
    nucleus_binary = ndi.binary_fill_holes(nucleus_binary > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(nucleus_binary, connectivity=8)
    if num_labels <= 1:
        largest_component = nucleus_binary
    else:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(component_areas)) + 1
        largest_component = (labels == largest_label).astype(np.uint8)

    if int(dilation_radius) > 0:
        kernel_size = int(dilation_radius) * 2 + 1
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        largest_component = cv2.dilate(largest_component, kernel_dilate, iterations=1)

    largest_component = ndi.binary_fill_holes(largest_component > 0).astype(np.uint8)
    return largest_component


def make_empty_rsfish_dataframe():
    return pd.DataFrame(columns=["x", "y", "t", "c", "intensity"])


def parse_rsfish_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"RS-FISH CSV not found: {csv_path}")

    if os.path.getsize(csv_path) == 0:
        empty_df = make_empty_rsfish_dataframe()
        empty_coords = np.asarray([], dtype=np.float32)
        return empty_coords, empty_coords.copy(), empty_df

    try:
        spot_df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        empty_df = make_empty_rsfish_dataframe()
        empty_coords = np.asarray([], dtype=np.float32)
        return empty_coords, empty_coords.copy(), empty_df

    if spot_df.empty and len(spot_df.columns) == 0:
        empty_df = make_empty_rsfish_dataframe()
        empty_coords = np.asarray([], dtype=np.float32)
        return empty_coords, empty_coords.copy(), empty_df

    lowered_columns = {column.lower(): column for column in spot_df.columns}

    x_candidates = ["x", "xpos", "x_pos", "x(px)", "x [px]", "x-coordinate"]
    y_candidates = ["y", "ypos", "y_pos", "y(px)", "y [px]", "y-coordinate"]

    x_column = next((lowered_columns[name] for name in x_candidates if name in lowered_columns), None)
    y_column = next((lowered_columns[name] for name in y_candidates if name in lowered_columns), None)

    if x_column is None or y_column is None:
        if spot_df.shape[1] >= 2:
            x_column = spot_df.columns[0]
            y_column = spot_df.columns[1]
        else:
            raise ValueError(f"Could not infer x/y columns from RS-FISH output: {csv_path}")

    x_coords = spot_df[x_column].to_numpy(dtype=np.float32)
    y_coords = spot_df[y_column].to_numpy(dtype=np.float32)
    return x_coords, y_coords, spot_df


def labels_from_spot_coordinates(image_shape, x_coords, y_coords, disk_radius=3):
    height, width = image_shape
    label_map = np.zeros((height, width), dtype=np.int32)
    yy, xx = np.ogrid[:height, :width]

    for label_id, (x_coord, y_coord) in enumerate(zip(x_coords, y_coords), start=1):
        center_x = int(round(float(x_coord)))
        center_y = int(round(float(y_coord)))
        if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
            continue
        disk_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= float(disk_radius) ** 2
        label_map[disk_mask & (label_map == 0)] = int(label_id)

    binary_mask = (label_map > 0).astype(np.uint8)
    return binary_mask, label_map


def resolve_rsfish_background_code(background_subtraction):
    background_map = {
        "No background subtraction": 0,
        "Mean": 1,
        "Median": 2,
        "RANSAC on Mean": 3,
        "RANSAC on Median": 4,
    }
    if background_subtraction not in background_map:
        raise ValueError(f"Unsupported background_subtraction: {background_subtraction}")
    return background_map[background_subtraction]


def resolve_rsfish_ransac_code(robust_fitting):
    ransac_map = {
        "No RANSAC": 0,
        "RANSAC": 1,
        "Multiconsensus RANSAC": 2,
    }
    if robust_fitting not in ransac_map:
        raise ValueError(f"Unsupported robust_fitting: {robust_fitting}")
    return ransac_map[robust_fitting]


def resolve_fiji_java_and_classpath(fiji_executable):
    fiji_root = os.path.dirname(os.path.abspath(fiji_executable))
    java_candidates = sorted(
        glob.glob(os.path.join(fiji_root, "java", "**", "java.exe"), recursive=True)
    )
    if not java_candidates:
        raise FileNotFoundError(f"Could not find bundled java.exe under Fiji root: {fiji_root}")

    plugin_candidates = sorted(
        glob.glob(os.path.join(fiji_root, "plugins", "Radial_SymmetryLocalization*.jar"))
    )
    if not plugin_candidates:
        raise FileNotFoundError(f"Could not find RS-FISH plugin jar under: {os.path.join(fiji_root, 'plugins')}")

    java_executable = java_candidates[-1]
    plugin_jar = plugin_candidates[-1]
    classpath = f"{plugin_jar};{os.path.join(fiji_root, 'jars', '*')}"
    return java_executable, classpath


def segment_puncta_rsfish_fiji(
        image,
        fiji_executable,
        rsfish_disk_radius=3,
        rsfish_timeout_sec=300,
        rsfish_work_dir=None,
        rsfish_anisotropy=1.0,
        rsfish_robust_fitting="RANSAC",
        rsfish_spot_intensity="Linear Interpolation",
        rsfish_sigma_dog=1.5,
        rsfish_threshold_dog=0.007,
        rsfish_support_radius=3,
        rsfish_inlier_ratio=0.1,
        rsfish_max_error=1.5,
        rsfish_intensity_threshold=0.0,
        rsfish_background_subtraction="No background subtraction",
        rsfish_background_max_error=0.05,
        rsfish_background_inlier_ratio=0.75,
        rsfish_use_multithreading=False,
        rsfish_num_threads=1,
        rsfish_block_size_x=128,
        rsfish_block_size_y=128,
        rsfish_block_size_z=16):
    if not fiji_executable:
        raise ValueError("fiji_executable must be provided.")
    if not os.path.exists(fiji_executable):
        raise FileNotFoundError(f"Fiji executable not found: {fiji_executable}")

    if rsfish_use_multithreading:
        print("RS-FISH official CLI does not expose block/multithreading flags here; using plugin defaults.")

    work_dir = rsfish_work_dir or tempfile.mkdtemp(prefix="rsfish_fiji_")
    check_existence_local(work_dir)

    input_tif = os.path.join(work_dir, "rsfish_input.tif")
    output_csv = os.path.join(work_dir, "rsfish_output.csv")
    save_array(input_tif, np.asarray(image, dtype=np.float32))

    java_executable, classpath = resolve_fiji_java_and_classpath(fiji_executable)
    command = [
        java_executable,
        "-cp",
        classpath,
        "cmd.RadialSymmetry",
        "-i",
        input_tif,
        "-o",
        output_csv,
        "-a",
        str(float(rsfish_anisotropy)),
        "-r",
        str(int(resolve_rsfish_ransac_code(rsfish_robust_fitting))),
        "-s",
        str(float(rsfish_sigma_dog)),
        "-t",
        str(float(rsfish_threshold_dog)),
        "-sr",
        str(int(rsfish_support_radius)),
        "-ir",
        str(float(rsfish_inlier_ratio)),
        "-e",
        str(float(rsfish_max_error)),
        "-it",
        str(float(rsfish_intensity_threshold)),
        "-bg",
        str(int(resolve_rsfish_background_code(rsfish_background_subtraction))),
        "-bge",
        str(float(rsfish_background_max_error)),
        "-bgir",
        str(float(rsfish_background_inlier_ratio)),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=int(rsfish_timeout_sec),
        cwd=work_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Fiji RS-FISH execution failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}"
        )
    if not os.path.exists(output_csv):
        raise FileNotFoundError(
            "RS-FISH did not produce the expected CSV output.\n"
            f"Expected: {output_csv}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}"
        )

    x_coords, y_coords, _ = parse_rsfish_csv(output_csv)
    binary_mask, label_map = labels_from_spot_coordinates(
        image_shape=np.asarray(image).shape,
        x_coords=x_coords,
        y_coords=y_coords,
        disk_radius=rsfish_disk_radius,
    )
    return binary_mask, label_map, output_csv


def build_component_table(label_map):
    label_map = np.asarray(label_map, dtype=np.int32)
    label_ids = [int(x) for x in np.unique(label_map) if int(x) != 0]
    rows = []

    for label_id in label_ids:
        comp_mask = label_map == label_id
        ys, xs = np.nonzero(comp_mask)
        rows.append(
            {
                "label_id": label_id,
                "area": int(len(ys)),
                "centroid_y": float(np.mean(ys)),
                "centroid_x": float(np.mean(xs)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["label_id", "area", "centroid_y", "centroid_x"])
    return pd.DataFrame(rows)


def build_iou_pairs(pred_labels, gt_labels):
    pred_ids = [int(x) for x in np.unique(pred_labels) if int(x) != 0]
    gt_ids = [int(x) for x in np.unique(gt_labels) if int(x) != 0]
    pred_areas = {label_id: int(np.count_nonzero(pred_labels == label_id)) for label_id in pred_ids}
    gt_areas = {label_id: int(np.count_nonzero(gt_labels == label_id)) for label_id in gt_ids}

    pairs = []
    for pred_id in pred_ids:
        pred_mask = pred_labels == pred_id
        overlapped_gt_ids = np.unique(gt_labels[pred_mask])
        overlapped_gt_ids = overlapped_gt_ids[overlapped_gt_ids != 0]
        for gt_id in overlapped_gt_ids:
            gt_id = int(gt_id)
            inter = int(np.count_nonzero(pred_mask & (gt_labels == gt_id)))
            if inter <= 0:
                continue
            union = pred_areas[pred_id] + gt_areas[gt_id] - inter
            pairs.append((pred_id, gt_id, inter / max(union, 1)))

    pairs.sort(key=lambda item: item[2], reverse=True)
    return pred_ids, gt_ids, pairs


def build_match_table(pred_labels, gt_labels, match_iou_threshold=0.1):
    pred_ids, gt_ids, pairs = build_iou_pairs(pred_labels, gt_labels)
    matched_pred = set()
    matched_gt = set()
    rows = []

    for pred_id, gt_id, iou in pairs:
        accepted = (
            iou >= float(match_iou_threshold)
            and pred_id not in matched_pred
            and gt_id not in matched_gt
        )
        if accepted:
            matched_pred.add(pred_id)
            matched_gt.add(gt_id)
        rows.append(
            {
                "pred_label_id": int(pred_id),
                "gt_label_id": int(gt_id),
                "iou": float(iou),
                "accepted": bool(accepted),
            }
        )

    for pred_id in pred_ids:
        if pred_id not in matched_pred:
            rows.append(
                {
                    "pred_label_id": int(pred_id),
                    "gt_label_id": -1,
                    "iou": np.nan,
                    "accepted": False,
                }
            )

    for gt_id in gt_ids:
        if gt_id not in matched_gt:
            rows.append(
                {
                    "pred_label_id": -1,
                    "gt_label_id": int(gt_id),
                    "iou": np.nan,
                    "accepted": False,
                }
            )

    return pd.DataFrame(rows)


def match_instances(pred_labels, gt_labels, match_iou_threshold=0.1):
    pred_ids, gt_ids, pairs = build_iou_pairs(pred_labels, gt_labels)
    matched_pred = set()
    matched_gt = set()
    tp = 0

    for pred_id, gt_id, iou in pairs:
        if iou < float(match_iou_threshold):
            continue
        if pred_id in matched_pred or gt_id in matched_gt:
            continue
        matched_pred.add(pred_id)
        matched_gt.add(gt_id)
        tp += 1

    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return tp, fp, fn, len(pred_ids), len(gt_ids)


def compute_detection_metrics(pred_labels, gt_labels, match_iou_threshold=0.1):
    tp, fp, fn, pred_count, gt_count = match_instances(
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        match_iou_threshold=match_iou_threshold,
    )
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    relative_count_error = abs(pred_count - gt_count) / max(gt_count, 1)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "relative_count_error": float(relative_count_error),
        "pred_count": int(pred_count),
        "gt_count": int(gt_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_pixel_metrics(pred_binary, gt_binary):
    pred_mask = np.asarray(pred_binary) > 0
    gt_mask = np.asarray(gt_binary) > 0

    tp = int(np.count_nonzero(pred_mask & gt_mask))
    fp = int(np.count_nonzero(pred_mask & (~gt_mask)))
    fn = int(np.count_nonzero((~pred_mask) & gt_mask))
    pred_positive_pixels = int(np.count_nonzero(pred_mask))
    gt_positive_pixels = int(np.count_nonzero(gt_mask))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pred_positive_pixels": int(pred_positive_pixels),
        "gt_positive_pixels": int(gt_positive_pixels),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def add_metric_prefix(metrics, prefix):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def build_long_metric_table(metric_df):
    rows = []
    metric_columns = [column for column in metric_df.columns if column not in ["sample", "model"]]

    for _, row in metric_df.iterrows():
        for column_name in metric_columns:
            analysis_type, metric_name = column_name.split("_", 1)
            rows.append(
                {
                    "sample": row["sample"],
                    "model": row["model"],
                    "analysis_type": analysis_type,
                    "metric": metric_name,
                    "value": row[column_name],
                }
            )
    return pd.DataFrame(rows)


def build_summary_metric_table(metric_df):
    rows = []
    metric_columns = [column for column in metric_df.columns if column not in ["sample", "model"]]

    for column_name in metric_columns:
        analysis_type, metric_name = column_name.split("_", 1)
        rows.append(
            {
                "analysis_type": analysis_type,
                "metric": metric_name,
                "mean_value": float(metric_df[column_name].mean()),
            }
        )
    return pd.DataFrame(rows)


def format_metric_value(metric_name, value):
    if metric_name in ["detection_tp", "detection_fp", "detection_fn", "detection_pred_count", "detection_gt_count"]:
        return f"{float(value):.1f}"
    if metric_name in ["detection_precision", "detection_recall", "detection_f1", "detection_relative_count_error",
                       "pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou"]:
        return f"{float(value):.3f}"
    return f"{float(value):.3f}"


def plot_metric_table(summary_df, metric_specs, title, save_path):
    if summary_df.empty:
        return

    header_color = "#6ea3d5"
    row_color_a = "#dce8f6"
    row_color_b = "#eef4fb"
    text_color = "#111111"
    best_color = "#1f7a3f"
    border_color = "white"

    display_df = summary_df.copy()
    display_df["model"] = display_df["model"].astype(str)

    metric_names = [metric_name for metric_name, _, _ in metric_specs]
    lower_is_better = {metric_name: is_lower_better for metric_name, _, is_lower_better in metric_specs}

    best_index_map = {}
    for metric_name in metric_names:
        values = display_df[metric_name].to_numpy(dtype=float)
        if lower_is_better[metric_name]:
            best_index_map[metric_name] = int(np.argmin(values))
        else:
            best_index_map[metric_name] = int(np.argmax(values))

    columns = ["Model"] + [display_name for _, display_name, _ in metric_specs]
    n_rows = len(display_df) + 1
    n_cols = len(columns)

    fig_width = max(8.5, 1.55 * n_cols)
    fig_height = 1.25 + 0.85 * n_rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 0.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        0,
        n_rows + 0.18,
        title,
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=text_color,
    )

    for col_index, column_name in enumerate(columns):
        rect = plt.Rectangle((col_index, n_rows - 1), 1, 1, facecolor=header_color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(
            col_index + 0.05,
            n_rows - 0.5,
            column_name,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

    for row_index, (_, row) in enumerate(display_df.iterrows()):
        y = n_rows - 2 - row_index
        row_color = row_color_a if row_index % 2 == 0 else row_color_b
        for col_index in range(n_cols):
            rect = plt.Rectangle((col_index, y), 1, 1, facecolor=row_color, edgecolor=border_color, linewidth=1.5)
            ax.add_patch(rect)

        ax.text(
            0.05,
            y + 0.5,
            row["model"],
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=text_color,
        )

        for metric_col_index, metric_name in enumerate(metric_names, start=1):
            value = row[metric_name]
            is_best = best_index_map[metric_name] == row_index
            ax.text(
                metric_col_index + 0.5,
                y + 0.5,
                format_metric_value(metric_name, value),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold" if is_best else "normal",
                color=best_color if is_best else text_color,
            )

    plt.tight_layout(pad=0.8)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_comparison_plots(summary_df, save_dir):
    detection_specs = [
        ("detection_tp", "TP", False),
        ("detection_fp", "FP", True),
        ("detection_fn", "FN", True),
        ("detection_precision", "Precision", False),
        ("detection_recall", "Recall", False),
        ("detection_f1", "F1", False),
    ]
    pixel_specs = [
        ("pixel_precision", "Pixel Precision", False),
        ("pixel_recall", "Pixel Recall", False),
        ("pixel_f1", "Pixel F1", False),
        ("pixel_iou", "Pixel IoU", False),
    ]
    plot_metric_table(
        summary_df=summary_df,
        metric_specs=detection_specs,
        title="NPC puncta detection comparison",
        save_path=os.path.join(save_dir, "model_comparison_detection_table.png"),
    )
    plot_metric_table(
        summary_df=summary_df,
        metric_specs=pixel_specs,
        title="NPC puncta pixel-wise comparison",
        save_path=os.path.join(save_dir, "model_comparison_pixel_table.png"),
    )


def gray_to_rgb_canvas(image):
    image_norm = normalize_to_unit(image)
    image_uint8 = np.uint8(np.clip(image_norm * 255, 0, 255))
    return np.repeat(image_uint8[:, :, np.newaxis], 3, axis=2)


def get_match_status_sets(match_table):
    if match_table.empty:
        return set(), set(), set(), set()

    accepted_rows = match_table[match_table["accepted"] == True]
    matched_pred_ids = {int(x) for x in accepted_rows["pred_label_id"].tolist() if int(x) > 0}
    matched_gt_ids = {int(x) for x in accepted_rows["gt_label_id"].tolist() if int(x) > 0}
    pred_ids = {int(x) for x in match_table["pred_label_id"].tolist() if int(x) > 0}
    gt_ids = {int(x) for x in match_table["gt_label_id"].tolist() if int(x) > 0}
    fp_pred_ids = pred_ids - matched_pred_ids
    fn_gt_ids = gt_ids - matched_gt_ids
    return matched_pred_ids, matched_gt_ids, fp_pred_ids, fn_gt_ids


def draw_label_set(canvas, label_map, label_ids, color, thickness=1, draw_centroid=False):
    output = np.asarray(canvas).copy()
    label_map = np.asarray(label_map, dtype=np.int32)

    for label_id in label_ids:
        mask = (label_map == int(label_id)).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(output, contours, -1, color, thickness)

        if draw_centroid:
            ys, xs = np.nonzero(mask)
            cy = int(round(float(np.mean(ys))))
            cx = int(round(float(np.mean(xs))))
            cv2.circle(output, (cx, cy), 3, color, 1)

    return output


def make_match_visualization(background_image, pred_labels, gt_labels, match_table):
    base_rgb = gray_to_rgb_canvas(background_image)
    matched_pred_ids, matched_gt_ids, fp_pred_ids, fn_gt_ids = get_match_status_sets(match_table)

    overlay = base_rgb.copy()
    overlay = draw_label_set(overlay, gt_labels, matched_gt_ids, color=(0, 255, 0), thickness=1, draw_centroid=False)
    overlay = draw_label_set(overlay, pred_labels, fp_pred_ids, color=(255, 0, 0), thickness=1, draw_centroid=True)
    overlay = draw_label_set(overlay, gt_labels, fn_gt_ids, color=(0, 0, 255), thickness=1, draw_centroid=True)
    return overlay


def save_sample_outputs(
        save_dir,
        sample_name,
        input_image,
        nucleus_mask,
        input_nucleus_masked,
        pred_npc,
        gt_npc,
        pred_npc_masked,
        gt_npc_masked,
        pred_binary,
        gt_binary,
        pred_labels,
        gt_labels,
        pred_components,
        gt_components,
        match_table,
        visualization_input,
        visualization_gt,
        visualization_output,
        pred_csv_path,
        gt_csv_path):
    sample_dir = os.path.join(save_dir, "samples", sample_name)
    check_existence_local(sample_dir)

    save_array(os.path.join(sample_dir, "input.tif"), np.asarray(input_image, dtype=np.float32))
    save_array(os.path.join(sample_dir, "nucleus_mask.tif"), np.asarray(nucleus_mask, dtype=np.uint8))
    save_array(os.path.join(sample_dir, "input_nucleus_masked.tif"), np.asarray(input_nucleus_masked, dtype=np.float32))
    save_array(os.path.join(sample_dir, "pred_npcs.tif"), np.asarray(pred_npc, dtype=np.float32))
    save_array(os.path.join(sample_dir, "gt_npcs.tif"), np.asarray(gt_npc, dtype=np.float32))
    save_array(os.path.join(sample_dir, "pred_npcs_nucleus_masked.tif"), np.asarray(pred_npc_masked, dtype=np.float32))
    save_array(os.path.join(sample_dir, "gt_npcs_nucleus_masked.tif"), np.asarray(gt_npc_masked, dtype=np.float32))
    save_array(os.path.join(sample_dir, "pred_binary.tif"), np.asarray(pred_binary, dtype=np.uint8))
    save_array(os.path.join(sample_dir, "gt_binary.tif"), np.asarray(gt_binary, dtype=np.uint8))
    save_array(os.path.join(sample_dir, "pred_labels.tif"), np.asarray(pred_labels, dtype=np.uint16))
    save_array(os.path.join(sample_dir, "gt_labels.tif"), np.asarray(gt_labels, dtype=np.uint16))

    pred_components.to_csv(os.path.join(sample_dir, "pred_components.csv"), index=False)
    gt_components.to_csv(os.path.join(sample_dir, "gt_components.csv"), index=False)
    match_table.to_csv(os.path.join(sample_dir, "matches.csv"), index=False)
    _, _, pred_spot_df = parse_rsfish_csv(pred_csv_path)
    _, _, gt_spot_df = parse_rsfish_csv(gt_csv_path)
    pred_spot_df.to_csv(os.path.join(sample_dir, "pred_rsfish.csv"), index=False)
    gt_spot_df.to_csv(os.path.join(sample_dir, "gt_rsfish.csv"), index=False)

    save_array(os.path.join(sample_dir, "match_visualization_input.tif"), np.asarray(visualization_input, dtype=np.uint8))
    save_array(os.path.join(sample_dir, "match_visualization_gt.tif"), np.asarray(visualization_gt, dtype=np.uint8))
    save_array(os.path.join(sample_dir, "match_visualization_output.tif"), np.asarray(visualization_output, dtype=np.uint8))


def evaluate_model_on_sample(
        net_main,
        sample_name,
        input_image,
        pred_tensor,
        gt_npc,
        nucleus_mask,
        npc_channel_index,
        model_save_dir,
        fiji_executable,
        rsfish_disk_radius,
        rsfish_timeout_sec,
        rsfish_base_work_dir,
        rsfish_anisotropy,
        rsfish_robust_fitting,
        rsfish_spot_intensity,
        rsfish_sigma_dog,
        rsfish_threshold_dog,
        rsfish_support_radius,
        rsfish_inlier_ratio,
        rsfish_max_error,
        rsfish_intensity_threshold,
        rsfish_background_subtraction,
        rsfish_background_max_error,
        rsfish_background_inlier_ratio,
        rsfish_use_multithreading,
        rsfish_num_threads,
        rsfish_block_size_x,
        rsfish_block_size_y,
        rsfish_block_size_z,
        match_iou_threshold):
    pred_npc = pred_tensor[0, npc_channel_index].detach().cpu().float().numpy()
    input_nucleus_masked = apply_binary_mask(input_image, nucleus_mask)
    pred_npc_masked = apply_binary_mask(pred_npc, nucleus_mask)
    gt_npc_masked = apply_binary_mask(gt_npc, nucleus_mask)

    pred_binary, pred_labels, pred_csv_path = segment_puncta_rsfish_fiji(
        image=pred_npc_masked,
        fiji_executable=fiji_executable,
        rsfish_disk_radius=rsfish_disk_radius,
        rsfish_timeout_sec=rsfish_timeout_sec,
        rsfish_work_dir=os.path.join(rsfish_base_work_dir, f"{sample_name}_pred"),
        rsfish_anisotropy=rsfish_anisotropy,
        rsfish_robust_fitting=rsfish_robust_fitting,
        rsfish_spot_intensity=rsfish_spot_intensity,
        rsfish_sigma_dog=rsfish_sigma_dog,
        rsfish_threshold_dog=rsfish_threshold_dog,
        rsfish_support_radius=rsfish_support_radius,
        rsfish_inlier_ratio=rsfish_inlier_ratio,
        rsfish_max_error=rsfish_max_error,
        rsfish_intensity_threshold=rsfish_intensity_threshold,
        rsfish_background_subtraction=rsfish_background_subtraction,
        rsfish_background_max_error=rsfish_background_max_error,
        rsfish_background_inlier_ratio=rsfish_background_inlier_ratio,
        rsfish_use_multithreading=rsfish_use_multithreading,
        rsfish_num_threads=rsfish_num_threads,
        rsfish_block_size_x=rsfish_block_size_x,
        rsfish_block_size_y=rsfish_block_size_y,
        rsfish_block_size_z=rsfish_block_size_z,
    )
    gt_binary, gt_labels, gt_csv_path = segment_puncta_rsfish_fiji(
        image=gt_npc_masked,
        fiji_executable=fiji_executable,
        rsfish_disk_radius=rsfish_disk_radius,
        rsfish_timeout_sec=rsfish_timeout_sec,
        rsfish_work_dir=os.path.join(rsfish_base_work_dir, f"{sample_name}_gt"),
        rsfish_anisotropy=rsfish_anisotropy,
        rsfish_robust_fitting=rsfish_robust_fitting,
        rsfish_spot_intensity=rsfish_spot_intensity,
        rsfish_sigma_dog=rsfish_sigma_dog,
        rsfish_threshold_dog=rsfish_threshold_dog,
        rsfish_support_radius=rsfish_support_radius,
        rsfish_inlier_ratio=rsfish_inlier_ratio,
        rsfish_max_error=rsfish_max_error,
        rsfish_intensity_threshold=rsfish_intensity_threshold,
        rsfish_background_subtraction=rsfish_background_subtraction,
        rsfish_background_max_error=rsfish_background_max_error,
        rsfish_background_inlier_ratio=rsfish_background_inlier_ratio,
        rsfish_use_multithreading=rsfish_use_multithreading,
        rsfish_num_threads=rsfish_num_threads,
        rsfish_block_size_x=rsfish_block_size_x,
        rsfish_block_size_y=rsfish_block_size_y,
        rsfish_block_size_z=rsfish_block_size_z,
    )

    pred_components = build_component_table(pred_labels)
    gt_components = build_component_table(gt_labels)
    detection_metrics = compute_detection_metrics(
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        match_iou_threshold=match_iou_threshold,
    )
    pixel_metrics = compute_pixel_metrics(
        pred_binary=pred_binary,
        gt_binary=gt_binary,
    )
    match_table = build_match_table(
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        match_iou_threshold=match_iou_threshold,
    )

    visualization_input = make_match_visualization(
        background_image=input_nucleus_masked,
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        match_table=match_table,
    )
    visualization_gt = make_match_visualization(
        background_image=gt_npc_masked,
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        match_table=match_table,
    )
    visualization_output = make_match_visualization(
        background_image=pred_npc_masked,
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        match_table=match_table,
    )

    save_sample_outputs(
        save_dir=model_save_dir,
        sample_name=sample_name,
        input_image=input_image,
        nucleus_mask=nucleus_mask,
        input_nucleus_masked=input_nucleus_masked,
        pred_npc=pred_npc,
        gt_npc=gt_npc,
        pred_npc_masked=pred_npc_masked,
        gt_npc_masked=gt_npc_masked,
        pred_binary=pred_binary,
        gt_binary=gt_binary,
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        pred_components=pred_components,
        gt_components=gt_components,
        match_table=match_table,
        visualization_input=visualization_input,
        visualization_gt=visualization_gt,
        visualization_output=visualization_output,
        pred_csv_path=pred_csv_path,
        gt_csv_path=gt_csv_path,
    )

    return detection_metrics, pixel_metrics


def run_puncta_analysis(
        model_dir,
        size,
        num_workers,
        gpu_rank="0",
        device_name="cuda:0",
        degradation_resolution=280,
        noise_level=0.280,
        average=1,
        num_file_eval=10,
        category=None,
        factor_list=None,
        npc_channel_index=0,
        use_nucleus_mask=True,
        compare_model_dir="",
        compare_model_category=None,
        compare_model_factor_list=None,
        primary_model_label="desm_decoupled",
        compare_model_label="desm_npcs_only",
        fiji_executable=DEFAULT_FIJI_EXECUTABLE,
        rsfish_disk_radius=3,
        rsfish_timeout_sec=300,
        rsfish_work_dir=None,
        rsfish_anisotropy=1.0,
        rsfish_robust_fitting="RANSAC",
        rsfish_spot_intensity="Linear Interpolation",
        rsfish_sigma_dog=1.5,
        rsfish_threshold_dog=0.007,
        rsfish_support_radius=3,
        rsfish_inlier_ratio=0.1,
        rsfish_max_error=1.5,
        rsfish_intensity_threshold=0.0,
        rsfish_background_subtraction="No background subtraction",
        rsfish_background_max_error=0.05,
        rsfish_background_inlier_ratio=0.75,
        rsfish_use_multithreading=False,
        rsfish_num_threads=1,
        rsfish_block_size_x=128,
        rsfish_block_size_y=128,
        rsfish_block_size_z=16,
        match_iou_threshold=0.1,
        save_dir=None):
    if category is None:
        category = ["Micro", "NPCs"]
    if factor_list is None:
        factor_list = [1, 1]
    if compare_model_category is None:
        compare_model_category = ["NPCs"]
    if compare_model_factor_list is None:
        compare_model_factor_list = [1]
    if save_dir is None or save_dir == "":
        save_dir = os.path.join(SCRIPT_DIR, "puncta_analysis_results")
    if not fiji_executable:
        raise ValueError("fiji_executable must be provided.")

    check_existence_local(save_dir)
    primary_save_dir = os.path.join(save_dir, primary_model_label)
    check_existence_local(primary_save_dir)
    check_existence_local(os.path.join(primary_save_dir, "samples"))
    primary_rsfish_base_work_dir = rsfish_work_dir or os.path.join(primary_save_dir, "rsfish_work")
    check_existence_local(primary_rsfish_base_work_dir)

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
    compare_net_main = None
    compare_rsfish_base_work_dir = None
    compare_save_dir = None
    if compare_model_dir:
        compare_opt = make_runtime_opt(
            model_dir=compare_model_dir,
            size=size,
            num_workers=num_workers,
            gpu_rank=gpu_rank,
            device_name=device_name,
            num_file_eval=num_file_eval,
            category=compare_model_category,
            factor_list=compare_model_factor_list,
        )
        compare_net_main = build_model(compare_opt, device)
        compare_save_dir = os.path.join(save_dir, compare_model_label)
        check_existence_local(compare_save_dir)
        check_existence_local(os.path.join(compare_save_dir, "samples"))
        compare_rsfish_base_work_dir = os.path.join(compare_save_dir, "rsfish_work")
        check_existence_local(compare_rsfish_base_work_dir)

    val_loader = gen_degradation_dataloader(
        GT_tag_list=category,
        noise_level=noise_level,
        factor_list=factor_list,
        SR_resolution_dict=opt["raw_data_resolution"],
        target_resolution=degradation_resolution,
        degradation_method="blur-composite-noise",
        average=average,
        generate_lifetime_flag=False,
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

    print("======================== puncta analysis ========================")
    metric_rows = []

    with torch.no_grad():
        bar = tqdm(total=num_file_eval)
        for batch_index, data in enumerate(val_loader):
            if batch_index >= num_file_eval:
                break

            Input, GT_DS, GT_D, _, _, _ = data
            pred = net_main.feed_data(Input=Input, GT=GT_DS)
            compare_pred = None
            if compare_net_main is not None:
                compare_pred = compare_net_main.feed_data(Input=GT_D[:,0:1,:,:], GT=GT_DS)

            input_image = Input[0, 0].detach().cpu().float().numpy()
            compare_input_image = GT_D[0, 0].detach().cpu().float().numpy()
            gt_npc = GT_DS[0, npc_channel_index].detach().cpu().float().numpy()
            sample_name = f"synthetic_{batch_index + 1:04d}"

            if use_nucleus_mask:
                nucleus_mask = estimate_nucleus_mask(gt_npc)
            else:
                nucleus_mask = np.ones_like(gt_npc, dtype=np.uint8)

            detection_metrics, pixel_metrics = evaluate_model_on_sample(
                net_main=net_main,
                sample_name=sample_name,
                input_image=input_image,
                pred_tensor=pred,
                gt_npc=gt_npc,
                nucleus_mask=nucleus_mask,
                npc_channel_index=0,
                model_save_dir=primary_save_dir,
                fiji_executable=fiji_executable,
                rsfish_disk_radius=rsfish_disk_radius,
                rsfish_timeout_sec=rsfish_timeout_sec,
                rsfish_base_work_dir=primary_rsfish_base_work_dir,
                rsfish_anisotropy=rsfish_anisotropy,
                rsfish_robust_fitting=rsfish_robust_fitting,
                rsfish_spot_intensity=rsfish_spot_intensity,
                rsfish_sigma_dog=rsfish_sigma_dog,
                rsfish_threshold_dog=rsfish_threshold_dog,
                rsfish_support_radius=rsfish_support_radius,
                rsfish_inlier_ratio=rsfish_inlier_ratio,
                rsfish_max_error=rsfish_max_error,
                rsfish_intensity_threshold=rsfish_intensity_threshold,
                rsfish_background_subtraction=rsfish_background_subtraction,
                rsfish_background_max_error=rsfish_background_max_error,
                rsfish_background_inlier_ratio=rsfish_background_inlier_ratio,
                rsfish_use_multithreading=rsfish_use_multithreading,
                rsfish_num_threads=rsfish_num_threads,
                rsfish_block_size_x=rsfish_block_size_x,
                rsfish_block_size_y=rsfish_block_size_y,
                rsfish_block_size_z=rsfish_block_size_z,
                match_iou_threshold=match_iou_threshold,
            )

            sample_metrics = {"sample": sample_name, "model": primary_model_label}
            sample_metrics.update(add_metric_prefix(detection_metrics, "detection"))
            sample_metrics.update(add_metric_prefix(pixel_metrics, "pixel"))
            metric_rows.append(sample_metrics)

            print(
                f"{primary_model_label} | {sample_name}: "
                f"detection_f1={detection_metrics['f1']:.4f}, "
                f"pixel_f1={pixel_metrics['f1']:.4f}"
            )

            if compare_pred is not None:
                compare_detection_metrics, compare_pixel_metrics = evaluate_model_on_sample(
                    net_main=compare_net_main,
                    sample_name=sample_name,
                    input_image=compare_input_image,
                    pred_tensor=compare_pred,
                    gt_npc=gt_npc,
                    nucleus_mask=nucleus_mask,
                    npc_channel_index=0,
                    model_save_dir=compare_save_dir,
                    fiji_executable=fiji_executable,
                    rsfish_disk_radius=rsfish_disk_radius,
                    rsfish_timeout_sec=rsfish_timeout_sec,
                    rsfish_base_work_dir=compare_rsfish_base_work_dir,
                    rsfish_anisotropy=rsfish_anisotropy,
                    rsfish_robust_fitting=rsfish_robust_fitting,
                    rsfish_spot_intensity=rsfish_spot_intensity,
                    rsfish_sigma_dog=rsfish_sigma_dog,
                    rsfish_threshold_dog=rsfish_threshold_dog,
                    rsfish_support_radius=rsfish_support_radius,
                    rsfish_inlier_ratio=rsfish_inlier_ratio,
                    rsfish_max_error=rsfish_max_error,
                    rsfish_intensity_threshold=rsfish_intensity_threshold,
                    rsfish_background_subtraction=rsfish_background_subtraction,
                    rsfish_background_max_error=rsfish_background_max_error,
                    rsfish_background_inlier_ratio=rsfish_background_inlier_ratio,
                    rsfish_use_multithreading=rsfish_use_multithreading,
                    rsfish_num_threads=rsfish_num_threads,
                    rsfish_block_size_x=rsfish_block_size_x,
                    rsfish_block_size_y=rsfish_block_size_y,
                    rsfish_block_size_z=rsfish_block_size_z,
                    match_iou_threshold=match_iou_threshold,
                )
                compare_sample_metrics = {"sample": sample_name, "model": compare_model_label}
                compare_sample_metrics.update(add_metric_prefix(compare_detection_metrics, "detection"))
                compare_sample_metrics.update(add_metric_prefix(compare_pixel_metrics, "pixel"))
                metric_rows.append(compare_sample_metrics)
                print(
                    f"{compare_model_label} | {sample_name}: "
                    f"detection_f1={compare_detection_metrics['f1']:.4f}, "
                    f"pixel_f1={compare_pixel_metrics['f1']:.4f}"
                )
            bar.update(1)
        bar.close()

    metric_df = pd.DataFrame(metric_rows)
    metric_df = metric_df[
        [
            "sample",
            "model",
            "detection_precision",
            "detection_recall",
            "detection_f1",
            "detection_relative_count_error",
            "detection_pred_count",
            "detection_gt_count",
            "detection_tp",
            "detection_fp",
            "detection_fn",
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
            "pixel_iou",
            "pixel_pred_positive_pixels",
            "pixel_gt_positive_pixels",
            "pixel_tp",
            "pixel_fp",
            "pixel_fn",
        ]
    ]
    metric_df_long = build_long_metric_table(metric_df)
    summary_df = metric_df.groupby("model", as_index=False).mean(numeric_only=True)
    if compare_net_main is not None:
        comparison_df = metric_df.pivot(index="sample", columns="model")
        comparison_df.columns = [f"{metric}_{model}" for metric, model in comparison_df.columns]
        comparison_df = comparison_df.reset_index()
        comparison_df.to_csv(os.path.join(save_dir, "puncta_metrics_model_comparison.csv"), index=False)

    metric_df.to_csv(os.path.join(save_dir, "puncta_metrics_per_sample_wide.csv"), index=False)
    metric_df_long.to_csv(os.path.join(save_dir, "puncta_metrics_per_sample.csv"), index=False)
    summary_df.to_csv(os.path.join(save_dir, "puncta_metrics_summary.csv"), index=False)
    save_comparison_plots(summary_df=summary_df, save_dir=save_dir)

    print("======================== puncta summary ========================")
    print(summary_df.to_string(index=False))
    return metric_df, summary_df


if __name__ == "__main__":
    model_dir = os.path.join(DESM_ROOT, "validation", "YOUR_DESM_MODEL_DIR", "weights", "1", "main_G.pth")
    save_dir = os.path.join(SCRIPT_DIR, "puncta_analysis_results")

    size = 384
    num_workers = 2
    gpu_rank = "0"
    device_name = "cuda:0"
    degradation_resolution = 225
    noise_level = 0
    average = 1
    num_file_eval = 100
    category = ["NPCs", "Micro"]
    factor_list = [1, 1]

    npc_channel_index = 0
    use_nucleus_mask = True
    compare_model_dir = os.path.join(
        DESM_ROOT,
        "validation",
        "YOUR_NPCS_ONLY_MODEL_DIR",
        "weights",
        "1",
        "main_G.pth",
    )
    compare_model_category = ["NPCs"]
    compare_model_factor_list = [1]
    primary_model_label = "desm_decoupled"
    compare_model_label = "desm_npcs_only"
    fiji_executable = DEFAULT_FIJI_EXECUTABLE
    rsfish_disk_radius = 3
    rsfish_timeout_sec = 300
    rsfish_work_dir = r""
    rsfish_anisotropy = 1.0
    rsfish_robust_fitting = "RANSAC"
    rsfish_spot_intensity = "Linear Interpolation"
    rsfish_sigma_dog = 1.5
    rsfish_threshold_dog = 0.007
    rsfish_support_radius = 3
    rsfish_inlier_ratio = 0.1
    rsfish_max_error = 1.5
    rsfish_intensity_threshold = 0.0
    rsfish_background_subtraction = "No background subtraction"
    rsfish_background_max_error = 0.05
    rsfish_background_inlier_ratio = 0.75
    rsfish_use_multithreading = False
    rsfish_num_threads = 1
    rsfish_block_size_x = 128
    rsfish_block_size_y = 128
    rsfish_block_size_z = 16
    match_iou_threshold = 0.1

    run_puncta_analysis(
        model_dir=model_dir,
        size=size,
        num_workers=num_workers,
        gpu_rank=gpu_rank,
        device_name=device_name,
        degradation_resolution=degradation_resolution,
        noise_level=noise_level,
        average=average,
        num_file_eval=num_file_eval,
        category=category,
        factor_list=factor_list,
        npc_channel_index=npc_channel_index,
        use_nucleus_mask=use_nucleus_mask,
        compare_model_dir=compare_model_dir,
        compare_model_category=compare_model_category,
        compare_model_factor_list=compare_model_factor_list,
        primary_model_label=primary_model_label,
        compare_model_label=compare_model_label,
        fiji_executable=fiji_executable,
        rsfish_disk_radius=rsfish_disk_radius,
        rsfish_timeout_sec=rsfish_timeout_sec,
        rsfish_work_dir=rsfish_work_dir,
        rsfish_anisotropy=rsfish_anisotropy,
        rsfish_robust_fitting=rsfish_robust_fitting,
        rsfish_spot_intensity=rsfish_spot_intensity,
        rsfish_sigma_dog=rsfish_sigma_dog,
        rsfish_threshold_dog=rsfish_threshold_dog,
        rsfish_support_radius=rsfish_support_radius,
        rsfish_inlier_ratio=rsfish_inlier_ratio,
        rsfish_max_error=rsfish_max_error,
        rsfish_intensity_threshold=rsfish_intensity_threshold,
        rsfish_background_subtraction=rsfish_background_subtraction,
        rsfish_background_max_error=rsfish_background_max_error,
        rsfish_background_inlier_ratio=rsfish_background_inlier_ratio,
        rsfish_use_multithreading=rsfish_use_multithreading,
        rsfish_num_threads=rsfish_num_threads,
        rsfish_block_size_x=rsfish_block_size_x,
        rsfish_block_size_y=rsfish_block_size_y,
        rsfish_block_size_z=rsfish_block_size_z,
        match_iou_threshold=match_iou_threshold,
        save_dir=save_dir,
    )
