import os
import sys

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import cv2
import natsort
import numpy as np
import tifffile
import torch
from copy import deepcopy
from matplotlib.colors import hsv_to_rgb
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from torchvision import transforms
from tqdm import tqdm


DESM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if DESM_ROOT not in sys.path:
    sys.path.append(DESM_ROOT)

from utils import check_existence, resize, to_cpu


size_list = [256, 512, 1024]
zoom_list = [
    "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11",
    "z12", "z13", "z14", "z15", "z16", "z18", "z22", "z25", "z28",
    "z38", "z48", "z60", "z90", "z150", "z225",
]
convert_table = [
    [664, 332, 166.0, "nm/pixel"],
    [320, 160, 80.0, "nm/pixel"],
    [210, 105, 52.0, "nm/pixel"],
    [164, 82, 41.0, "nm/pixel"],
    [125, 62, 31.0, "nm/pixel"],
    [105, 53, 26.0, "nm/pixel"],
    [94, 47, 23.0, "nm/pixel"],
    [78, 39, 20.0, "nm/pixel"],
    [70, 35, 18.0, "nm/pixel"],
    [63, 31, 16.0, "nm/pixel"],
    [59, 29, 15.0, "nm/pixel"],
    [55, 27, 14.0, "nm/pixel"],
    [51, 25, 13.0, "nm/pixel"],
    [47, 23, 12.0, "nm/pixel"],
    [43, 21, 11.0, "nm/pixel"],
    [39, 20, 10.0, "nm/pixel"],
    [35, 18, 9.0, "nm/pixel"],
    [31, 16, 8.0, "nm/pixel"],
    [27, 14, 7.0, "nm/pixel"],
    [23, 12, 6.0, "nm/pixel"],
    [20, 10, 5.0, "nm/pixel"],
    [16, 8, 4.0, "nm/pixel"],
    [12, 6, 3.0, "nm/pixel"],
    [8, 4, 2.0, "nm/pixel"],
    [4, 3, 1.5, "nm/pixel"],
    [1, 2, 1.0, "nm/pixel"],
]


def norm_statistic(Input, device, std=None):
    mean = torch.mean(Input).to(device)
    mean_zero = torch.zeros_like(mean).to(device)
    std = torch.std(Input).to(device) if std is None else std
    output = transforms.Normalize(mean_zero, std)(Input)
    return output, mean_zero, std


def gen_FLIM(tm_data, intensity_image, t_range=(0, 6000)):
    tm_data = tm_data.copy()
    tm_data[tm_data < t_range[0]] = t_range[0]
    tm_data[tm_data > t_range[1]] = t_range[1]
    intensity_image = np.expand_dims(intensity_image, axis=-1)
    intensity_image = np.repeat(intensity_image, axis=-1, repeats=3)
    tm_DATA = (tm_data - np.min(tm_data)) / (np.max(tm_data) - np.min(tm_data))
    in_DATA = (intensity_image - np.min(intensity_image)) / (np.max(intensity_image) - np.min(intensity_image))
    hue_channel = 2 * tm_DATA / 3
    value_channel = in_DATA

    h, w = intensity_image.shape[0:2]
    hsv_img = np.zeros((h, w, 3))
    hsv_img[:, :, 0] = hue_channel
    hsv_img[:, :, 1] = 1
    hsv_img[:, :, 2] = value_channel[:, :, 0]

    rgb_color = hsv_to_rgb(hsv_img)
    return np.uint16(rgb_color * 255)


def make_threshold_otsu(Input):
    thresh = threshold_otsu(Input)
    Output = deepcopy(Input)
    Output[Output < thresh] = 0
    Output[Output >= thresh] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    Output = cv2.dilate(
        Output,
        kernel,
        anchor=None,
        iterations=3,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return Output


def label_to_uint8(label_img):
    label_img = np.asarray(label_img, dtype=np.float32)
    max_value = float(np.max(label_img))
    if max_value <= 0:
        return np.zeros_like(label_img, dtype=np.uint8)
    return np.uint8(label_img / max_value * 255)


def relabel_background_to_zero(clustered_input, IO_prior_background):
    clustered = np.asarray(clustered_input)
    bg_mask = np.asarray(IO_prior_background) > 0

    labels = [int(x) for x in np.unique(clustered)]
    masks = {label: (clustered == label) for label in labels}
    score_table = []

    for label in labels:
        mask_l = masks[label]
        area_l = np.count_nonzero(mask_l)
        overlap_bg = np.count_nonzero(mask_l & bg_mask)
        purity_bg = overlap_bg / (area_l + 1e-8)
        score_table.append({
            "label": label,
            "overlap_bg": overlap_bg,
            "purity_bg": purity_bg,
        })

    score_table.sort(key=lambda item: (item["purity_bg"], item["overlap_bg"]), reverse=True)
    bg_label = int(score_table[0]["label"])

    relabeled = np.zeros_like(clustered, dtype=np.int32)
    next_id = 1
    for label in labels:
        if label == bg_label:
            continue
        relabeled[masks[label]] = next_id
        next_id += 1
    return relabeled


def relabel_foreground_by_priors(clustered_input_bg0, IO_prior_list):
    clustered = np.asarray(clustered_input_bg0)
    priors = [(np.asarray(p) > 0) for p in IO_prior_list]

    fixed = np.zeros_like(clustered, dtype=np.int32)
    labels = [int(x) for x in np.unique(clustered) if int(x) != 0]
    masks = {label: (clustered == label) for label in labels}
    unused = labels.copy()

    for new_id, prior_mask in enumerate(priors, start=1):
        if not unused:
            break

        current_scores = []
        for label in unused:
            mask_l = masks[label]
            inter = np.count_nonzero(mask_l & prior_mask)
            union = np.count_nonzero(mask_l | prior_mask)
            iou = inter / (union + 1e-8)
            current_scores.append((label, iou, inter))

        current_scores.sort(key=lambda item: (item[1], item[2]), reverse=True)
        best_old = int(current_scores[0][0])
        fixed[masks[best_old]] = new_id
        unused.remove(best_old)

    next_id = len(priors) + 1
    for label in unused:
        fixed[masks[label]] = next_id
        next_id += 1
    return fixed


def clean_labels_fill_bg_then_smooth(lbl_in, ksize=5, open_iter=2, close_iter=2, propagate_iters=100):
    lbl = np.asarray(lbl_in)
    out = lbl.copy().astype(np.int32)
    labels = [int(x) for x in np.unique(out) if int(x) != 0]
    if not labels:
        return out

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    fg_ref = out != 0
    smooth = np.zeros_like(out, dtype=np.int32)

    for label in labels:
        mask = (out == label).astype(np.uint8)
        if open_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        if close_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        smooth[mask > 0] = label

    missing = fg_ref & (smooth == 0)
    if np.any(missing):
        for _ in range(propagate_iters):
            if not np.any(missing):
                break
            filled_any = False
            for label in labels:
                dil = cv2.dilate((smooth == label).astype(np.uint8), kernel, iterations=1).astype(bool)
                cand = missing & dil
                if np.any(cand):
                    smooth[cand] = label
                    missing[cand] = False
                    filled_any = True
            if not filled_any:
                break
        if np.any(missing):
            smooth[missing] = out[missing]
    return smooth


def DESM_IL_inference(
        read_dir,
        read_dir_asc,
        weights_dir,
        prior_weights_dir,
        save_dir,
        resize_to_const=False,
        tm_min=0,
        tm_max=5500,
        org_list=None,
        n_clusters=None,
        device="cuda"):
    if org_list is None:
        org_list = ["Micro", "Mito", "Lyso"]
    if n_clusters is None:
        n_clusters = len(org_list) + 1

    read_dir = os.path.join(DESM_ROOT, read_dir)
    read_dir_asc = os.path.join(DESM_ROOT, read_dir_asc)
    weights_dir = os.path.join(DESM_ROOT, weights_dir)
    prior_weights_dir = os.path.join(DESM_ROOT, prior_weights_dir)
    save_dir = os.path.join(DESM_ROOT, save_dir)

    file_list = natsort.natsorted(os.listdir(read_dir))
    asc_list = natsort.natsorted(os.listdir(read_dir_asc))
    check_existence(save_dir)

    model = torch.load(weights_dir, weights_only=False)
    prior_model = torch.load(prior_weights_dir, weights_only=False)
    model.eval()
    prior_model.eval()

    bar = tqdm(total=len(file_list))
    with torch.no_grad():
        for file_index, file_name in enumerate(file_list):
            read_dir_file = os.path.join(read_dir, file_name)
            read_dir_ASC = os.path.join(read_dir_asc, asc_list[file_index])

            zoom = ""
            for item in zoom_list:
                if item in read_dir_file:
                    zoom = item
                    break

            img = tifffile.imread(read_dir_file)
            tm_data = np.loadtxt(read_dir_ASC)

            raw_h, raw_w = img.shape
            for zoom_index, zoom_name in enumerate(zoom_list):
                if zoom == zoom_name:
                    for size_index, size_value in enumerate(size_list):
                        if raw_h == size_value:
                            convert_ratio = convert_table[zoom_index][size_index] / 20
                            resized_h = int(raw_h * convert_ratio)
                            resized_w = int(raw_w * convert_ratio)
                            img = resize(img, (resized_h, resized_w))
                            tm_data = resize(tm_data, (resized_h, resized_w))
                            h_slice = slice(int(0.05 * resized_h), int(0.95 * resized_h))
                            w_slice = slice(int(0.05 * resized_w), int(0.95 * resized_w))
                            img = img[h_slice, w_slice]
                            tm_data = tm_data[h_slice, w_slice]
                            break

            FLIM_image = gen_FLIM(tm_data=tm_data, intensity_image=img, t_range=(tm_min, tm_max))

            vector = tm_data.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(vector)
            clustered_image = kmeans.labels_.reshape(tm_data.shape).astype(np.uint8)
            tifffile.imwrite(
                os.path.join(save_dir, f"{file_index}_clustered_original.tif"),
                label_to_uint8(clustered_image),
            )

            h, w = img.shape
            size = min(h, w)
            upper_limit = 1440
            if size > upper_limit:
                num_row = (h // upper_limit) + 1
                num_col = (w // upper_limit) + 1
                size = upper_limit
            else:
                upper_limit = (size // 16) * 16
                num_row = h // upper_limit
                num_col = w // upper_limit
                size = upper_limit

            pred_list = [[] for _ in range(num_row)]
            row_cood_list = []
            col_cood_list = []
            for row in range(num_row):
                row_cood_list.append([row * size, (row + 1) * size] if (row + 1) * size < h else [h - size, h])
            for col in range(num_col):
                col_cood_list.append([col * size, (col + 1) * size] if (col + 1) * size < w else [w - size, w])

            for row in range(num_row):
                for col in range(num_col):
                    intensity_input = img[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                    clustered_input = clustered_image[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]

                    intensity_input = torch.tensor(np.float64(intensity_input), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                    intensity_input, mean, std = norm_statistic(intensity_input, device)

                    IO_output = prior_model(intensity_input)
                    IO_prior = np.zeros_like(clustered_input)
                    IO_prior_list = []
                    for org_index in range(len(org_list)):
                        IO_per_c = IO_output[:, org_index:org_index + 1, :, :].squeeze(0).squeeze(0).cpu().numpy()
                        tifffile.imwrite(
                            os.path.join(save_dir, f"{file_index}_{org_index}_IO_output.tif"),
                            np.asarray(IO_per_c, dtype=np.float32),
                        )
                        IO_per_c_thresholded = make_threshold_otsu(IO_per_c)
                        tifffile.imwrite(
                            os.path.join(save_dir, f"{file_index}_{org_index}_IO_output_thresholded.tif"),
                            np.asarray(IO_per_c_thresholded, dtype=np.uint8),
                        )
                        IO_prior[IO_per_c_thresholded != 0] = 1
                        IO_prior_list.append(IO_per_c_thresholded)

                    IO_prior_background = np.zeros_like(IO_prior)
                    IO_prior_background[IO_prior == 0] = 1
                    tifffile.imwrite(
                        os.path.join(save_dir, f"{file_index}_IO_prior.tif"),
                        np.asarray(IO_prior, dtype=np.uint8),
                    )

                    if len(org_list) == 3:
                        IO_prior_list = [IO_prior_list[0] + IO_prior_list[2], IO_prior_list[1]]
                    elif len(org_list) == 2:
                        IO_prior_list = [IO_prior_list[0], IO_prior_list[1]]

                    clustered_bg0 = relabel_background_to_zero(clustered_input, IO_prior_background)
                    permuted_clustered_input = relabel_foreground_by_priors(clustered_bg0, IO_prior_list)
                    permuted_clustered_input = IO_prior * permuted_clustered_input
                    tifffile.imwrite(
                        os.path.join(save_dir, f"{file_index}_clustered_permuted.tif"),
                        label_to_uint8(permuted_clustered_input),
                    )
                    permuted_clustered_input = clean_labels_fill_bg_then_smooth(
                        permuted_clustered_input,
                        ksize=5,
                        open_iter=2,
                        close_iter=2,
                    )
                    tifffile.imwrite(
                        os.path.join(save_dir, f"{file_index}_clustered_image_final.tif"),
                        label_to_uint8(permuted_clustered_input),
                    )

                    clustered_input_tensor = torch.tensor(
                        np.float64(permuted_clustered_input),
                        dtype=torch.float,
                        device=device,
                    ).unsqueeze(0).unsqueeze(0)
                    clustered_input_tensor, _, _ = norm_statistic(clustered_input_tensor, device=device)
                    model_input = torch.concat([intensity_input, clustered_input_tensor], dim=1)

                    pred = model(model_input)
                    pred = pred * std + mean
                    pred[pred < 0] = 0
                    pred_list[row].append(to_cpu(pred.squeeze(0).permute(1, 2, 0)))

            full_pred = np.zeros((h, w, pred_list[0][0].shape[-1]))
            for row in range(num_row):
                for col in range(num_col):
                    full_pred[
                        row_cood_list[row][0]:row_cood_list[row][1],
                        col_cood_list[col][0]:col_cood_list[col][1],
                    ] = pred_list[row][col]

            img_to_save = np.uint8(img / np.max(img) * 255)
            if resize_to_const:
                img_to_save = resize(img_to_save, (1024, 1024))
                full_pred = resize(full_pred, (1024, 1024, full_pred.shape[-1]))
                FLIM_image = resize(FLIM_image, (1024, 1024))
                tm_data = resize(tm_data, (1024, 1024))

            tifffile.imwrite(os.path.join(save_dir, f"{file_index}_intensity_input.tif"), img_to_save)
            tifffile.imwrite(os.path.join(save_dir, f"{file_index}_FLIM_image.tif"), FLIM_image)
            tifffile.imwrite(os.path.join(save_dir, f"{file_index}_tm_data.tif"), tm_data)

            full_pred_uint8 = np.uint8(full_pred / np.max(full_pred) * 255)
            for org_index in range(full_pred_uint8.shape[-1]):
                tifffile.imwrite(
                    os.path.join(save_dir, f"{file_index}_{org_index}_SR.tif"),
                    full_pred_uint8[:, :, org_index],
                )

            SR_image = np.zeros_like(img_to_save, dtype=np.uint16)
            for org_index in range(full_pred_uint8.shape[-1]):
                SR_image += full_pred_uint8[:, :, org_index]
            tifffile.imwrite(os.path.join(save_dir, f"{file_index}_SR_only.tif"), SR_image)

            bar.set_description(f"{zoom}")
            bar.update(1)
    bar.close()


    # 第一个模型
if __name__ == "__main__":
    DESM_IL_inference(
        read_dir=r"eval_tools\Experimental_ablation\DESM_IL_Mitochondria_Lysosomes\Intensity",
        read_dir_asc=r"eval_tools\Experimental_ablation\DESM_IL_Mitochondria_Lysosomes\ASC",
        weights_dir=r"eval_tools\Experimental_ablation\weights\DESM_IL_Mito_Lyso.pth",
        prior_weights_dir=r"eval_tools\Experimental_ablation\weights\DESM_IO_Prior_Mito_Lyso.pth",
        save_dir=r"eval_tools\Experimental_ablation\DESM_IL_Mitochondria_Lysosomes\Results_random_overlap",
        resize_to_const=False,
        tm_min=0,
        tm_max=5500,
        org_list=["Mito", "Lyso"],
        n_clusters=3,
        device="cuda",
    )

    DESM_IL_inference(
        read_dir=r"eval_tools\Experimental_ablation\DESM_IL_Mitochondria_Lysosomes\Intensity",
        read_dir_asc=r"eval_tools\Experimental_ablation\DESM_IL_Mitochondria_Lysosomes\ASC",
        weights_dir=os.path.join(
            DESM_ROOT,
            "validation",
            "YOUR_EXTRA_OVERLAP_MODEL_DIR",
            "weights",
            "1",
            "main_G.pth",
        ),
        prior_weights_dir=r"eval_tools\Experimental_ablation\weights\DESM_IO_Prior_Mito_Lyso.pth",
        save_dir=r"eval_tools\Experimental_ablation\DESM_IL_Mitochondria_Lysosomes\Results_extra_overlap_class",
        resize_to_const=False,
        tm_min=0,
        tm_max=5500,
        org_list=["Mito", "Lyso"],
        n_clusters=4,
        device="cuda",
    )
    # 第二个模型
