import os
import sys
import cv2
import copy
import tifffile
import numpy as np
import torch

from random import randint
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm
from skimage.filters import threshold_otsu

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from loss.SSIM_loss import SSIM


# =========================
# 基础绘图与形变函数
# =========================

def rand_lines(img_shape, num_lines, min_len=50, max_len=150, max_tries=1000):
    H, W = img_shape
    lines = []
    tries = 0
    while len(lines) < num_lines and tries < max_tries:
        x1, y1 = np.random.randint(W), np.random.randint(H)
        angle = np.random.rand() * 2 * np.pi
        length = np.random.randint(min_len, max_len + 1)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        if 0 <= x2 < W and 0 <= y2 < H:
            lines.append(((x1, y1), (x2, y2)))
        tries += 1
    return lines


def draw_lines(img_shape, lines, int_range, width=1):
    intensity = randint(*int_range)
    img = np.zeros(img_shape, dtype=np.uint8)
    for p0, p1 in lines:
        cv2.line(img, p0, p1, color=intensity, thickness=width)
    return img


def draw_ellipse_like_lines_fixed(img_shape, lines, alpha, int_range, max_thickness=8, min_thickness=2):
    """
    alpha = 0 时是椭圆，alpha = 1 时是线段，中间为过渡状态
    使用传入的 lines 位置绘图
    """
    H, W = img_shape
    img = np.zeros(img_shape, dtype=np.uint8)
    intensity = randint(*int_range)

    for p0, p1 in lines:
        x1, y1 = p0
        x2, y2 = p1

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))

        major = int(np.hypot(dx, dy))
        minor = int(max_thickness * (1 - alpha)) + min_thickness

        if alpha < 1.0:
            cv2.ellipse(img, (cx, cy), (major // 2, minor), angle, 0, 360, intensity, -1)
        else:
            pt1 = (np.clip(x1, 0, W - 1), np.clip(y1, 0, H - 1))
            pt2 = (np.clip(x2, 0, W - 1), np.clip(y2, 0, H - 1))
            cv2.line(img, pt1, pt2, color=intensity, thickness=2)

    return img


def draw_ellipse_like_lines(img_shape, num_shapes, alpha, int_range, min_len=12, max_len=15, max_thickness=8, min_thickness=2):
    """
    alpha = 0 时是椭圆，alpha = 1 时是线段，中间为过渡状态
    """
    H, W = img_shape
    img = np.zeros(img_shape, dtype=np.uint8)
    intensity = randint(*int_range)

    for _ in range(num_shapes):
        cx, cy = np.random.randint(W), np.random.randint(H)
        angle = np.random.rand() * 360

        major = np.random.randint(min_len, max_len + 1)
        minor = int(max_thickness * (1 - alpha)) + min_thickness

        if alpha < 1.0:
            cv2.ellipse(img, (cx, cy), (major // 2, minor), angle, 0, 360, intensity, -1)
        else:
            angle_rad = np.deg2rad(angle)
            dx = int((major / 2) * np.cos(angle_rad))
            dy = int((major / 2) * np.sin(angle_rad))
            pt1 = (np.clip(cx - dx, 0, W - 1), np.clip(cy - dy, 0, H - 1))
            pt2 = (np.clip(cx + dx, 0, W - 1), np.clip(cy + dy, 0, H - 1))
            cv2.line(img, pt1, pt2, color=255, thickness=2)
    return img


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1
    dx = gaussian_filter(dx, sigma, mode="reflect") * alpha
    dy = gaussian_filter(dy, sigma, mode="reflect") * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    distorted = map_coordinates(image.astype(np.float32), indices, order=1, mode='reflect').reshape(shape)
    return distorted


def add_poisson_gaussian_noise(img, poisson_scale=30, gauss_sigma=0.04):
    vals = np.random.poisson(img * poisson_scale) / float(poisson_scale)
    gauss = np.random.normal(0, gauss_sigma, img.shape)
    noisy = vals + gauss
    return np.clip(noisy, 0, 1)


def check_existence(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        remove_list = os.listdir(dir_path)
        for i in range(len(remove_list)):
            remove_dir = os.path.join(dir_path, remove_list[i])
            if os.path.isfile(remove_dir):
                os.remove(remove_dir)


# =========================
# 可选：GPU KMeans（保留但当前不再使用）
# =========================

def kmeans_torch(X, n_clusters, n_iters=100, n_init=10, device='cuda'):
    X = torch.tensor(X, device=device)
    X = X.to(device)
    N, D = X.shape
    best_inertia = float('inf')
    best_labels = None
    best_centers = None

    for _ in range(n_init):
        indices = torch.randperm(N)[:n_clusters]
        centers = X[indices]

        for _ in range(n_iters):
            dists = torch.cdist(X, centers, p=2)
            labels = dists.argmin(dim=1)
            new_centers = torch.stack([
                X[labels == k].mean(dim=0) if (labels == k).sum() > 0 else centers[k]
                for k in range(n_clusters)
            ])
            if torch.allclose(centers, new_centers, rtol=1e-4, atol=1e-4):
                break
            centers = new_centers

        inertia = torch.sum(torch.min(torch.cdist(X, centers, p=2) ** 2, dim=1).values)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
            best_centers = centers.clone()

    return best_labels.cpu(), best_centers.cpu()


# =========================
# 新版 lifetime map 生成函数
# =========================

def find_overlap_background(thresholded_list):
    """
    输入:
      thresholded_list: list of binary masks (0/1)
    输出:
      overlapped: 重叠区域 mask
      background: 背景区域 mask
    """
    stack = np.stack([(np.asarray(m) > 0).astype(np.uint8) for m in thresholded_list], axis=0)
    count = stack.sum(axis=0)

    overlapped = (count > 1).astype(np.uint8)
    background = (count == 0).astype(np.uint8)
    return overlapped, background


def all_components_cv2(label_img, connectivity=8, background_value=0):
    """
    对 label_img 中每个取值分别做连通域分析。
    不同类别不会混在一起。

    返回:
      cc_labels: 每个连通域一个唯一 id
      comp_value: dict, component_id -> 原始类别值
      comp_area: dict, component_id -> 连通域面积
    """
    label_img = np.asarray(label_img)
    cc_labels = np.zeros(label_img.shape, dtype=np.int32)

    comp_value = {0: background_value}
    comp_area = {0: int((label_img == background_value).sum())}

    next_id = 1
    unique_vals = np.unique(label_img)

    for val in unique_vals:
        if val == background_value:
            continue

        mask = (label_img == val).astype(np.uint8)
        num_cc, labels = cv2.connectedComponents(mask, connectivity=connectivity)

        for cid in range(1, num_cc):
            comp_mask = (labels == cid)
            cc_labels[comp_mask] = next_id
            comp_value[next_id] = int(val)
            comp_area[next_id] = int(comp_mask.sum())
            next_id += 1

    return cc_labels, comp_value, comp_area


def fill_overlapped_components_from_neighbors(
    simulated_lifetime,
    overlapped,
    cc_labels,
    comp_value,
    background_value,
    connectivity,
    max_radius,
    seed,
    fallback_global
):
    sl = np.asarray(simulated_lifetime).copy()
    ov = (np.asarray(overlapped) != 0)

    H, W = sl.shape
    if max_radius is None:
        max_radius = max(H, W)

    rng = np.random.default_rng(seed)

    # 目标连通域：overlapped!=0 的像素所在的 component id
    target_ids = np.unique(cc_labels[ov])

    # donor 像素：overlapped==0 且 非背景
    donor_pixel_mask = (~ov) & (sl != background_value)

    # donor_labels：只保留 donor 区域内的 component id，其他地方置 0
    donor_labels = np.where(donor_pixel_mask, cc_labels, 0)

    # 形态学核：决定“扩圈”的邻接方式
    if connectivity == 4:
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
    elif connectivity == 8:
        kernel = np.ones((3, 3), dtype=np.uint8)
    else:
        raise ValueError("connectivity 只能是 4 或 8")

    # 全局 donor 值池（用于 fallback_global）
    if fallback_global:
        global_donor_ids = np.unique(donor_labels)
        global_donor_ids = global_donor_ids[global_donor_ids != 0]

    for tid in target_ids:
        if tid == 0:
            continue

        target_value = comp_value[tid]
        comp_mask = (cc_labels == tid)

        prev = comp_mask.copy().astype(np.uint8)
        cur = prev.copy()

        chosen_donor_id = None

        for _ in range(max_radius):
            cur = cv2.dilate(cur, kernel, iterations=1)
            ring = (cur.astype(bool) & (~prev.astype(bool)))
            prev = cur

            neighbor_ids = np.unique(donor_labels[ring])
            neighbor_ids = neighbor_ids[neighbor_ids != 0]

            if neighbor_ids.size > 0:
                neighbor_ids = neighbor_ids[[comp_value[nid] != target_value for nid in neighbor_ids]]

            if neighbor_ids.size > 0:
                chosen_donor_id = int(rng.choice(neighbor_ids))
                break

        if chosen_donor_id is None:
            if fallback_global:
                cand = global_donor_ids[[comp_value[nid] != target_value for nid in global_donor_ids]]
                if cand.size > 0:
                    chosen_donor_id = int(rng.choice(cand))
                else:
                    continue
            else:
                continue

        sl[comp_mask] = comp_value[chosen_donor_id]

    return sl


def make_lifetime_distribution(num_org, thresholded_list, size, seed=123):
    """
    新版 lifetime map 生成方式：
    1. 每个结构先按类别值写入 simulated_lifetime
    2. 重叠区域标成 -1
    3. 连通域分析
    4. 重叠区域按邻近 donor component 回填

    返回:
      filled: uint16 lifetime/class map
    """
    fac_list = [1, 2, 3, 4, 5, 6, 7, 8, 9][:num_org]

    simulated_lifetime = np.zeros((size, size), dtype=np.int32)
    overlapped, background = find_overlap_background(thresholded_list)

    for index_thresh in range(len(thresholded_list)):
        mask = (np.asarray(thresholded_list[index_thresh]) > 0)
        simulated_lifetime[mask] = fac_list[index_thresh] + simulated_lifetime[mask]

    # 重叠区域单独标记，防止和普通类别值撞上
    simulated_lifetime[overlapped != 0] = -1

    cc_labels, comp_value, comp_area = all_components_cv2(
        simulated_lifetime,
        connectivity=8,
        background_value=0
    )

    filled = fill_overlapped_components_from_neighbors(
        simulated_lifetime=simulated_lifetime,
        overlapped=overlapped,
        cc_labels=cc_labels,
        comp_value=comp_value,
        background_value=0,
        connectivity=8,
        max_radius=None,
        seed=seed,
        fallback_global=True
    )

    # 背景保持 0
    filled[background != 0] = 0

    return filled.astype(np.uint16)


# =========================
# 数据集生成主函数
# =========================

def make_dataset(num_levels,
                 save_dir,
                 num_file,
                 sigma,
                 image_size=512,
                 intensity_range_1=[128, 255],
                 intensity_range_2=[128, 255],
                 num_structure_range_1=[60, 80],
                 num_structure_range_2=[60, 80],
                 length_range_1=[100, 200],
                 length_range_2=[20, 80],
                 min_thickness=2,
                 max_thickness=8):

    ssim_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SSIM_criterion = SSIM().to(device=ssim_device)

    for i in tqdm(range(1, num_levels + 1), desc="Simulating transition from ellipse to line"):
        if i == 4:
            sub_dir = os.path.join(save_dir, f"{sigma}_level_{i}")
            os.makedirs(sub_dir, exist_ok=True)

            GT_DS_dir = os.path.join(sub_dir, "GT_DS")
            Input_dir = os.path.join(sub_dir, "Input")
            GT_S_dir = os.path.join(sub_dir, "GT_S")
            denoised_dir = os.path.join(sub_dir, "denoised")
            lifetime_dir = os.path.join(sub_dir, "lifetime")

            check_existence(GT_DS_dir)
            check_existence(Input_dir)
            check_existence(GT_S_dir)
            check_existence(denoised_dir)
            check_existence(lifetime_dir)

            # 控制椭圆向线过渡
            alpha = (i - 1) / (num_levels - 2)
            SSIM_list = []

            for index in tqdm(range(1, num_file + 1), leave=False):
                n_lines = np.random.randint(*num_structure_range_1)
                n_ellipses = np.random.randint(*num_structure_range_2)

                lines = rand_lines(image_size, n_lines, min_len=length_range_1[0], max_len=length_range_1[1])
                line_img = draw_lines(image_size, lines, int_range=intensity_range_1, width=min_thickness)

                if i < num_levels:
                    ellipse_img = draw_ellipse_like_lines(
                        image_size,
                        n_ellipses,
                        alpha,
                        int_range=intensity_range_2,
                        min_len=length_range_2[0],
                        max_len=length_range_2[1],
                        max_thickness=max_thickness,
                        min_thickness=min_thickness
                    )

                    SSIM_index = SSIM_criterion(
                        torch.tensor(np.float32(line_img), dtype=torch.float32, device=ssim_device).unsqueeze(0).unsqueeze(0),
                        torch.tensor(np.float32(ellipse_img), dtype=torch.float32, device=ssim_device).unsqueeze(0).unsqueeze(0),
                    )
                    SSIM_list.append(SSIM_index.item())
                else:
                    lines = rand_lines(image_size, n_lines, min_len=length_range_1[0], max_len=length_range_1[1])
                    ellipse_img = draw_lines(image_size, lines, int_range=intensity_range_2, width=min_thickness)

                line_img = np.clip(line_img, 0, 255)
                ellipse_img = np.clip(ellipse_img, 0, 255)

                # 合并图像（线 + 椭圆/线）
                combined = line_img.astype(np.uint16) + ellipse_img.astype(np.uint16)

                # 高斯模糊模拟 PSF
                blurred = gaussian_filter(combined, sigma=sigma)

                # 模糊 GT
                line_img = gaussian_filter(line_img, sigma=1)
                ellipse_img = gaussian_filter(ellipse_img, sigma=1)

                # 二值化，用于 lifetime map
                line_img_thresholded = copy.deepcopy(line_img)
                ellipse_img_thresholded = copy.deepcopy(ellipse_img)

                thresh_line = threshold_otsu(line_img_thresholded)
                thresh_ellipse = threshold_otsu(ellipse_img_thresholded)

                line_img_thresholded[line_img_thresholded < thresh_line] = 0
                line_img_thresholded[line_img_thresholded >= thresh_line] = 1

                ellipse_img_thresholded[ellipse_img_thresholded < thresh_ellipse] = 0
                ellipse_img_thresholded[ellipse_img_thresholded >= thresh_ellipse] = 1

                lifetime_list = [line_img_thresholded, ellipse_img_thresholded]
                lifetime_image = make_lifetime_distribution(
                    num_org=len(lifetime_list),
                    thresholded_list=lifetime_list,
                    size=image_size[0],
                    seed=index
                )

                # 保存图像
                line_img = np.expand_dims(line_img, axis=0)
                ellipse_img = np.expand_dims(ellipse_img, axis=0)
                stack = np.stack((line_img, ellipse_img), axis=0)

                tifffile.imwrite(os.path.join(GT_DS_dir, f"{index}.tif"), np.uint16(stack), imagej=True)
                tifffile.imwrite(os.path.join(GT_S_dir, f"{index}.tif"), np.uint16(combined))
                tifffile.imwrite(os.path.join(Input_dir, f"{index}.tif"), np.uint16(blurred))
                tifffile.imwrite(os.path.join(denoised_dir, f"{index}.tif"), np.uint16(blurred))
                tifffile.imwrite(os.path.join(lifetime_dir, f"{index}.tif"), np.uint16(lifetime_image))

            if len(SSIM_list) > 0:
                print(f"the SSIM of level {i}: {np.mean(SSIM_list):.6f}")
            else:
                print(f"the SSIM of level {i}: N/A")


# =========================
# 主程序
# =========================

num_levels = 4
sigma = 5.0
image_size = (512, 512)
intensity_range_1 = [200, 255]
intensity_range_2 = [200, 255]
num_structure_range_1 = [60, 80]
num_structure_range_2 = [60, 80]
length_range_1 = [100, 200]
length_range_2 = [100, 200]
min_thickness = 2
max_thickness = 8

make_dataset(
    num_levels=num_levels,
    save_dir=r'data\simulated_data\train',
    num_file=1000,
    sigma=sigma,
    image_size=image_size,
    intensity_range_1=intensity_range_1,
    intensity_range_2=intensity_range_2,
    num_structure_range_1=num_structure_range_1,
    num_structure_range_2=num_structure_range_2,
    length_range_1=length_range_1,
    length_range_2=length_range_2,
    min_thickness=min_thickness,
    max_thickness=max_thickness
)

make_dataset(
    num_levels=num_levels,
    save_dir=r'data\simulated_data\val',
    num_file=50,
    sigma=sigma,
    image_size=image_size,
    intensity_range_1=intensity_range_1,
    intensity_range_2=intensity_range_2,
    num_structure_range_1=num_structure_range_1,
    num_structure_range_2=num_structure_range_2,
    length_range_1=length_range_1,
    length_range_2=length_range_2,
    min_thickness=min_thickness,
    max_thickness=max_thickness
)