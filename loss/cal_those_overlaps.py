import cv2
import numpy as np
import torch 
from torch import nn


def _to_mask2d_numpy(mask):
    """mask 2D uint8 numpy, 0/1 Input: - np.ndarray: (H,W) / (1,H,W) / (1,1,H,W) - torch.Tensor:"""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)

    while mask.ndim > 2:
        mask = np.squeeze(mask, axis=0)

    mask = (mask > 0).astype(np.uint8)
    return mask

def masked_nrmae(pred, gt, mask, norm_mode="range", eps=1e-12):
    """mask Calculate NRMAE Parameter ---- pred, gt : np.ndarray 2D Image, shape mask : np.ndarray 2D mask, 0 Calculate norm_mode : str "range" : gt mask Normalize "mean" : gt mask Normalize eps : float Prevent 0 Return ---- nrmae : float mask Pixel, Return np.nan"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.asarray(mask) > 0

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    if mask.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {gt.shape}")

    valid = np.count_nonzero(mask)
    if valid == 0:
        return np.nan

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    mae = np.mean(np.abs(pred_valid - gt_valid))

    if norm_mode == "range":
        denom = gt_valid.max() - gt_valid.min()
    elif norm_mode == "mean":
        denom = np.mean(np.abs(gt_valid))
    else:
        raise ValueError("norm_mode must be 'range' or 'mean'")

    if denom <= eps:
        return np.nan

    return float(mae / denom)


def _pairwise_normalize_torch(pred, gt, eps=1e-8):
    """pred / gt min-max Normalize [0,1]"""
    pair_min = torch.minimum(pred.min(), gt.min())
    pair_max = torch.maximum(pred.max(), gt.max())
    denom = (pair_max - pair_min).clamp_min(eps)

    pred_norm = (pred - pair_min) / denom
    gt_norm = (gt - pair_min) / denom
    return pred_norm, gt_norm


def connected_components_weighted_ssim(pred,
                                       gt,
                                       overlap_mask,
                                       ssim_criterion,
                                       min_area=20,
                                       connectivity=8,
                                       eps=1e-8):
    """overlap mask SSIM, . Parameter ---- pred, gt : torch.Tensor shape = [1,1,H,W] overlap_mask : np.ndarray or torch.Tensor shape = [H,W] / [1,H,W] / [1,1,H,W] overlap Evaluation ssim_criterion : SSIM() import SSIM min_area : int connectivity : int 4 8 Return ---- weighted_ssim : float num_components : int component_areas : list[int]"""
    if pred.ndim != 4 or gt.ndim != 4:
        raise ValueError(f"pred and gt must be [B,C,H,W], got {pred.shape}, {gt.shape}")
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.size(0) != 1 or pred.size(1) != 1:
        raise ValueError("This helper expects single-image single-channel input: [1,1,H,W]")

    mask2d = _to_mask2d_numpy(overlap_mask)
    if mask2d.sum() == 0:
        return np.nan, 0, []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask2d, connectivity=connectivity)

    weighted_sum = 0.0
    total_area = 0
    kept_areas = []

    for comp_id in range(1, num_labels):  # 0 Background
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

        pred_roi = pred[:, :, y:y+h, x:x+w]
        gt_roi = gt[:, :, y:y+h, x:x+w]

        # ROI SSIM( win_size Default 11)
        if pred_roi.size(-2) < 11 or pred_roi.size(-1) < 11:
            continue

        pred_roi, gt_roi = _pairwise_normalize_torch(pred_roi, gt_roi, eps=eps)
        ssim_value = ssim_criterion(pred_roi, gt_roi).item()

        weighted_sum += ssim_value * area
        total_area += area
        kept_areas.append(area)

    if total_area == 0:
        return np.nan, 0, []

    return weighted_sum / total_area, len(kept_areas), kept_areas


def multichannel_overlap_ssim(fake_main,
                              GT_DS,
                              overlap_mask,
                              ssim_criterion,
                              min_area=20,
                              connectivity=8):
    """ChannelOutputChannelCalculate overlap-SSIM, Channel. Parameter ---- fake_main, GT_DS : torch.Tensor shape = [1,C,H,W] overlap_mask : : - [H,W]: Channel overlap mask - [C,H,W]: Channel overlap mask - [1,C,H,W] / [1,1,H,W]"""
    if fake_main.shape != GT_DS.shape:
        raise ValueError(f"Shape mismatch: {fake_main.shape} vs {GT_DS.shape}")

    num_channels = fake_main.size(1)
    channel_ssim_values = []

    if torch.is_tensor(overlap_mask):
        overlap_mask_np = overlap_mask.detach().cpu().numpy()
    else:
        overlap_mask_np = np.asarray(overlap_mask)

    for c in range(num_channels):
        pred_c = fake_main[:, c:c+1, :, :].detach()
        gt_c = GT_DS[:, c:c+1, :, :].detach()

        # mask Channel mask
        if overlap_mask_np.ndim == 2:
            mask_c = overlap_mask_np
        elif overlap_mask_np.ndim == 3:
            # [C,H,W] [1,H,W]
            if overlap_mask_np.shape[0] == num_channels:
                mask_c = overlap_mask_np[c]
            else:
                mask_c = np.squeeze(overlap_mask_np, axis=0)
        elif overlap_mask_np.ndim == 4:
            # [1,C,H,W] [1,1,H,W]
            if overlap_mask_np.shape[1] == num_channels:
                mask_c = overlap_mask_np[0, c]
            else:
                mask_c = overlap_mask_np[0, 0]
        else:
            raise ValueError(f"Unsupported overlap_mask shape: {overlap_mask_np.shape}")

        ssim_c, _, _ = connected_components_weighted_ssim(
            pred=pred_c,
            gt=gt_c,
            overlap_mask=mask_c,
            ssim_criterion=ssim_criterion,
            min_area=min_area,
            connectivity=connectivity
        )

        if not np.isnan(ssim_c):
            channel_ssim_values.append(ssim_c)

    if len(channel_ssim_values) == 0:
        return np.nan

    return float(np.mean(channel_ssim_values))

def masked_psnr(pred, gt, mask, data_range=None, eps=1e-12):
    """mask Calculate PSNR Parameter ---- pred, gt : np.ndarray 2D Image, shape mask : np.ndarray 2D mask, 0 Calculate data_range : float or None - None, mask pred/gt min-max - Image, [0,1] [0,255], eps : float Prevent Return ---- psnr : float mask Pixel, Return np.nan mask mse=0, Return inf"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.asarray(mask) > 0

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    if mask.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {gt.shape}")

    valid = np.count_nonzero(mask)
    if valid == 0:
        return np.nan

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    mse = np.mean((pred_valid - gt_valid) ** 2)
    if mse <= eps:
        return float("inf")

    if data_range is None:
        data_min = min(pred_valid.min(), gt_valid.min())
        data_max = max(pred_valid.max(), gt_valid.max())
        data_range = data_max - data_min

    if data_range <= eps:
        return np.nan

    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(psnr)

def masked_pearson_numpy(pred, gt, mask):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.asarray(mask) > 0

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")

    valid = np.count_nonzero(mask)
    if valid < 2:
        return np.nan

    x = pred[mask]
    y = gt[mask]

    x_mean = x.mean()
    y_mean = y.mean()

    x_center = x - x_mean
    y_center = y - y_mean

    denom = np.sqrt(np.sum(x_center ** 2) * np.sum(y_center ** 2))
    if denom <= 1e-12:
        return np.nan

    return float(np.sum(x_center * y_center) / denom)


def build_overlap_mask_from_gtd(GT_D, threshold=0):
    """Default overlap mask Function: GT_D: [1,C,H,W] [C,H,W] Return: [H,W], "Channel" If overlap mask, , NeedFunction."""
    if torch.is_tensor(GT_D):
        x = GT_D.detach().cpu().numpy()
    else:
        x = np.asarray(GT_D)

    if x.ndim == 4:
        x = x[0]  # [C,H,W]
    elif x.ndim != 3:
        raise ValueError(f"GT_D shape should be [1,C,H,W] or [C,H,W], got {x.shape}")

    fg = (x > threshold).astype(np.uint8)
    overlap = (np.sum(fg, axis=0) >= 2).astype(np.uint8)
    return overlap


