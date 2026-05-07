import numpy as np

def calculate_psnr(img1, img2, data_range=None):
    """
    计算两张图像之间的 PSNR

    参数
    ----
    img1, img2 : np.ndarray
        两张待比较的图像，shape 必须一致
    data_range : float or None
        图像动态范围。
        - 如果是 8-bit 图像，通常设为 255
        - 如果是 [0,1] 浮点图，通常设为 1.0
        - 如果为 None，则自动用 max(img1, img2) - min(img1, img2)

    返回
    ----
    psnr : float
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float("inf")

    if data_range is None:
        data_min = min(img1.min(), img2.min())
        data_max = max(img1.max(), img2.max())
        data_range = data_max - data_min

    if data_range <= 0:
        raise ValueError("data_range must be positive")

    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)