import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import copy
import torch
import tifffile
import joblib
from random import uniform
#import cupy as cp
from utils import *
from sklearn.cluster import KMeans
from skimage.filters import threshold_local


# morphology processing functions
# -----------------------------------------------
def all_components_cv2(img: np.ndarray, connectivity: int = 8):
    img = np.asarray(img)
    assert img.ndim == 2

    H, W = img.shape
    cc_labels = np.zeros((H, W), dtype=np.int32)

    comp_value = [0]
    comp_area  = [0]

    next_id = 1
    for v in np.unique(img):
        mask = (img == v).astype(np.uint8)

        n, lab = cv2.connectedComponents(mask, connectivity=connectivity)  # 顺序是 n, lab

        areas = np.bincount(lab.ravel())  # lab 是 ndarray 了

        for k in range(1, n):
            comp = (lab == k)
            cc_labels[comp] = next_id
            comp_value.append(int(v))
            comp_area.append(int(areas[k]))
            next_id += 1

    return cc_labels, np.array(comp_value), np.array(comp_area)


def fill_overlapped_components_from_neighbors(
    simulated_lifetime,
    overlapped,
    cc_labels,
    comp_value,
    background_value,
    connectivity,
    max_radius,
    seed,
    fallback_global   # 找不到邻居时，是否全局随机选 donor
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
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    elif connectivity == 8:
        kernel = np.ones((3,3), dtype=np.uint8)
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
        cur  = prev.copy()

        chosen_donor_id = None

        for _ in range(max_radius):
            cur = cv2.dilate(cur, kernel, iterations=1)              # 扩一圈
            ring = (cur.astype(bool) & (~prev.astype(bool)))         # 当前圈环带
            prev = cur

            neighbor_ids = np.unique(donor_labels[ring])
            neighbor_ids = neighbor_ids[neighbor_ids != 0]

            # 排除 donor 值等于 target 值（比如 target=-1，不要再选 -1）
            if neighbor_ids.size > 0:
                neighbor_ids = neighbor_ids[comp_value[neighbor_ids] != target_value]

            if neighbor_ids.size > 0:
                chosen_donor_id = int(rng.choice(neighbor_ids))
                break

        if chosen_donor_id is None:
            if fallback_global:
                # 全局随机挑一个（同样排除同值）
                cand = global_donor_ids[comp_value[global_donor_ids] != target_value]
                if cand.size > 0:
                    chosen_donor_id = int(rng.choice(cand))
                else:
                    continue
            else:
                continue

        sl[comp_mask] = comp_value[chosen_donor_id]

    return sl
# -----------------------------------------------


class Degradation_base_model():
    def __init__(self, target_resolution, noise_level, average, STED_resolution_list, size, factor_list, device):
        super(Degradation_base_model, self).__init__()
        self.target_resolution = target_resolution
        self.noise_level = noise_level
        self.average = average
        self.STED_resolution_list = STED_resolution_list
        self.size = size
        self.factor_list = factor_list
        self.device = device

        self.temp_count = 1
    
    def fwhm_gaussian_1d_fit(self, y):
        """
        ImageJ Gaussian fit:
            y = a + (b-a) * exp(-(x-c)^2 / (2*d^2))
        返回：
            fwhm（单位：采样点 index）
        """
        y = np.asarray(y, dtype=float).ravel()
        x = np.arange(y.size, dtype=float)

        # ---- 初值（按 ImageJ 参数含义）----
        i0 = int(np.argmax(y))
        a0 = float(np.min(y))
        b0 = float(np.max(y))
        c0 = float(x[i0])

        # 粗估 d：先用半高宽估 fwhm0 -> d0
        half = a0 + 0.5 * (b0 - a0)
        left = np.where(y[:i0] < half)[0]
        right = np.where(y[i0:] < half)[0]
        if left.size == 0 or right.size == 0:
            d0 = max(1.0, y.size / 10.0) / 2.354820045
        else:
            xL0 = float(left[-1])
            xR0 = float(i0 + right[0])
            fwhm0 = max(1.0, xR0 - xL0)
            d0 = fwhm0 / 2.354820045

        a, b, c, d = a0, b0, c0, float(d0)

        # ---- LM 迭代拟合 ----
        for _ in range(60):
            d = max(d, 1e-6)
            A = (b - a)  # 振幅
            t = (x - c) / d
            e = np.exp(-0.5 * t * t)

            y_hat = a + A * e
            r = y - y_hat

            # Jacobian: da, db, dc, dd
            # y = a + (b-a)e = a(1-e) + b e
            J_a = (1.0 - e)
            J_b = e
            # dy/dc = (b-a)*e*(x-c)/d^2 = A*e*t/d
            J_c = A * e * (t / d)
            # dy/dd = (b-a)*e*(x-c)^2/d^3 = A*e*t^2/d
            J_d = A * e * ((t * t) / d)

            J = np.vstack([J_a, J_b, J_c, J_d]).T  # (N,4)

            H = J.T @ J
            g = J.T @ r
            lam = 1e-3 * np.trace(H) / 4.0 + 1e-12
            H_lm = H + lam * np.eye(4)

            try:
                dp = np.linalg.solve(H_lm, g)
            except np.linalg.LinAlgError:
                break

            a_new, b_new, c_new, d_new = a + dp[0], b + dp[1], c + dp[2], d + dp[3]

            # 约束：d>0；若是正峰，通常 b>=a（不强制也行）
            if d_new <= 0:
                d_new = 1e-6

            # 收敛
            if np.max(np.abs(dp)) < 1e-8:
                a, b, c, d = a_new, b_new, c_new, d_new
                break

            a, b, c, d = a_new, b_new, c_new, d_new

        fwhm = 2.354820045 * d
        return float(fwhm)  # 单位：index（采样点）
    
    def set_image_params(self, grid_size=36, dx=20e-9):
        self.N = grid_size
        self.dx = dx

        self.x_image = np.arange(-self.N / 2, self.N / 2, dtype=np.float32) * self.dx
        self.y_image = np.arange(-self.N / 2, self.N / 2, dtype=np.float32) * self.dx

        self.x_image_grid, self.y_image_grid = np.meshgrid(
            self.x_image,
            -self.y_image,
            indexing="ij"
        )

        self.r_image = np.sqrt(self.x_image_grid**2 + self.y_image_grid**2).astype(np.float32)
        self.phi_image = np.arctan2(self.y_image_grid, self.x_image_grid).astype(np.float32)

    def generate_2D_PSF(self, target_fwhm_xy_nm):
        """
        target_fwhm_xy_nm: target intensity FWHM, unit = nm
        return:
            PSF_2d: numpy.ndarray, shape = (N, N), dtype = float32
        """
        # E(r) = exp(-(r/w0)^2)
        # I(r) = |E(r)|^2 = exp(-2(r/w0)^2)
        # intensity FWHM = 1.177 * w0

        fwhm_xy_m = float(target_fwhm_xy_nm) * 1e-9
        w0_xy = fwhm_xy_m / 1.177

        E_2d = np.exp(-(self.r_image / w0_xy) ** 2)
        PSF_2d = np.abs(E_2d) ** 2
        PSF_2d = PSF_2d / (np.sum(PSF_2d) + 1e-12)

        return PSF_2d.astype(np.float32)
    
    def fwhm_gaussian_1d_fit(self, y):
        y = np.asarray(y, dtype=float)

        # 如果输入是 2D，取中心横截面
        if y.ndim == 2:
            y = y[y.shape[0] // 2, :]
        elif y.ndim != 1:
            raise ValueError(f"Input must be 1D or 2D, but got shape {y.shape}")

        y = y.ravel()

        # 去背景并归一化
        y = y - np.min(y)
        peak = np.max(y)
        if peak <= 0:
            raise ValueError("Invalid profile: peak <= 0")

        y = y / peak

        i0 = int(np.argmax(y))

        # ---- 左半高点 ----
        i = i0
        while i > 0 and y[i] >= 0.5:
            i -= 1
        if i == 0 and y[i] >= 0.5:
            raise ValueError("Left half-maximum crossing not found")

        # 在线段 [i, i+1] 上插值
        xL = i + (0.5 - y[i]) / (y[i + 1] - y[i] + 1e-12)

        # ---- 右半高点 ----
        j = i0
        while j < len(y) - 1 and y[j] >= 0.5:
            j += 1
        if j == len(y) - 1 and y[j] >= 0.5:
            raise ValueError("Right half-maximum crossing not found")

        # 在线段 [j-1, j] 上插值
        xR = (j - 1) + (0.5 - y[j - 1]) / (y[j] - y[j - 1] + 1e-12)

        fwhm = xR - xL
        return float(fwhm)   # 单位：pixel

    def find_psf_for_resolution(self, resolution):
        """
        resolution_nm: target FWHM in nm
        """
        psf = self.generate_2D_PSF(target_fwhm_xy_nm=resolution)
        #tifffile.imwrite(r"E:\Onedrive\Work\codes\DESM\temp\PSF_image.tif", psf)

        # 自动支持 2D，内部会取中心横截面
        measured_fwhm_pix = self.fwhm_gaussian_1d_fit(psf)
        measured_fwhm_nm = measured_fwhm_pix * self.dx * 1e9

        print(f"Target FWHM:   {resolution:.3f} nm")
        print(f"Measured FWHM: {measured_fwhm_nm:.3f} nm")
        print(f"Measured FWHM: {measured_fwhm_pix:.3f} pixels")

        return psf, measured_fwhm_pix, measured_fwhm_nm
    
    def _generate_psf_legacy(self, m=0, N=None, span=12.0, lamb=635e-9, w0=2.0):
        """
        旧版本 generate_psf 的迁移版：
            E = (r / w0)^m * exp(-r^2 / w0^2) * exp(i*beta) * exp(-i*m*theta)
            I = |E|^2

        这里默认 m=0，因此实际就是旧版里最常用的 Gaussian-like PSF 形式。
        """
        if N is None:
            N = int(self.N)

        w0 = max(float(w0), 1e-6)
        beta = np.deg2rad(50.0)

        x = np.linspace(-span, span, N, dtype=np.float32)
        y = np.linspace(-span, span, N, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")

        r = np.sqrt(X**2 + Y**2).astype(np.float32)
        theta = np.arctan2(Y, X).astype(np.float32)

        E = (
            np.power(r / w0, m) *
            np.exp(-(r**2) / (w0**2)) *
            np.exp(1j * beta) *
            np.exp(-1j * m * theta)
        )

        I = np.real(E * np.conj(E)).astype(np.float32)
        I = I / (np.sum(I) + 1e-12)
        return I


    def _search_w0_for_target_fwhm(self,
                                target_fwhm_pix,
                                span=12.0,
                                w0_min=0.2,
                                w0_max=15.0,
                                n_coarse=200,
                                n_fine=120):
        """
        给定目标 FWHM（单位：pixel），搜索旧版 PSF 公式中的 w0，
        使生成出来的 PSF 的测得 FWHM 最接近 target_fwhm_pix。
        """
        # ---------- coarse search ----------
        coarse_grid = np.linspace(w0_min, w0_max, n_coarse, dtype=np.float32)

        best_w0 = None
        best_psf = None
        best_fwhm_pix = None
        best_diff = np.inf

        for w0 in coarse_grid:
            psf = self._generate_psf_legacy(m=0, N=self.N, span=span, w0=float(w0))
            fwhm_pix = self.fwhm_gaussian_1d_fit(psf)
            diff = abs(fwhm_pix - target_fwhm_pix)

            if diff < best_diff:
                best_diff = diff
                best_w0 = float(w0)
                best_psf = psf
                best_fwhm_pix = float(fwhm_pix)

        # ---------- fine search ----------
        coarse_step = (w0_max - w0_min) / max(n_coarse - 1, 1)
        fine_left = max(w0_min, best_w0 - 2.0 * coarse_step)
        fine_right = min(w0_max, best_w0 + 2.0 * coarse_step)
        fine_grid = np.linspace(fine_left, fine_right, n_fine, dtype=np.float32)

        for w0 in fine_grid:
            psf = self._generate_psf_legacy(m=0, N=self.N, span=span, w0=float(w0))
            fwhm_pix = self.fwhm_gaussian_1d_fit(psf)
            diff = abs(fwhm_pix - target_fwhm_pix)

            if diff < best_diff:
                best_diff = diff
                best_w0 = float(w0)
                best_psf = psf
                best_fwhm_pix = float(fwhm_pix)
        print(best_w0, best_fwhm_pix * 20)

        return best_w0, best_psf.astype(np.float32), best_fwhm_pix


    def _generate_2D_PSF(self, target_fwhm_xy_nm, span=12.0, use_cache=True):
        """
        用旧版本 generate_psf 的方式生成 2D PSF，
        但对外接口保持不变：输入仍然是 target_fwhm_xy_nm。

        参数
        ----
        target_fwhm_xy_nm : float
            目标强度 FWHM，单位 nm

        返回
        ----
        PSF_2d : np.ndarray, shape=(N, N), dtype=float32
        """
        if not hasattr(self, "N") or not hasattr(self, "dx"):
            raise AttributeError("Please call set_image_params(...) before generate_2D_PSF(...).")

        target_fwhm_xy_nm = float(target_fwhm_xy_nm)
        if target_fwhm_xy_nm <= 0:
            raise ValueError("target_fwhm_xy_nm must be > 0")

        target_fwhm_pix = target_fwhm_xy_nm / (self.dx * 1e9)

        # 可选 cache，避免同一个分辨率反复搜索
        if use_cache:
            if not hasattr(self, "_psf_cache"):
                self._psf_cache = {}

            cache_key = (round(target_fwhm_xy_nm, 6), int(self.N), float(span), float(self.dx))
            if cache_key in self._psf_cache:
                return self._psf_cache[cache_key].copy()

        best_w0, psf, measured_fwhm_pix = self._search_w0_for_target_fwhm(
            target_fwhm_pix=target_fwhm_pix,
            span=span,
            w0_min=0.2,
            w0_max=15.0,
            n_coarse=200,
            n_fine=120
        )

        psf = psf / (np.sum(psf) + 1e-12)
        psf = psf.astype(np.float32)

        if use_cache:
            self._psf_cache[cache_key] = psf.copy()

        return psf
    
    
    def generate_cal_psf(self, resolution_LR, resolution_SR):
        N = 36
        span = 12
        self.SR_psf = self.generate_2D_PSF(target_fwhm_xy_nm=resolution_SR)
        self.LR_psf = self.generate_2D_PSF(target_fwhm_xy_nm=resolution_LR)

    
        # 计算OTF
        SR_otf = (np.fft.fftshift(np.fft.fft2(self.SR_psf)))
        LR_otf = (np.fft.fftshift(np.fft.fft2(self.LR_psf)))
 
        LR_otf[np.abs(LR_otf) < 1e-3] = 0
        SR_otf[np.abs(SR_otf) < 1e-3] = 0
        cal_otf = (LR_otf) / ((SR_otf) + 1e-9)
        
        cal_psf = np.fft.fftshift(np.fft.fft2(cal_otf))
        cal_psf = np.abs(cal_psf)
        
        self.cal_psf = cal_psf / np.sum(cal_psf)
        self.cal_psf /= np.sum(self.cal_psf)

        if 0:
            tifffile.imwrite(r'E:\Onedrive\Work\codes\DESM\temp\confocal_psf.tif', self.LR_psf)
            tifffile.imwrite(r'E:\Onedrive\Work\codes\DESM\temp\confocal_otf.tif', np.abs(LR_otf))
            tifffile.imwrite(r'E:\Onedrive\Work\codes\DESM\temp\STED_psf.tif', self.SR_psf)
            tifffile.imwrite(r'E:\Onedrive\Work\codes\DESM\temp\STED_otf.tif', np.abs(SR_otf))
            tifffile.imwrite(r'E:\Onedrive\Work\codes\DESM\temp\cal_psf.tif', self.cal_psf)
            tifffile.imwrite(r'E:\Onedrive\Work\codes\DESM\temp\cal_OTF.tif', (cal_otf))
            
            self.SR_psf = self.SR_psf / np.max(self.SR_psf) * 255
            self.LR_psf = self.LR_psf / np.max(self.LR_psf) * 255
            self.cal_psf = self.cal_psf / np.max(self.cal_psf) * 255
            fwhm_SR = self.fwhm_gaussian_1d_fit(self.SR_psf)
            fwhm_LR = self.fwhm_gaussian_1d_fit(self.LR_psf)
            fwhm_cal = self.fwhm_gaussian_1d_fit(self.cal_psf)
            print(f"SR PSF FWHM: {fwhm_SR*20:.3f} nm")
            print(f"LR PSF FWHM: {fwhm_LR*20:.3f} nm")
            print(f"Cal PSF FWHM: {fwhm_cal*20:.3f} nm")

        self.cal_psf = np.expand_dims(self.cal_psf, axis=-1)
        return self.cal_psf

    
    def map_values_numpy(self, image, new_min=20, new_max=255, percentile=99):
        # 计算对称百分位数
        low_p = (100 - percentile) / 2
        high_p = 100 - low_p

        min_val = np.percentile(image, low_p)
        max_val = np.percentile(image, high_p)

        if max_val == min_val:
            return np.full_like(image, new_min, dtype=np.float32)

        # 转 float32
        image = image.astype(np.float32)

        # 一次性线性缩放
        scale = (new_max - new_min) / (max_val - min_val)
        mapped = (image - min_val) * scale + new_min

        # clip 限幅（比手动 mask +赋值更快）
        np.clip(mapped, new_min, new_max, out=mapped)

        return mapped
    
    # 20260408
    def add_poisson_gaussian_numpy(self,
                               Input,
                               poisson_gain=0.05,
                               gauss_sigma=1,
                               average=3):
        #poisson_gain = 0.28
        #gauss_sigma = 0.001
        Output = np.zeros_like(Input)
        for i in range(average):
            scaled = np.clip(Input * poisson_gain, 0, None)
            poisson_sample = np.random.poisson(scaled) / float(poisson_gain)

            gauss_noise = np.random.normal(loc=0.0,
                                        scale=gauss_sigma,
                                        size=Input.shape)
            
            #Output += (poisson_sample + gauss_noise)
            Output += (Input + poisson_sample + gauss_noise)
        Output /= average
        Output = self.map_values_numpy(Output, new_min=0, new_max=np.max(Input), percentile=99.9)
        return Output
    
    def map_values_torch(self, image, new_min=20, new_max=255, percentile=99):
        image = image.to(torch.float32)

        # 对称百分位
        low_p = (100.0 - percentile) / 2.0
        high_p = 100.0 - low_p

        flat = image.reshape(-1)
        min_val = torch.quantile(flat, low_p / 100.0)
        max_val = torch.quantile(flat, high_p / 100.0)

        if torch.isclose(max_val, min_val):
            return torch.full_like(image, fill_value=float(new_min), dtype=torch.float32)

        new_min_t = torch.as_tensor(new_min, dtype=image.dtype, device=image.device)
        new_max_t = torch.as_tensor(new_max, dtype=image.dtype, device=image.device)

        scale = (new_max_t - new_min_t) / (max_val - min_val)
        mapped = (image - min_val) * scale + new_min_t
        mapped = torch.clamp(mapped, min=new_min_t, max=new_max_t)

        return mapped

    def add_poisson_gaussian_torch(
        self, 
        Input,
        poisson_gain=0.05,
        gauss_sigma=1.0,
        average=3,
    ):
        """
        More physically reasonable version:
        noisy_sample = Poisson(Input * gain) / gain + Gaussian noise
        """
        if poisson_gain <= 0:
            raise ValueError("poisson_gain must be > 0")
        if average <= 0:
            raise ValueError("average must be > 0")

        Input = Input.to(torch.float32)
        Output = torch.zeros_like(Input)

        for _ in range(average):
            scaled = torch.clamp(Input * poisson_gain, min=0.0)
            poisson_sample = torch.poisson(scaled) / float(poisson_gain)
            gauss_noise = torch.randn_like(Input) * gauss_sigma

            noisy = poisson_sample + gauss_noise
            Output += noisy

        Output = Output / average
        Output = self.map_values_torch(
            Output,
            new_min=0,
            new_max=Input.max(),
            percentile=99.9
        )
        return Output
    
    def make_threshold(self, Input):
        thresh = threshold_otsu(Input)
        Output = copy.deepcopy(Input)
        Output[Output<thresh] = 0
        Output[Output>=thresh] = 1
        kernel_size = (5, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        Output = cv2.dilate(Output, kernel, 
                 anchor=None, 
                 iterations=3, 
                 borderType=cv2.BORDER_CONSTANT, 
                 borderValue=0)
        return Output
       
    def find_overlap_background(self, thresholded_list):
        masks = np.stack(thresholded_list, axis=0)  # (K, H, W) bool
        mask = masks.sum(axis=0)
        overlap =  mask >= 2                            # (H, W) bool
        background = mask == 0
        return overlap, background

    def make_lifetime_distribution(self, num_org, thresholded_list, size):
        self.temp_count += 1
        # for convention combination, assume the lifetime values are distinct. 
        fac_list = [1,2,3,4,5,6,7,8,9]
        # for combinations in which the lifetime values are not distinct, assign the same factor to different organlles，e.g. 1,2,1,4....
        #fac_list = [1,2,1,4,5,6,7,8,9]
        fac_list = fac_list[:num_org]
        # size - crop size
        simulated_lifetime = np.zeros((size, size))
        overlapped, background = self.find_overlap_background(thresholded_list)

        for index_thresh in range(len(thresholded_list)):
            temp_ones = np.ones((size, size))
            temp_ones[thresholded_list[index_thresh] == 0] = 0
            temp_ones *= fac_list[index_thresh]
            simulated_lifetime += temp_ones

        cc_labels, comp_value, comp_area = all_components_cv2(simulated_lifetime, connectivity=8)
        filled = fill_overlapped_components_from_neighbors(
            simulated_lifetime,
            overlapped,
            cc_labels,
            comp_value,
            background_value=0,
            connectivity=8,
            max_radius=None,     # 自动用 max(H,W)
            seed=123,
            fallback_global=True # 周围一直找不到就全局随机找 donor
        )
        return filled
    
    def degrade_resolution_numpy(self, Input, psf):
        blurred = cv2.filter2D(Input, -1, psf)    
        return blurred    
    
    def degrade_noise(self, Input, version, average=1, noise_scale=0, intensity=200, percentile=90):
        if version == "numpy":
            noised = self.add_poisson_gaussian_numpy(Input=Input, poisson_gain=noise_scale, average=average)
        elif version == "pytorch":
            noised = self.add_poisson_gaussian_torch(Input=Input, poisson_gain=noise_scale, average=average)
        noised[noised<0] = 0
        return noised
    
    def create_stack(self):
        self.stack_LR = []
        self.stack_HR = []

    def add_image(self, Input_LR, Input_HR):
        self.stack_LR.append(np.expand_dims(Input_LR, axis=-1))
        self.stack_HR.append(np.expand_dims(Input_HR, axis=-1))
    
    def images_concatenation(self):
        self.GT_DS = np.concatenate([*self.stack_HR], axis=-1)
        self.GT_D = np.concatenate([*self.stack_LR], axis=-1)
        return self.GT_DS, self.GT_D
    
    def generate_plain(self, size):
        self.plain_blurred = np.zeros((size, size, 1))
        self.plain_GT_S = np.zeros((size, size, 1))
    
    def composition(self, factor_list):
        for i in range(len(self.stack_HR)):
            self.plain_blurred += factor_list[i] * self.stack_LR[i]
            self.plain_GT_S += factor_list[i] * self.stack_HR[i]
        return self.plain_blurred, self.plain_GT_S
    
    def composition_LR(self, x):
        self.plain_blurred = np.zeros((self.size, self.size))
        for index in range(len(x)):
            self.plain_blurred += x[index]
        return self.plain_blurred
    
    def composition_HR(self, x):
        self.plain_GT_S = np.zeros((self.size, self.size))
        for index in range(len(x)):
            self.plain_GT_S += x[index]
        return self.plain_GT_S



    
    