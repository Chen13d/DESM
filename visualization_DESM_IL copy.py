import os
# changes through the equipments
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

from utils import *
from options.options import *
import tifffile
from tqdm import tqdm
from torchvision import transforms
from copy import deepcopy

from dataset.degradation_model import *


# test: thresholding method
# ---------------------------------------
import numpy as np
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, remove_small_holes

from skimage.filters import threshold_isodata
from skimage.util import img_as_ubyte


from sklearn.cluster import KMeans
from matplotlib.colors import hsv_to_rgb

device = 'cuda'

def adaptive_binarize_auto(img, min_obj=0, fill_holes=0):
    """
    img: 2D ndarray (float / uint16 都行)
    返回: 0/1 uint8 mask
    参数 min_obj / fill_holes 可选（你不想调就保持0）
    """
    x = img.astype(np.float32, copy=False)

    # 1) 稳健归一化，避免极亮点影响阈值
    p1, p99 = np.percentile(x, (1, 99))
    x = np.clip(x, p1, p99)
    x = (x - p1) / (p99 - p1 + 1e-8)

    H, W = x.shape

    # 2) 自动决定局部尺度：取 min(H,W)/16，且为奇数，并限定最小31
    win = max(31, (min(H, W) // 16) | 1)
    sigma = win / 6.0  # 与窗口对应的平滑尺度（经验上很稳）

    # 3) 局部均值（背景） + 自适应偏置（由噪声估计得到）
    mu = gaussian(x, sigma=sigma, preserve_range=True)
    res = x - mu
    mad = np.median(np.abs(res - np.median(res)))
    sigma_n = 1.4826 * mad  # robust noise std

    # 偏置：噪声越大，阈值越往上抬；这个系数一般不需要改
    thr = mu + 1.0 * sigma_n

    mask = x > thr

    # 4) 可选清理（你不想调参就先都设0）
    if min_obj > 0:
        mask = remove_small_objects(mask, min_size=min_obj)
    if fill_holes > 0:
        mask = remove_small_holes(mask, area_threshold=fill_holes)

    return mask.astype(np.uint8)


def imagej_auto_default_mask(img, clip_percentile=(0.5, 99.5)):
    """
    尽量复现 ImageJ Threshold -> Auto (Default) 的感觉
    返回: mask(uint8 0/1), threshold(0..255)
    """
    x = img.astype(np.float32, copy=False)

    lo, hi = np.percentile(x, clip_percentile)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)

    x8 = img_as_ubyte(x)  # 0..255
    t = threshold_isodata(x8)  # IsoData阈值
    mask = (x8 > t).astype(np.uint8)
    return mask, t

def make_threshold_otsu(Input):
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

def relabel_by_priors(clustered_input, IO_prior_background, IO_prior_list):
    """
    clustered_input: 2D/3D label image, values are arbitrary (n+1 classes)
    IO_prior_background: background mask (bg=1 or >0)
    IO_prior_list: list of n class masks (each=1 or >0), order matters

    return:
      fixed: background=0, class i -> i (i=1..n) following IO_prior_list order
      mapping: dict old_label -> new_label
    """
    clustered = np.asarray(clustered_input)
    bg = (np.asarray(IO_prior_background) > 0)
    priors = [(np.asarray(p) > 0) for p in IO_prior_list]  # n masks

    # 把背景也当成第 0 类 prior：顺序为 [background, class1, class2, ...]
    priors_all = [bg] + priors
    new_ids = list(range(len(priors_all)))  # [0,1,2,...,n]

    labels = [int(x) for x in np.unique(clustered)]
    masks = {l: (clustered == l) for l in labels}

    mapping = {}
    fixed = np.zeros_like(clustered, dtype=np.int32)
    unused = labels.copy()

    for prior_mask, new_id in zip(priors_all, new_ids):
        if not unused:
            break

        # “乘算”重叠：bool 下用 & 等价于相乘后>0；这里直接 count_nonzero
        scores = [np.count_nonzero(masks[l] & prior_mask) for l in unused]
        best_idx = int(np.argmax(scores))
        best_old = unused.pop(best_idx)   # 一一对应：用过的旧 label 移除

        mapping[best_old] = new_id
        fixed[masks[best_old]] = new_id

    # 可选：如果你的背景 prior 很可靠，强制背景区域为 0（防止边缘误分）
    fixed[bg] = 0

    return fixed, mapping


def clean_labels_fill_bg_then_smooth(lbl_in,
                                    ksize=3,
                                    fill_close_iter=1,   # 第一步：填背景洞（对前景mask做closing）
                                    open_iter=1,         # 第二步：每类开运算（去毛刺）
                                    close_iter=1,        # 第二步：每类闭运算（补小孔）
                                    propagate_iters=100  # 标签传播填洞的迭代上限
                                    ):
    """
    lbl_in: 2D label image, background=0, classes=1..n (int)
    目标：
      A) 先把前景内部的 0 小洞补掉（不让背景“渗进来”）
      B) 再对每个前景类做开/闭运算去毛刺
      C) 尽量保证前景不被处理后变成背景（不产生新的0洞）

    return: lbl_out (int32)
    """
    lbl = np.asarray(lbl_in)
    out = lbl.copy().astype(np.int32)

    labels = [int(x) for x in np.unique(out) if int(x) != 0]
    if len(labels) == 0:
        return out  # 全背景

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # ---------- Step 1: 先补“背景洞”(0) ----------
    fg0 = (out != 0).astype(np.uint8)
    if fill_close_iter > 0:
        fg_filled = cv2.morphologyEx(fg0, cv2.MORPH_CLOSE, kernel, iterations=fill_close_iter)
    else:
        fg_filled = fg0

    holes = (fg_filled > 0) & (out == 0)  # 这些是需要从0变成某个前景类的像素

    # 用“邻近类扩张传播”给 holes 赋类（谁先扩张到就归谁，基本等价最近邻传播）
    if np.any(holes):
        for _ in range(propagate_iters):
            if not np.any(holes):
                break
            filled_any = False
            for l in labels:
                dil = cv2.dilate((out == l).astype(np.uint8), kernel, iterations=1).astype(bool)
                cand = holes & dil
                if np.any(cand):
                    out[cand] = l
                    holes[cand] = False
                    filled_any = True
            if not filled_any:
                break
        # 兜底：如果还有 holes（极少），直接放弃填充（或你也可改成填最近类）
        # out[holes] = 0

    # 记录“补洞后”的前景区域：后续不希望这里变回0
    fg_ref = (out != 0)

    # ---------- Step 2: 对前景每类做开闭 ----------
    smooth = np.zeros_like(out, dtype=np.int32)

    for l in labels:
        m = (out == l).astype(np.uint8)

        if open_iter > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        if close_iter > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        m = (m > 0)
        # 写回策略：不抢其它类像素（只写空白，或写回自己原来的位置）
        write = m & ((smooth == 0) | (out == l))
        smooth[write] = l

    # ---------- Step 3: 防止产生新的0洞（前景不应变背景） ----------
    missing = fg_ref & (smooth == 0)
    if np.any(missing):
        # 用 smooth 里已有的类向 missing 扩张填回去
        for _ in range(propagate_iters):
            if not np.any(missing):
                break
            filled_any = False
            for l in labels:
                dil = cv2.dilate((smooth == l).astype(np.uint8), kernel, iterations=1).astype(bool)
                cand = missing & dil
                if np.any(cand):
                    smooth[cand] = l
                    missing[cand] = False
                    filled_any = True
            if not filled_any:
                break
        # 兜底：还有 missing 就用原 out 填回（至少不变成0）
        if np.any(missing):
            smooth[missing] = out[missing]

    return smooth
# ---------------------------------------



size_list = [256, 512, 1024]
zoom_list = ["z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z18", "z22", "z25", "z28", "z38", "z48", "z60", "z90", "z150", "z225"]           

convert_table = [
    [664, 332, 166.0, 'nm/pixel'],
    [320, 160, 80.0, 'nm/pixel'],
    [210, 105, 52.0, 'nm/pixel'],
    [164, 82, 41.0, 'nm/pixel'],
    [125, 62, 31.0, 'nm/pixel'],
    [105, 53, 26.0, 'nm/pixel'],
    [94, 47, 23.0, 'nm/pixel'],
    [78, 39, 20.0, 'nm/pixel'],
    [70, 35, 18.0, 'nm/pixel'],
    [63, 31, 16.0, 'nm/pixel'],
    [59, 29, 15.0, 'nm/pixel'],
    [55, 27, 14.0, 'nm/pixel'],
    [51, 25, 13.0, 'nm/pixel'],
    [47, 23, 12.0, 'nm/pixel'],
    [43, 21, 11.0, 'nm/pixel'],
    [39, 20, 10.0, 'nm/pixel'],
    [35, 18, 9.0, 'nm/pixel'],
    [31, 16, 8.0, 'nm/pixel'],
    [27, 14, 7.0, 'nm/pixel'],
    [23, 12, 6.0, 'nm/pixel'],
    [20, 10, 5.0, 'nm/pixel'],
    [16, 8, 4.0, 'nm/pixel'],
    [12, 6, 3.0, 'nm/pixel'],
    [8, 4, 2.0, 'nm/pixel'],
    [4, 3, 1.5, 'nm/pixel'],
    [1, 2, 1.0, 'nm/pixel']
]

def norm_statistic(Input, device, std=None):
    mean = torch.mean(Input).to(device)
    mean_zero = torch.zeros_like(mean).to(device)
    std = torch.std(Input).to(device) if std == None else std
    output = transforms.Normalize(mean_zero, std)(Input)
    return output, mean_zero, std


def gen_FLIM(tm_data, intensity_image, t_range=[0, 6000]):
    #print(intensity_dir, asc_dir)
    #tm_data = np.loadtxt(asc_dir)[:512, :512]
    #tm_data = resize(np.loadtxt(asc_dir)[160:864, 160:864], (1024, 1024))
    tm_data[tm_data < t_range[0]] = t_range[0]
    tm_data[tm_data > t_range[1]] = t_range[1]
    #in_data = cv2.imdecode(np.fromfile(intensity_dir, dtype=np.uint8), flags=cv2.IMREAD_COLOR).astype(np.float64)
    #in_data = np.array(Image.open(intensity_dir))
    intensity_image = np.expand_dims(intensity_image, axis=-1)
    intensity_image = np.repeat(intensity_image, axis=-1, repeats=3)
    tm_DATA = (tm_data - np.min(tm_data)) / (np.max(tm_data) - np.min(tm_data))
    in_DATA = (intensity_image - np.min(intensity_image)) / (np.max(intensity_image) - np.min(intensity_image))
    hue_channel = 2 * tm_DATA / 3  # Map normalized lifetime to [0, 2/3]
    value_channel = in_DATA  # Use normalized intensity as the value channel

    h, w = intensity_image.shape[0:2]
    # Create HSV image
    hsv_img = np.zeros((h, w, 3))
    hsv_img[:, :, 0] = hue_channel  # Hue channel
    hsv_img[:, :, 1] = 1  # Fixed saturation at 1 (max saturation)
    hsv_img[:, :, 2] = value_channel[:,:,0]  # Value channel

    # Convert HSV to RGB
    rgb_color1 = hsv_to_rgb(hsv_img)
    rgb_color2 = np.zeros_like(rgb_color1)
    rgb_color2[:,:,0] = rgb_color1[:,:,0]
    rgb_color2[:,:,1] = rgb_color1[:,:,1]
    rgb_color2[:,:,2] = rgb_color1[:,:,2]

    return np.uint16(rgb_color2*255)


def calibrate_labels_by_thresh(thresh_image, sorted_image):
    """
    使用重合率逻辑校准标签，确保背景永远是 0。
    """
    unique_labels = np.unique(sorted_image)
    foreground_mask = (thresh_image > 0)
    
    label_scores = []
    for l in unique_labels:
        mask_l = (sorted_image == l)
        total_pixels = np.sum(mask_l)
        if total_pixels == 0: continue
        
        # 计算该标签在 thresh 前景区域的重合比例
        # 比例越低，越有可能是背景
        overlap_score = np.sum(mask_l & foreground_mask) / total_pixels
        label_scores.append({'label': l, 'score': overlap_score})
    
    # 1. 找到得分最低的 label 作为背景 (True Background)
    label_scores.sort(key=lambda x: x['score'])
    true_bkg_label = label_scores[0]['label']
    
    # 2. 准备重新映射
    calibrated_image = np.zeros_like(sorted_image)
    
    # 3. 将除背景外的其他标签按得分高低（或者原顺序）重新排列
    # 这里我们将剩下的标签标记为 1, 2, 3...
    new_label_idx = 1
    # 除去背景标签，按剩余标签在原图中的数值大小排序，保证逻辑一致性
    other_labels = [ls['label'] for ls in label_scores if ls['label'] != true_bkg_label]
    other_labels.sort() 
    
    for l in other_labels:
        calibrated_image[sorted_image == l] = new_label_idx
        new_label_idx += 1
        
    #print(f"校准完成: 原始标签 {true_bkg_label} 被识别为背景(0)")
    return calibrated_image


def DESM_IL_inference(read_dir, read_dir_asc, weights_dir, prior_weights_dir=None, device='cuda', 
                          save_dir=None, resize_to_const=False, tm_min=0, tm_max=5000, 
                          org_list=['Micro', 'Mito', 'Lyso'], factor_list=[1,1,1], show_image=False
                          ):
    file_list = natsort.natsorted(os.listdir(read_dir))    
    asc_list = natsort.natsorted(os.listdir(read_dir_asc))
    check_existence(save_dir)
    
    model = torch.load(weights_dir, weights_only=False)
    # Using DESM-IO as prior to define the lifetime distribution
    prior_model = torch.load(prior_weights_dir, weights_only=False)
    
    bar = tqdm(total=len(file_list))
    with torch.no_grad():
        for file_index in range(len(file_list)):
            #if index + 1 == 16 or index + 1 ==17:
            if 1:
                read_dir_file = os.path.join(read_dir, file_list[file_index])
                read_dir_ASC = os.path.join(read_dir_asc, asc_list[file_index])
                zoom = 0
                for i in range(len(zoom_list)):
                        if read_dir_file.find(zoom_list[i]) != -1:
                            zoom = zoom_list[i]
                #img = np.array(Image.open(read_dir_file))
                img = tifffile.imread(read_dir_file)
                tm_data = np.loadtxt(read_dir_ASC)#[:1008, :1008]

                # resize into a constant pixel size
                raw_h, raw_w = img.shape
                for zoom_index in range(len(zoom_list)):
                        if zoom == zoom_list[zoom_index]:
                            for size_index in range(len(size_list)):
                                if raw_h == size_list[size_index]:
                                    convert_ratio = convert_table[zoom_index][size_index]/20
                                    img = resize(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                                    img = img[int(0.05*(raw_h*convert_ratio)):int(0.95*(raw_h*convert_ratio)), int(0.05*(raw_w*convert_ratio)):int(0.95*(raw_w*convert_ratio))]
                                    tm_data = resize(tm_data, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                                    tm_data = tm_data[int(0.05*(raw_h*convert_ratio)):int(0.95*(raw_h*convert_ratio)), int(0.05*(raw_w*convert_ratio)):int(0.95*(raw_w*convert_ratio))]


                FLIM_image = gen_FLIM(tm_data=tm_data, intensity_image=img, t_range=[tm_min, tm_max])

                # write a FLIM color bar for reference
                temp_bar_int = np.linspace(5000, 0, 150)[:, np.newaxis] * np.ones((1, 15))
                temp_bar_lifetime = np.linspace(5000, 0, 150)[:, np.newaxis] * np.ones((1, 15))
                FLIM_bar = gen_FLIM(tm_data=temp_bar_lifetime, intensity_image=temp_bar_int, t_range=[tm_min, tm_max])
                #tifffile.imwrite(r"E:\Onedrive\Work\DSRM_paper\FLIM\Different tm range Experiment\Mito_Lyso_280_0.070_1\Unet_FLIM\0-5000.tif", FLIM_bar)

                #vector = FLIM_image.reshape(-1, 3)
                vector = tm_data.reshape(-1, 1)
                n_cluster = 3
                kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
                kmeans.fit(vector)
                labels = kmeans.labels_
                #segmented_image = labels.reshape(lifetime_image.shape)
                clustered_image = labels.reshape(tm_data.shape)

                # using thresh to make sure the background is zero
                thresh_image = deepcopy(img)
                thresh = threshold_otsu(thresh_image)
                thresh_image[thresh_image < thresh] = 0
                thresh_image[thresh_image >= thresh] = 1
                clustered_image = calibrate_labels_by_thresh(thresh_image, clustered_image)

                # to ensure dtype
                clustered_image = clustered_image.astype(np.uint8)
                thresh_image = thresh_image.astype(np.uint8)

                # most stable one so far
                '''clustered_image *= thresh_image
                clustered_image = cv2.GaussianBlur(clustered_image, (5, 5), 3)
                clustered_image = smooth_mask_edges(clustered_image, morph_ksize=(3, 3), iterations=5, blur_ksize=(5, 5))
                clustered_image *= thresh_image'''

                h, w = img.shape
                size = min(h, w)
                upper_limit = 1440
                if size > upper_limit:
                    num_row = (h // upper_limit) + 1
                    num_col = (w // upper_limit) + 1
                    size = upper_limit
                else:
                    upper_limit = (size // 16) * 16
                    num_row = (h // upper_limit)
                    num_col = (w // upper_limit)
                    size = upper_limit
                
                image_list = [[] for _ in range(num_row)]
                pred_list = [[] for _ in range(num_row)]
                FLIM_list = [[] for _ in range(num_row)]
                row_cood_list = []
                col_cood_list = []
                for row in range(num_row):
                    if (row+1)*size < h:
                        row_cood_list.append([row*size, (row+1)*size])
                    else:
                        row_cood_list.append([h-size, h])
                for col in range(num_col):
                    if (col+1)*size < w:
                        col_cood_list.append([col*size, (col+1)*size])
                    else:
                        col_cood_list.append([w-size, w])
                for row in range(num_row):
                    for col in range(num_col):
                        intensity_input = img[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        clustered_Input = clustered_image[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        tifffile.imwrite(os.path.join(save_dir, "{}_clustered_original.tif".format(file_index)), 
                                         np.uint8(clustered_Input/np.max(clustered_Input)*255))
                        #temp = image / np.max(image) * 255
                        #estimate_poisson_gaussian_noise(img=temp)
                        intensity_input = torch.tensor(np.float64(intensity_input), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                        intensity_input, mean, std = norm_statistic(intensity_input, device)

                        IO_output = prior_model(intensity_input)
                        IO_per_c_list = []
                        IO_prior_list = []
                        # test: generate the mPSF-convolved prior
                        IO_prior = np.zeros_like(clustered_Input)
                        for org_index in range(len(org_list)):
                            IO_per_c = IO_output[:, org_index:org_index+1, :, :]
                            #IO_per_c = torch.nn.functional.conv2d(IO_per_c, mPSF_list[org_index], padding='same')
                            IO_per_c = IO_per_c.squeeze(0).squeeze(0).cpu().numpy()
                            tifffile.imwrite(os.path.join(save_dir, "{}_{}_IO_output.tif".format(file_index, org_index)), IO_per_c)
                            IO_per_c_thresholded = deepcopy(IO_per_c)
                            #IO_per_c_thresholded, t = imagej_auto_default_mask(IO_per_c_thresholded)
                            IO_per_c_thresholded = make_threshold_otsu(IO_per_c)
                            tifffile.imwrite(os.path.join(save_dir, "{}_{}_IO_output_thresholded.tif".format(file_index, org_index)), IO_per_c_thresholded)
                            IO_prior[IO_per_c_thresholded != 0] = 1
                            IO_per_c_list.append(IO_per_c)
                            IO_prior_list.append(IO_per_c_thresholded)

                        IO_prior_background = np.zeros_like(IO_prior)
                        IO_prior_background[IO_prior == 0] = 1
                        tifffile.imwrite(os.path.join(save_dir, "{}_IO_prior.tif".format(file_index)), IO_prior)
                        clustered_Input = IO_prior * clustered_Input
                        
                        permuted_clustered_Input = clustered_Input.copy()
                        if len(org_list) == 3:
                            IO_prior_list = [IO_prior_list[0] + IO_prior_list[2], IO_prior_list[1]]
                        elif len(org_list) == 2:
                            IO_prior_list = [IO_prior_list[0], IO_prior_list[1]]

                        # remapping the labels, according to IO priors
                        permuted_clustered_Input, mapping = relabel_by_priors(
                            clustered_input=clustered_Input, 
                            IO_prior_background=IO_prior_background, 
                            IO_prior_list=IO_prior_list
                        )

                        tifffile.imwrite(os.path.join(save_dir, "{}_clustered_permuted.tif".format(file_index)), permuted_clustered_Input)
                        # fill morphological holes and smooth the edges
                        # default: permuted_clustered_Input = clean_labels_fill_bg_then_smooth(permuted_clustered_Input, ksize=5, open_iter=2, close_iter=2)
                        permuted_clustered_Input = clean_labels_fill_bg_then_smooth(permuted_clustered_Input, ksize=5, open_iter=2, close_iter=2)

                        clustered_Input = permuted_clustered_Input
                        tifffile.imwrite(os.path.join(save_dir, f"{file_index}_clustered_image_final.tif"), np.uint8(clustered_Input/np.max(clustered_Input)*255))
                        
                        clustered_Input = torch.tensor(np.float64(clustered_Input), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                        clustered_Input, _, _  = norm_statistic(clustered_Input, device=device)
                        Input = torch.concat([intensity_input, clustered_Input], dim=1)

                        pred = model(Input)


                        intensity_input = intensity_input*std+mean
                        pred = pred*std+mean

                        intensity_input[intensity_input<0] = 0
                        pred[pred<0] = 0
                        image_list[row].append(to_cpu(intensity_input.squeeze(0).squeeze(0)))
                        pred_list[row].append(to_cpu(pred.squeeze(0).permute(1,2,0)))
                        FLIM_list[row].append(to_cpu(clustered_Input.squeeze(0).squeeze(0)))
                full_pred = np.zeros((h, w ,pred_list[0][0].shape[-1]))
                for row in range(num_row):
                    for col in range(num_col):
                        full_pred[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]] = pred_list[row][col]
                h, w = img.shape
                img = np.uint8(img/np.max(img)*255)
                clustered_image = np.uint8(clustered_image/np.max(clustered_image)*255)
                if resize_to_const:
                    img = resize(img, (1024, 1024))
                    full_pred = resize(full_pred, (1024,1024,full_pred.shape[-1]))
                    clustered_image = resize(clustered_image, (1024, 1024))
                    FLIM_image = resize(FLIM_image, (1024, 1024))
                    tm_data = resize(tm_data, (1024, 1024))
                if file_list[file_index].find('tif') != -1:
                    tifffile.imwrite(os.path.join(save_dir, f"{file_index}_intensity_input.tif"), img)
                    tifffile.imwrite(os.path.join(save_dir, f"{file_index}_FLIM_image.tif"), FLIM_image)
                    tifffile.imwrite(os.path.join(save_dir, f"{file_index}_tm_data.tif"), tm_data)
                    tifffile.imwrite(os.path.join(save_dir, f"{file_index}_intensity_input_thresholded.tif"), thresh_image)
                    full_pred = np.uint8(full_pred/np.max(full_pred)*255)
                    for org_index in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, f"{file_index}_{org_index}_SR.tif"), np.uint8(full_pred[:,:,org_index]))
                
                
                SR_image = np.zeros_like(img, dtype=np.uint16)
                for i in range(full_pred.shape[-1]):
                    SR_image += full_pred[:,:,i]
                tifffile.imwrite(os.path.join(save_dir, f"{file_index}_SR_only.tif"), SR_image)

                bar.set_description("{}".format(zoom))
                bar.update(1)

                if show_image == "Layout_1":
                    if file_index == 0:
                        plt.figure(figsize=(12, 12))
                        plt.subplot(221)
                        plt.imshow(img, cmap='gray')
                        plt.title('Intensity Input (single-channel LR)')
                        plt.axis('off')

                        plt.subplot(222)
                        plt.imshow(tm_data, cmap='gray')
                        plt.title('lifetime distribution')
                        plt.axis('off')

                        plt.subplot(223)
                        plt.imshow(FLIM_image)
                        plt.title('FLIM Image')
                        plt.axis('off')

                        plt.subplot(224)
                        if full_pred.shape[-1] == 3:
                            pred_rgb = np.zeros_like(full_pred)
                            pred_rgb[:,:,0] = full_pred[:,:,0]
                            pred_rgb[:,:,1] = full_pred[:,:,1]
                            pred_rgb[:,:,2] = full_pred[:,:,2]
                            plt.imshow(pred_rgb.astype(np.uint8))
                        else:
                            plt.imshow(full_pred[:,:,0], cmap='gray')
                        plt.title('DESM-IL Prediction (multi-channel SR)')
                        plt.axis('off')

                        plt.tight_layout()
                        plt.show()
                elif show_image == "Layout_2":
                    if file_index == 0:
                        plt.figure(figsize=(12, 12))
                        plt.subplot(231)
                        plt.imshow(img, cmap='gray')
                        plt.title('Intensity Input (single-channel LR)')
                        plt.axis('off')

                        plt.subplot(232)
                        plt.imshow(tm_data, cmap='gray')
                        plt.title('lifetime distribution')
                        plt.axis('off')

                        plt.subplot(233)
                        
                        plt.imshow(full_pred[:,:,0], cmap='gray')
                        plt.title('DESM-IL Prediction (Mitochondria)')
                        plt.axis('off')

                        plt.subplot(234)
                        plt.imshow(full_pred[:,:,1], cmap='gray')
                        plt.title('DESM-IL Prediction (Lysosomes)')
                        plt.axis('off')

                        plt.subplot(235)
                        if FLIM_image.shape[-1] == 3:
                            plt.imshow(FLIM_image.astype(np.uint8))
                        else:
                            plt.imshow(FLIM_image, cmap='gray')
                        plt.title('FLIM Image')
                        plt.axis('off')

                        plt.tight_layout()
                        plt.show()


if __name__ == "__main__":
    # Mito Lyso dynamics 1
    if 0:
        read_dir = r'E:\Onedrive\Work\DSRM_paper\FLIM\Dynamics\Mito_Lyso_20251225\Intensity'
        read_dir_asc = r'E:\Onedrive\Work\DSRM_paper\FLIM\Dynamics\Mito_Lyso_20251225\ASC'
        #weights_dir = r"E:\Onedrive\Work\codes\microscopy_decouple\validation\DSCM_Mito_Lyso_280_0.070_1_DSCM_FLIM_384_Unet_fea_loss_0.01_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth"
        weights_dir = r"E:\Onedrive\Work\codes\microscopy_decouple\validation\DSCM_FLIM_new_Mito_Lyso_280_0.070_1_DSCM_384_Unet_fea_loss_0.01_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth"
        prior_weights_dir = r"E:\Onedrive\Work\codes\microscopy_decouple\validation\DSCM_Mito_Lyso_280_0.070_1_DSCM_384_Unet_fea_loss_0.1_SSIM_loss_1_grad_loss_1_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth"
        save_dir = r'E:\Onedrive\Work\DSRM_paper\FLIM\Dynamics\Mito_Lyso_20251225\Results'
        DESM_IL_inference(read_dir=read_dir, read_dir_asc=read_dir_asc, weights_dir=weights_dir, prior_weights_dir=prior_weights_dir,
                              save_dir=save_dir, resize_to_const=False, tm_min=0, tm_max=5000, 
                              org_list=['Mito', 'Lyso'], factor_list=[1, 1]) 
    # Mito Lyso dynamics 2
    if 0:
        read_dir = r'D:\Onedrive\Work\DSRM_paper\FLIM\Dynamics\Mito_Lyso_Previous\20260121\Intensity'
        read_dir_asc = r'D:\Onedrive\Work\DSRM_paper\FLIM\Dynamics\Mito_Lyso_Previous\20260121\ASC'
        #weights_dir = r"E:\Onedrive\Work\codes\microscopy_decouple\validation\DSCM_Mito_Lyso_280_0.070_1_DSCM_FLIM_384_Unet_fea_loss_0.01_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth"
        weights_dir = r"D:\Onedrive\Work\codes\microscopy_decouple\validation\DSCM_FLIM_new_Mito_Lyso_280_0.070_1_DSCM_384_Unet_fea_loss_0.01_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth"
        prior_weights_dir = r"D:\Onedrive\Work\codes\microscopy_decouple\validation\DSCM_Mito_Lyso_280_0.070_1_DSCM_384_Unet_fea_loss_0.1_SSIM_loss_1_grad_loss_1_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth"
        save_dir = r'D:\Onedrive\Work\DSRM_paper\FLIM\Dynamics\Mito_Lyso_Previous\20260121\Results'
        DESM_IL_inference(read_dir=read_dir, read_dir_asc=read_dir_asc, weights_dir=weights_dir, prior_weights_dir=prior_weights_dir,
                              save_dir=save_dir, resize_to_const=False, tm_min=0, tm_max=5500, 
                              org_list=['Mito', 'Lyso'], factor_list=[1, 1]) 
    # Micro Mito Lyso FLIM 20260202
    if 1:
        # Micro Mito Lyso
        read_dir_intensity = r'visualization\real-world data\DESM_IL_Microtubules_Mitochondria_Lysosomes\Intensity'
        save_dir = r'visualization\real-world data\DESM_IL_Microtubules_Mitochondria_Lysosomes\Results'
        read_dir_asc = r'visualization\real-world data\DESM_IL_Microtubules_Mitochondria_Lysosomes\ASC'
        weights_dir = r"Trained_models\DESM_IL\DESM_IL_Micro_Mito_Lyso_resolution_280nm_Poisson_0.280_avg_1.pth"
        prior_weights_dir = r"Trained_models\DESM_IL\DESM_IO_Micro_Mito_Lyso_resolution_280nm_Poisson_0.280_avg_1.pth"
        
        DESM_IL_inference(read_dir=read_dir_intensity, read_dir_asc=read_dir_asc, weights_dir=weights_dir, prior_weights_dir=prior_weights_dir,
                                save_dir=save_dir, resize_to_const=False, tm_min=0, tm_max=5500, 
                                org_list=['Mito', 'Lyso'], factor_list=[1, 1]) 

    