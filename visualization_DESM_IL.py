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
from threadpoolctl import threadpool_limits
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb

device = 'cuda'

def adaptive_binarize_auto(img, min_obj=0, fill_holes=0):
    """img: 2D ndarray (float / uint16 ) Return: 0/1 uint8 mask Parameter min_obj / fill_holes (0)"""
    x = img.astype(np.float32, copy=False)

    # 1) Normalize, AvoidThreshold
    p1, p99 = np.percentile(x, (1, 99))
    x = np.clip(x, p1, p99)
    x = (x - p1) / (p99 - p1 + 1e-8)

    H, W = x.shape

    # 2) : min(H,W)/16, , 31
    win = max(31, (min(H, W) // 16) | 1)
    sigma = win / 6.0  # Window()

    # 3) (Background) + (Noise)
    mu = gaussian(x, sigma=sigma, preserve_range=True)
    res = x - mu
    mad = np.median(np.abs(res - np.median(res)))
    sigma_n = 1.4826 * mad  # robust noise std

    # : Noise, Threshold; Need
    thr = mu + 1.0 * sigma_n

    mask = x > thr

    # 4) (0)
    if min_obj > 0:
        mask = remove_small_objects(mask, min_size=min_obj)
    if fill_holes > 0:
        mask = remove_small_holes(mask, area_threshold=fill_holes)

    return mask.astype(np.uint8)


def imagej_auto_default_mask(img, clip_percentile=(0.5, 99.5)):
    """ImageJ Threshold -> Auto (Default) Return: mask(uint8 0/1), threshold(0..255)"""
    x = img.astype(np.float32, copy=False)

    lo, hi = np.percentile(x, clip_percentile)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)

    x8 = img_as_ubyte(x)  # 0..255
    t = threshold_isodata(x8)  # IsoDataThreshold
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

def relabel_background_to_zero(clustered_input, IO_prior_background):
    """" label Background", 0. labels 1,2,3,... Parameter ---- clustered_input : ndarray 2D/3D label image, IO_prior_background : ndarray Background prior mask, >0 Background Return ---- relabeled : ndarray(int32) Background 0, Class 1..K bg_label : int clustered_input Background label mapping : dict old_label -> new_label score_table : list[dict] label Background, debug"""
    clustered = np.asarray(clustered_input)
    bg_mask = (np.asarray(IO_prior_background) > 0)

    labels = [int(x) for x in np.unique(clustered)]
    masks = {l: (clustered == l) for l in labels}

    score_table = []
    for l in labels:
        mask_l = masks[l]
        area_l = np.count_nonzero(mask_l)
        overlap_bg = np.count_nonzero(mask_l & bg_mask)

        # "Background"BackgroundClass: ClassBackground prior
        purity_bg = overlap_bg / (area_l + 1e-8)

        score_table.append({
            "label": l,
            "area": int(area_l),
            "overlap_bg": int(overlap_bg),
            "purity_bg": float(purity_bg),
        })

    # purity_bg , overlap_bg , Avoid
    score_table.sort(key=lambda d: (d["purity_bg"], d["overlap_bg"]), reverse=True)
    bg_label = int(score_table[0]["label"])

    relabeled = np.zeros_like(clustered, dtype=np.int32)
    mapping = {bg_label: 0}

    next_id = 1
    for l in labels:
        if l == bg_label:
            continue
        relabeled[masks[l]] = next_id
        mapping[l] = next_id
        next_id += 1

    return relabeled, bg_label, mapping, score_table


def relabel_foreground_by_priors(clustered_input_bg0, IO_prior_list, keep_unmatched=True):
    """"Background 0", labels prior . Background. Parameter ---- clustered_input_bg0 : ndarray Background 0 label image IO_prior_list : list of ndarray Class prior masks, Output 1,2,3,... keep_unmatched : bool : True -> False -> 0() Return ---- fixed : ndarray(int32) Background=0, prior i Class -> i+1 mapping : dict old_fg_label -> new_fg_label score_debug : dict"""
    clustered = np.asarray(clustered_input_bg0)
    priors = [(np.asarray(p) > 0) for p in IO_prior_list]

    fixed = np.zeros_like(clustered, dtype=np.int32)
    fixed[clustered == 0] = 0

    labels = [int(x) for x in np.unique(clustered) if int(x) != 0]
    masks = {l: (clustered == l) for l in labels}

    unused = labels.copy()
    mapping = {0: 0}
    score_debug = {}

    for new_id, prior_mask in enumerate(priors, start=1):
        if not unused:
            break

        current_scores = []
        for l in unused:
            mask_l = masks[l]
            inter = np.count_nonzero(mask_l & prior_mask)
            union = np.count_nonzero(mask_l | prior_mask)
            iou = inter / (union + 1e-8)

            current_scores.append({
                "old_label": l,
                "intersection": int(inter),
                "iou": float(iou),
            })

        # IoU, intersection
        current_scores.sort(key=lambda d: (d["iou"], d["intersection"]), reverse=True)
        best_old = int(current_scores[0]["old_label"])

        mapping[best_old] = new_id
        fixed[masks[best_old]] = new_id
        unused.remove(best_old)

        score_debug[new_id] = current_scores

    if keep_unmatched and len(unused) > 0:
        next_id = len(priors) + 1
        for l in unused:
            mapping[l] = next_id
            fixed[masks[l]] = next_id
            next_id += 1

    return fixed, mapping, score_debug


def clean_labels_fill_bg_then_smooth(lbl_in,
                                    ksize=3,
                                    fill_close_iter=1,   # : Background(maskclosing)
                                    open_iter=1,         # : Class()
                                    close_iter=1,        # : Class()
                                    propagate_iters=100  # Implementation note.
                                    ):
    """lbl_in: 2D label image, background=0, classes=1..n (int) : A) 0 (Background"") B) Class/ C) Background(0) return: lbl_out (int32)"""
    lbl = np.asarray(lbl_in)
    out = lbl.copy().astype(np.int32)

    labels = [int(x) for x in np.unique(out) if int(x) != 0]
    if len(labels) == 0:
        return out  # Background

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # ---------- Step 1: "Background"(0) ----------
    fg0 = (out != 0).astype(np.uint8)
    if fill_close_iter > 0:
        fg_filled = cv2.morphologyEx(fg0, cv2.MORPH_CLOSE, kernel, iterations=fill_close_iter)
    else:
        fg_filled = fg0

    holes = (fg_filled > 0) & (out == 0)  # Need0ClassPixel

    # "Class" holes Class(, )
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
        # : If holes(), (Class)
        # out[holes] = 0

    # "": 0
    fg_ref = (out != 0)

    # ---------- Step 2: Class ----------
    smooth = np.zeros_like(out, dtype=np.int32)

    for l in labels:
        m = (out == l).astype(np.uint8)

        if open_iter > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        if close_iter > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        m = (m > 0)
        # : ClassPixel(, )
        write = m & ((smooth == 0) | (out == l))
        smooth[write] = l

    # ---------- Step 3: Prevent0(Background) ----------
    missing = fg_ref & (smooth == 0)
    if np.any(missing):
        # smooth Class missing
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
        # : missing out (0)
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
    """, EnsureBackground 0."""
    unique_labels = np.unique(sorted_image)
    foreground_mask = (thresh_image > 0)
    
    label_scores = []
    for l in unique_labels:
        mask_l = (sorted_image == l)
        total_pixels = np.sum(mask_l)
        if total_pixels == 0: continue
        
        # Calculate thresh
        # , Background
        overlap_score = np.sum(mask_l & foreground_mask) / total_pixels
        label_scores.append({'label': l, 'score': overlap_score})
    
    # 1. label Background (True Background)
    label_scores.sort(key=lambda x: x['score'])
    true_bkg_label = label_scores[0]['label']
    
    # 2.
    calibrated_image = np.zeros_like(sorted_image)
    
    # 3. Background()
    # 1, 2, 3...
    new_label_idx = 1
    # Background, SizeSort,
    other_labels = [ls['label'] for ls in label_scores if ls['label'] != true_bkg_label]
    other_labels.sort() 
    
    for l in other_labels:
        calibrated_image[sorted_image == l] = new_label_idx
        new_label_idx += 1
        
    # print(f": {true_bkg_label} Background(0)")
    return calibrated_image


def DESM_IL_inference(read_dir, read_dir_asc, weights_dir, prior_weights_dir=None, device='cuda', 
                          save_dir=None, resize_to_const=False, tm_min=0, tm_max=5000, 
                          IO_prior_dilate_iter=0, smooth_iter=2, 
                          org_list=['Micro', 'Mito', 'Lyso'], show_image=False,
                          full_field=False, resample_input=True, normalize_to_255=False,
                          crop_to_multiple=False, save_float=False
                          ):
    file_list = natsort.natsorted(f for f in os.listdir(read_dir) if f.lower().endswith(('.tif', '.tiff')))
    asc_list = natsort.natsorted(f for f in os.listdir(read_dir_asc) if f.lower().endswith('.asc'))
    if len(file_list) != len(asc_list):
        raise ValueError('Intensity and ASC file counts differ')
    os.makedirs(save_dir, exist_ok=True)
    records = []
    
    model = torch.load(weights_dir, map_location='cpu', weights_only=False).to(device).eval().requires_grad_(False)
    # Using DESM-IO as prior to define the lifetime distribution
    prior_model = torch.load(prior_weights_dir, map_location='cpu', weights_only=False).to(device).eval().requires_grad_(False)
    for network in (model, prior_model):
        for layer in network.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.train()
                layer.track_running_stats = False
    
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
                intensity_image = tifffile.imread(read_dir_file)
                lifetime_distribution = np.loadtxt(read_dir_ASC)#[:1008, :1008]

                # resize into a constant pixel size
                raw_h, raw_w = intensity_image.shape
                for zoom_index in range(len(zoom_list)):
                        if zoom == zoom_list[zoom_index]:
                            for size_index in range(len(size_list)):
                                if raw_h == size_list[size_index]:
                                    convert_ratio = convert_table[zoom_index][size_index]/20
                                    if resample_input:
                                        intensity_image = resize(intensity_image, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                                    intensity_image = intensity_image[int(0.05*(raw_h*convert_ratio)):int(0.95*(raw_h*convert_ratio)), int(0.05*(raw_w*convert_ratio)):int(0.95*(raw_w*convert_ratio))]
                                    if resample_input:
                                        lifetime_distribution = resize(lifetime_distribution, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                                    lifetime_distribution = lifetime_distribution[int(0.05*(raw_h*convert_ratio)):int(0.95*(raw_h*convert_ratio)), int(0.05*(raw_w*convert_ratio)):int(0.95*(raw_w*convert_ratio))]

                prior_intensity = intensity_image.astype(np.float32)
                if normalize_to_255:
                    if resample_input:
                        intensity_image = prior_intensity * np.float32(255 / intensity_image.max())
                    else:
                        intensity_image = (intensity_image.astype(np.float64) / intensity_image.max() * 255).astype(np.float32)
                        prior_intensity = intensity_image.copy()
                FLIM_image = gen_FLIM(tm_data=lifetime_distribution, intensity_image=intensity_image, t_range=[tm_min, tm_max])

                # write a FLIM color bar for reference
                temp_bar_int = np.linspace(5000, 0, 150)[:, np.newaxis] * np.ones((1, 15))
                temp_bar_lifetime = np.linspace(5000, 0, 150)[:, np.newaxis] * np.ones((1, 15))
                FLIM_bar = gen_FLIM(tm_data=temp_bar_lifetime, intensity_image=temp_bar_int, t_range=[tm_min, tm_max])
                # tifffile.imwrite(os.path.join(save_dir, "FLIM_bar.tif"), FLIM_bar)

                #vector = FLIM_image.reshape(-1, 3)
                vector = lifetime_distribution.reshape(-1, 1)
                n_cluster = 3
                kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
                with threadpool_limits(limits=4):
                    kmeans.fit(vector)
                labels = kmeans.labels_
                clustered_image = labels.reshape(lifetime_distribution.shape)

                # to ensure dtype
                clustered_image = clustered_image.astype(np.uint8)

                if crop_to_multiple:
                    hh, ww = (intensity_image.shape[0] // 16) * 16, (intensity_image.shape[1] // 16) * 16
                    intensity_image = intensity_image[:hh, :ww]
                    prior_intensity = prior_intensity[:hh, :ww]
                    clustered_image = clustered_image[:hh, :ww]
                    lifetime_distribution = lifetime_distribution[:hh, :ww]
                    FLIM_image = FLIM_image[:hh, :ww]
                h, w = intensity_image.shape
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
                final_clustered_image_list = [[] for _ in range(num_row)]
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
                if full_field:
                    num_row = num_col = 1
                    row_cood_list = [[0, h]]
                    col_cood_list = [[0, w]]
                    image_list = [[]]
                    pred_list = [[]]
                    final_clustered_image_list = [[]]
                for row in range(num_row):
                    for col in range(num_col):
                        intensity_input = intensity_image[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        clustered_Input = clustered_image[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        tifffile.imwrite(os.path.join(save_dir, "{}_clustered_original.tif".format(file_index)), 
                                         np.uint8(clustered_Input/np.max(clustered_Input)*255))
                        
                        intensity_input = torch.tensor(intensity_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        std = intensity_input.std().clamp_min(1e-8)
                        mean = torch.zeros_like(std)
                        intensity_input = intensity_input / std
                        tile_h, tile_w = intensity_input.shape[-2:]
                        prior_tile = intensity_image[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]].astype(np.float32)
                        prior_input = torch.from_numpy(prior_tile)[None, None]
                        # Preserve the arithmetic used to generate the original priors.
                        # The resampled three-channel example used CPU normalization
                        # and restored intensity units; the native live example used GPU statistics.
                        if not resample_input:
                            prior_input = prior_input.to(device)
                        prior_std = prior_input.std().clamp_min(1e-8)
                        prior_input = prior_input / prior_std
                        prior_input = F.pad(prior_input.to(device), (0, (-tile_w) % 16, 0, (-tile_h) % 16), mode='reflect')
                        IO_output = prior_model(prior_input)[:, :, :tile_h, :tile_w]
                        if resample_input:
                            IO_output = IO_output * prior_std
                        IO_per_c_list = []
                        IO_prior_list = []
                        # test: generate the mPSF-convolved prior
                        IO_prior = np.zeros_like(clustered_Input)
                        for org_index in range(len(org_list)):
                            IO_per_c = IO_output[:, org_index:org_index+1, :, :]
                            IO_per_c = IO_per_c.squeeze(0).squeeze(0).cpu().numpy()
                            
                            tifffile.imwrite(os.path.join(save_dir, "{}_{}_IO_output.tif".format(file_index, org_index)), IO_per_c)
                            IO_per_c_thresholded = deepcopy(IO_per_c)
                            
                            IO_per_c_thresholded = make_threshold_otsu(IO_per_c)
                            if IO_prior_dilate_iter > 0:
                                IO_per_c_thresholded = cv2.dilate(IO_per_c_thresholded.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=IO_prior_dilate_iter)
                            tifffile.imwrite(os.path.join(save_dir, "{}_{}_IO_output_thresholded.tif".format(file_index, org_index)), IO_per_c_thresholded)
                            IO_prior[IO_per_c_thresholded != 0] = 1
                            
                            IO_per_c_list.append(IO_per_c)
                            IO_prior_list.append(IO_per_c_thresholded)

                        IO_prior_background = np.zeros_like(IO_prior)
                        IO_prior_background[IO_prior == 0] = 1
                        tifffile.imwrite(os.path.join(save_dir, "{}_IO_prior.tif".format(file_index)), IO_prior)
   
                        permuted_clustered_Input = clustered_Input.copy()
                        if len(org_list) == 3:
                            IO_prior_list = [IO_prior_list[0] + IO_prior_list[2], IO_prior_list[1]]
                        elif len(org_list) == 2:
                            IO_prior_list = [IO_prior_list[0], IO_prior_list[1]]


                        # Step 1: BackgroundClass, Background 0
                        clustered_bg0, bg_label, bg_mapping, bg_scores = relabel_background_to_zero(
                            clustered_input=clustered_Input,
                            IO_prior_background=IO_prior_background
                        )

                        # Step 2: foreground labels prior
                        permuted_clustered_Input, fg_mapping, fg_scores = relabel_foreground_by_priors(
                            clustered_input_bg0=clustered_bg0,
                            IO_prior_list=IO_prior_list,
                            keep_unmatched=True
                        )

                        permuted_clustered_Input = IO_prior * permuted_clustered_Input

                        tifffile.imwrite(os.path.join(save_dir, "{}_clustered_permuted.tif".format(file_index)), permuted_clustered_Input)
                        # fill morphological holes and smooth the edges
                        if smooth_iter > 0:
                            permuted_clustered_Input = clean_labels_fill_bg_then_smooth(permuted_clustered_Input, ksize=5, open_iter=smooth_iter, close_iter=smooth_iter)

                        clustered_Input = permuted_clustered_Input
                        tifffile.imwrite(os.path.join(save_dir, f"{file_index}_clustered_image_final.tif"), np.uint8(clustered_Input/np.max(clustered_Input)*255))
                        
                        if save_float:
                            tifffile.imwrite(os.path.join(save_dir, f"{file_index}_lifetime_labels.tif"), clustered_Input.astype(np.float32))
                            tifffile.imwrite(os.path.join(save_dir, f"{file_index}_input_float.tif"), prior_tile)
                        clustered_Input = torch.tensor(clustered_Input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        clustered_Input = clustered_Input / clustered_Input.std().clamp_min(1e-8)
                        Input = torch.concat([intensity_input, clustered_Input], dim=1)
                        Input = F.pad(Input.to(device), (0, (-tile_w) % 16, 0, (-tile_h) % 16), mode='reflect')
                        pred = model(Input)[:, :, :tile_h, :tile_w]
                        intensity_input = intensity_input.to(device)
                        clustered_Input = clustered_Input.to(device)

                        intensity_input = intensity_input*std+mean
                        pred = pred*std+mean

                        intensity_input[intensity_input<0] = 0
                        pred[pred<0] = 0
                        image_list[row].append(to_cpu(intensity_input.squeeze(0).squeeze(0)))
                        pred_list[row].append(to_cpu(pred.squeeze(0).permute(1,2,0)))
                        final_clustered_image_list[row].append(to_cpu(clustered_Input.squeeze(0).squeeze(0)))
                full_pred = np.zeros((h, w ,pred_list[0][0].shape[-1]))
                for row in range(num_row):
                    for col in range(num_col):
                        full_pred[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]] = pred_list[row][col]
                if save_float:
                    for channel in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, f"{file_index}_{channel}_SR_float.tif"), full_pred[:, :, channel].astype(np.float32))
                records.append({'index': file_index, 'source': file_list[file_index], 'asc': asc_list[file_index],
                                'background_mapping': bg_mapping, 'foreground_mapping': fg_mapping,
                                'shape': [h, w], 'from_raw_ASC': True})
                h, w = intensity_image.shape
                intensity_image = np.uint8(intensity_image/np.max(intensity_image)*255)
                clustered_image = np.uint8(clustered_image/np.max(clustered_image)*255)
                if resize_to_const:
                    intensity_image = resize(intensity_image, (1024, 1024))
                    full_pred = resize(full_pred, (1024,1024,full_pred.shape[-1]))
                    clustered_image = resize(clustered_image, (1024, 1024))
                    FLIM_image = resize(FLIM_image, (1024, 1024))
                    lifetime_distribution = resize(lifetime_distribution, (1024, 1024))
                if file_list[file_index].find('tif') != -1:
                    tifffile.imwrite(os.path.join(save_dir, f"{file_index}_intensity_input.tif"), intensity_image)
                    tifffile.imwrite(os.path.join(save_dir, f"{file_index}_FLIM_image.tif"), FLIM_image)
                    tifffile.imwrite(os.path.join(save_dir, f"{file_index}_tm_data.tif"), lifetime_distribution)
                    full_pred = np.uint8(full_pred/np.max(full_pred)*255)
                    for org_index in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, f"{file_index}_{org_index}_SR.tif"), np.uint8(full_pred[:,:,org_index]))
                
                SR_image = np.zeros_like(intensity_image, dtype=np.uint16)
                for i in range(full_pred.shape[-1]):
                    SR_image += full_pred[:,:,i]
                tifffile.imwrite(os.path.join(save_dir, f"{file_index}_SR_only.tif"), SR_image)

                bar.set_description("{}".format(zoom))
                bar.update(1)

                if show_image == "Layout_1":
                    if file_index == 0:
                        plt.figure(figsize=(12, 12))
                        plt.subplot(221)
                        plt.imshow(intensity_image, cmap='gray')
                        plt.title('Intensity Input (single-channel LR)')
                        plt.axis('off')

                        plt.subplot(222)
                        plt.imshow(to_cpu(clustered_Input)[0,0], cmap='gray')
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
                        #plt.show()
                elif show_image == "Layout_2":
                    if file_index == 0:
                        plt.figure(figsize=(12, 12))
                        plt.subplot(231)
                        plt.imshow(intensity_image, cmap='gray')
                        plt.title('Intensity Input (single-channel LR)')
                        plt.axis('off')

                        plt.subplot(232)
                        plt.imshow(lifetime_distribution, cmap='gray')
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
                        #plt.show()
            
            #output_stack.append(full_pred.astype(np.uint16))
            #clustered_stack.append(to_cpu(clustered_Input)[0,0].astype(np.uint16))
            #tifffile.imwrite(os.path.join(Comparison_dir, f"{prior_dilate_index}_{smooth_index}_Output.tif"), full_pred.astype(np.uint16))
            #tifffile.imwrite(os.path.join(Comparison_dir, f"{prior_dilate_index}_{smooth_index}_clustered_image.tif"), to_cpu(clustered_Input)[0,0].astype(np.uint16))

    return records
