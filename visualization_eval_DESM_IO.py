import tifffile, time
from tqdm import tqdm

from utils import *

from torchvision import transforms
import torch.nn.functional as F


def norm_statistic(Input, device, std=None):
    mean = torch.mean(Input).to(device)
    mean_zero = torch.zeros_like(mean).to(device)
    std = torch.std(Input).to(device) #if std == None else std
    output = transforms.Normalize(mean_zero, std)(Input)
    return output, mean_zero, std


def DESM_IO_inference_with_reference(read_dir, weights_dir, weights_dir_dn=None, 
                     device='cuda', save_dir=False, name=None, show_image=False, full_field=False, save_float=False):
    file_list = natsort.natsorted(os.listdir(read_dir))    
    os.makedirs(save_dir, exist_ok=True)
    
    model = torch.load(weights_dir, map_location='cpu', weights_only=False).to(device).eval().requires_grad_(False)
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.train()
            layer.track_running_stats = False
    save_list = []
    bar = tqdm(total=len(file_list))
    with torch.no_grad():
        for file_index in range(len(file_list)):
            if file_list[file_index].find('.tif') != -1 or file_list[file_index].find('.png') != -1:
                read_dir_file = os.path.join(read_dir, file_list[file_index])
                img = tifffile.imread(read_dir_file)
                if len(img.shape) == 3:
                    GT_2 = img[0,:,:]
                    GT_1 = img[2,:,:]
                    img = img[1,:,:]
                raw_h, raw_w = img.shape
                # the pixel size of the raw image is 22.7 nm, and the pixel size of the training data is 20 nm
                convert_ratio = 22.7 / 20
                img = resize(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                GT_1 = resize(GT_1, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                GT_2 = resize(GT_2, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))

                h, w = img.shape
                size = min(h, w)
                upper_limit = 2048
                if size > upper_limit:
                    num_row = (h // upper_limit)
                    num_col = (w // upper_limit)
                    size = upper_limit
                else:
                    upper_limit = (size // 16) * 16
                    size = upper_limit
                    num_row = 1
                    num_col = 1
                
                image_list = []
                pred_list = []
                GT_list = []
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
                    row_cood_list, col_cood_list = [[0, h]], [[0, w]]
                for row in range(num_row):
                    image_list.append([])
                    pred_list.append([])
                    GT_list.append([])
                    for col in range(num_col):
                        image = img[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        GT_1_image = GT_1[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        GT_2_image = GT_2[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        
                        image = torch.from_numpy(np.asarray(image, np.float32)[None, None])
                        image, mean, std = norm_statistic(image, 'cpu')
                        tile_h, tile_w = image.shape[-2:]
                        padded = F.pad(image.to(device), (0, (-tile_w) % 16, 0, (-tile_h) % 16), mode='reflect')
                        pred = model(padded)[:, :, :tile_h, :tile_w]
                        image = image*std+mean
                        pred = pred*std+mean
                        image[image<0] = 0
                        if not save_float:
                            pred[pred<0] = 0
                        image = to_cpu(image.squeeze(0).squeeze(0))
                        pred = to_cpu((pred.squeeze(0).permute(1,2,0))) 
                        GT = np.transpose(np.stack((GT_1_image, GT_2_image), axis=0), (1,2,0))

                        image_list[row].append(image)
                        pred_list[row].append(pred)
                        GT_list[row].append(GT)
                full_pred = np.zeros((h, w ,pred_list[0][0].shape[-1]))
                for row in range(num_row):
                    for col in range(num_col):
                        full_pred[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]] = pred_list[row][col]
                full_GT = np.zeros((h, w ,GT_list[0][0].shape[-1]))
                for row in range(num_row):
                    for col in range(num_col):
                        full_GT[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]] = GT_list[row][col]
                if save_float:
                    stem = os.path.splitext(file_list[file_index])[0]
                    for channel in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, f'{stem}_{channel}_SR_float.tif'), full_pred[:, :, channel].astype(np.float32))
                full_pred[full_pred < 0] = 0
                h, w = img.shape
                img = np.uint8(img/np.max(img)*255)
                if file_list[file_index].find('tif') != -1:
                    tifffile.imwrite(os.path.join(save_dir, file_list[file_index].replace(".tif", "_Input.tif")), img)
                    save_list.append(img)
                    full_pred = np.uint8(full_pred/np.max(full_pred)*255)
                    for i in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, file_list[file_index].replace(".tif", f"_{i}_SR.tif")), (full_pred[:,:,i]))
                        save_list.append(full_pred[:,:,i])
                    full_GT = np.uint8(full_GT/np.max(full_GT)*255)
                    for i in range(full_GT.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, file_list[file_index].replace(".tif", f"_{i}_GT.tif")), (full_GT[:,:,i]))
                        save_list.append(full_GT[:,:,i])
                
                if show_image:
                    if file_index == 0:
                        plt.figure(figsize=(10, 6))

                        plt.subplot(231)
                        plt.imshow(img, cmap='gray')
                        plt.title("Input Image")

                        plt.subplot(232)
                        plt.imshow(full_GT[:,:,0], cmap='gray')
                        plt.title("Confocal reference - Microtubules")

                        plt.subplot(233)
                        plt.imshow(full_GT[:,:,1], cmap='gray')
                        plt.title("Confocal reference - Mitochondria")

                        plt.subplot(234)
                        plt.imshow(full_pred[:,:,0], cmap='gray')
                        plt.title("DESM-IO - Microtubules")

                        plt.subplot(235)
                        plt.imshow(full_pred[:,:,1], cmap='gray')
                        plt.title("DESM-IO - Mitochondria")

                        plt.tight_layout()
                        plt.show()
            bar.update(1)

        # save in stack if needed
        if name:
            stack = np.stack(save_list, axis=0)
            tifffile.imwrite(r"visualization\Experimental_data_eval\Micro_Mito\resized results\{}.tif".format(name), stack, imagej=True)
if __name__ == "__main__":

    read_dir = r"visualization\Experimental_data_eval\Micro_Mito\data"
    save_dir = r"visualization\Experimental_data_eval\Micro_Mito\results"
    resolution_list = ["225", "245", "265"]
    Poisson_noise_list = ["0.040", "0.120", "0.200", "0.280"]
    for resolution_selected in resolution_list:
        for Poisson_noise_selected in Poisson_noise_list:
            weights_dir = r"validation\DESM_Micro_Mito_{}_{}_4_DSCM_384_Unet_fea_loss_0.01_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches_new_noise\weights\1\main_G.pth".format(resolution_selected, Poisson_noise_selected)
            DESM_IO_inference_with_reference(read_dir, weights_dir, save_dir=save_dir, show_image=False, name="{}_{}".format(resolution_selected, Poisson_noise_selected))
