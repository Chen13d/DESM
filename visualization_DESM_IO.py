from utils import *
from options.options import *
import tifffile
from tqdm import tqdm
from torchvision import transforms
from skimage.util import view_as_windows
from sklearn.linear_model import LinearRegression


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
    std = torch.std(Input).to(device) #if std == None else std
    output = transforms.Normalize(mean_zero, std)(Input)
    return output, mean_zero, std


def DESM_IO_inference(read_dir, weights_dir, device='cuda', save_dir=False, resize_to_const=False, show_image=False):
    file_list = natsort.natsorted(os.listdir(read_dir))    
    folder_zoom = 0
    for file_index in range(len(zoom_list)):
        if read_dir.find(zoom_list[file_index]) != -1:
            folder_zoom = zoom_list[file_index]
    check_existence(save_dir)
    model = torch.load(weights_dir, weights_only=False)
    
    bar = tqdm(total=len(file_list))
    with torch.no_grad():
        for file_index in range(len(file_list)):
            if file_list[file_index].find('.tif') != -1 or file_list[file_index].find('.png') != -1:
                read_dir_file = os.path.join(read_dir, file_list[file_index])
                zoom = folder_zoom
                for i in range(len(zoom_list)):
                    if read_dir_file.find(zoom_list[i]) != -1:
                        zoom = zoom_list[i]
                img = tifffile.imread(read_dir_file)
                if len(img.shape) == 3: img = img[1,:,:]
                raw_h, raw_w = img.shape
                for zoom_index in range(len(zoom_list)):
                        if zoom == zoom_list[zoom_index]:
                            for size_index in range(len(size_list)):
                                if raw_h == size_list[size_index]:
                                    convert_ratio = convert_table[zoom_index][size_index]/20
                                    img = resize(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                #convert_ratio = 0.95
                img = resize(img, (int(raw_h*convert_ratio), int(raw_w*convert_ratio)))
                
                h, w = img.shape
                size = min(h, w)
                upper_limit = 2000
                if size > upper_limit:
                    num_row = (h // upper_limit) + 1
                    num_col = (w // upper_limit) + 1
                    size = upper_limit
                else:
                    upper_limit = (size // 16) * 16
                    num_row = (h // upper_limit)
                    num_col = (w // upper_limit)
                    size = upper_limit
                
                image_list = []
                pred_list = []
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
                    image_list.append([])
                    pred_list.append([])
                    for col in range(num_col):
                        image = img[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]]
                        image = torch.tensor(np.float64(image), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                        image, mean, std = norm_statistic(image, device)
                        
                        pred = model(image)
                        image = image*std+mean
                        pred = pred*std+mean
                        image[image<0] = 0
                        pred[pred<0] = 0
                        image_list[row].append(to_cpu(image.squeeze(0).squeeze(0)))
                        pred_list[row].append(to_cpu(pred.squeeze(0).permute(1,2,0)))
                full_pred = np.zeros((h, w ,pred_list[0][0].shape[-1]))
                for row in range(num_row):
                    for col in range(num_col):
                        full_pred[row_cood_list[row][0]:row_cood_list[row][1], col_cood_list[col][0]:col_cood_list[col][1]] = pred_list[row][col]
                h, w = img.shape
                img = np.uint8(img/np.max(img)*255)
                if resize_to_const:
                    img = resize(img, ((1024, 1024)))
                    full_pred = resize(full_pred, (1024,1024,full_pred.shape[-1]))
                if file_list[file_index].find('tif') != -1:
                    tifffile.imwrite(os.path.join(save_dir, file_list[file_index].replace(".tif", "_Input.tif")), img)
                    full_pred = np.uint8(full_pred/np.max(full_pred)*255)
                    for i in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, file_list[file_index].replace(".tif", f"_{i}_SR.tif")), np.uint8(full_pred[:,:,i]))
                elif file_list[file_index].find('png') != -1:
                    tifffile.imwrite(os.path.join(save_dir, file_list[file_index].replace(".png", "_Input.tif")), img)
                    full_pred = np.uint8(full_pred/np.max(full_pred)*255)
                    for i in range(full_pred.shape[-1]):
                        tifffile.imwrite(os.path.join(save_dir, file_list[file_index].replace(".png", f"_{i}_SR.tif")), np.uint8(full_pred[:,:,i]))
                bar.set_description(f"File done: {file_list[file_index]}")
                bar.update(1)

                if show_image:
                    if file_index == 0:
                        plt.figure(figsize=(12, 12))
                        plt.subplot(131)
                        plt.imshow(img, cmap='gray')
                        plt.title('Intensity Input (single-channel LR)')
                        plt.axis('off')

                        plt.subplot(132)
                        plt.imshow(full_pred[:,:,0], cmap='gray')
                        plt.title('DESM-IL Prediction (Microtubules)')
                        plt.axis('off')

                        plt.subplot(133)
                        plt.imshow(full_pred[:,:,1], cmap='gray')
                        plt.title('DESM-IL Prediction (Lysosomes)')
                        plt.axis('off')

                        plt.tight_layout()
                        plt.show()



if __name__ == "__main__":
    
    # Micro Lyso
    if 0:
        read_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Lysosomes\data'
        weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Lyso_280_0.070_1_DSCM_384_Unet_fea_loss_0.1_SSIM_loss_1_grad_loss_1_GAN_loss_1_real-time_1000_epoches\weights\1\main_G.pth"
        #weights_dir = r"D:\CQL\codes\microscopy_decouple\validation\DSCM_Micro_Lyso_360_0.030_1_DSCM_384_Unet_fea_loss_0.1_SSIM_loss_1_grad_loss_0_GAN_loss_1_real-time_1000_epoches_test\weights\1\main_G.pth"
        save_dir = r'D:\CQL\codes\microscopy_decouple\visualization\Multi_structure\Microtubes_Lysosomes\results'
        DESM_IO_inference(read_dir=read_dir, weights_dir=weights_dir, save_dir=save_dir, resize_to_const=False)
    
            

    
    
    
    