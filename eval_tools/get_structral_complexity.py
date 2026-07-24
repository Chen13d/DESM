import os, tifffile, natsort, sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage import io, color, img_as_float32

# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *

def make_thresh_image(image):
    thresh_image = deepcopy(image)
    thresh = threshold_otsu(thresh_image)
    thresh_image[thresh_image < thresh] = 0
    thresh_image[thresh_image >= thresh] = 1
    return thresh_image

def compute_structural_complexity(image_path: str) -> float:
    """Calculate(Mean Gradient, MG) : MG = 1/(M*N) * Σ Σ sqrt( ((ΔI/Δx)^2 + (ΔI/Δy)^2) / 2 ) Parameter: image_path: str Input imagePath, Image. Return: mg: float Image(MG)."""
    # 1. ReadConvert, Normalize [0, 1]
    img = tifffile.imread(image_path)
    img = np.float64(img)
    
    thresh_image = make_thresh_image(img)

    img /= np.max(img)
    # 2. Calculate ( ΔI/Δx, ΔI/Δy)
    gy, gx = np.gradient(img)

    # 3. CalculatePixel
    grad_mag = np.sqrt((gx ** 2 + gy ** 2) / 2)
    
    # 4. Pixel
    mg = np.mean(grad_mag)
    mg /= np.sum(thresh_image)
    
    #plt.figure()
    #plt.imshow(grad_mag)
    #plt.show()

    return mg

def structural_complexity(folder_dir):
    file_dir = os.listdir(folder_dir)
    file_dir = natsort.natsorted(file_dir)
    complexity_list = []
    for index in range(len(file_dir)):
        if file_dir[index].find("STED") != -1 and file_dir[index].find("deconv") == -1:
            complexity = compute_structural_complexity(os.path.join(folder_dir, file_dir[index]))
            complexity_list.append(complexity)
    print(np.mean(complexity_list))


if __name__ == "__main__":
    print("get_structral_complexity.py is a utility helper. Please edit the input folder before running it directly.")

    
