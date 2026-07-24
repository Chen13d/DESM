import os, sys
import pandas as pd
from skimage.metrics import structural_similarity as ssim_
# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *
from SSIM_loss import *

def measure_ssim(read_dir, size=512):
    imlist = os.listdir(read_dir)
    for image_name in imlist:
        image = np.array(Image.open(os.path.join(read_dir, image_name)))
        for j in range(3):
            GT = image[:size, (j + 1) * size:(j + 2) * size]
            Output = image[size:size * 2, (j + 1) * size:(j + 2) * size]
            temp_max = max(np.max(GT), np.max(Output))
            GT = GT / temp_max * 255
            Output = Output / temp_max * 255
            GT[GT < 0] = 0
            Output[Output < 0] = 0
            GT = np.uint8(GT)
            Output = np.uint8(Output)
            score = ssim_(GT, Output)
            print(f"{image_name} channel {j}: {score}")


if __name__ == "__main__":
    print("measure_SSIM.py is a quick utility script. Please edit the input folder path before running it directly.")
