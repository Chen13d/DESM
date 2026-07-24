import argparse
import os, sys, tifffile, copy
import pandas as pd
from skimage.metrics import structural_similarity as ssim_
# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *


def make_threshold(Input):
        thresh = threshold_otsu(Input)
        Output = copy.deepcopy(Input)
        Output[Output<thresh] = 0
        Output[Output>=thresh] = 255
        kernel_size = (3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        Output = cv2.dilate(Output, kernel, 
                 anchor=None, 
                 iterations=1, 
                 borderType=cv2.BORDER_CONSTANT, 
                 borderValue=0)
        return Output


def generate_overlapped_map(image_list):
    for i in range(len(image_list)):
        img = image_list[i]
        img_thresh = make_threshold(img)
        if i == 0: img_overlapped = np.zeros_like(img)
        img_overlapped += img_thresh
    img_overlapped[img_overlapped == 255] = 0
    img_overlapped[img_overlapped != 0] = 1
    return img_overlapped

def main(image_paths):
    image_list = [tifffile.imread(path) for path in image_paths]
    thresh_list = [make_threshold(img) for img in image_list]

    plt.figure()
    for idx, thresh_img in enumerate(thresh_list, start=1):
        plt.subplot(1, len(thresh_list), idx)
        plt.imshow(thresh_img)
        plt.axis("off")

    img_overlapped = generate_overlapped_map(image_list)
    plt.figure()
    plt.imshow(img_overlapped)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize thresholded channels and their overlap map.")
    parser.add_argument("image_paths", nargs="+", help="Input TIFF paths to combine into an overlap map.")
    args = parser.parse_args()
    main(args.image_paths)


