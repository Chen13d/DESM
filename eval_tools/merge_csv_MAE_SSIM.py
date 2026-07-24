import os, sys
import pandas as pd

# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *



def merge_csv_MAE_SSIM(read_dir_list, save_dir, name_list):
    data_list = [[] for i in range(len(name_list))]
    name_count = 0
    # enumerate combination]
    for i in range(len(read_dir_list)):
        read_csv_dir = os.path.join(read_dir_list[i], 'MAE_SSIM.csv')
        # enumerate noise_list
        df = pd.read_csv(read_csv_dir)
        # cal num_org, discard index column
        num_col = len(df.iloc[0]) - 1
        num_org = num_col // 2
        for j in range(num_col):
            #data_list[name_count] = [name_list[name_count]]
            data_list[name_count] = pd.to_numeric(df.iloc[:,j+1]).tolist()
            name_count += 1
    data_dict = {}
    for i in range(len(data_list)):
        data_dict[f'{name_list[i]}'] = data_list[i]
    data_frame = pd.DataFrame(data_dict)

    data_frame.to_csv(save_dir, index=False)
        
if __name__ == "__main__":
    print("merge_csv_MAE_SSIM.py is a utility helper. Please edit the input folders and metric names before running it directly.")
