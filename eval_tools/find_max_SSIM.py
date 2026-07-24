import os, sys, shutil
import pandas as pd

# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *


def find_max_SSIM(read_dir, save_dir, name_list):
    data_frame = pd.read_csv(read_dir)
    data_col_list = [[] for i in range(len(name_list))]
    sta_list = [[] for i in range(len(name_list))]
    sta_name_list = ['argmin', 'min', 'argmax', 'max', 'mean', 'std']
    df_dict = {}
    # enumerate in name_list to obtain data in column
    for i in range(len(name_list)):
        data_col_list[i].append(pd.to_numeric(data_frame.iloc[1:,i]).tolist())
        sta_list[i] = [np.argmin(data_col_list[i][0])+1, np.min(data_col_list[i][0]), np.argmax(data_col_list[i][0])+1, np.max(data_col_list[i][0]), np.mean(data_col_list[i][0]), np.std(data_col_list[i][0])]
        df_dict[f'{name_list[i]}'] = sta_list[i]
    df = pd.DataFrame(df_dict, index=sta_name_list)
    df.to_csv(save_dir)

    return df_dict

        
if __name__ == "__main__":
    print("find_max_SSIM.py is a utility helper. Please provide your own CSV paths and metric names before running it directly.")
