import os, sys
 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

import pandas as pd
from utils import *



def sort_colocalization(read_dir, num_org=2, save_dir=None):
    df = pd.read_csv(read_dir)
    filtered_df = df[df['Images'].str.contains('Output')]
    image_count = len(df['Images'].dropna())
    # DisplayFilterData
    org_pearson_list = [[] for i in range(num_org)]
    org_col_list = [[] for i in range(num_org)]
    for i in range(image_count):
        for j in range(num_org):
            pearson = filtered_df.iloc[i].iloc[3]
            coloc = filtered_df.iloc[i].iloc[8]
            if (i % num_org) == j:    
                org_pearson_list[j].append(pearson)
                org_col_list[j].append(coloc)
    print("Pearson")
    for i in range(num_org):
        print(np.mean(org_pearson_list[i]), end=' ')
    for i in range(num_org):
        print(np.mean(org_col_list[i]), end=' ')


if __name__ == "__main__":
    print("sort_colocalization.py is a utility helper. Please edit the input CSV before running it directly.")
