import os, sys
import pandas as pd
from tqdm import tqdm

# add parent dir for "utils.py"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import *

def merge_csv(read_dir, save_dir, Org_list, name_list, combination_list, noise_list=[0]):
    name_count = 0
    data_list = [[] for i in range(len(name_list))]
    half_len_name_list = len(name_list) // 2
    # Iterate over
    for combination_index in range(len(combination_list)):
        folder_path = os.path.join(read_dir, "{}\\noise_{}\\Resolution_csv".format(combination_list[combination_index], noise_list[0]))
        org_list = Org_list[combination_index]
        num_org = len(org_list)
        Input_list = []
        GT_list = [[] for i in range(num_org)]
        Output_list = [[] for i in range(num_org)]

        # DirectoryCSVFile name
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        csv_files = natsort.natsorted(csv_files)
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            # ReadCSV
            df = pd.read_csv(file_path)
            # File nameData()
            name = file[:file.find(".csv")]
            #df['source_file'] = file[:file.find(".csv")]
            if name.find("Input") != -1:
                Input_list.append(df)
            for i in range(num_org):
                if name.find("Output") != -1 and name.find(org_list[i]) != -1:
                    Output_list[i].append(df)
                    #print(len(Output_list))
                elif name.find("GT") != -1 and name.find(org_list[i]) != -1:
                    GT_list[i].append(df)

        # Dataall_data
        Input_df = pd.concat([*Input_list], ignore_index=True)
        Output_df = [[] for i in range(num_org)]
        GT_df = [[] for i in range(num_org)]
        for i in range(num_org):
            Output_df[i] = pd.concat([*Output_list[i]], ignore_index=True)
            GT_df[i] = pd.concat([*GT_list[i]], ignore_index=True)
        for i in range(num_org):
            data_list[name_count] = (pd.to_numeric(GT_df[i].iloc[:,0]).tolist())
            data_list[name_count+half_len_name_list] = (pd.to_numeric(GT_df[i].iloc[:,2]).tolist())
            #if name_count == 16 or name_count+half_len_name_list-1 == 16: print(name_count)
            
            name_count += 1
            data_list[name_count] = (pd.to_numeric(Output_df[i].iloc[:,0]).tolist())
            data_list[name_count+half_len_name_list] = (pd.to_numeric(Output_df[i].iloc[:,2]).tolist())
            #if name_count == 16 or name_count+half_len_name_list-1 == 16: print(name_count)
            name_count += 1
    #print(len(name_list), len(data_list))
    #for i in range(len(name_list)):
    #    print(name_list[i], len(data_list[i]))
    #print(name_list[16], len(data_list[16]))
    data_dict = {}
    for i in range(len(name_list)):
        data_dict[f'{name_list[i]}'] = data_list[i]
    data_frame = pd.DataFrame(data_dict)
    data_frame.to_csv(save_dir, index=False)


def arrange_csv_from_single_organelle(read_dir, save_dir):
    file_list = os.listdir(read_dir)
    file_list = natsort.natsorted(file_list)
    #name_list = []
    rFRC_value_list = []
    mean_rFRC_list = []
    min_rFRC_list = []
    max_rFRC_list = []
    # read files
    for i in tqdm(range(0, len(file_list)), desc="Processing files"):
    #for i in range(len(file_list)):
        try:
            file_dir = os.path.join(read_dir, file_list[i])
            df = pd.read_csv(file_dir)
            mean_FRC = df["Mean FRC (nm)"].values[0]
            rFRC_value = df["rFRC value"].values[0]
            min_FRC = df["Min FRC (nm)"].values[0]
            max_FRC = df["Max FRC (nm)"].values[0]
            if rFRC_value >=0.01 and max_FRC < 200:
            #if 1:
                mean_rFRC_list.append(mean_FRC)
                rFRC_value_list.append(rFRC_value)
                min_rFRC_list.append(min_FRC)
                max_rFRC_list.append(max_FRC)
                print(i)
            
        except:
            print("error in {}".format(i))

    
    # According torFRC_value_listListSort
    sorted_lists = sorted(zip(rFRC_value_list, mean_rFRC_list, min_rFRC_list, max_rFRC_list), key=lambda x: x[0])
    # SortList
    rFRC_value_list, mean_rFRC_list, min_rFRC_list, max_rFRC_list = zip(*sorted_lists)
     # SaveSortResult, Save asCSV
    result_df = pd.DataFrame({
        'rFRC value': rFRC_value_list,
        'Mean FRC (nm)': mean_rFRC_list,
        'Min FRC (nm)': min_rFRC_list,
        'Max FRC (nm)': max_rFRC_list
    })
    print(np.mean(mean_rFRC_list[0:20]))
    result_df.to_csv(save_dir, index=False)

    
if __name__ == "__main__":
    print("merge_csv_resolution.py is a utility helper. Please edit the input folders and metric names before running it directly.")
                


