import os
import sys

import medmnist
from medmnist.info import INFO, DEFAULT_ROOT
from medmnist.utils import create_sub_path, WriteCsv


print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
print("==============================================")
print("===    This is Xeniro custimized version   ===")
print("===   for 2D and 3D npy data file convert  ===")
print("==============================================")

MED_2D = ['pathmnist', 'chestmnist', 'dermamnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'octmnist','organamnist','organcmnist','organsmnist']
MED_3D = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d' , 'fracturemnist3d' , 'vesselmnist3d', 'synapsemnist3d']
DATA_SPLIT_FLAG = ['train', 'val', 'test']
D3T_D2F = None

MIN_Samples = 20
CSV  = "labels.csv"
     
# USER_PATH = os.path.expanduser('~')
CUR_PATH = os.getcwd()
DATA_PATH = os.path.join(CUR_PATH, 'data')
assert os.path.exists(DATA_PATH) , f'{DATA_PATH} does not exist! please create it mannually!'

def print_dataset_info(data_flag: str):
    '''print dataset infor'''

    info = INFO[data_flag]
    description = info['description']
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    label_list = info['label']
    spacing = info['3d_spacing']
    # DataClass = getattr(medmnist, info['python_class'])
    # download = False
    # dataset = DataClass(split='train', download=download)
    # print(dataset)
    # print(type(dataset))
    # print(label_list)
    print(description)
    print(f"task = {task}, n_channels = {n_channels}, n_classes = {n_classes}")
    print(f"spacing = {spacing}, note this is only valid for 3D dataset")

    print("label mapping:")
    for key, value in label_list.items():
        print(f"label[{key}] : value is [{value}]" )

def check_valid_dataflag(data_flag: str) -> bool :
    '''check given data_flag is valid, if valid, return False(2D) or True (3D)'''
    if data_flag in MED_2D:
        return  False
    elif data_flag in MED_3D:
        return True
    else:
        assert False, f"data_flag {data_flag} not recongnized!"

def download_origin_npy(data_flag: str):
    '''doanload the original npy dataset'''
    check_valid_dataflag(data_flag)
    path = DEFAULT_ROOT
    print(f"Downloading {data_flag}, save to {path}:")
    _ = getattr(medmnist, INFO[data_flag]['python_class'])(
        split="train", root=path, download=True)

def full_generate():
    '''convert all dataset'''
    print_dataset_info(dataset_name)
    for data_flag_2d in MED_2D:
        specific_dataset(data_flag_2d)

    for data_flag_3d in MED_3D:
        specific_dataset(data_flag_3d)

def specific_dataset(data_flag: str):
    '''convert specific dataset, dataset name: data_flag'''
    D3T_D2F = check_valid_dataflag(data_flag)
    download_origin_npy(data_flag)

    download = False
    info = INFO[data_flag]
    # task = info['task']
    # n_channels = info['n_channels']
    # n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    for split_flag in DATA_SPLIT_FLAG:
        # load the data
        dataset = DataClass(split=split_flag, download=download)
        # print information
        # print(dataset)

        ds_imgs_path, name, label = create_sub_path(data_flag, split_flag, DATA_PATH, D3T_D2F)
        # print(f"ds_imgs_path = {ds_imgs_path}")
        csv_file = os.path.join(os.path.dirname(ds_imgs_path), CSV)
        WriteCsv(csv_file, "w", "ID", "label", "data_path", "")

        if D3T_D2F:
            #### 3D convert
            dataset.save(ds_imgs_path, postfix="dcm", write_csv=True, customize=True)
        else:
            #### 2D convert
            dataset.save(ds_imgs_path, postfix="png", write_csv=True, customize=True)

###############################################################################

# data_flag = 'nodulemnist3d'       #3D  #BC
# data_flag = 'breastmnist'         #2D  #BC
# data_flag = 'pneumoniamnist'      #2D  #BC
# data_flag = 'dermamnist'            #2D  #MC(7)

# specific_dataset(data_flag)
# full_generate()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No enough parameters provided..")
        print(
            "Usage: python "
            + __file__
            + " <dataset_name>"
        )
        sys.exit(1)
    else:
        dataset_name = sys.argv[1]
        
    if dataset_name == 'all':
        full_generate()
    elif dataset_name == '2d':
        for actual_dataset_name in MED_2D:
            print_dataset_info(actual_dataset_name)
            specific_dataset(actual_dataset_name)

    elif dataset_name == '3d':
        for actual_dataset_name in MED_3D:
            print_dataset_info(actual_dataset_name)
            specific_dataset(actual_dataset_name)
    else:
        print_dataset_info(dataset_name)
        specific_dataset(dataset_name)

