# from tqdm import tqdm


import os


import medmnist
from medmnist import INFO, Evaluator
from medmnist.utils import create_sub_path, WriteCsv

import SimpleITK as sitk
import numpy as np
import os


print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

MED_2D = ['pathmnist', 'chestmnist', 'dermamnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist']
MED_3D = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d' , 'fracturemnist3d' , 'vesselmnist3d', 'synapsemnist3d']
DATA_SPLIT_FLAG = ['train', 'val', 'test']
D3T_D2F = None

MIN_Samples = 20
CSV  = "labels.csv"
     
USER_PATH = os.path.expanduser('~')
CUR_PATH = os.getcwd()
DATA_PATH = os.path.join(CUR_PATH, 'data')
assert os.path.exists(DATA_PATH) , f'{DATA_PATH} does not exist! please create it mannually!'

data_flag = 'nodulemnist3d'
# data_flag = 'breastmnist'

split_flag = 'test'

if data_flag in MED_2D:
    D3T_D2F = False
elif data_flag in MED_3D:
    D3T_D2F = True
else:
    assert False

if split_flag not in DATA_SPLIT_FLAG:
    assert False

download = True


info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

def full_generate():
    for split_flag in DATA_SPLIT_FLAG:
        # load the data
        train_dataset = DataClass(split=split_flag, download=download)
        # print information
        # print(train_dataset)

        ds_imgs_path, name, label = create_sub_path(data_flag, split_flag, DATA_PATH, D3T_D2F)
        # print(f"ds_imgs_path = {ds_imgs_path}")
        csv_file = os.path.join(os.path.dirname(ds_imgs_path), CSV)
        WriteCsv(csv_file, "w", "ID", "label", "data_path", "")

        if D3T_D2F:
            #### 3D convert
            train_dataset.save(ds_imgs_path, postfix="dcm", write_csv=True, customize=True)
        else:
            #### 2D convert
            train_dataset.save(ds_imgs_path)

# load the data
train_dataset = DataClass(split=split_flag, download=download)
# print information
# print(train_dataset)

ds_imgs_path, name, label = create_sub_path(data_flag, split_flag, DATA_PATH, D3T_D2F)
# print(f"ds_imgs_path = {ds_imgs_path}")
csv_file = os.path.join(os.path.dirname(ds_imgs_path), CSV)
WriteCsv(csv_file, "w", "ID", "label", "data_path", "")

if D3T_D2F:
    #### 3D convert
    train_dataset.save(ds_imgs_path, postfix="dcm", write_csv=True, customize=True)
else:
    #### 2D convert
    train_dataset.save(ds_imgs_path)


# load the data
# train_dataset = DataClass(split='train',  download=download)
# encapsulate data into dataloader form
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_dataset.save(ds_imgs_path)

# train_dataset.save(ds_imgs_path, postfix="dcm", write_csv=True, customize=True)