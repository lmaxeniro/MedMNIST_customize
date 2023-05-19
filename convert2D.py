# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import os
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

import SimpleITK as sitk
import numpy as np
import os
# import csv

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

MED_2D = ['pathmnist', 'chestmnist', 'dermamnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist']
MED_3D = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d' , 'fracturemnist3d' , 'vesselmnist3d', 'synapsemnist3d']
DATA_CAT = ['train', 'val', 'test']
D3T_D2F = None

MIN_Samples = 20
CSV  = "labels.csv"
     
USER_PATH = os.path.expanduser('~')
CUR_PATH = os.getcwd()
DATA_PATH = os.path.join(CUR_PATH, 'data')
assert os.path.exists(DATA_PATH) , f'{DATA_PATH} does not exist! please create it mannually!'

data_flag = 'organmnist3d'
# data_flag = 'breastmnist'

if data_flag in MED_2D:
    D3T_D2F = False
elif data_flag in MED_3D:
    D3T_D2F = True
else:
    assert False

download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

#### 2D convert

# # preprocessing
# data_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[.5], std=[.5])
# ])

# # load the data
# train_dataset = DataClass(split='train', transform=data_transform, download=download)
# test_dataset = DataClass(split='test', transform=data_transform, download=download)

# pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
# test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# print(train_dataset)
# print("===================")
# print(test_dataset)

# # visualization
# train_dataset.montage(length=1)
# # montage
# train_dataset.montage(length=20)

from medmnist.utils import create_sub_path, WriteCsv

ds_imgs_path, name, label = create_sub_path(data_flag, 'train', DATA_PATH, D3T_D2F)
# print(f"ds_imgs_path = {ds_imgs_path}")
csv_file = os.path.join(os.path.dirname(ds_imgs_path), CSV)
WriteCsv(csv_file, "w", "ID", "label", "data_path", "")
# train_dataset.save(ds_imgs_path)

#### 3D convert

# load the data
train_dataset = DataClass(split='train',  download=download)
# encapsulate data into dataloader form
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_dataset.save(ds_imgs_path)

train_dataset.save(ds_imgs_path, postfix="dcm", write_csv=True, customize=True)