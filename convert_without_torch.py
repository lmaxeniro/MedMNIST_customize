import SimpleITK as sitk
import numpy as np
import os
import csv

MED_2D = ['pathmnist', 'chestmnist', 'dermamnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist']
MED_3D = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d' , 'fracturemnist3d' , 'vesselmnist3d', 'synapsemnist3d']

DATA_CAT = ['train', 'val', 'test']

MIN_Samples = 20
CSV  = "labels.csv"
     
USER_PATH = os.path.expanduser('~')
CUR_PATH = os.getcwd()
DATA_PATH = os.path.join(CUR_PATH, 'data')
assert os.path.exists(DATA_PATH) , f'{DATA_PATH} does not exist!'


def check_dir_exit(dir_name: str) -> bool:
    if os.path.exists(dir_name):
        return True
    else:
        print(f"{dir_name} does not exist!")
        return False

def create_dir(dir_path: str):
    if check_dir_exit(dir_path):
        return
    else:
        os.mkdir(dir_path)
        return
        

def WriteCsv(csv_name: str, write_method:str, val_1, val_2, val_3, val_4):
    '''EXAMPLE: WriteCsv(CSV, "w", "ID", "label", "data_path", "mask_path")'''
    with open (csv_name, write_method) as labels_csv:
        writer = csv.writer(labels_csv)
        writer.writerow([val_1, val_2, val_3, val_4])

#####################################################################]
# load the dataset and check np values
DS = 'organmnist3d'
# DS = 'breastmnist'

file = os.path.join(USER_PATH, '.medmnist' , DS + '.npz')
np_data =  np.load(file)

num_train_data = len(np_data['train_images'])
num_val_data   = len(np_data['val_images'])
num_test_data  = len(np_data['test_images'])
print(f"train data size = {num_train_data}, eval data size = {num_val_data}, test data size = {num_test_data}")

assert len(np_data['train_images']) == len(np_data['train_labels'])
assert len(np_data['val_images'])   == len(np_data['val_labels'])
assert len(np_data['test_images'])   == len(np_data['test_labels'])


data = np.load(file)
# print(data.files)
# print(data['train_images'][0])
# print(data['train_labels'][0])
x, y = data['train_images'][0], data['train_labels'][0][0]

print(x.shape, y.shape)

# for i in range (10):
#     x, y = data['train_images'][i], data['train_labels'][i][0]
#     print(f"The #{i} img, label = {y}")

###################################################################################
def create_sub_path(ds_name:str, ds_cat: str, D3T_D2F:bool):
    assert ds_cat in DATA_CAT
    img_folder_name = 'dicoms' if D3T_D2F else 'pngs'
    
    if ds_cat == 'train':
        name, label = 'train_images' , 'train_labels'
    if ds_cat == 'val':
        name, label = 'val_images' , 'val_labels'
    if ds_cat == 'test':
        name, label = 'test_images' , 'test_labels'
    
    ds_path = os.path.join(DATA_PATH, ds_name)
    ds_sub_path = os.path.join(ds_path, ds_name+'_'+name)
    ds_imgs_path = os.path.join(ds_sub_path, img_folder_name)
       
    create_dir(ds_path)
    create_dir(ds_sub_path)
    create_dir(ds_imgs_path)

    return ds_imgs_path, name, label

def get_np_data(ds_name:str):
    file = os.path.join(USER_PATH, '.medmnist', ds_name + '.npz')
    np_data =  np.load(file)
    
    return np_data

def convert_npz_onedcm(ds_name:str, ds_cat:str, D3T_D2F:bool = True):
   
    ds_imgs_path, name, label = create_sub_path(ds_name, ds_cat, D3T_D2F)
   
    np_data = get_np_data(ds_name)
    
    assert len(np_data[name]) == len(np_data[label])
    
    sample_num = len(np_data[name])
    print(f"ds_name: [{ds_name}], ds_cat = {ds_cat}, data size = {sample_num}, name = {name}, label = {label}")
    
    # CREATE labels.csv once time, put under ds_imgs_path temprarily
    temp_file = os.path.join(ds_imgs_path, 'temp.csv')
    WriteCsv(temp_file, "w", "ID", "label", "data_path", "mask_path")
    
    # change working dir to img path
    os.chdir(ds_imgs_path)
    img_ext_name = '.dcm' if D3T_D2F else '.png'
    for n in range(sample_num):
        x, y = np_data[name][n], np_data[label][n][0]
        img = sitk.GetImageFromArray(x)
        this_img_id   = ds_cat + '_' + str(n)
        this_img_name = this_img_id + img_ext_name
        sitk.WriteImage(img, this_img_name)
        #print(this_img_name)
        WriteCsv(temp_file, "a", this_img_id, y, './'+os.path.basename(ds_imgs_path), "")
        if n > MIN_Samples:
            break
    
    #move the lables.csv to parent folder
    csv_file = os.path.join (os.path.dirname(os.getcwd()), CSV)
    os.rename(temp_file, csv_file)
    # change working dir to prj path

    os.chdir(CUR_PATH)

def convert_npz_multidcm(ds_name:str, ds_cat:str, D3T_D2F:bool = True):
   
    ds_imgs_path, name, label = create_sub_path(ds_name, ds_cat, D3T_D2F)
   
    np_data = get_np_data(ds_name)
    
    assert len(np_data[name]) == len(np_data[label])
    
    sample_num = len(np_data[name])
    print(f"ds_name: [{ds_name}], ds_cat = {ds_cat}, data size = {sample_num}, name = {name}, label = {label}")
    
    # CREATE labels.csv once time, put under ds_imgs_path temprarily
    temp_file = os.path.join(ds_imgs_path, 'temp.csv')
    WriteCsv(temp_file, "w", "ID", "label", "data_path", "mask_path")
    
    # change working dir to img path

    img_ext_name = '.dcm' if D3T_D2F else '.png'
    for n in range(sample_num):
        x_arry, y = np_data[name][n], np_data[label][n][0]
        
        dcm_path = os.path.join(ds_imgs_path, str(n))
        create_dir(dcm_path)
        os.chdir(dcm_path)
        i = 0
        for x in x_arry:
            img = sitk.GetImageFromArray(x)
            this_img_id   = ds_cat + '_' + str(n)+'_' + str(i)
            this_img_name = this_img_id + img_ext_name
            sitk.WriteImage(img, this_img_name)
            i+=1
            print(f"array #{i}")
        WriteCsv(temp_file, "a", this_img_id, y, './'+os.path.basename(ds_imgs_path), "")
        if n > MIN_Samples:
            break
    
    #move the lables.csv to parent folder
    csv_file = os.path.join (os.path.dirname(os.getcwd()), CSV)
    os.rename(temp_file, csv_file)
    # change working dir to prj path

    os.chdir(CUR_PATH)

# 3d_dataset
for ds_name in MED_3D:
    # train/val/test
    for data_cat in DATA_CAT:
        convert_npz_onedcm(ds_name, data_cat)
