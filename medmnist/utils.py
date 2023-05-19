import os
from PIL import Image
from tqdm import trange
import skimage
from skimage.util import montage as skimage_montage


SPLIT_DICT = {
    "train": "TRAIN",
    "val": "VALIDATION",
    "test": "TEST"
}  # compatible for Google AutoML Vision

import csv

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

def create_sub_path(ds_name:str, ds_cat: str, data_path:str, D3T_D2F:bool):
    
    if ds_cat == 'train':
        name, label = 'train_images' , 'train_labels'
    elif ds_cat == 'val':
        name, label = 'val_images' , 'val_labels'
    elif ds_cat == 'test':
        name, label = 'test_images' , 'test_labels'
    else:
        assert False, f"incorrect ds_cat = {ds_cat} provided!"
    
    img_folder_name = 'dcms' if D3T_D2F else 'pngs'
    
    ds_path = os.path.join(data_path, ds_name)
    ds_sub_path = os.path.join(ds_path, ds_name+'_'+name)
    ds_imgs_path = os.path.join(ds_sub_path, img_folder_name)
       
    create_dir(ds_path)
    create_dir(ds_sub_path)
    create_dir(ds_imgs_path)

    return ds_imgs_path, name, label

####################################

def save2d(imgs, labels, img_folder,
           split, postfix, csv_path, customize= False):
    if customize:
        return customize_save_fn(imgs, labels, img_folder,
            split, postfix, csv_path,
            load_fn=lambda arr: Image.fromarray(arr),
            save_fn=lambda img, path: img.save(path))
    
    else:
        return save_fn(imgs, labels, img_folder,
                    split, postfix, csv_path,
                    load_fn=lambda arr: Image.fromarray(arr),
                    save_fn=lambda img, path: img.save(path))


def montage2d(imgs, n_channels, sel):
    sel_img = imgs[sel]

    # version 0.20.0 changes the kwarg `multichannel` to `channel_axis`
    if skimage.__version__ >= "0.20.0":
        montage_arr = skimage_montage(
            sel_img, channel_axis=3 if n_channels == 3 else None)
    else:
        montage_arr = skimage_montage(sel_img, multichannel=(n_channels == 3))
    montage_img = Image.fromarray(montage_arr)

    return montage_img


def save3d(imgs, labels, img_folder,
           split, postfix, csv_path, customize=False):
    if customize:
        return customize_save_fn(imgs, labels, img_folder,
                   split, postfix, csv_path,
                   load_fn=sitk_load_frames,
                   save_fn=save_frames_as_dcm)
    else:
        return save_fn(imgs, labels, img_folder,
                   split, postfix, csv_path,
                   load_fn=load_frames,
                   save_fn=save_frames_as_gif)


def montage3d(imgs, n_channels, sel):

    montage_frames = []
    for frame_i in range(imgs.shape[1]):
        montage_frames.append(montage2d(imgs[:, frame_i], n_channels, sel))

    return montage_frames


def save_fn(imgs, labels, img_folder,
            split, postfix, csv_path,
            load_fn, save_fn):

    assert imgs.shape[0] == labels.shape[0]

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    if csv_path is not None:
        csv_file = open(csv_path, "a")
        print(f"save_fn: csv_file = {csv_path}")

    for idx in trange(imgs.shape[0]):

        img = load_fn(imgs[idx])

        label = labels[idx]

        file_name = f"{split}{idx}_{'_'.join(map(str,label))}.{postfix}"

        save_fn(img, os.path.join(img_folder, file_name))

        if csv_path is not None:
            line = f"{SPLIT_DICT[split]},{file_name},{','.join(map(str,label))}\n"
            csv_file.write(line)

    if csv_path is not None:
        csv_file.close()


def customize_save_fn(imgs, labels, img_folder,
            split, postfix, csv_path,
            load_fn, save_fn):

    assert imgs.shape[0] == labels.shape[0], f"imgs.shape[0] = {imgs.shape[0]} labels.shape[0] = {labels.shape[0]}"
    print("customize saving function")
    
    # if not os.path.exists(img_folder):
    #     os.makedirs(img_folder)

    if csv_path is not None:
        csv_file = open(csv_path, "a")

    print (f"csv_path = {csv_path}")

    for idx in trange(imgs.shape[0]):

        img = load_fn(imgs[idx])

        label = labels[idx]

        file_name = f"{split}{idx}_{'_'.join(map(str,label))}.{postfix}"

        save_fn(img, os.path.join(img_folder, file_name))

        if csv_path is not None:
            #line = f"{SPLIT_DICT[split]},{file_name},{','.join(map(str,label))}\n"
            line = f"{file_name},{','.join(map(str,label))},{'./'+os.path.basename(img_folder)+'/'+file_name}\n"
            csv_file.write(line)

    if csv_path is not None:
        csv_file.close()

def load_frames(arr):
    frames = []
    for frame in arr:
        frames.append(Image.fromarray(frame))
    return frames


def save_frames_as_gif(frames, path, duration=200):
    assert path.endswith(".gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)


def sitk_load_frames(arr):
    import SimpleITK as sitk
    imgs_arry = sitk.GetImageFromArray(arr)
    return imgs_arry


def save_frames_as_dcm(frames, path):
    assert path.endswith(".dcm")
    import SimpleITK as sitk
    sitk.WriteImage(frames, path)

    #print(f"save_frames_as_dcm: frames has {len(frames)} frames")
    # frames[0].save(path, save_all=True, append_images=frames[1:],
    #                duration=duration, loop=0)
