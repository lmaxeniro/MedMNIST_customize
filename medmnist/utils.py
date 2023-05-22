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
import time


def writeSlices(Writer, series_tag_values, new_img, out_dir, i):
    image_slice = new_img[:, :, i]

    # Tags shared by the series.
    list(
        map(
            lambda tag_value: image_slice.SetMetaData(
                tag_value[0], tag_value[1]
            ),
            series_tag_values,
        )
    )

    # Slice specific tags.
    #   Instance Creation Date
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
    #   Instance Creation Time
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

    # Setting the type to CT so that the slice location is preserved and
    # the thickness is carried over.
    # Modality
    image_slice.SetMetaData("0008|0060", "CT")
    # the Manufacturer
    image_slice.SetMetaData("0008|0070", "Xeniro")

    # (0020, 0032) image position patient determines the 3D spacing between
    # slices.
    #   Image Position (Patient)
    image_slice.SetMetaData(
        "0020|0032",
        "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))),
    )
    #   Instance Number
    image_slice.SetMetaData("0020|0013", str(i))

    # Write to the output directory and add the extension dcm, to force
    # writing in DICOM format.
    Writer.SetFileName(os.path.join(out_dir, str(i) + ".dcm"))
    Writer.Execute(image_slice)


def check_dir_exit(dir_name: str) -> bool:
    if os.path.exists(dir_name):
        return True
    else:
        print(f"{dir_name} does not exist! considering create it..")
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
       
    create_dir(ds_path)         # exp: breastmnist
    create_dir(ds_sub_path)     # exp: breastmnist/breastmnist_test_images
    create_dir(ds_imgs_path)    # exp: breastmnist/breastmnist_test_images/pngs or dcms

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
                   load_fn=load_frames_for_dcm,
                   save_fn=write_dcms)
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

        if save_fn == write_dcms:
            ext_path = '/'
        else:
            ext_path = ''

        if csv_path is not None:
            #line = f"{SPLIT_DICT[split]},{file_name},{','.join(map(str,label))}\n"
            line = f"{file_name},{','.join(map(str,label))},{'./'+os.path.basename(img_folder)+'/'+file_name+ext_path}\n"
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
    return sitk.GetImageFromArray(arr)


def save_frames_as_dcm(imgs_arry, path):
    '''write frames into one dcms'''
    assert path.endswith(".dcm")
    #print(f"save_frames_as_dcms: dcm_path = {path}, num of img : {len(imgs_arry)}")
    import SimpleITK as sitk
    sitk.WriteImage(imgs_arry, path)   

#############################################

def load_frames_for_dcm(arr):
    import SimpleITK as sitk

    new_img = sitk.GetImageFromArray(arr)
    new_img.SetSpacing([1, 1, 1]) #todo: read from dic for each dataset

    return new_img

def write_dcms(new_img, path):
    import SimpleITK as sitk
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = new_img.GetDirection()

    series_tag_values = [
    ("0008|0031", modification_time),  # Series Time
    ("0008|0021", modification_date),  # Series Date
    ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
    (
        "0020|000e",
        "1.2.826.0.1.3680043.2.1125."
        + modification_date
        + ".1"
        + modification_time,
    ),  # Series Instance UID
    (
        "0020|0037",
        "\\".join(
            map(
                str,
                (
                    direction[0],
                    direction[3],
                    direction[6],
                    direction[1],
                    direction[4],
                    direction[7],
                ),
            )
        ),
    ),  # Image Orientation
    # (Patient)
    ("0008|103e", "ConvertedfromNumpy"),  # Series Description
    ]

    # If we want to write floating point values, we need to use the rescale
    # slope, "0028|1053", to select the number of digits we want to keep. We
    # also need to specify additional pixel storage and representation
    # information.

    # rescale_slope = 0.001  # keep three digits after the decimal point
    # series_tag_values = series_tag_values + [
    #     ("0028|1053", str(rescale_slope)),  # rescale slope
    #     ("0028|1052", "0"),  # rescale intercept
    #     ("0028|0100", "16"),  # bits allocated
    #     ("0028|0101", "16"),  # bits stored
    #     ("0028|0102", "15"),  # high bit
    #     ("0028|0103", "1"),
    # ]  # pixel representation

    create_dir(path)
    # os.chdir(path)

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    list(
    map(
        lambda i: writeSlices(writer, series_tag_values, new_img, path, i),
        range(new_img.GetDepth()),
    )
    )
