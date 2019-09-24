from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
import pandas as pd
from PIL import Image
import os
import sys


def indentity_func(x, **kwargs):
    return x


def load_images_from_dir_and_downscale(dir_loc, ext, limit = np.inf, prog_func = indentity_func, downscale_factor = 4, desc = "Loading files"):
    file_list = os.listdir(dir_loc)
    file_list = sorted(list(filter(lambda s: s.endswith(ext), file_list)))
    limit = min(len(file_list), limit)
    file_list = file_list[0:limit]
    
    files = pd.DataFrame({
        'filename': [],
        'image_hr': [],
        'image_lr': []
    })
    for f in prog_func(file_list, desc = desc):
        path = os.path.join(dir_loc,f)
        
        img_hr = Image.open(path).copy()
        
        lr_width = img_hr.width // downscale_factor
        lr_height = img_hr.height // downscale_factor
        
        img_lr = img_hr.resize((lr_width, lr_height), Image.LANCZOS)
        
        files = files.append({
            'filename': f,
            'image_hr': img_hr,
            'image_lr': img_lr
        }, ignore_index=True)
    
    return files


def split_images_train_test(images_df, train_test_ratio = 0.8):
    number_of_images = images_df.shape[0]
    number_of_train_images = int(round(number_of_images * train_test_ratio))
    
    images_df_train = images_df[:number_of_train_images].reset_index(drop=True)
    images_df_test  = images_df[number_of_train_images:number_of_images].reset_index(drop=True)
    
    return images_df_train, images_df_test





def select_random_rows(images_df, n = 1):
    row_indices = np.random.randint(0, images_df.shape[0], size=n)
    
    return images_df.iloc[row_indices]




def convert_array_to_image(array):
    array = np.uint8(np.around((array + 1) * 127.5))
    array = array.swapaxes(0,1)
    
    return Image.fromarray(array)


def convert_image_to_array(img):
    array = np.array(img)
    array = (array / 127.5) - 1
    array = array.swapaxes(0,1)
    
    return array
    
    
def convert_image_series_to_array(image_series):
    array = np.array([np.array(img).swapaxes(0,1) for img in image_series])
    array = (array / 127.5) - 1
    
    return array


def convert_imagesdf_to_arrays(images_df):
    array_hr = convert_image_series_to_array(images_df.image_hr)
    array_lr = convert_image_series_to_array(images_df.image_lr)
    
    return array_hr, array_lr





    
def save_array_as_image(a, filename, quality = 100):
    a_img = convert_array_to_image(a)
    a_img.save(filename, quality = quality)

    
def rescale_save_array_as_image(a, filename, target_size = (1920, 1080), quality = 100):
    a_img = convert_array_to_image(a)
    a_img = a_img.resize(target_size, Image.BICUBIC)
    a_img.save(filename, quality = quality)

    
def save_images_orig(images_df, idx_start, idx_stop, path, prefix, target_size = (1920, 1080), quality = 100):
    
    idx_stop = min(idx_stop, images_df.shape[0])
    
    for idx in range(idx_start, idx_stop):
        img = images_df.image_lr[idx]
        img = img.resize(target_size, Image.BICUBIC)
        img.save(path + '/' + prefix + "_im%04d_lowres.jpg" % idx, quality = quality)
        
        img = images_df.image_hr[idx]
        img.save(path + '/' + prefix + "_im%04d_orig.jpg" % idx, quality = quality)


def save_images_predicted(images_df, upscaler, idx_start, idx_stop, path, prefix, batch, quality = 100):
    
    idx_stop = min(idx_stop, images_df.shape[0])
    
    for idx in range(idx_start, idx_stop):
        ex = convert_image_series_to_array(images_df.image_lr[[idx]])
        ex = upscaler.predict(ex)[0]
        save_array_as_image(ex, path + '/' + prefix + "_im%04d_upscaled_%06d.jpg" % (idx, batch), quality = quality)


    


    
    
    
    




def load_data_from_dir(dir_loc, ext, limit = np.inf, prog_func = indentity_func):
    files = []
    file_list = os.listdir(dir_loc)
    limit = min(len(file_list), limit)
    file_list = file_list[0:limit]
    for f in prog_func(file_list, desc = 'Loading files'): 
        if f.endswith(ext):
            image = data.imread(os.path.join(dir_loc,f))
            if len(image.shape) > 2:
                files.append(image)
    return files

def load_train_test_data(directory, ext, number_of_images = 1000, train_test_ratio = 0.8, lr_shape = (480,270,3), prog_func = indentity_func):
    
    files = load_data_from_dir(directory, ext, limit = number_of_images, prog_func = prog_func)
    
    number_of_images = min(len(files), number_of_images)
    
    number_of_train_images = int(number_of_images * train_test_ratio)
    
    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]
    
    x_train_hr = hr_images(x_train)
    x_train_hr = normalize(x_train_hr)
    
    x_train_lr = lr_images(x_train, lr_shape, prog_func = prog_func)
    x_train_lr = normalize(x_train_lr)
    
    x_test_hr = hr_images(x_test)
    x_test_hr = normalize(x_test_hr)
    
    x_test_lr = lr_images(x_test, lr_shape, prog_func = prog_func)
    x_test_lr = normalize(x_test_lr)
    
    return x_train_lr, x_train_hr, x_test_lr, x_test_hr

def load_data_rescale(directory, ext, number_of_images = 1000, lr_shape = (480,270,3), prog_func = indentity_func):
    
    files = load_data_from_dir(directory, ext, limit = number_of_images, prog_func = prog_func)
    
    files_hr = hr_images(files)
    files_hr = normalize(files_hr)
    
    files_lr = lr_images(files, lr_shape, prog_func = prog_func)
    files_lr = normalize(files_lr)
    
    return files_lr, files_hr



def hr_images(images):
    images_hr = array(images)
    # conversion between PIL and array swaps height and width
    images_hr = np.swapaxes(images_hr, 1, 2)
    return images_hr

def lr_images(images_real , lr_shape, prog_func = indentity_func):
    images = []
    for img in  prog_func(range(len(images_real)), desc = 'Converting to low-res'):
        img_lr = np.array(
                Image.fromarray(images_real[img]).resize(
                    [lr_shape[0],lr_shape[1]],
                    resample=Image.BICUBIC
                )
            )
        images.append(img_lr)
    images_lr = array(images)
    # conversion between PIL and array swaps height and width
    images_lr = np.swapaxes(images_lr, 1, 2)
    return images_lr



def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

