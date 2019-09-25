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





    
def save_array_as_image(a, filename, **kwargs):
    a_img = convert_array_to_image(a)
    a_img.save(filename, **kwargs)

    
def rescale_save_array_as_image(a, filename, target_size = (1920, 1080), **kwargs):
    a_img = convert_array_to_image(a)
    a_img = a_img.resize(target_size, Image.BICUBIC)
    a_img.save(filename, **kwargs)

    
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


                        
def save_images_orig_png(images_df, idx_start, idx_stop, path, prefix, target_size = (1920, 1080)):
    
    idx_stop = min(idx_stop, images_df.shape[0])
    
    for idx in range(idx_start, idx_stop):
        img = images_df.image_lr[idx]
        img = img.resize(target_size, Image.BICUBIC)
        img.save(path + '/' + prefix + "_im%04d_lowres.png" % idx)
        
        img = images_df.image_hr[idx]
        img.save(path + '/' + prefix + "_im%04d_orig.png" % idx)


def save_images_predicted_png(images_df, upscaler, idx_start, idx_stop, path, prefix, batch):
    
    idx_stop = min(idx_stop, images_df.shape[0])
    
    for idx in range(idx_start, idx_stop):
        ex = convert_image_series_to_array(images_df.image_lr[[idx]])
        ex = upscaler.predict(ex)[0]
        save_array_as_image(ex, path + '/' + prefix + "_im%04d_upscaled_%06d.png" % (idx, batch))
                        


