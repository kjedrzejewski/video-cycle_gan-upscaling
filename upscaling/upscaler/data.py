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
        img_hr = img_hr.convert('RGB')
        
        lr_width = img_hr.width // downscale_factor
        lr_height = img_hr.height // downscale_factor
        
        img_lr = img_hr.resize((lr_width, lr_height), Image.LANCZOS)
        
        files = files.append({
            'filename': f,
            'image_hr': img_hr,
            'image_lr': img_lr
        }, ignore_index=True)
    
    return files


def load_images_from_dir(dir_loc, ext, limit = np.inf, prog_func = indentity_func, min_shape = (256, 256), desc = "Loading files"):
    
    file_list = os.listdir(dir_loc)
    file_list = sorted(list(filter(lambda s: s.endswith(ext), file_list)))
    limit = min(len(file_list), limit)
    file_list = file_list[0:limit]
    
    files = pd.DataFrame({
        'filename': [],
        'image_size': [],
        'image_hr': []
    })
    for f in prog_func(file_list, desc = desc):
        path = os.path.join(dir_loc,f)
        
        img_hr = Image.open(path).copy()
        img_hr = img_hr.convert('RGB')
        img_shape = img_hr.size
        
        if img_shape[0] >= min_shape[0] and img_shape[1] >= min_shape[1]:
            files = files.append({
                'filename': f,
                'image_size': img_hr.size,
                'image_hr': img_hr
            }, ignore_index=True)
            
    return files


def crop_images(hq_images, prog_func = indentity_func, target_shape = (256, 256), downscale_ratio = np.nan, seed = np.nan, desc = "Processing files"):
    
    if not np.isnan(seed):
        rand_state = np.random.get_state()
        np.random.seed(seed)
    
    cropped = []
    crop_shapes = []
    if not np.isnan(downscale_ratio):
        img_crop_lr = []
    
    for id, img in prog_func(hq_images.iterrows(), desc = desc):

        img_hr = img['image_hr']         
        img_shape = img_hr.size
        
        width_range  = img_hr.size[0] - target_shape[0]
        height_range = img_hr.size[1] - target_shape[1]
            
        left = np.random.randint(0, width_range + 1, 1)[0]
        top  = np.random.randint(0, height_range + 1, 1)[0]
        
        crop_shape = (left, top, left + target_shape[0], top + target_shape[1])
        img_cropped = img_hr.crop(crop_shape)

        cropped.append(img_cropped)
        crop_shapes.append(crop_shape)
        
        if not np.isnan(downscale_ratio):
            img_lr = img_cropped.resize((target_shape[0] // 4, target_shape[1] // 4), Image.LANCZOS)
            img_crop_lr.append(img_lr)
    
    res = hq_images.assign(
        crop_shape = crop_shapes,
        image_cropped = cropped
    )
        
    if not np.isnan(downscale_ratio):
        res = res.assign(
            image_cropped_lr = img_crop_lr
        )
    
    if not np.isnan(seed):
        np.random.set_state(rand_state)
    
    return res


def split_images_train_test(images_df, train_test_ratio = 0.8, seed = np.nan):
    
    if not np.isnan(seed):
        rand_state = np.random.get_state()
        np.random.seed(seed)
    
    number_of_images = images_df.shape[0]
    number_of_train_images = int(round(number_of_images * train_test_ratio))
    
    train_ids = np.random.choice(images_df.shape[0], size=number_of_train_images, replace=False)
    
    images_df_train = images_df.iloc[train_ids].reset_index(drop=True)
    images_df_test  = images_df[~images_df.index.isin(train_ids)].reset_index(drop=True)
    
    if not np.isnan(seed):
        np.random.set_state(rand_state)
    
    return images_df_train, images_df_test





def select_random_rows(images_df, n = 1, seed = np.nan):
    if not np.isnan(seed):
        rand_state = np.random.get_state()
        np.random.seed(seed)
    
    row_indices = np.random.randint(0, images_df.shape[0], size=n)

    if not np.isnan(seed):
        np.random.set_state(rand_state)
    
    return images_df.iloc[row_indices].reset_index(drop=True)




def convert_array_to_image(array):
    array = np.uint8(np.around((array + 1) * 127.5))
    
    return Image.fromarray(array)


def convert_image_to_array(img):
    array = np.array(img)
    array = (array / 127.5) - 1
    
    return array
    
    
def convert_image_series_to_array(image_series):
    array = np.array([np.array(img) for img in image_series])
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

    
def save_images_orig(images_df, idx_start, idx_stop, path, prefix, target_size = (1920, 1080), quality = 95):
    
    idx_stop = min(idx_stop, images_df.shape[0])
    
    for idx in range(idx_start, idx_stop):
        img = images_df.image_lr[idx]
        img = img.resize(target_size, Image.BICUBIC)
        img.save(path + '/' + prefix + "_im%04d_lowres.jpg" % idx, quality = quality)
        
        img = images_df.image_hr[idx]
        img.save(path + '/' + prefix + "_im%04d_orig.jpg" % idx, quality = quality)


def save_images_predicted(images_df, upscaler, idx_start, idx_stop, path, prefix, batch, quality = 95):
    
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
                        




def save_img_orig(images_series, path, prefix, quality = 95):
    
    for idx, img in enumerate(images_series):
        img.save(path + '/' + prefix + "_im%04d_orig.jpg" % idx, quality = quality)


def save_img_resize(images_series, path, prefix, sufix = '', target_size = (1080, 1920), quality = 95):
    
    for idx, img in enumerate(images_series):
        img = img.resize(target_size, Image.BICUBIC)
        img.save(path + '/' + prefix + "_im%04d_lowres%s.jpg" % (idx, sufix), quality = quality)

        
def save_img_predict(images_series, upscaler, path, prefix, batch, sufix = '', quality = 95):
    
    for idx, img in enumerate(images_series):
        ex = convert_image_series_to_array([img])
        ex = upscaler.predict(ex)[0]
        save_array_as_image(ex, path + '/' + prefix + "_im%04d_upscaled_%06d%s.jpg" % (idx, batch, sufix), quality = quality)
