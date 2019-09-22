from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from PIL import Image
import os
import sys


def indentity_func(x, **kwargs):
    return x



def load_data_from_dir(dir_loc, ext, limit = np.inf, prog_func = indentity_func):
    files = []
    count = 0
    file_list = os.listdir(dir_loc)
    limit = min(len(file_list), limit)
    file_list = file_list[0:limit]
    for f in prog_func(file_list, desc = 'Loading files'): 
        if f.endswith(ext):
            image = data.imread(os.path.join(dir_loc,f))
            if len(image.shape) > 2:
                files.append(image)
            count = count + 1
        if count >= limit:
            break
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



def save_array_as_image(a, filename, quality = 100):
    a = np.uint8(np.around((a + 1) * 127.5))
    a = np.swapaxes(a, 0, 1)
    a_img = Image.fromarray(a)
    a_img.save(filename, quality = quality)
    
def rescale_save_array_as_image(a, filename, quality = 100):
    a = np.uint8(np.around((a + 1) * 127.5))
    a = np.swapaxes(a, 0, 1)
    a_img = Image.fromarray(a)
    a_img = a_img.resize((1920, 1080), Image.BICUBIC)
    a_img.save(filename, quality = quality)

def save_images_orig(lowres, highres, idx_start, idx_stop, path, prefix, quality = 100):
    
    for idx in range(idx_start, idx_stop + 1):
        ex = lowres[idx]
        rescale_save_array_as_image(ex, path + '/' + prefix + "_im%04d_lowres.jpg" % idx, quality)

        ex = highres[idx]
        save_array_as_image(ex, path + '/' + prefix + "_im%04d_orig.jpg" % idx, quality)

def save_images_predicted(lowres, upscaler, idx_start, idx_stop, path, prefix, batch, quality = 100):
    
    for idx in range(idx_start, idx_stop + 1):
        ex = upscaler.predict(lowres[[idx]])[0]
        save_array_as_image(ex, path + '/' + prefix + "_im%04d_upscaled_%06d.jpg" % (idx, batch), quality)
