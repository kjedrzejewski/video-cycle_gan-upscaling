from Network import Generator, Discriminator
import Utils
from keras.applications.vgg19 import VGG19

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Add, Concatenate
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

from skimage import data, io
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm, tqdm_notebook
import numpy as np

downscale_factor = 4
output_image_shape = (1920,1080,3)
input_image_shape = (480,270,3)


def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization()(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization()(model)
        
    model = Add()([gen, model])
    
    return model


def up_sampling_block(model, kernal_size, filters, strides):
    
    model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


class VGG_MSE_LOSS(object):

    def __init__(self, image_shape, mse_loss_rate = 0.1):
        
        self.image_shape = image_shape
        self.mse_loss_rate = mse_loss_rate

    # computes VGG loss or content loss
    def loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
        
        return K.mean(K.square(model(y_true) - model(y_pred))) + self.mse_loss_rate * K.mean(K.square(y_true - y_pred))


def make_upscaler(input_shape):
        
    upscaler_input = Input(shape = input_image_shape)

    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(upscaler_input)
    model = PReLU(shared_axes=[1,2])(model)

    upsc_model = model

    for index in range(16):
        model = res_block_gen(model, 3, 64, 1)

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = BatchNormalization()(model)
    model = Add()([upsc_model, model])

    for index in range(2):
        model = up_sampling_block(model, 3, 256, 2)

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)

    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model






model_save_dir = 'trained_model'
images_dir = 'example_images'
subdir = 'ukiyo'
model_prefix = "orig_vgg-mse"

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

def save_images_orig(lowres, highres, idx_start, idx_stop, prefix, quality = 100):
    
    for idx in range(idx_start, idx_stop + 1):
        ex = lowres[idx]
        rescale_save_array_as_image(ex, images_dir + '/' + subdir + '/' + model_prefix + '/' + prefix + "_im%04d_lowres.jpg" % idx, quality)

        ex = highres[idx]
        save_array_as_image(ex, images_dir + '/' + subdir + '/' + model_prefix + prefix + "_im%04d_orig.jpg" % idx, quality)

def save_images_predicted(lowres, upscaler, idx_start, idx_stop, prefix, batch, quality = 100):
    
    for idx in range(idx_start, idx_stop + 1):
        ex = upscaler.predict(lowres[[idx]])[0]
        save_array_as_image(ex, images_dir + '/' + subdir + '/' + model_prefix + '/' + prefix + "_im%04d_upscaled_%06d.jpg" % (idx, batch), quality)








input_dir = '../images/ukiyo-e_fullhd'
number_of_images = 1000
train_test_ratio = 0.95

x_train_lr, x_train_hr, x_test_lr, x_test_hr = Utils.load_training_data(input_dir, '.jpg', number_of_images, train_test_ratio, downscale_factor, prog_func=tqdm)




upscaler = make_upscaler(input_image_shape)
input_layer = Input(shape=input_image_shape)
upscaled_layer = upscaler(input_layer)
upscaler_training_model = Model(inputs=input_layer, outputs=upscaled_layer)
vgg_loss = VGG_MSE_LOSS(output_image_shape, 0.01).loss
upscaler_training_model.compile(loss=vgg_loss, optimizer=Adam())





batch_count = 40001
batch_size = 1

loss_file_name = model_save_dir + '/' + subdir + '/' + model_prefix + '/' + 'losses_upscaler_' + model_prefix + '.txt'
best_loss_file_name = model_save_dir + '/' + subdir + '/' + model_prefix + '/' + 'losses_upscaler_' + model_prefix + '_best.txt'
model_file_name_tpl = model_save_dir + '/' + subdir + '/' + model_prefix + '/' + 'model_upscaler_' + model_prefix + '_%06db.h5'
model_file_name_best = model_save_dir + '/' + subdir + '/' + model_prefix + '/' + 'model_upscaler_' + model_prefix + '_best.h5'
model_save_batches = 500

agg_loss = 0.03
loss_update_rate = 0.01

loss_file = open(loss_file_name, 'w+')
loss_file.write('batch\tloss\tagg_loss\n')
loss_file.close()

loss_file = open(best_loss_file_name, 'w+')
loss_file.write('batch\tloss\tagg_loss\n')
loss_file.close()

save_images_orig(x_train_lr, x_train_hr, 0, 10, model_prefix + '_train')
save_images_orig(x_test_lr, x_test_hr, 0, 10, model_prefix + '_test')

best_loss = np.inf

for b in tqdm(range(batch_count), desc = 'Batch'):

    rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

    image_batch_hr = x_train_hr[rand_nums]
    image_batch_lr = x_train_lr[rand_nums]
    loss = upscaler_training_model.train_on_batch(image_batch_lr, image_batch_hr)

    agg_loss = (1 - loss_update_rate) * agg_loss + loss_update_rate * loss
    
    loss_file = open(loss_file_name, 'a')
    loss_file.write('%d\t%f\t%f\n' %(b, loss, agg_loss) )
    loss_file.close()
    
    if agg_loss < best_loss:
        best_loss = agg_loss
        
        upscaler.save(model_file_name_best)
        
        loss_file = open(best_loss_file_name, 'a')
        loss_file.write('%d\t%f\t%f\n' %(b, loss, agg_loss) )
        loss_file.close()
    
    if(b % model_save_batches == 0):
        upscaler.save(model_file_name_tpl % b)
        save_images_predicted(x_train_lr, upscaler, 0, 10, model_prefix + '_train', b)
        save_images_predicted(x_test_lr, upscaler, 0, 10, model_prefix + '_test', b)
