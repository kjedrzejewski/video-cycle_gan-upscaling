import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Input, Lambda, Add, Concatenate
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, Cropping2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import math

def residual_block(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization()(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization()(model)
        
    model = Add()([gen, model])
    
    return model


def residual_block_simple(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        
    model = Add()([gen, model])
    
    return model

def downsampling_block(model, kernel_size, filters, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

def upsampling_block(model, kernel_size, filters, strides):
    
    model = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

    
class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
        
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    def loss(self, y_true, y_pred):
        return K.mean(K.square(self.model(y_true) - self.model(y_pred)))


class VGG_MSE_LOSS(object):

    def __init__(self, image_shape, mse_loss_rate = 0.1):
        
        self.image_shape = image_shape
        self.mse_loss_rate = mse_loss_rate
        
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    def loss(self, y_true, y_pred):
        return K.mean(K.square(self.model(y_true) - self.model(y_pred))) + self.mse_loss_rate * K.mean(K.square(y_true - y_pred))


class VGG_MAE_LOSS(object):

    def __init__(self, image_shape, mae_loss_rate = 0.1):
        
        self.image_shape = image_shape
        self.mae_loss_rate = mae_loss_rate
        
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    def loss(self, y_true, y_pred):
        return K.mean(K.abs(self.model(y_true) - self.model(y_pred))) + self.mae_loss_rate * K.mean(K.abs(y_true - y_pred))



def make_upscaler_orig(output_image_shape, upscale_factor = 4):
    
    input_image_shape = (output_image_shape[0] // upscale_factor, output_image_shape[1] // upscale_factor, output_image_shape[2])
        
    upscale_times = int(math.log(upscale_factor,2))
    
    upscaler_input = Input(shape = input_image_shape)

    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(upscaler_input)
    model = PReLU(shared_axes=[1,2])(model)

    upsc_model = model

    for index in range(16):
        model = residual_block(model, 3, 64, 1)

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = BatchNormalization()(model)
    model = Add()([upsc_model, model])

    for index in range(upscale_times):
        model = upsampling_block(model, 3, 256, 2)

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)

    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model



def make_upscaler_skip_con(output_image_shape, upscale_factor = 4):
    
    input_image_shape = (output_image_shape[0] // upscale_factor, output_image_shape[1] // upscale_factor, output_image_shape[2])
    
    upscale_times = int(math.log(upscale_factor,2))
    
    upscaler_input = Input(shape = input_image_shape)

    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(upscaler_input)
    model = PReLU(shared_axes=[1,2])(model)

    upsc_model = model

    for index in range(16):
        model = residual_block(model, 3, 64, 1)

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = BatchNormalization()(model)
    model = Add()([upsc_model, model])

    for index in range(upscale_times):
        model = upsampling_block(model, 3, 224, 2) # smaller number of filters due to OOM error
        
    resized_input = (Lambda(lambda x: K.resize_images(x, 2 ** upscale_times, 2 ** upscale_times, "channels_last", "bilinear")))(upscaler_input)
    model = Concatenate(axis = 3)([resized_input, model])

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)

    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model








def same_size_unetish_block(model, kernal_size, filters, strides, name):
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", name = name+'/Conv2D')(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/PReLU')(model)
    
    return model

def downsampling_unetish_block(model, kernel_size, filters, strides, name):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/Conv2D')(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/PReLU')(model)
    
    return model

def upsampling_unetish_block(model, kernel_size, filters, strides, name):
    
    model = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/Conv2DTrans')(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/PReLU')(model)
    
    return model


def find_crop_shape(input_layer, output_down, output_up):
    # this way the shapes of the output is calcualted
    down_shape = Model(inputs = input_layer, outputs = output_down).output_shape
    up_shape   = Model(inputs = input_layer, outputs = output_up).output_shape
    
    height_diff = up_shape[1] - down_shape[1]
    width_diff  = up_shape[2] - down_shape[2]
    
    top_crop = height_diff // 2
    left_crop = width_diff // 2
    
    crop_shapes = ((top_crop, height_diff - top_crop),(left_crop, width_diff - left_crop))
    
    return crop_shapes
    
    
def concatenate_layers(input_layer, output_down, output_up, name):
    
    crop_shapes = find_crop_shape(input_layer, output_down, output_up)
    
    model = Cropping2D(cropping=crop_shapes, name = name+"/Cropping2D")(output_up)
    model = Concatenate(axis = 3, name = name+"/Concatenate")([output_down, model])
    
    return model



def make_upscaler_unetish(output_image_shape, upscale_factor = 4, step_size = 2, downscale_times = 2, initial_step_filter_count = 128): 

    upscale_times = int(math.log(upscale_factor,2)) + downscale_times
    input_image_shape = (output_image_shape[0] // upscale_factor, output_image_shape[1] // upscale_factor, output_image_shape[2])
    
    upscaler_input = Input(shape = input_image_shape, name = 'input')
    
    model = Conv2D(filters = initial_step_filter_count, kernel_size = 9, strides = 1, padding = "same", name = 'initial/Conv2D')(upscaler_input)
    model = PReLU(shared_axes=[1,2], name = 'initial/PReLU')(model)

    upsc_model = model
    
    
    outputs = []
    step_filter_count = initial_step_filter_count
    
    # downsampling steps
    for step in range(downscale_times):
        
        for index in range(step_size):
            model = same_size_unetish_block(model, 3, step_filter_count, 1, "down/"+str(step)+"/same/"+str(index))
        
        outputs.append(model)
        model = downsampling_unetish_block(model, 3, step_filter_count, 2, "down/"+str(step)+"/down")
        step_filter_count = step_filter_count * 2
    
    
    # steps at the bottom of U
    for index in range(step_size):
        model = same_size_unetish_block(model, 3, step_filter_count, 1, "bottom/"+str(step)+"/same"+str(index))
    
    
    down_outputs_len = len(outputs)
    
    # upsampling steps
    for step in range(upscale_times):
        model = upsampling_unetish_block(model, 3, step_filter_count, 2, "up/"+str(step)+"/up")
        
        if step < down_outputs_len:
            model = concatenate_layers(upscaler_input, outputs[down_outputs_len - step - 1], model, "up/"+str(step)+"/concat")
            step_filter_count = step_filter_count // 2
            
        for index in range(step_size):
            model = same_size_unetish_block(model, 3, step_filter_count, 1, "up/"+str(step)+"/same/"+str(index))
    
    
    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", name = 'final/Conv2D')(model)
    model = Activation('tanh', name = 'final/tanh')(model)
    
    
    # making sure the output is of the right shape
    
    # to extract the output shape
    output_shape = Model(inputs = upscaler_input, outputs = model).output_shape

    height_diff = output_shape[1] - output_image_shape[0]
    width_diff  = output_shape[2] - output_image_shape[1]
    
    top_crop = height_diff // 2
    left_crop = width_diff // 2
    
    crop_shapes = ((top_crop, height_diff - top_crop),(left_crop, width_diff - left_crop))
    
    model = Cropping2D(cropping=crop_shapes, name = 'final/Cropping2D')(model)
    
    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model










def compile_training_model(upscaler, loss, optimizer=Adam()):
    
    input_layer = Input(shape=upscaler.input_shape[1:4])
    upscaled_layer = upscaler(input_layer)
    upscaler_training_model = Model(inputs=input_layer, outputs=upscaled_layer)
    upscaler_training_model.compile(loss=loss, optimizer=optimizer)
    
    return upscaler_training_model