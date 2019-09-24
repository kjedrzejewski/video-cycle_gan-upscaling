import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Input, Lambda, Add, Concatenate
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model


def residual_block(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization()(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization()(model)
        
    model = Add()([gen, model])
    
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



def make_upscaler_orig(input_image_shape, upscale_times = 2):
        
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



def make_upscaler_skip_con(input_image_shape, upscale_times = 2):
        
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
        
    resized_input = (Lambda(lambda x: K.resize_images(x, 4, 4, "channels_last", "bilinear")))(upscaler_input)
    model = Concatenate(axis = 3)([resized_input, model])

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)

    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model

def compile_training_model(upscaler, loss, optimizer=Adam()):
    
    input_layer = Input(shape=upscaler.input_shape[1:4])
    upscaled_layer = upscaler(input_layer)
    upscaler_training_model = Model(inputs=input_layer, outputs=upscaled_layer)
    upscaler_training_model.compile(loss=loss, optimizer=optimizer)
    
    return upscaler_training_model