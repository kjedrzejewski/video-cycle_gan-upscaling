import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Input, Lambda, Add, Concatenate, Dropout, Multiply, Dense
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, Cropping2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import tensorflow as tf
import math
from abc import abstractmethod, ABCMeta

def residual_block(model, kernel_size, filters, strides, name=""):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/conv_pre')(model)
    model = BatchNormalization(name = name+'/batch_norm_pre')(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/prelu')(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/conv_post')(model)
    model = BatchNormalization(name = name+'/batch_norm_post')(model)
        
    model = Add(name = name+'/final_add')([gen, model])
    
    return model


def residual_block_attention(model, input_, kernel_size, filters, strides, batch_norm=True, name=""):
    
    gen = model
    
    attention = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/attention')(input_)
    attention = Activation('sigmoid', name = name+'/attention_sigmoid')(attention)
    model = Multiply(name = name+'/attention_multiply')([attention, model])
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/conv_pre')(model)
    if batch_norm:
        model = BatchNormalization(name = name+'/batch_norm_pre')(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/prelu')(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/conv_post')(model)
    if batch_norm:
        model = BatchNormalization(name = name+'/batch_norm_post')(model)
        
    model = Add(name = name+'/final_add')([gen, model])
    
    return model


def residual_block_simple(model, kernel_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
        
    model = Add()([gen, model])
    
    return model

def downsampling_block(model, kernel_size, filters, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

def upsampling_block(model, kernel_size, filters, strides, name=""):
    
    model = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same",name=name+"/conv_transp")(model)
    model = LeakyReLU(alpha = 0.2, name = name + "/leaky_relu")(model)
    
    return model


def upsampling_block_attention(model, input_, scale, kernel_size, filters, name=""):
    
    input_upscaled_nearest = (Lambda(lambda x: K.resize_images(x, scale//2, scale//2, "channels_last", "nearest"), name= name + '/nearest'))(input_)
    input_upscaled_bilinear = (Lambda(lambda x: K.resize_images(x, scale//2, scale//2, "channels_last", "bilinear"), name= name + '/resize_bilinear'))(input_)
    input_upscaled = Concatenate(name = name + '/upscaled_concat')([input_upscaled_nearest, input_upscaled_bilinear])
    
    model_cnt = Model(inputs = input_, outputs = model).output_shape[3]
    
    attention = Conv2D(filters = model_cnt, kernel_size = kernel_size, strides = 1, padding = "same",name = name + "/attention")(input_upscaled)
    attention = Activation('sigmoid', name = name +'/attention_sigmoid')(attention)
    
    model = Multiply(name = name + '/attention_multiply')([attention, model])
    model = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = 2, padding = "same", name = name + "/conv_transp")(model)
    model = LeakyReLU(alpha = 0.2, name = name + "/leaky_relu")(model)
    
    to_add_input = (Lambda(lambda x: tf.math.atanh(0.99999 * x), name =  name + "/to_add_input_atanh"))(input_)
    to_add_input = Conv2DTranspose(filters = filters, kernel_size = scale + 1, strides = scale, padding = "same", name = name + "/to_add_input_conv_transp")(to_add_input)
    
    model = Add(name = name + '/add_input')([model, to_add_input])
    
    return model

    
class VGG_LOSS(object):

    def __init__(self, image_shape, vgg19 = None):
        
        self.image_shape = image_shape
        
        if vgg19 is None:
            vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
            vgg19.trainable = False
            for l in vgg19.layers:
                l.trainable = False
        
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    def loss(self, y_true, y_pred):
        return K.mean(K.square(self.model(y_true) - self.model(y_pred)))


class VGG_MSE_LOSS(object):

    def __init__(self, image_shape, mse_loss_rate = 0.1, vgg19 = None):
        
        self.image_shape = image_shape
        self.mse_loss_rate = mse_loss_rate
        
        if vgg19 is None:
            vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
            vgg19.trainable = False
            for l in vgg19.layers:
                l.trainable = False
        
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    def loss(self, y_true, y_pred):
        return K.mean(K.square(self.model(y_true) - self.model(y_pred))) + self.mse_loss_rate * K.mean(K.square(y_true - y_pred))


class VGG_MAE_LOSS(object):

    def __init__(self, image_shape, mae_loss_rate = 0.1, vgg19 = None):
        
        self.image_shape = image_shape
        self.mae_loss_rate = mae_loss_rate
        
        if vgg19 is None:
            vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
            vgg19.trainable = False
            for l in vgg19.layers:
                l.trainable = False
        
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    def loss(self, y_true, y_pred):
        return K.mean(K.abs(self.model(y_true) - self.model(y_pred))) + self.mae_loss_rate * K.mean(K.abs(y_true - y_pred))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)





class GanLosses(metaclass = ABCMeta):
        
    def __init__(self, loss_activation = 'log-sigm', real_output = None, fake_output = None):
        self._real_output = real_output
        self._fake_output = fake_output
        
        if loss_activation == 'sigmoid':
            self.loss_activation = lambda x: K.sigmoid(x)
        elif loss_activation == 'log-sigm':
            self.loss_activation = lambda x: K.log(K.sigmoid(x))
        elif loss_activation == 'tanh':
            self.loss_activation = lambda x: K.tanh(x)
        elif loss_activation == 'bi-log':
            self.loss_activation = lambda x: (x / (1 + K.abs(x))) * K.log(K.abs(x) + 2)
        else:
            self.loss_activation = lambda x: x
    
    
    @property
    def real_output(self):
        return self._real_output

    @real_output.setter
    def real_output(self, real_output):
        self._real_output = real_output
        
    
    @property
    def fake_output(self):
        return self._fake_output

    @fake_output.setter
    def fake_output(self, fake_output):
        self._fake_output = fake_output

            
    @property
    @abstractmethod
    def discriminator_loss(self):
        pass
    
    @property
    @abstractmethod
    def generator_loss(self):
        pass
        
    


class WassersteinLosses(GanLosses):
    
    @property
    def discriminator_loss(self):
        
        def loss(y_true,y_pred):
            l = K.mean(self._real_output) - K.mean(self._fake_output)
            
            return l
        
        return loss
    
    @property
    def generator_loss(self):
        
        def loss(y_true,y_pred):
            l = K.mean(self._fake_output)
            
            return l
        
        return loss

    

class RelativisticLosses(GanLosses):
    
    @property
    def discriminator_loss(self):
        
        def loss(y_true,y_pred):
            l = K.mean(self._real_output) - K.mean(self._fake_output)
            l = self.loss_activation(l)
            
            return l
        
        return loss
    
    @property
    def generator_loss(self):
        
        def loss(y_true,y_pred):
            l = K.mean(self._fake_output) - K.mean(self._real_output)
            l = self.loss_activation(l)
            
            return l
        
        return loss
    
    



def make_upscaler_orig(output_image_shape, kernel_size = 5, filters = 64, upscale_factor = 4, res_block_num=16):
    
    input_image_shape = (output_image_shape[0] // upscale_factor, output_image_shape[1] // upscale_factor, output_image_shape[2])
        
    upscale_times = int(math.log(upscale_factor,2))
    
    upscaler_input = Input(shape = input_image_shape, name="initial/input")

    model = Conv2D(filters = filters, kernel_size = 9, strides = 1, padding = "same", name="initial/conv")(upscaler_input)
    model = PReLU(shared_axes=[1,2], name="initial/prelu")(model)

    upsc_model = model

    for index in range(res_block_num):
        model = residual_block(model, kernel_size, filters, 1, name="res_block/"+str(index))

    model = Conv2D(filters = 64, kernel_size = kernel_size, strides = 1, padding = "same", name='prefinal/conv2d')(model)
    model = BatchNormalization(name='prefinal/batch_norm')(model)
    model = Add(name='prefinal/tanh')([upsc_model, model])

    for index in range(upscale_times):
        model = upsampling_block(model, kernel_size, 256, 2, name='upscaling/'+str(index)+'/block')

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", name='final/conv')(model)
    model = Activation('tanh', name='final/tanh')(model)

    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model



def make_upscaler_attention(output_image_shape, kernel_size = 5, filters = 64, upscale_factor = 4, res_block_num=16):
    
    input_image_shape = (output_image_shape[0] // upscale_factor, output_image_shape[1] // upscale_factor, output_image_shape[2])
    
    upscale_times = int(math.log(upscale_factor,2))
    
    upscaler_input = Input(shape = input_image_shape, name="initial/input")

    model = Conv2D(filters = filters, kernel_size = 9, strides = 1, padding = "same", name="initial/conv")(upscaler_input)
    model = PReLU(shared_axes=[1,2], name="initial/prelu")(model)

    upsc_model = model

    for index in range(res_block_num):
        model = residual_block_attention(model, upscaler_input, kernel_size, filters, 1, name="res_block/"+str(index))

    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = 1, padding = "same", name="after_res/conv")(model)
    model = BatchNormalization(name="after_res/batch_norm")(model)
    model = Add(name="after_res/add")([upsc_model, model])
    
    for index in range(upscale_times):
        scale = 2 ** (index + 1)
        model = upsampling_block_attention(model, upscaler_input, scale, kernel_size, 128, name='upscaling/'+str(index)+'/block')

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", name='final/conv')(model)
    model = Activation('tanh', name='final/tanh')(model)

    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model



def make_upscaler_skip_con(output_image_shape, kernel_size = 5, filters = 64, upscale_factor = 4):
    
    input_image_shape = (output_image_shape[0] // upscale_factor, output_image_shape[1] // upscale_factor, output_image_shape[2])
    
    upscale_times = int(math.log(upscale_factor,2))
    
    upscaler_input = Input(shape = input_image_shape)

    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(upscaler_input)
    model = PReLU(shared_axes=[1,2])(model)

    upsc_model = model

    for index in range(16):
        model = residual_block(model, kernel_size, filters, 1)

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








def inception_mini_resblock(model, filters, name, kernel_size, batch_normalisation = True):
    
    if batch_normalisation:
        model = BatchNormalization(name = name+'/batch_norm')(model)
        
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/prelu')(model)

    model = Conv2D(filters = filters, kernel_size = kernel_size, padding = "same", name = name+'/%dx%d' % (kernel_size[0], kernel_size[1]))(model)
    
    return model




def inception_resblock_3path(model, filters, name, kernel_size = 3, batch_normalisation = True):
    
    gen = model
    
    path_a_filters = int(filters * 0.5)
    path_b_filters = int(filters * 0.5)
    path_c_filters1 = int(filters * 0.5)
    path_c_filters2 = int(filters * 0.75)
    path_c_filters3 = filters
    

    path_a_model = inception_mini_resblock(model, path_a_filters, name = name+'/a/1', kernel_size = (1,1), batch_normalisation = batch_normalisation)
    
    path_b_model = inception_mini_resblock(model, path_b_filters, name = name+'/b/1', kernel_size = (1,1), batch_normalisation = batch_normalisation)
    path_b_model = inception_mini_resblock(path_b_model, path_b_filters, name = name+'/b/2', kernel_size = (kernel_size,kernel_size), batch_normalisation = batch_normalisation)
    
    path_c_model = inception_mini_resblock(model, path_c_filters1, name = name+'/c/1', kernel_size = (1,1), batch_normalisation = batch_normalisation)
    path_c_model = inception_mini_resblock(path_c_model, path_c_filters2, name = name+'/c/2', kernel_size = (kernel_size,kernel_size), batch_normalisation = batch_normalisation)
    path_c_model = inception_mini_resblock(path_c_model, path_c_filters3, name = name+'/c/3', kernel_size = (kernel_size,kernel_size), batch_normalisation = batch_normalisation)

    model = Concatenate(axis = 3, name = name+'/final/concat')([path_a_model, path_b_model, path_c_model])
    model = Conv2D(filters = filters, kernel_size = 1, padding = "same", name = name+'/final/1x1')(model)
    
    model = Add(name = name+'/final/add')([gen, model])

    return model




def inception_resblock_2path(model, filters, name, kernel_size = 7, batch_normalisation = True):
    
    gen = model

    path_a_filters = int(filters * 0.5)
    path_b_filters1 = int(filters * 0.3)
    path_b_filters2 = int(filters * 0.4)
    path_b_filters3 = int(filters * 0.5)
    
    path_a_model = inception_mini_resblock(model, path_a_filters, name = name+'/a/1', kernel_size = (1,1), batch_normalisation = batch_normalisation)
    
    path_b_model = inception_mini_resblock(model, path_b_filters1, name = name+'/b/1', kernel_size = (1,1), batch_normalisation = batch_normalisation)
    path_b_model = inception_mini_resblock(path_b_model, path_b_filters2, name = name+'/b/2', kernel_size = (1,kernel_size), batch_normalisation = batch_normalisation)
    path_b_model = inception_mini_resblock(path_b_model, path_b_filters3, name = name+'/b/3', kernel_size = (kernel_size,1), batch_normalisation = batch_normalisation)
    
    model = Concatenate(axis = 3, name = name+'/final/concat')([path_a_model, path_b_model])
    model = Conv2D(filters = filters, kernel_size = 1, padding = "same", name = name+'/final/1x1')(model)
    
    model = Add(name = name+'/final/add')([gen, model])

    return model






def make_upscaler_incep_resnet(
    output_image_shape, filters = 64, upscale_factor = 4,
    a_block_type = '3path', a_block_num = 5, a_block_kernel = 3,
    b_block_type = '2path', b_block_num = 10, b_block_kernel = 7,
    c_block_type = '2path', c_block_num = 5, c_block_kernel = 3
):
    
    input_image_shape = (output_image_shape[0] // upscale_factor, output_image_shape[1] // upscale_factor, output_image_shape[2])
    upscale_times = int(math.log(upscale_factor,2))
    upscaler_input = Input(shape = input_image_shape, name="initial/input")

    
    model = Conv2D(filters = filters, kernel_size = 9, strides = 1, padding = "same", name="initial/conv/9x9")(upscaler_input)

    upsc_model = model

    for index in range(a_block_num):
        if a_block_type == '3path':
            model = inception_resblock_3path(model, filters, name="inc_res_block/A/3p/"+str(index), kernel_size = a_block_kernel, batch_normalisation = True)
        elif a_block_type == '2path':
            model = inception_resblock_2path(model, filters, name="inc_res_block/A/2p/"+str(index), kernel_size = a_block_kernel, batch_normalisation = True)
    
    for index in range(b_block_num):
        if b_block_type == '3path':
            model = inception_resblock_3path(model, filters, name="inc_res_block/B/3p/"+str(index), kernel_size = b_block_kernel, batch_normalisation = True)
        elif b_block_type == '2path':
            model = inception_resblock_2path(model, filters, name="inc_res_block/B/2p/"+str(index), kernel_size = b_block_kernel, batch_normalisation = True)

    for index in range(c_block_num):
        if c_block_type == '3path':
            model = inception_resblock_3path(model, filters, name="inc_res_block/c/3p/"+str(index), kernel_size = c_block_kernel, batch_normalisation = True)
        elif c_block_type == '2path':
            model = inception_resblock_2path(model, filters, name="inc_res_block/c/2p/"+str(index), kernel_size = c_block_kernel, batch_normalisation = True)

    model = Conv2D(filters = filters, kernel_size = c_block_kernel, strides = 1, padding = "same", name='prefinal/conv2d')(model)
    model = BatchNormalization(name='prefinal/batch_norm')(model)
    model = Add(name='prefinal/tanh')([upsc_model, model])

    for index in range(upscale_times):
        model = upsampling_block(model, c_block_kernel, 256, 2, name='upscaling/'+str(index)+'/block')

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", name='final/conv')(model)
    model = Activation('tanh', name='final/tanh')(model)

    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model















def same_size_unetish_block(model, kernel_size, filters, strides, name, dropout_rate=0.1):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/Conv2D')(model)
    model = BatchNormalization()(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/PReLU')(model)
    model = Dropout(rate=dropout_rate, name = name+'/Dropout')(model)
    
    return model

def downsampling_unetish_block(model, kernel_size, filters, strides, name, dropout_rate=0.1):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/Conv2D')(model)
    model = BatchNormalization()(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/PReLU')(model)
    model = Dropout(rate=dropout_rate, name = name+'/Dropout')(model)
    
    return model

def upsampling_unetish_block(model, kernel_size, filters, strides, name, dropout_rate=0.1):
    
    model = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", name = name+'/Conv2DTrans')(model)
    model = BatchNormalization()(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2], name = name+'/PReLU')(model)
    model = Dropout(rate=dropout_rate, name = name+'/Dropout')(model)
    
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


def sum_layers(input_layer, output_down, output_up, name):
    
    crop_shapes = find_crop_shape(input_layer, output_down, output_up)
    
    model = Cropping2D(cropping=crop_shapes, name = name+"/Cropping2D")(output_up)
    model = Add(name = name+"/Add")([output_down, model])
    
    return model



def make_upscaler_unetish(output_image_shape, kernel_size = 5, upscale_factor = 4, step_size = 4, downscale_times = 5, initial_step_filter_count = 32, dropout_rate=0.1): 

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
            model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "down/"+str(step)+"/same/"+str(index))
        
        outputs.append(model)
        model = downsampling_unetish_block(model, kernel_size, step_filter_count, 2, "down/"+str(step)+"/down", dropout_rate=dropout_rate)
        step_filter_count = step_filter_count * 2
    
    
    # steps at the bottom of U
    for index in range(step_size):
        model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "bottom/"+str(step)+"/same/"+str(index), dropout_rate=dropout_rate)
    
    
    down_outputs_len = len(outputs)
    
    # upsampling steps
    for step in range(upscale_times):
        model = upsampling_unetish_block(model, kernel_size, step_filter_count, 2, "up/"+str(step)+"/up", dropout_rate=dropout_rate)
        
        if step < down_outputs_len:
            model = concatenate_layers(upscaler_input, outputs[down_outputs_len - step - 1], model, "up/"+str(step)+"/concat")
            step_filter_count = step_filter_count // 2
            
        for index in range(step_size):
            model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "up/"+str(step)+"/same/"+str(index), dropout_rate=dropout_rate)
    

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)
    
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





def make_upscaler_unetish_add(output_image_shape, kernel_size = 5, upscale_factor = 4, step_size = 4, downscale_times = 5, initial_step_filter_count = 48, dropout_rate=0.1): 

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
            model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "down/"+str(step)+"/same/"+str(index))
        
        outputs.append(model)
        model = downsampling_unetish_block(model, kernel_size, step_filter_count, 2, "down/"+str(step)+"/down", dropout_rate=dropout_rate)
        step_filter_count = step_filter_count * 2
    
    
    # steps at the bottom of U
    for index in range(step_size):
        model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "bottom/"+str(step)+"/same/"+str(index), dropout_rate=dropout_rate)
    
    step_filter_count = step_filter_count // 2
    
    down_outputs_len = len(outputs)
    
    # upsampling steps
    for step in range(upscale_times):
        model = upsampling_unetish_block(model, kernel_size, step_filter_count, 2, "up/"+str(step)+"/up", dropout_rate=dropout_rate)
        
        if step < down_outputs_len:
            model = sum_layers(upscaler_input, outputs[down_outputs_len - step - 1], model, "up/"+str(step)+"/add")
            step_filter_count = step_filter_count // 2
            
        for index in range(step_size):
            model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "up/"+str(step)+"/same/"+str(index), dropout_rate=dropout_rate)
    

    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)
    
    # making sure the output is of the right shape
    
    # to extract the output shape
    output_shape = Model(inputs = upscaler_input, outputs = model).output_shape

    height_diff = output_shape[1] - output_image_shape[0]
    width_diff  = output_shape[2] - output_image_shape[1]
    
    top_crop = height_diff // 2
    left_crop = width_diff // 2
    
    crop_shapes = ((top_crop, height_diff - top_crop),(left_crop, width_diff - left_crop))
    
    model = Cropping2D(cropping=crop_shapes, name = 'prefinal/Cropping2D')(model)
    
    resized_input = (Lambda(lambda x: K.resize_images(x, upscale_factor, upscale_factor, "channels_last", "bilinear"), name = 'final/input_resize/resize'))(upscaler_input)
    resized_input = (Lambda(lambda x: tf.math.atanh(0.99999 * x), name = 'final/input_resize/atanh'))(resized_input)
    
    model = sum_layers(upscaler_input, model, resized_input, "final/concat")
    
    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)
    
    upscaler_model = Model(inputs = upscaler_input, outputs = model)

    return upscaler_model

























def make_upscaler_unetish_complex(output_image_shape, kernel_size = 5, upscale_factor = 4, step_size = 4, downscale_times = 3, initial_step_filter_count = 32, dropout_rate=0.1): 

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
            model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "down/"+str(step)+"/same/"+str(index))
        
        outputs.append(model)
        model = downsampling_unetish_block(model, kernel_size, step_filter_count, 2, "down/"+str(step)+"/down", dropout_rate=dropout_rate)
        step_filter_count = step_filter_count * 2
    
    
    # steps at the bottom of U
    for index in range(step_size):
        model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "bottom/"+str(step)+"/same/"+str(index), dropout_rate=dropout_rate)
    
    
    down_outputs_len = len(outputs)
    
    # upsampling steps
    for step in range(upscale_times):
        model = upsampling_unetish_block(model, kernel_size, step_filter_count, 2, "up/"+str(step)+"/up", dropout_rate=dropout_rate)
        
        if step < down_outputs_len:
            model = concatenate_layers(upscaler_input, outputs[down_outputs_len - step - 1], model, "up/"+str(step)+"/concat")
            step_filter_count = step_filter_count // 2
            
        for index in range(step_size):
            model = same_size_unetish_block(model, kernel_size, step_filter_count, 1, "up/"+str(step)+"/same/"+str(index), dropout_rate=dropout_rate)
    

    # adding upscaled original image
    resized_input = (Lambda(lambda x: K.resize_images(x, upscale_factor, upscale_factor, "channels_last", "bilinear"), name = 'input_resize/resize'))(upscaler_input)
    #resized_input = (Lambda(lambda x: tf.math.atanh(0.999 * x), name = 'input_resize/atanh'))(resized_input)
    attention = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", name = 'final/initial/attention')(resized_input)
    
    for step in range(3):
        attention = Concatenate(axis = 3, name = 'final/'+str(step)+'/input_concat')([resized_input, attention])
        attention = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", name = 'final/'+str(step)+'/attention')(attention)
        attention = Activation('sigmoid', name = 'final/'+str(step)+'/att_sigmoid')(attention)
        
        model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", name = 'final/'+str(step)+'/Conv2D')(model)
        att_model = Multiply(name = 'final/'+str(step)+'/att_Conv2D')([attention, model])
        
        model = Concatenate(axis = 3, name = 'final/'+str(step)+'/input_att_concat')([att_model, model])
        model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", name = 'final/'+str(step)+'/Conv2D_after_att')(model)
        model = Activation('tanh', name = 'final/'+str(step)+'/tanh')(model)
        
        if step < 2:
            model = Dropout(rate=dropout_rate, name = 'final/'+str(step)+'/Dropout')(model)
    
    
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








def make_discriminator_simple_512(input_shape, activation = 'none'):
    input = Input(input_shape, name = 'discriminator/input')

    layer = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", name = 'discriminator/block_1/Conv2d')(input)
    layer = BatchNormalization(name = 'discriminator/block_1/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_1/LeakyReLU')(layer)
    
    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_2/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_2/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_2/LeakyReLU')(layer)

    layer = Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_3/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_3/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_3/LeakyReLU')(layer)

    layer = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_4/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_4/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_4/LeakyReLU')(layer)

    layer = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_5/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_5/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_5/LeakyReLU')(layer)

    layer = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_6/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_6/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_6/LeakyReLU')(layer)

    layer = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_7/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_7/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_7/LeakyReLU')(layer)

    layer = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_8/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_8/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_8/LeakyReLU')(layer)
    
    layer = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_9/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_9/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_9/LeakyReLU')(layer)

    layer = Flatten(name = 'discriminator/final/Flatten')(layer)
    layer = Dense(1024, name = 'discriminator/final/Dense_1')(layer)
    layer = BatchNormalization(name = 'discriminator/final/BatchNorm_1')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/final/LeakyReLU_1')(layer)
    
    layer = Dense(32, name = 'discriminator/final/Dense_2')(layer)
    layer = BatchNormalization(name = 'discriminator/final/BatchNorm_2')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/final/LeakyReLU_2')(layer)

    layer = Dense(1, name = 'discriminator/final/Dense_3')(layer)
    if activation == 'sigmoid':
        layer = Activation('sigmoid', name = 'discriminator/final/sigmoid')(layer)
    elif activation == 'log-sigm':
        layer = Lambda(lambda x: K.log(K.sigmoid(x)), name = 'discriminator/final/log-sigmoid')(layer)
    elif activation == 'tanh':
        layer = Activation('tanh', name = 'discriminator/final/tanh')(layer)
    elif activation == 'bi-log':
        layer = Lambda(lambda x: (x / (1 + K.abs(x))) * K.log(K.abs(x) + 2), name = 'discriminator/final/bi-log')(layer)

    model  = Model(inputs = input, outputs = layer)
    
    return model




def make_discriminator_thin_512(input_shape, activation = 'none'):
    input = Input(input_shape, name = 'discriminator/input')

    layer = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", name = 'discriminator/block_1/Conv2d')(input)
    layer = BatchNormalization(name = 'discriminator/block_1/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_1/LeakyReLU')(layer)
    
    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_2/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_2/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_2/LeakyReLU')(layer)

    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_3/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_3/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_3/LeakyReLU')(layer)

    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_4/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_4/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_4/LeakyReLU')(layer)

    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_5/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_5/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_5/LeakyReLU')(layer)

    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_6/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_6/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_6/LeakyReLU')(layer)

    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_7/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_7/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_7/LeakyReLU')(layer)

    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_8/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_8/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_8/LeakyReLU')(layer)
    
    layer = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", name = 'discriminator/block_9/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_9/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_9/LeakyReLU')(layer)

    layer = Flatten(name = 'discriminator/final/Flatten')(layer)
    layer = Dense(1024, name = 'discriminator/final/Dense_1')(layer)
    layer = BatchNormalization(name = 'discriminator/final/BatchNorm_1')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/final/LeakyReLU_1')(layer)
    
    layer = Dense(32, name = 'discriminator/final/Dense_2')(layer)
    layer = BatchNormalization(name = 'discriminator/final/BatchNorm_2')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/final/LeakyReLU_2')(layer)

    layer = Dense(1, name = 'discriminator/final/Dense_3')(layer)
    if activation == 'sigmoid':
        layer = Activation('sigmoid', name = 'discriminator/final/sigmoid')(layer)
    elif activation == 'log-sigm':
        layer = Lambda(lambda x: K.log(K.sigmoid(x)), name = 'discriminator/final/log-sigmoid')(layer)
    elif activation == 'tanh':
        layer = Activation('tanh', name = 'discriminator/final/tanh')(layer)
    elif activation == 'bi-log':
        layer = Lambda(lambda x: (x / (1 + K.abs(x))) * K.log(K.abs(x) + 2), name = 'discriminator/final/bi-log')(layer)

    model  = Model(inputs = input, outputs = layer)
    
    return model


def make_discriminator_sparse_512(input_shape, activation = 'none'):
    input = Input(input_shape, name = 'discriminator/input')

    layer = Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = "valid", name = 'discriminator/block_1/Conv2d')(input)
    layer = BatchNormalization(name = 'discriminator/block_1/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_1/LeakyReLU')(layer)
    
    layer = Conv2D(filters = 128, kernel_size = 5, strides = 3 , padding = "valid", name = 'discriminator/block_2/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_2/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_2/LeakyReLU')(layer)

    layer = Conv2D(filters = 256, kernel_size = 5, strides = 3, padding = "valid", name = 'discriminator/block_3/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_3/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_3/LeakyReLU')(layer)
    
    layer = Conv2D(filters = 256, kernel_size = 5, strides = 3, padding = "valid", name = 'discriminator/block_4/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_4/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_4/LeakyReLU')(layer)

    layer = Conv2D(filters = 256, kernel_size = 5, strides = 3, padding = "valid", name = 'discriminator/block_5/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_5/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_5/LeakyReLU')(layer)
    
    layer = Conv2D(filters = 256, kernel_size = 5, strides = 3, padding = "valid", name = 'discriminator/block_6/Conv2d')(layer)
    layer = BatchNormalization(name = 'discriminator/block_6/BatchNorm')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/block_6/LeakyReLU')(layer)

    layer = Flatten(name = 'discriminator/final/Flatten')(layer)
    layer = Dense(128, name = 'discriminator/final/Dense_1')(layer)
    layer = BatchNormalization(name = 'discriminator/final/BatchNorm_1')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/final/LeakyReLU_1')(layer)
    
    layer = Dense(32, name = 'discriminator/final/Dense_2')(layer)
    layer = BatchNormalization(name = 'discriminator/final/BatchNorm_2')(layer)
    layer = LeakyReLU(alpha = 0.1, name = 'discriminator/final/LeakyReLU_2')(layer)

    layer = Dense(1, name = 'discriminator/final/Dense_3')(layer)
    if activation == 'sigmoid':
        layer = Activation('sigmoid', name = 'discriminator/final/sigmoid')(layer)
    elif activation == 'log-sigm':
        layer = Lambda(lambda x: K.log(K.sigmoid(x)), name = 'discriminator/final/log-sigmoid')(layer)
    elif activation == 'tanh':
        layer = Activation('tanh', name = 'discriminator/final/tanh')(layer)
    elif activation == 'bi-log':
        layer = Lambda(lambda x: (x / (1 + K.abs(x))) * K.log(K.abs(x) + 2), name = 'discriminator/final/bi-log')(layer)

    model  = Model(inputs = input, outputs = layer)
    
    return model




def make_and_compile_gan(
    generator,
    discriminator,
    input_shape,
    output_shape,
    content_loss,
    content_loss_weight,
    discriminator_loss,
    discriminator_loss_weight,
    optimizer=Adam()
):
    
    generator_input_layer = Input(shape=input_shape)
    generator_layer = generator(generator_input_layer)
    generator_training_model = Model(inputs=generator_input_layer, outputs=generator_layer)
    generator_training_model.compile(loss=content_loss, optimizer=optimizer)
    
    discriminator.trainable = True
    discriminator_input_layer = Input(shape=output_shape)
    discriminator_layer = discriminator(discriminator_input_layer)
    discriminator_training_model = Model(inputs=discriminator_input_layer, outputs=discriminator_layer)
    discriminator_training_model.compile(loss=discriminator_loss, optimizer=optimizer)
    
    discriminator.trainable = False
    gan_input_layer = Input(shape=input_shape)
    gan_generator_layer = generator(gan_input_layer)
    gan_discriminator_layer = discriminator(gan_generator_layer)
    gan_training_model = Model(inputs=gan_input_layer, outputs=[gan_generator_layer, gan_discriminator_layer])
    gan_training_model.compile(
        loss=[content_loss, discriminator_loss],
        loss_weights=[content_loss_weight, discriminator_loss_weight],
        optimizer=optimizer
    )
    
    return generator_training_model, discriminator_training_model, gan_training_model





def make_and_compile_gan2(
    generator,
    discriminator,
    input_shape,
    output_shape,
    content_loss,
    content_loss_weight,
    discriminator_losses,
    discriminator_loss_weight,
    optimizer=Adam()
):


    generator_input = Input(shape=input_shape)
    generator_output = generator(generator_input)
    generator_training_model = Model(inputs=generator_input, outputs=generator_output)
    generator_training_model.compile(loss=content_loss, optimizer=optimizer)
    
    
    
    discriminator.trainable = True
    
    discriminator_real_input  = Input(shape=output_shape)
    discriminator_real_output = discriminator(discriminator_real_input)
    
    discriminator_fake_input  = Input(shape=output_shape)
    discriminator_fake_output = discriminator(discriminator_fake_input)
    
    disc_loss = discriminator_losses()
    disc_loss.real_output = discriminator_real_output
    disc_loss.fake_output = discriminator_fake_output
    disc_loss = disc_loss.discriminator_loss
    
    discriminator_training_model = Model(
        inputs=[discriminator_real_input, discriminator_fake_input],
        outputs=discriminator_fake_output
    )
    discriminator_training_model.compile(
        loss=disc_loss,
        optimizer=optimizer
    )
    
    
    
    discriminator.trainable = False
    
    gan_real_input  = Input(shape=output_shape)
    gan_disc_real_output = discriminator(gan_real_input)
    
    gan_input_layer = Input(shape=input_shape)
    gan_generator_output = generator(gan_input_layer)
    gan_disc_fake_output = discriminator(gan_generator_output)
    
    gan_loss = discriminator_losses()
    gan_loss.real_output = gan_disc_real_output
    gan_loss.fake_output = gan_disc_fake_output
    gan_loss = gan_loss.generator_loss
          
    gan_training_model = Model(
        inputs=[gan_input_layer, gan_real_input],
        outputs=[gan_generator_output, gan_disc_fake_output]
    )
    gan_training_model.compile(
        loss=[content_loss, gan_loss],
        loss_weights=[content_loss_weight, discriminator_loss_weight],
        optimizer=optimizer
    )
    
    return generator_training_model, discriminator_training_model, gan_training_model




def compile_training_model(upscaler, loss, optimizer=Adam()):
    
    input_layer = Input(shape=upscaler.input_shape[1:4])
    upscaled_layer = upscaler(input_layer)
    upscaler_training_model = Model(inputs=input_layer, outputs=upscaled_layer)
    upscaler_training_model.compile(loss=loss, optimizer=optimizer)
    
    return upscaler_training_model