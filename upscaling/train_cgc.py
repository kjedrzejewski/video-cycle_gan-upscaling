from upscaler.data import load_images_from_dir, split_images_train_test, downscale_images
from upscaler.data import select_random_rows, convert_image_series_to_array, convert_array_to_image
from upscaler.data import save_img_orig, save_img_resize, save_img_predict
from upscaler.model import make_upscaler_skip_con, make_upscaler_orig, make_upscaler_unetish, make_upscaler_unetish_add, make_upscaler_attention
from upscaler.model import VGG_LOSS, VGG_MSE_LOSS, VGG_MAE_LOSS
from upscaler.model import compile_training_model
from upscaler.json import PandasEncoder
from keras.utils.vis_utils import plot_model

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import argparse
import json

if __name__== "__main__":
    
    ###########################################################
    ## Preparing call arguments parsing
    ###########################################################
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--image_input_dir', action='store', dest='image_input_dir', default='ukiyo-e_fullhd', help='Path to load images from (subdir of "../images/")')
    
    parser.add_argument('-i1g', '--image_input_dir_1gen', action='store', dest='image_input_dir_1gen', default='ukiyo-e_1gen', help='Path to load images 1 generator converted messages from (subdir of "../images/")')
    
    parser.add_argument('-i2g', '--image_input_dir_2gen', action='store', dest='image_input_dir_2gen', default='ukiyo-e_2gen', help='Path to load images 2 generator converted messages from (subdir of "../images/")')
    
    parser.add_argument('-s', '--subdir', action='store', dest='subdir', default='ukiyo', help='Subdir to put generated images, trained models, etc to')
    
    parser.add_argument('-p', '--output_prefix', action='store', dest='output_prefix', default='auto', help='Prefix to put in names of generated files (and sometimes also a subdir). Default value \'auto\' means generate it automatically')
    
    parser.add_argument('-ic', '--image_count', action='store', dest='image_count', default='3000', help='Number of images to be used (and split into training and test sets)', type=int)
    
    parser.add_argument('-tr', '--train_test_ratio', action='store', dest='train_test_ratio', default='0.95', help='Ratio of splitting images into the test and train sets', type=float)
    
    parser.add_argument('-m', '--model', action='store', dest='model', default='orig', choices=['orig','skip-con','resnet-att','unetish','unetish-add'], help='Model to be used')
    
    parser.add_argument('-l', '--loss', action='store', dest='loss', default='vgg-only', choices=['vgg-only','vgg-mae','vgg-mse'], help='Loss function to be used for the training')
    
    parser.add_argument('-lw', '--non_vgg_loss_weight', action='store', dest='non_vgg_loss_weight', default='0.001', help='Weight of the loss other than VGG (if there is any)', type=float)
    
    parser.add_argument('-msf', '--model_save_freq', action='store', dest='model_save_freq', default='500', help='How frequently a model should be saved? (number of batches)', type=int)
    
    parser.add_argument('-bs', '--batch_size', action='store', dest='batch_size', default='1', help='Number of examples to be put in the batch', type=int)
    
    parser.add_argument('-nb', '--number_of_batches', action='store', dest='number_of_batches', default='40001', help='Number batches to be run', type=int)
    
    parser.add_argument('-d', '--downscale_factor', action='store', dest='downscale_factor', default='4', help='Downscale factor', type=int)
    
    parser.add_argument('-ks', '--kernel_size', action='store', dest='kernel_size', default='5', help='Kernel size', type=int)
    
    parser.add_argument('-dr', '--dropout_rate', action='store', dest='dropout_rate', default='0.0', help='Dropout rate to be used (if supported by given model)', type=float)

    parser.add_argument('-ss', '--split_seed', action='store', dest='split_seed', default='42', help='Dropout rate to be used (if supported by given model)', type=int)
    
    values = parser.parse_args()
    
    
    ###########################################################
    ## Parameters of input and output images
    ###########################################################
    
    downscale_factor = values.downscale_factor
    kernel_size = values.kernel_size
    # upscale_times = int(math.log(downscale_factor,2))
    output_image_shape = (1080, 1920,3)
    input_image_shape = (
        output_image_shape[0] // downscale_factor,
        output_image_shape[1] // downscale_factor,
        output_image_shape[2]
    )
    target_shape = (
        output_image_shape[1],
        output_image_shape[0]
    )
    if int(math.log(downscale_factor,2)) != math.log(downscale_factor,2):
        print("Downscale factor needs to be a power of 2. It was %d." % downscale_factor)
        sys.exit(0)
    
    
    ###########################################################
    ## Reading call arguments and setting up paths
    ###########################################################
    
    script_dirname = os.path.dirname(sys.argv[0])
    if script_dirname == '':
        script_dirname = '.'
    input_dir = script_dirname + '/../images/' + values.image_input_dir
    input_dir_1gen = script_dirname + '/../images/' + values.image_input_dir_1gen
    input_dir_2gen = script_dirname + '/../images/' + values.image_input_dir_2gen
    model_save_dir = script_dirname + '/' + 'trained_model'
    loss_save_dir = script_dirname + '/' + 'losses'
    images_dir = script_dirname + '/' + 'example_images'
    subdir = values.subdir
    #model_prefix = "orig_vgg-mse"
    model_prefix = values.output_prefix
    if model_prefix == 'auto':
        model_prefix = "cgc_" + values.model + "_" + values.loss + ("_x%d" % downscale_factor)
        print("Prefix generated automatically: '" + model_prefix + "'")
    
    number_of_images = values.image_count
    train_test_ratio = values.train_test_ratio

    # where the examples of images will be saved
    image_path = images_dir + '/' + subdir + '/' + model_prefix
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    print("Generated images will be saved to: '" + image_path + "'")

    # where models and loss values will be saved
    model_path = model_save_dir + '/' + subdir + '/' + model_prefix
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("Trained models will be saved to: '" + model_path + "'")
    model_file_name_tpl = model_path + '/' + 'model_upscaler_' + model_prefix + '_%06db.h5'
    model_file_name_best = model_path + '/' + 'model_upscaler_' + model_prefix + '_best.h5'
    
    loss_path = loss_save_dir + '/' + subdir + '/' + model_prefix
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    print("Loss values and training parameters will be saved to: '" + loss_path + "'")
    loss_file_name = loss_path + '/' + 'losses_upscaler_' + model_prefix + '.txt'
    best_loss_file_name = loss_path + '/' + 'losses_upscaler_' + model_prefix + '_best.txt'
    param_file_path = loss_path + '/' + 'parameters.json'
    progress_file_path = loss_path + '/' + 'progress.json'

    ###########################################################
    ## Loading the data
    ###########################################################
    
    images_fullhd = load_images_from_dir(
        input_dir,
        '.jpg',
        limit = number_of_images,
        prog_func = tqdm
    )
    images_fullhd = downscale_images(
        images_fullhd,
        prog_func = tqdm,
        downscale_ratio = downscale_factor
    )
    images_fullhd = images_fullhd.rename(columns={'image_hr': 'fullhd', 'downscaled' : 'scaled'})
    
    images_1gen = load_images_from_dir(
        input_dir_1gen,
        '.jpg',
        limit = number_of_images,
        prog_func = tqdm
    )
    images_1gen = images_1gen.rename(columns={'image_hr': 'gen1'}).drop(columns='image_size')
    
    images_2gen = load_images_from_dir(
        input_dir_2gen,
        '.jpg',
        limit = number_of_images,
        prog_func = tqdm
    )
    images_2gen = images_2gen.rename(columns={'image_hr': 'gen2'}).drop(columns='image_size')
    
    
    images_all = (images_fullhd
                      .join(images_1gen.set_index('filename'), on = 'filename', how = 'inner')
                  .join(images_2gen.set_index('filename'), on = 'filename', how = 'inner')
                 )
    images_train, images_test = split_images_train_test(images_all, train_test_ratio, seed = values.split_seed)
    
    ###########################################################
    ## Saving train parameters file
    ###########################################################
    
    parameters = vars(values)
    parameters['model_prefix'] = model_prefix
    parameters['train_set'] = images_train.filename
    parameters['test_set'] = images_test.filename
    
    with open(param_file_path, 'w+') as param_file:
        json.dump(parameters, param_file, indent = 4, cls = PandasEncoder)

    
    ###########################################################
    ## Setting up the model for training
    ###########################################################
    
    # create the model instance
    if values.model == 'orig':
        upscaler = make_upscaler_orig(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor)
    elif values.model == 'skip-con':
        upscaler = make_upscaler_skip_con(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor)
    elif values.model == 'resnet-att':
        upscaler = make_upscaler_attention(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor)
    elif values.model == 'unetish':
        upscaler = make_upscaler_unetish(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor, dropout_rate=values.dropout_rate)
    elif values.model == 'unetish-add':
        upscaler = make_upscaler_unetish_add(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor, dropout_rate=values.dropout_rate)
    
    plot_model(upscaler, to_file=loss_path+'/model_plot.png', show_shapes=True, show_layer_names=True)
    #upscaler.summary(line_length=200)
    #sys.exit(0)
    
    # create the loss
    if values.loss == 'vgg-only':
        loss = VGG_LOSS(output_image_shape).loss
    elif values.loss == 'vgg-mae':
        loss = VGG_MAE_LOSS(output_image_shape, values.non_vgg_loss_weight).loss
    elif values.loss == 'vgg-mse':
        loss = VGG_MSE_LOSS(output_image_shape, values.non_vgg_loss_weight).loss
    
    # setting up the model for training
    upscaler_training_model = compile_training_model(upscaler, loss)
    
    ###########################################################
    ## Model training
    ###########################################################
    
    agg_loss = 0.0
    loss_update_rate = 0.01
    best_loss = np.inf
    
    # progress log initialisation
    progress = {
        'best_model': None,
        'saved_models': None
    }
    
    saved_models = pd.DataFrame({
        'batch': [],
        'loss_1gen': [],
        'loss_2gen': [],
        'loss_scaled': [],
        'agg_loss': [],
        'path': []
    })
    
    # loss logs initialisations
    with open(loss_file_name, 'w+') as loss_file:
        loss_file.write('batch\tloss_1gen\tloss_2gen\tloss_scaled\tagg_loss\n')

    with open(best_loss_file_name, 'w+') as loss_file:
        loss_file.write('batch\tloss_1gen\tloss_2genn\tloss_scaled\tagg_loss\n')
    
    
    # saving lowres and highres examples
    save_img_orig(images_train.fullhd[0:10],   image_path, model_prefix + '_train', quality = 95)
    save_img_resize(images_train.gen1[0:10],   image_path, model_prefix + '_train', sufix = '_1gen', target_size = target_shape, quality = 95)
    save_img_resize(images_train.gen2[0:10],   image_path, model_prefix + '_train', sufix = '_2gen', target_size = target_shape, quality = 95)
    save_img_resize(images_train.scaled[0:10], image_path, model_prefix + '_train', sufix = '_scal', target_size = target_shape, quality = 95)
    
    save_img_orig(images_test.fullhd[0:10],   image_path, model_prefix + '_test', quality = 95)
    save_img_resize(images_test.gen1[0:10],   image_path, model_prefix + '_test', sufix = '_1gen', target_size = target_shape, quality = 95)
    save_img_resize(images_test.gen2[0:10],   image_path, model_prefix + '_test', sufix = '_2gen', target_size = target_shape, quality = 95)
    save_img_resize(images_test.scaled[0:10], image_path, model_prefix + '_test', sufix = '_scal', target_size = target_shape, quality = 95)

    
    # actual training loop
    for b in tqdm(range(values.number_of_batches), desc = 'Batch'):

        batch_df = select_random_rows(images_train, n=values.batch_size)
        
        image_batch_hr = convert_image_series_to_array(batch_df.fullhd)
        image_batch_1gen = convert_image_series_to_array(batch_df.gen1)
        image_batch_2gen = convert_image_series_to_array(batch_df.gen2)
        image_batch_scal = convert_image_series_to_array(batch_df.scaled)

        loss_1gen = upscaler_training_model.train_on_batch(image_batch_1gen, image_batch_hr)
        loss_2gen = upscaler_training_model.train_on_batch(image_batch_2gen, image_batch_hr)
        loss_scal = upscaler_training_model.train_on_batch(image_batch_scal, image_batch_hr)
        loss = (loss_1gen + loss_2gen + loss_scal) / 3
        
        agg_loss = (1 - loss_update_rate) * agg_loss + loss_update_rate * loss

        with open(loss_file_name, 'a') as loss_file:
            loss_file.write('%d\t%f\t%f\t%f\t%f\n' %(b, loss_1gen, loss_2gen, loss_scal, agg_loss) )
        
        # update progress log with best model
        if b > values.model_save_freq and agg_loss < best_loss:
            best_loss = agg_loss

            upscaler.save(model_file_name_best)

            with open(best_loss_file_name, 'a') as loss_file:
                loss_file.write('%d\t%f\t%f\t%f\t%f\n' %(b, loss_1gen, loss_2gen, loss_scal, agg_loss) )
            
            best_model_progress = {
                'batch': b,
                'loss_1gen': float(loss_1gen),
                'loss_2gen': float(loss_2gen),
                'loss_scaled': float(loss_scal),
                'agg_loss': float(agg_loss),
                'saved': model_file_name_best
            }
            progress['best_model'] = best_model_progress
            
            with open(progress_file_path, 'w+') as progress_file:
                json.dump(progress, progress_file, indent = 4, cls = PandasEncoder)

        # saving a next state of the model
        if(b % values.model_save_freq == 0):
            model_file_name = model_file_name_tpl % b
            upscaler.save(model_file_name)
            
            # update progress log with next model saved
            saved_models = saved_models.append({
                'batch': b,
                'loss_1gen': float(loss_1gen),
                'loss_2gen': float(loss_2gen),
                'loss_scaled': float(loss_scal),
                'agg_loss': float(agg_loss),
                'path': model_file_name
            }, ignore_index=True)
            
            progress['saved_models'] = saved_models
            
            with open(progress_file_path, 'w+') as progress_file:
                json.dump(progress, progress_file, indent = 4, cls = PandasEncoder)
            
            save_img_predict(images_train.gen1[0:10],   upscaler, image_path, model_prefix + '_train', b, sufix = '_1gen', quality = 95)
            save_img_predict(images_train.gen2[0:10],   upscaler, image_path, model_prefix + '_train', b, sufix = '_2gen', quality = 95)
            save_img_predict(images_train.scaled[0:10], upscaler, image_path, model_prefix + '_train', b, sufix = '_scal', quality = 95)
            save_img_predict(images_test.gen1[0:10],   upscaler, image_path, model_prefix + '_test', b, sufix = '_1gen', quality = 95)
            save_img_predict(images_test.gen2[0:10],   upscaler, image_path, model_prefix + '_test', b, sufix = '_2gen', quality = 95)
            save_img_predict(images_test.scaled[0:10], upscaler, image_path, model_prefix + '_test', b, sufix = '_scal', quality = 95)
            
