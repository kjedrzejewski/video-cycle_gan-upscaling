from upscaler.data import load_images_from_dir_and_downscale, split_images_train_test
from upscaler.data import select_random_rows, convert_imagesdf_to_arrays, convert_array_to_image
from upscaler.data import save_images_orig, save_images_predicted
from upscaler.data import save_images_orig_png, save_images_predicted_png
from upscaler.model import make_upscaler_skip_con, make_upscaler_orig
from upscaler.model import VGG_LOSS, VGG_MSE_LOSS, VGG_MAE_LOSS
from upscaler.model import compile_training_model
from upscaler.json import DataFrameEncoder

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
    
    parser.add_argument('-s', '--subdir', action='store', dest='subdir', default='ukiyo', help='Subdir to put generated images, trained models, etc to')
    
    parser.add_argument('-p', '--output_prefix', action='store', dest='output_prefix', default='auto', help='Prefix to put in names of generated files (and sometimes also a subdir). Default value \'auto\' means generate it automatically')
    
    parser.add_argument('-ic', '--image_count', action='store', dest='image_count', default='1000', help='Number of images to be used (and split into training and test sets)', type=int)
    
    parser.add_argument('-tr', '--train_test_ratio', action='store', dest='train_test_ratio', default='0.95', help='Ratio of splitting images into the test and train sets', type=float)
    
    parser.add_argument('-m', '--model', action='store', dest='model', default='orig', choices=['orig','skip-con'], help='Model to be used')
    
    parser.add_argument('-l', '--loss', action='store', dest='loss', default='vgg-only', choices=['vgg-only','vgg-mae','vgg-mse'], help='Loss function to be used for the training')
    
    parser.add_argument('-lw', '--non_vgg_loss_weight', action='store', dest='non_vgg_loss_weight', default='1.0', help='Weight of the loss other than VGG (if there is any)', type=float)
    
    parser.add_argument('-msf', '--model_save_freq', action='store', dest='model_save_freq', default='500', help='How frequently a model should be saved? (number of batches)', type=int)
    
    parser.add_argument('-bs', '--batch_size', action='store', dest='batch_size', default='1', help='Number of examples to be put in the batch', type=int)
    
    parser.add_argument('-nb', '--number_of_batches', action='store', dest='number_of_batches', default='40001', help='Number batches to be run', type=int)
    
    parser.add_argument('-d', '--downscale_factor', action='store', dest='downscale_factor', default='4', help='Downscale factor', type=int)
    
    values = parser.parse_args()
    
    
    ###########################################################
    ## Parameters of input and output images
    ###########################################################
    
    downscale_factor = values.downscale_factor
    upscale_times = int(math.log(downscale_factor,2))
    output_image_shape = (1920,1080,3)
    input_image_shape = (
        output_image_shape[0] // downscale_factor,
        output_image_shape[1] // downscale_factor,
        output_image_shape[2]
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
    model_save_dir = script_dirname + '/' + 'trained_model'
    images_dir = script_dirname + '/' + 'example_images'
    subdir = values.subdir
    #model_prefix = "orig_vgg-mse"
    model_prefix = values.output_prefix
    if model_prefix == 'auto':
        model_prefix = values.model + "_" + values.loss + ("_x%d" % downscale_factor)
        print("Prefix generated automatically: '" + model_prefix + "'")
    
    number_of_images = values.image_count
    train_test_ratio = values.train_test_ratio

    # where the examples of images will be saved
    image_path = images_dir + '/' + subdir + '/' + model_prefix
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    print("Generated images will be saved to: '" + image_path + "'")

    # where models and loss values will be saved
    model_loss_path = model_save_dir + '/' + subdir + '/' + model_prefix
    if not os.path.exists(model_loss_path):
        os.makedirs(model_loss_path)
    print("Loss values and trained models will be saved to: '" + model_loss_path + "'")
    loss_file_name = model_loss_path + '/' + 'losses_upscaler_' + model_prefix + '.txt'
    best_loss_file_name = model_loss_path + '/' + 'losses_upscaler_' + model_prefix + '_best.txt'
    model_file_name_tpl = model_loss_path + '/' + 'model_upscaler_' + model_prefix + '_%06db.h5'
    model_file_name_best = model_loss_path + '/' + 'model_upscaler_' + model_prefix + '_best.h5'
    param_file_path = model_loss_path + '/' + 'parameters.json'
    progress_file_path = model_loss_path + '/' + 'progress.json'

    
    ###########################################################
    ## Saving train parameters file
    ###########################################################
    
    parameters = vars(values)
    parameters['model_prefix'] = model_prefix
    
    with open(param_file_path, 'w+') as param_file:
        json.dump(parameters, param_file, indent = 4)


    ###########################################################
    ## Loading the data
    ###########################################################
        
    images_all = load_images_from_dir_and_downscale(
        input_dir,
        '.jpg',
        limit = number_of_images,
        downscale_factor = downscale_factor,
        prog_func = tqdm
    )
    
    images_train, images_test = split_images_train_test(images_all, train_test_ratio)
    
    ###########################################################
    ## Setting up the model for training
    ###########################################################
    
    # create the model instance
    if values.model == 'orig':
        upscaler = make_upscaler_orig(input_image_shape, upscale_times)
    elif values.model == 'skip-con':
        upscaler = make_upscaler_skip_con(input_image_shape, upscale_times)
    
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
    prev_loss = -1
    loss_update_rate = 0.01
    best_loss = np.inf
    loss_decreases = False
    
    # progress log initialisation
    progress = {
        'best_model': None,
        'saved_models': None
    }
    
    saved_models = pd.DataFrame({
        'batch': [],
        'loss': [],
        'agg_loss': [],
        'path': []
    })
    
    # loss logs initialisations
    with open(loss_file_name, 'w+') as loss_file:
        loss_file.write('batch\tloss\tagg_loss\n')

    with open(best_loss_file_name, 'w+') as loss_file:
        loss_file.write('batch\tloss\tagg_loss\n')
    
    
    # saving lowres and highres examples
    save_images_orig(images_train, 0, 10, image_path, model_prefix + '_train', quality = 95)
    save_images_orig(images_test, 0, 10, image_path, model_prefix + '_test', quality = 95)

    
    # actual training loop
    for b in tqdm(range(values.number_of_batches), desc = 'Batch'):

        batch_df = select_random_rows(images_train, n=values.batch_size)

        image_batch_hr, image_batch_lr = convert_imagesdf_to_arrays(batch_df)
        loss = upscaler_training_model.train_on_batch(image_batch_lr, image_batch_hr)

        # check if aggregated loss started to decrease, only then we start looking for the best model
        prev_loss = agg_loss
        agg_loss = (1 - loss_update_rate) * agg_loss + loss_update_rate * loss
        loss_decreases = loss_decreases or (prev_loss > agg_loss)

        with open(loss_file_name, 'a') as loss_file:
            loss_file.write('%d\t%f\t%f\n' %(b, loss, agg_loss) )
        
        # update progress log with best model
        if loss_decreases and agg_loss < best_loss:
            best_loss = agg_loss

            upscaler.save(model_file_name_best)

            with open(best_loss_file_name, 'a') as loss_file:
                loss_file.write('%d\t%f\t%f\n' %(b, loss, agg_loss) )
            
            best_model_progress = {
                'batch': b,
                'loss': float(loss),
                'agg_loss': float(agg_loss),
                'saved': model_file_name_best
            }
            progress['best_model'] = best_model_progress
            
            with open(progress_file_path, 'w+') as progress_file:
                json.dump(progress, progress_file, indent = 4, cls = DataFrameEncoder)

        # saving a next state of the model
        if(b % values.model_save_freq == 0):
            model_file_name = model_file_name_tpl % b
            upscaler.save(model_file_name)
            
            # update progress log with next model saved
            saved_models = saved_models.append({
                'batch': b,
                'loss': float(loss),
                'agg_loss': float(agg_loss),
                'path': model_file_name
            }, ignore_index=True)
            
            progress['saved_models'] = saved_models
            
            with open(progress_file_path, 'w+') as progress_file:
                json.dump(progress, progress_file, indent = 4, cls = DataFrameEncoder)
            
            save_images_predicted(images_train, upscaler, 0, 10, image_path, model_prefix + '_train', b, quality = 95)
            save_images_predicted(images_test, upscaler, 0, 10, image_path, model_prefix + '_test', b, quality = 95)
            
