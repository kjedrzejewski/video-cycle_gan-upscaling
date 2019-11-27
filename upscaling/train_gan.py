from upscaler.data import load_images_from_dir, split_images_train_test, downscale_images, crop_images_cgc
from upscaler.data import select_random_rows, convert_image_series_to_array, convert_array_to_image
from upscaler.data import save_img_orig, save_img_resize, save_img_predict
from upscaler.model import make_upscaler_skip_con, make_upscaler_orig, make_upscaler_unetish, make_upscaler_unetish_add, make_upscaler_attention
from upscaler.model import make_discriminator_simple_512
from upscaler.model import VGG_LOSS, VGG_MSE_LOSS, VGG_MAE_LOSS, wasserstein_loss
from upscaler.model import make_and_compile_gan
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
    
    parser.add_argument('-gm', '--generator_model', action='store', dest='generator_model', default='resnet-att', choices=['orig','skip-con','resnet-att','unetish','unetish-add'], help='Generator model to be used')
    
    parser.add_argument('-dm', '--discriminator_model', action='store', dest='discriminator_model', default='simple-512', choices=['simple-512'], help='Discriminator model to be used')
    
    parser.add_argument('-da', '--discriminator_activation', dest='discriminator_activation', action='store_true', help='Use activtation for the discriminator output')
    parser.add_argument('-nda', '--no-discriminator_activation', dest='discriminator_activation', action='store_false', help='Don\'t use activtation for the discriminator output')
    parser.set_defaults(discriminator_activation=False)
    
    parser.add_argument('-cl', '--content_loss', action='store', dest='content_loss', default='vgg-only', choices=['vgg-only','vgg-mae','vgg-mse'], help='Content loss function to be used for the training')
    
    parser.add_argument('-dl', '--discriminator_loss', action='store', dest='discriminator_loss', default='wasserstein', choices=['wasserstein'], help='Discriminator loss function to be used for the training')
    
    parser.add_argument('-dlw', '--discriminator_loss_weight', action='store', dest='discriminator_loss_weight', default='1e-10', type=float, help='Weight of the discriminator loss function to be used for the training GAN. Generator loss weight is considered to be equal 1')
    
    parser.add_argument('-lw', '--non_vgg_loss_weight', action='store', dest='non_vgg_loss_weight', default='0.001', help='Weight of the loss other than VGG (if there is any)', type=float)
    
    parser.add_argument('-msf', '--model_save_freq', action='store', dest='model_save_freq', default='500', help='How frequently a model should be saved? (number of batches)', type=int)
    
    parser.add_argument('-bs', '--batch_size', action='store', dest='batch_size', default='2', help='Number of examples to be put in the batch', type=int)
    
    parser.add_argument('-oh', '--output_height', action='store', dest='output_height', default='512', help='Height of the output during training', type=int)
    
    parser.add_argument('-ow', '--output_width', action='store', dest='output_width', default='512', help='Width of the output during training', type=int)
    
    parser.add_argument('-nb', '--number_of_batches', action='store', dest='number_of_batches', default='400001', help='Number batches to be run', type=int)
    
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
    output_image_shape = (values.output_height, values.output_width,3)
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
        model_prefix = "gan_" + values.generator_model + "_" + values.content_loss + "_" + values.discriminator_model + "_" + values.discriminator_loss + ("_x%d" % downscale_factor)
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
    upscaler_model_file_name_tpl = model_path + '/' + 'model_upscaler_' + model_prefix + '_%06db.h5'
    discriminator_model_file_name_tpl = model_path + '/' + 'model_discriminator_' + model_prefix + '_%06db.h5'
    upscaler_model_file_name_best = model_path + '/' + 'model_upscaler_' + model_prefix + '_best.h5'
    discriminator_model_file_name_best = model_path + '/' + 'model_discriminator_' + model_prefix + '_best.h5'
    
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
        limit = number_of_images * 10000,
        prog_func = tqdm
    )
    images_1gen = images_1gen.rename(columns={'image_hr': 'gen1'}).drop(columns='image_size')
    
    images_2gen = load_images_from_dir(
        input_dir_2gen,
        '.jpg',
        limit = number_of_images * 10000,
        prog_func = tqdm
    )
    images_2gen = images_2gen.rename(columns={'image_hr': 'gen2'}).drop(columns='image_size')
    
    
    images_all = (images_fullhd
                    .join(images_1gen.set_index('filename'), on = 'filename', how = 'inner')
                    .join(images_2gen.set_index('filename'), on = 'filename', how = 'inner')
                 )
    
    images_all = crop_images_cgc(images_all, target_shape = target_shape, seed = values.split_seed, downscale_ratio = downscale_factor)
    
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
    
    # create the generator instance
    if values.generator_model == 'orig':
        upscaler = make_upscaler_orig(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor)
    elif values.generator_model == 'skip-con':
        upscaler = make_upscaler_skip_con(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor)
    elif values.generator_model == 'resnet-att':
        upscaler = make_upscaler_attention(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor)
    elif values.generator_model == 'unetish':
        upscaler = make_upscaler_unetish(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor, dropout_rate=values.dropout_rate)
    elif values.generator_model == 'unetish-add':
        upscaler = make_upscaler_unetish_add(output_image_shape, kernel_size = kernel_size, upscale_factor = downscale_factor, dropout_rate=values.dropout_rate)
    
    plot_model(upscaler, to_file=loss_path+'/model_plot.png', show_shapes=True, show_layer_names=True)
    #upscaler.summary(line_length=200)
    #sys.exit(0)
    
    # create the discriminator instance
    if values.discriminator_model == 'simple-512':
        discriminator = make_discriminator_simple_512(output_image_shape, final_sigmoid = values.discriminator_activation)
    
    # create the content loss
    if values.content_loss == 'vgg-only':
        content_loss = VGG_LOSS(output_image_shape).loss
    elif values.content_loss == 'vgg-mae':
        content_loss = VGG_MAE_LOSS(output_image_shape, values.non_vgg_loss_weight).loss
    elif values.content_loss == 'vgg-mse':
        content_loss = VGG_MSE_LOSS(output_image_shape, values.non_vgg_loss_weight).loss
    
    # create the discriminator loss
    if values.discriminator_loss == 'wasserstein':
        discriminator_loss = wasserstein_loss
    
    # setting up the model for training
    gen_train, disc_train, gan_train = make_and_compile_gan(
        generator = upscaler, discriminator = discriminator,
        input_shape = input_image_shape, output_shape = output_image_shape,
        content_loss = content_loss, content_loss_weight = 1,
        discriminator_loss = discriminator_loss, discriminator_loss_weight = values.discriminator_loss_weight
    )
    
    ###########################################################
    ## Model training
    ###########################################################
    
    agg_loss_disc = 0.0
    agg_loss_gan_gen = 0.0
    agg_loss_gan_disc = 0.0
    loss_update_rate = 0.01
    best_loss = np.inf
    
    # progress log initialisation
    progress = {
        'best_model': None,
        'saved_models': None
    }
    
    saved_models = pd.DataFrame({
        'batch': [],
        'loss_disc': [],
        'agg_loss_disc': [],
        'loss_gan_gen': [],
        'agg_loss_gan_gen': [],
        'loss_gan_disc': [],
        'agg_loss_gan_disc': [],
        'path_upscaler': [],
        'path_discriminator': []
    })
    
    # loss logs initialisations
    with open(loss_file_name, 'w+') as loss_file:
        loss_file.write('batch\tloss_disc\tagg_loss_disc\tloss_gan_gen\tagg_loss_gan_gen\tloss_gan_disc\tagg_loss_gan_disc\n')

    with open(best_loss_file_name, 'w+') as loss_file:
        loss_file.write('batch\tloss_disc\tagg_loss_disc\tloss_gan_gen\tagg_loss_gan_gen\tloss_gan_disc\tagg_loss_gan_disc\n')
    
    
    # saving lowres and highres examples
    save_img_orig(images_train.cropped_hd[0:10],   image_path, model_prefix + '_train', quality = 95)
    save_img_resize(images_train.cropped_gen1[0:10],   image_path, model_prefix + '_train', sufix = '_1gen', target_size = target_shape, quality = 95)
    save_img_resize(images_train.cropped_gen2[0:10],   image_path, model_prefix + '_train', sufix = '_2gen', target_size = target_shape, quality = 95)
    save_img_resize(images_train.cropped_scaled[0:10], image_path, model_prefix + '_train', sufix = '_scal', target_size = target_shape, quality = 95)
    
    save_img_orig(images_test.cropped_hd[0:10],   image_path, model_prefix + '_test', quality = 95)
    save_img_resize(images_test.cropped_gen1[0:10],   image_path, model_prefix + '_test', sufix = '_1gen', target_size = target_shape, quality = 95)
    save_img_resize(images_test.cropped_gen2[0:10],   image_path, model_prefix + '_test', sufix = '_2gen', target_size = target_shape, quality = 95)
    save_img_resize(images_test.cropped_scaled[0:10], image_path, model_prefix + '_test', sufix = '_scal', target_size = target_shape, quality = 95)
    
    
    # actual training loop
    for b in tqdm(range(values.number_of_batches), desc = 'Batch'):

        batch_df = select_random_rows(images_train, n=values.batch_size)
        
        batch_hr = pd.concat([batch_df.cropped_hd, batch_df.cropped_hd, batch_df.cropped_hd], ignore_index=True)
        batch_lr = pd.concat([batch_df.cropped_gen1, batch_df.cropped_gen2, batch_df.cropped_scaled], ignore_index=True)
        
        image_batch_hr  = convert_image_series_to_array(batch_hr)
        image_batch_lr  = convert_image_series_to_array(batch_lr)
        image_batch_gen = gen_train.predict(image_batch_lr)
        # batch provided on the input of the discriminator training
        image_batch_disc = np.concatenate((image_batch_hr, image_batch_gen), axis=0)
        
        # expected discriminator outputs: 1 - original image, -1 - upscaled
        disc_real_y = np.ones(3 * values.batch_size)
        disc_fake_y = -disc_real_y
        # expected output provided for the discriminator training
        image_batch_y = np.concatenate((disc_real_y, disc_fake_y), axis=0)
        
        # training
        loss_disc = disc_train.train_on_batch(image_batch_disc, image_batch_y)
        loss_gan, loss_gan_gen, loss_gan_disc = gan_train.train_on_batch(image_batch_lr, [image_batch_hr,disc_real_y])
        
        agg_loss_disc = (1 - loss_update_rate) * agg_loss_disc + loss_update_rate * loss_disc
        agg_loss_gan_gen = (1 - loss_update_rate) * agg_loss_gan_gen + loss_update_rate * loss_gan_gen
        agg_loss_gan_disc = (1 - loss_update_rate) * agg_loss_gan_disc + loss_update_rate * loss_gan_disc

        with open(loss_file_name, 'a') as loss_file:
            loss_file.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' %(b, loss_disc, agg_loss_disc, loss_gan_gen, agg_loss_gan_gen, loss_gan_disc, agg_loss_gan_disc))
        
        # update progress log with best model
        if b > values.model_save_freq and agg_loss_gan_gen < best_loss:
            best_loss = agg_loss_gan_gen

            upscaler.save(upscaler_model_file_name_best)
            discriminator.save(discriminator_model_file_name_best)

            with open(best_loss_file_name, 'a') as loss_file:
                loss_file.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' %(b, loss_disc, agg_loss_disc, loss_gan_gen, agg_loss_gan_gen, loss_gan_disc, agg_loss_gan_disc))
            
            best_model_progress = {
                'batch': b,
                'loss_disc': float(loss_disc),
                'agg_loss_disc': float(agg_loss_disc),
                'loss_gan_gen': float(loss_gan_gen),
                'agg_loss_gan_gen': float(agg_loss_gan_gen),
                'loss_gan_disc': float(loss_gan_disc),
                'agg_loss_gan_disc': float(agg_loss_gan_disc),
                'saved_upscaler': upscaler_model_file_name_best,
                'saved_discriminator': discriminator_model_file_name_best
            }
            progress['best_model'] = best_model_progress
            
            with open(progress_file_path, 'w+') as progress_file:
                json.dump(progress, progress_file, indent = 4, cls = PandasEncoder)

        # saving a next state of the model
        if(b % values.model_save_freq == 0):
            upscaler_model_file_name = upscaler_model_file_name_tpl % b
            upscaler.save(upscaler_model_file_name)
            
            discriminator_model_file_name = discriminator_model_file_name_tpl % b
            discriminator.save(discriminator_model_file_name)
            
            # update progress log with next model saved
            saved_models = saved_models.append({
                'batch': b,
                'loss_disc': float(loss_disc),
                'agg_loss_disc': float(agg_loss_disc),
                'loss_gan_gen': float(loss_gan_gen),
                'agg_loss_gan_gen': float(agg_loss_gan_gen),
                'loss_gan_disc': float(loss_gan_disc),
                'agg_loss_gan_disc': float(agg_loss_gan_disc),
                'path_upscaler': upscaler_model_file_name,
                'path_discriminator': discriminator_model_file_name
            }, ignore_index=True)
            
            progress['saved_models'] = saved_models
            
            with open(progress_file_path, 'w+') as progress_file:
                json.dump(progress, progress_file, indent = 4, cls = PandasEncoder)
            
            save_img_predict(images_train.cropped_gen1[0:10],   upscaler, image_path, model_prefix + '_train', b, sufix = '_1gen', quality = 95)
            save_img_predict(images_train.cropped_gen2[0:10],   upscaler, image_path, model_prefix + '_train', b, sufix = '_2gen', quality = 95)
            save_img_predict(images_train.cropped_scaled[0:10], upscaler, image_path, model_prefix + '_train', b, sufix = '_scal', quality = 95)
            save_img_predict(images_test.cropped_gen1[0:10],   upscaler, image_path, model_prefix + '_test', b, sufix = '_1gen', quality = 95)
            save_img_predict(images_test.cropped_gen2[0:10],   upscaler, image_path, model_prefix + '_test', b, sufix = '_2gen', quality = 95)
            save_img_predict(images_test.cropped_scaled[0:10], upscaler, image_path, model_prefix + '_test', b, sufix = '_scal', quality = 95)
            
