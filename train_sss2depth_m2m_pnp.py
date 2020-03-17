"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from options.val_options import ValOptions
from data import create_dataset
from models import create_model
from util.visualizer_sssd import Visualizer, save_images, cal_scores
from copy import deepcopy
import os
from util import html


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # opt.sample_nums = 100
    # opt.dataset_mode = "alignedm2md"
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    # hard-code some parameters for val
    opt_val = TrainOptions().parse()
#     print("opt type is: ", type(opt), "opt_val type is: ", type(opt_val))
    opt_val.num_threads = 1   # test code only supports num_threads = 1
    opt_val.batch_size = 2    # test code only supports batch_size = 1
    opt_val.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt_val.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt_val.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt_val.phase = 'val'
    opt_val.isTrain = False        # get validating options
    opt_val.results_dir= './results/'
    opt_val.aspect_ratio = 1.0
    # opt_val.sample_nums = 100
    # opt_val.dataset_mode = "alignedm2md"
    
    dataset_val = create_dataset(opt_val)
#     opt_val.print_options(opt_val)
    print(opt_val)
    web_dir_val = os.path.join(opt_val.results_dir, opt_val.name, '{}_{}'.format(opt_val.phase, opt_val.epoch))  # define the website directory
    webpage_val = html.HTML(web_dir_val, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt_val.name, opt_val.phase, opt_val.epoch))
    
    min_mae = 100.
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model and validate every <save_epoch_freq> epochs
            abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for i_val, data_val in enumerate(dataset_val):
                model.set_input(data_val)
                model.test()           # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()     # get image paths
#                 save_images(webpage_val, visuals, img_path, aspect_ratio=opt_val.aspect_ratio, width=opt.display_winsize)
                (abs_rel_, sq_rel_, rmse_, rmse_log10_, mae_, mae_log10_, a1_, a2_, a3_) = cal_scores(visuals)
    
                abs_rel += abs_rel_
                sq_rel += sq_rel
                rmse += rmse_
                rmse_log10 += rmse_log10_
                mae += mae_
                mae_log10 += mae_log10_
                a1 += a1_
                a2 += a2_
                a3 += a3_
            webpage_val.save()  # save the HTML
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
#             (abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3)=(abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3)/(i_val+1)
            abs_rel /= (i_val+1)
            sq_rel /= (i_val+1)
            rmse /= (i_val+1)
            rmse_log10 /= (i_val+1)
            mae /= (i_val+1)
            mae_log10 /= (i_val+1)
            a1 /= (i_val+1)
            a2 /= (i_val+1)
            a3 /= (i_val+1)
            losses = {"abs_rel": abs_rel, "sq_rel":sq_rel, "rmse": rmse, "rmse_log10":rmse_log10, "mae":mae, "mae_log10":mae_log10, "a1":a1, "a2":a2, "a3":a3}
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
#             print("mae: ", mae)
            if mae<min_mae:
                min_mae = mae
                model.save_networks('best')
                print('saving the best model at the end of epoch %d, iters %d, mae %3.3f' % (epoch, total_iters, mae))
                
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
