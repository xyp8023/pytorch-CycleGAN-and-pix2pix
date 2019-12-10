"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer_m2o import save_images, cal_scores
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#     opt.load_best = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    
    # by default it is True
#     if opt.load_best:
#         web_dir = '{:s}_iter{:s}'.format(web_dir, "best")
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
#         if i % 5 == 0:  # save images to an HTML file
#         print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
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
    print('saving the model at the end of epoch: ' , opt.epoch)
    print('i+1 is: ', i)
#             (abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3)=(abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3)/(i_val+1)
    webpage.save()  # save the HTML
    abs_rel /= (i+1)
    sq_rel /= (i+1)
    rmse /= (i+1)
    rmse_log10 /= (i+1)
    mae /= (i+1)
    mae_log10 /= (i+1)
    a1 /= (i+1)
    a2 /= (i+1)
    a3 /= (i+1)
    losses = {"abs_rel": abs_rel, "sq_rel":sq_rel, "rmse": rmse, "rmse_log10":rmse_log10, "mae":mae, "mae_log10":mae_log10, "a1":a1, "a2":a2, "a3":a3}
    message = '(test phase) '
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log_test.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message
        
