import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
EPS = 1e-12
depth_max = -9.64964580535888671875 + EPS
depth_min = -21.6056976318359375 - EPS

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

    
def compute_errors(height_ori, height_pre_masked, mask):
    # mask now: valid data is true
    d, d_hat = height_ori[mask], height_pre_masked[mask]  # [-1, 1]
#     d, d_hat = reverse_norm(d), reverse_norm(d_hat)  # [-21, -9]

    thresh = np.maximum((d / d_hat), (d_hat / d))
    a1 = (thresh < 1.05).mean()
    a2 = (thresh < 1.05 ** 2).mean()
    a3 = (thresh < 1.05 ** 3).mean()

    rmse = (d - d_hat) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log10 = (np.log10(-d) - np.log10(-d_hat)) ** 2
    rmse_log10 = np.sqrt(rmse_log10.mean())

    mae = np.mean(np.abs((d - d_hat)))
    mae_log10 = np.mean(np.abs(np.log10(-d) - np.log10(-d_hat)))

    abs_rel = np.mean(np.abs(d - d_hat) / -d)
    sq_rel = np.mean(((d - d_hat) ** 2) / -d)

    return abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3

def cal_scores(visuals):
    """ Calculate scores 
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
    """
#     image_dir = webpage.get_image_dir()
#     short_path = ntpath.basename(image_path[0])
#     name = os.path.splitext(short_path)[0]
#     txt_name = '%s_%s.txt' % (name, label)

#     webpage.add_header(name)
#     ims, txts, links = [], [], []
    for label, im_data in visuals.items():
#         print('fake_B:')    
        
        if "fake_B" == label:
            im_fake_B = util.tensor2im(im_data, imtype=np.float64, keep_grayscale=True)
        if "real_B" == label:
            im_real_B = util.tensor2im(im_data, imtype=np.float64, keep_grayscale=True)
#     util.print_multi_numpy(im_fake_B, val=True, shp=True)# max 255.0 min 0.0
    
#     print('im_fake_B:')    
#     util.print_numpy(im_fake_B, val=True, shp=True)# max 255.0 min 0.0
#     print('im_real_B:')    
#     util.print_numpy(im_real_B, val=True, shp=True)
    mask = (im_real_B<255.)
    depth_ori = im_real_B/255.*(depth_max-depth_min)+depth_min
    depth_pre = im_fake_B/255.*(depth_max-depth_min)+depth_min
    
    abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3 = compute_errors(
        depth_ori, depth_pre, mask)  # epsilon = 1.05
#     print("abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3:\n", abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3)
#     print("mae: ", mae)
    return abs_rel, sq_rel, rmse, rmse_log10, mae, mae_log10, a1, a2, a3

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        if 'B' in label:
            im = util.tensor2im(im_data, color_map=True)
#             util.save_image(im, save_path, aspect_ratio=aspect_ratio, color_map=True)
            
        else:
            # im = util.tensor2im(im_data)
            im = util.tensor2im(im_data[:,0,:,:].unsqueeze(1))
            im_numpy_sparse = util.tensor2im(im_data[:,1,:,:].unsqueeze(1), color_map=True)
        
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        if 'A' in label:
            util.save_image(im_numpy_sparse, os.path.join(image_dir, 'sparse_'+image_name), aspect_ratio=aspect_ratio)
        ims.append(image_name)
        if 'A' in label:
            ims.append('sparse_'+image_name)
        txts.append(label)
        if 'A' in label:
            txts.append('sparse_'+label)
        links.append(image_name)
        if 'A' in label:
            links.append('sparse_'+image_name)

    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
#                     print('label : ', label)
                    if 'B' in label:
                        image_numpy = util.tensor2im(image, color_map=True)    
                    else:    
                        # print("label, image shape: ", label, image[:,0,:,:].shape)
                        image_numpy = util.tensor2im(image[:,0,:,:].unsqueeze(1))
                        image_numpy_sparse = util.tensor2im(image[:,1,:,:].unsqueeze(1), color_map=True)
                        # print("label, image numpy shape: ", label, image_numpy.shape) (256, 256, 3)

                        # print("label, image numpy sparse shape: ", label, image_numpy_sparse.shape)
                        # image_numpy = np.concatenate((image_numpy, image_numpy_sparse), axis=0)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    if 'A' in label:
                        images.append(image_numpy_sparse.transpose([2, 0, 1]))

                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
#                         print('label :', label)
                        
                        if 'B' in label:
                            image_numpy = util.tensor2im(image, color_map=True)    
                        else:    
                            image_numpy = util.tensor2im(image)
#                         image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
#                 print('label :', label)
                
                if 'B' in label:
                    image_numpy = util.tensor2im(image, color_map=True)    
                else:    
                    image_numpy = util.tensor2im(image)
#                 image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
#                     print('label :', label)
                    
                    if 'B' in label:
                        image_numpy = util.tensor2im(image, color_map=True)    
                    else:    
                        image_numpy = util.tensor2im(image)
#                     image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
