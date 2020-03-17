"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from cv2 import applyColorMap, COLORMAP_JET 


def multi_tensor2im(input_image, imtype=np.float64):
    """"Converts a 4D Tensor array into a numpy array. For cal scores

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        
        print_multi_numpy(image_numpy, val=True, shp=True)# max 255.0 min 0.0
        
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1.0) / 2.0 * 255.0  # post-processing: tranpose and scaling
            
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2im_raw(input_image, imtype=np.float64):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # print(image_numpy.shape)
        # image_numpy = np.transpose(image_numpy, (1, 2, 0)) 
        
    return image_numpy.astype(imtype)

def tensor2im(input_image, imtype=np.uint8, keep_grayscale=False, color_map=False):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            if not keep_grayscale:
                image_numpy = np.tile(image_numpy, (3, 1, 1))
#         print("before post-processing\n")
#         print_numpy(image_numpy, val=True, shp=True)
        
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1.0) / 2.0 * 255.0  # post-processing: tranpose and scaling
        
        if color_map:
#             import matplotlib.pyplot as plt
#             cm = plt.get_cmap('jet')
#             print('before colormap: ')
#             print_numpy(image_numpy, val=True, shp=True)
#             image_numpy = cm(image_numpy[:,:,0])[:,:,:3]
            image_numpy = applyColorMap(image_numpy.astype(np.uint8), COLORMAP_JET)
#             print('after colormap \n')
#             print_numpy(image_numpy, val=True, shp=True)
            
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_numpy_array(image_numpy, image_path):
    """Save a numpy array to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    np.save(image_path, image_numpy)

def save_image(image_numpy, image_path, aspect_ratio=1.0, color_map=False):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    if color_map:
        import matplotlib.pyplot as plt
        cm = plt.get_cmap('jet')
        colored_image = cm(image_numpy[:,:,0])[:,:,:3]
#         print_numpy(colored_image, val=True, shp=True) # max 1.0 min 0.0 shape (256,256,3)
        
        image_pil = Image.fromarray((colored_image*255.).astype(np.uint8))
    else:
#         print_numpy(image_numpy, val=True, shp=True)
        image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
        
def print_multi_numpy(x_, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    for i in range(x_.shape[0]):
        x = x_[i].astype(np.float64)
        if shp:
            print('shape,', x.shape)
        if val:
            x = x.flatten()
            print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
                np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
