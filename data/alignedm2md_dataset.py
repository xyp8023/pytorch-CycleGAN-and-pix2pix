import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
# from PIL import Image
import numpy as np
from copy import deepcopy

class AlignedM2MDDataset(BaseDataset):
    """A dataset class for paired image dataset. (SSS-sparse-depth, depth)

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
#         print("self.AB_paths are: ", self.AB_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.sample_nums = self.opt.sample_nums
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
#         AB = Image.open(AB_path).convert('RGB')
        AB = np.load(AB_path)
        # split AB image into A and B
        h, w = AB.shape
#         h, w = AB.size
        
        w2 = int(w / 2)
        # h2 = int(h / 2)
        
        # N, C, H, W
#         A = AB.crop((0, 0, w2, h))
#         B = AB.crop((w2, 0, w, h)) 
        
        # for sss2depth
        # sss = AB[:,:w2] # (256, 256)
        # depth = AB[:,w2:] # (256, 256)
        
        # for sssd2depth
        sss = AB[:,:h]
        cos = AB[:,h:2*h]
        sparse_cos = AB[:, 2*h:3*h]

        slant = AB[:, 3*h:4*h]
        depth = AB[:, 4*h:5*h]

        sparse_cos = np.expand_dims(sparse_cos, axis=0) # (1, 256, 256)
        slant = np.expand_dims(slant, axis=0) # (1, 256, 256)
        depth = np.expand_dims(depth, axis=0) # (1, 256, 256)

        """
        # sample sparse_depth
        mask_keep = depth!=1.0
        n_keep = np.count_nonzero(mask_keep)
        prob = float(self.sample_nums)/n_keep
        # sample_mask = np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
        sample_mask = np.bitwise_and(mask_keep, np.random.uniform(0, 1, (256,)) < prob)
        sparse_depth = deepcopy(depth)
        sparse_depth[~sample_mask] = 1. # (256, 256)
        sparse_depth = np.expand_dims(sparse_depth, axis=0) # (1, 256, 256)
        """
        sss = sss.reshape((1,) + sss.shape) # (1, 256, 256)
        # print("sss shape: ", sss.shape)
        # print("sparse depth shape: ", sparse_depth.shape)

        sss_sparse_cos = np.append(sss, sparse_cos, axis=0)
        cos = cos.reshape((1,) + cos.shape)
        

        # apply the same transform to both A and B
#         transform_params = get_params(self.opt, A.size)
#         A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
#         B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

#         A = A_transform(A)
#         B = B_transform(B)

        return {'A': sss_sparse_cos, 'B': cos, 'A_paths': AB_path, 'B_paths': AB_path, 'slant': slant, 'depth': depth}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
