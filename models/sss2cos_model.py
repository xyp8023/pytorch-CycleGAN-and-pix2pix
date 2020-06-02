import torch
from .base_model import BaseModel
from . import networks

from torch.autograd import Variable
from torch.autograd import grad as Grad
import numpy as np

import torch



class MaskedL1Loss(torch.nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        # print("predi, ", pred.dim(), "target, ", target.dim())
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        # valid_mask = (target>-1.0).detach()
        valid_mask = (target!=0.0).detach()
        
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class MaskedL2Loss(torch.nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()

    def forward(self, pred, target):
        # print("predi, ", pred.dim(), "target, ", target.dim())
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        # valid_mask = (target>-1.0).detach()
        valid_mask = (target!=0.0).detach()
        
        diff = (target - pred)**2
        diff = diff[valid_mask]
        self.loss = diff.mean()
        return self.loss 
    
class Sss2CosModel(BaseModel):
    """ This class implements the sss2depth model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG resnet_9blocks' Res-Net generator,
    and a '--gan_mode' as none: L1 loss with total variation reluarization

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For sss2depth, we do not use image buffer
        The training objective is: Ln_loss + lambda_tv * TV Loss: Ln_loss = ||G(A)-B||_n
        By default, we use L1 loss, ResNet with instance_norm, and aligned datasets.
        """
        # changing the default values to match our case
        # if parser.is_pnp:
            # parser.set_defaults(norm='instance', netG='resnet_7blocks_pnp', dataset_mode='aligned')
        # else:
        parser.set_defaults(norm='instance', netG='resnet_7blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='none')
            parser.add_argument('--lambda_TV', type=float, default=1e-6, help='weight for tv regularation')

        return parser

    def __init__(self, opt):
        """Initialize the sss2depth class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['TV',  'G_L1']
#         self.loss_names = ['TV', 'G_L1']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
#         self.visual_names = ['real_A', 'fake_B', 'real_B', 'mask']
        self.visual_names = ['real_A', 'real_B', 'fake_B_show', 'fake_B']
        self.visual_names_test = ['real_A', 'real_B', 'fake_B_show', 'fake_B', 'real_slant', 'real_depth'] # real_A: SSS, real_B: cos
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        self.model_names = ['G']
        # define networks (generator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            # define loss functions
#             self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1 =  MaskedL1Loss().cuda()
            
            #self.criterionL2 =  MaskedL2Loss().cuda()
            
            self.criterionTV = networks.TVLoss(self.opt.lambda_TV)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.pnp_iters = 1
        else:
            self.pnp_iters = opt.pnp_iters


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # sss
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # cos
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.mask = (self.real_B>-1.0) # type bool
        self.real_slant = input['slant'].to(self.device)
        self.real_depth = input['depth'].to(self.device)
#         slant = torch.from_numpy((np.arange(256)*170.0/256).reshape(1,1,256)).float().to(self.device)
#         torch.from_numpy(slant_).float().to(device)
        # print("real_A shape: ", self.real_A.shape)
        # print("real_B shape: ", self.real_B.shape)
        # print("real_mask shape: ", self.mask.shape)
        # print("real_slant shape: ", self.real_slant.shape)
        # print("real_depth shape: ", self.real_depth.shape)
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.fake_B = self.netG(self.real_A) 
        self.fake_B_show = self.fake_B * (self.mask.float())
        self.fake_B_show[~self.mask]=-1.0



    def forward_pnp_front(self):
        # print("self.__dict__ is ", self.__dict__)
        # print("self.netG.__dict__ is ",self.netG.__dict__)
        # self.netG.forward(self.real_A)
        self.pnp_z =self.netG.module.forward_pnp_front(self.real_A) 

    def forward_pnp_rear(self):
        self.fake_B =self.netG.module.forward_pnp_rear(self.pnp_z) 
        # self.sparse_mask = (self.sparse_target<1.0)
        self.fake_B_show = self.fake_B * (self.mask.float())
        self.fake_B_show[~self.mask]=-1.0

    def backward_G(self):
        """Calculate TV and L1 loss for the generator"""

        # Second, G(A) = B
#         self.loss_CNT = self.mask.size()[0]*self.mask.size()[2]*self.mask.size()[3]/torch.sum(self.mask.float())
        #slant = torch.from_numpy((np.arange(256)*169.0/256).reshape(1,1,256)).float().to(self.device)
        
#         self.loss_G_L1 = self.criterionL1( -(self.fake_B_show+1.0)/2.0 * slant, self.real_depth) 
#         self.loss_G_L1 = self.criterionL1( (self.fake_B_show+1) *slant, (self.real_B+1)*slant) 
        #self.loss_G_L1 = self.criterionL2( (self.fake_B_show+1) *slant, (self.real_B+1)*slant) 
#         self.loss_G_L1 = self.criterionL2( (self.fake_B_show+1) *slant, (self.real_B+1)*slant) 
        self.loss_G_L1 = self.criterionL1((self.fake_B_show+1) * self.real_slant, (self.real_B+1) * self.real_slant) 
        
        
        # self.loss_G_L1 = self.criterionL1( self.fake_B_show, self.real_B) 

        # self.loss_G_L1 = self.criterionL1(self.fake_B_show , self.real_B) 

#         self.loss_G_L1_show = self.criterionL1(self.fake_B_show , self.real_B) * 2.0 *(depth_max-depth_min) * self.loss_CNT
        
        self.loss_TV = self.criterionTV(self.fake_B) 
        # combine loss and calculate gradients
        self.loss_G = self.loss_TV + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def optimize_parameters_pnp(self):
        # self.forward_pnp_front()    
        # self.forward_pnp_rear()               # compute fake images: G(A)
        self.sparse_target = self.real_A[:,-1:] # NOTE: written for rgbd input
        # self.sparse_target = self.real_depth
        # criterion = criteria.MaskedL1Loss().cuda() # NOTE: criterion function defined here only for clarity
        # pnp_iters = 10 # number of iterations
        pnp_alpha = 0.01 # update/learning rate
        # pnp_z = model.pnp_forward_front(input)
        self.forward_pnp_front() 
        self.criterionMaskedL1 = MaskedL1Loss().cuda()
        # self.criterionTV = networks.TVLoss(self.opt.lambda_TV)
        for pnp_i in range(self.pnp_iters):
            
            if pnp_i != 0:
                self.pnp_z = self.pnp_z - pnp_alpha * torch.sign(pnp_z_grad) # iFGM
            self.pnp_z = Variable(self.pnp_z, requires_grad=True)
            # pred = model.pnp_forward_rear(pnp_z)
            self.forward_pnp_rear() 
            if pnp_i < self.pnp_iters - 1:
                # if pnp_i==0:
                    # print("pnp iteration: ", pnp_i)
                self.loss_G_L1 = self.criterionMaskedL1(self.fake_B, self.sparse_target ) 
                
                # self.loss_G_L1 = self.criterionMaskedL1( -(self.fake_B+1.0)/2.0*self.real_slant, -(self.sparse_target+1.0)/2.0*self.real_slant ) 
                # self.loss_TV = self.criterionTV(self.fake_B)
                # self.loss_G = self.loss_TV + self.loss_G_L1
                pnp_z_grad = Grad([self.loss_G_L1], [self.pnp_z], create_graph=True)[0]
