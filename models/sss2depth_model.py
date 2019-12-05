import torch
from .base_model import BaseModel
from . import networks

EPS = 1e-12
depth_max = -9.64964580535888671875 + EPS
depth_min = -21.6056976318359375 - EPS


class Sss2DepthModel(BaseModel):
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
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        self.model_names = ['G']
        # define networks (generator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)



        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss(self.opt.lambda_TV)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.mask = (self.real_B<1.0) # type bool
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A) 
        self.fake_B_show = self.fake_B * (self.mask.float())
        self.fake_B_show[~self.mask]=1.0

    def backward_G(self):
        """Calculate TV and L1 loss for the generator"""

        # Second, G(A) = B
#         self.loss_CNT = self.mask.size()[0]*self.mask.size()[2]*self.mask.size()[3]/torch.sum(self.mask.float())
        
        self.loss_G_L1 = self.criterionL1(self.fake_B_show, self.real_B) * 2.0 *(depth_max-depth_min)
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
