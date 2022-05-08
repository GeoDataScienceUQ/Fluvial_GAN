import torch
import os
from models.generator import Generator
from models.discriminator import PatchGANDiscriminator
from models.ganloss import GANLoss
from models.weights_init import init_weights
import torch.autograd as ag

def save_network(net, label, epoch, args):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(save_filename)
    torch.save(net.cpu().state_dict(),save_path)
    if len(args.gpu_ids) and torch.cuda.is_available():
        net.cuda()

def load_network(net, label, epoch):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net

class GANModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.FloatTensor = torch.cuda.FloatTensor
        self.netG, self.netD = self.initialize_networks(args)
        # set loss functions
        if args.isTrain:
            self.criterionGAN = GANLoss(gan_mode='hinge', tensor=self.FloatTensor)

    def forward(self, noise, data, mode):
        real_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                noise, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                noise, real_image)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image,fake_cube = self.generate_fake(noise)
            return fake_cube

    def create_optimizers(self, args):

        G_params = list(self.netG.parameters())
        if args.isTrain:
            D_params = list(self.netD.parameters())

        optimizer_G = torch.optim.AdamW(G_params, lr=args.lr, betas=(0.5, 0.9))
        optimizer_D = torch.optim.AdamW(D_params, lr=args.lr, betas=(0.5, 0.9))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        save_network(self.netG, 'G', epoch, self.args)
        save_network(self.netD, 'D', epoch, self.args)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, args):
        netG = Generator(args)
        netD = PatchGANDiscriminator(args)
        
        netG.apply(init_weights)
        netD.apply(init_weights)

        if args.is_continue:
            netG = load_network(netG, 'G', args.which_epoch)
            netD = load_network(netD, 'D', args.which_epoch)
        #print(netD)
        return netG, netD

    def preprocess_input(self, data):
        # move to GPU and change data types
        #data['image'] = data['image']/3.0 - 1.0
        data['image'] = data['image'].cuda()

        return data['image']

    def compute_generator_loss(self, noise, real_image):
        G_losses = {}

        fake_image,fake_cube= self.generate_fake(noise)

        pred_fake, pred_real = self.discriminate(fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        return G_losses, fake_cube

    def cal_gp(self,interpolates_concat,disc_interpolates,center=0):
        device=torch.device('cuda')
        if isinstance(disc_interpolates, list):
            gradients=0
            for pred_i in disc_interpolates:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                    new_gradients = ag.grad(outputs=pred_i, inputs=interpolates_concat,
                                            grad_outputs=torch.ones(pred_i.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
                    gradients += new_gradients
            gradients=gradients / len(disc_interpolates)
        else:
            gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates_concat,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = ((gradients.norm(2, dim=1) - center) ** 2).mean()

        return gp

    def cal_interpolate(self,real_image,fake_image,alpha=None):
        device=torch.device('cuda')
        if alpha is not None:
            alpha = torch.tensor(alpha, device=device)
        else:
            alpha = torch.rand(real_image.size(0), device=device)
        bs, ch, w, h = real_image.size()
        alpha = alpha.expand([ch,w,h,bs]).permute(3,0,1,2)

        interpolates = alpha * real_image + ((1 - alpha) * fake_image)
        interpolates.requires_grad_(True)
        interpolates_concat = interpolates
        return interpolates_concat

    def gradientpenalty(self,real_image,fake_image):
        #from https://github.com/htt210/GeneralizationAndStabilityInGANs/blob/master/GradientPenaltiesGAN.py
        LAMBDA = 10
        interpolates_concat = self.cal_interpolate(real_image,fake_image)
        disc_interpolates = self.netD(interpolates_concat)
        gp = self.cal_gp(interpolates_concat,disc_interpolates) * LAMBDA

        return gp

    def compute_discriminator_loss(self, noise, real_image, use_gp=True):
        D_losses = {}
        with torch.no_grad():
            fake_image,fake_cube = self.generate_fake(noise)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)

        #from https://github.com/htt210/GeneralizationAndStabilityInGANs/blob/master/GradientPenaltiesGAN.py
        if use_gp:
            D_losses['D_GP'] = self.gradientpenalty(real_image, fake_image)

        return D_losses

    def generate_fake(self, noise):
        fake_image = self.netG(noise)
        fake_cube = 3.0*(fake_image+1.0)
        return fake_image,fake_cube

    def discriminate(self, fake_image, real_image):
        pred_fake = self.netD(fake_image)
        pred_real = self.netD(real_image)
        #print(pred_fake[0][0].size())
        return pred_fake, pred_real