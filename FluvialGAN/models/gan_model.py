import torch
import os
from models.generator import Generator
from models.discriminator import HybridDiscriminator
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
        netD = HybridDiscriminator(args)
        
        netG.apply(init_weights)
        netD.apply(init_weights)

        if args.is_continue:
            netG = load_network(netG, 'G', args.which_epoch)
            netD = load_network(netD, 'D', args.which_epoch)
        #print(netD)
        return netG, netD

    def preprocess_input(self, data):
        # move to GPU and change data types
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

        real_group1 = self.group(real_image+1,[1.0,2.0,3.0,7.0],[4.0,5.0])
        real_group2 = self.group(real_image+1,[3.0,7.0],[1.0,2.0])

        fake_group1 = self.group_G(fake_image,[1.0,2.0,3.0,7.0],[4.0,5.0])
        fake_group2 = self.group_G(fake_image,[3.0,7.0],[1.0,2.0])

        real_image = self.onehot(real_image)
        LAMBDA = 10

        fake_add = torch.cat([fake_group1, fake_group2], dim=1)
        real_add = torch.cat([real_group1, real_group2], dim=1)

        fake_concat = torch.cat([fake_add, fake_image[::,[0,1,2,3,4,6],::,::]], dim=1)
        real_concat = torch.cat([real_add, real_image[::,[0,1,2,3,4,6],::,::]], dim=1)

        interpolates_concat = self.cal_interpolate(real_concat,fake_concat)

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

    def _tolong(self,input_onehot):
        bs, _, h, w = input_onehot.size()
        out=input_onehot.argmax(1).view([bs,1,h,w])
        return out

    def generate_fake(self, noise):
        fake_image = self.netG(noise)
        fake_cube = self._tolong(fake_image)
        return fake_image,fake_cube

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.
    def onehot(self, input_image):
        input_image = input_image.long()
        input_image = input_image.cuda()
        f_map = input_image
        bs, c, h, w = f_map.size()
        nc = 7
        out_label = self.FloatTensor(bs, nc*c, h, w).zero_()
        for i in range(c):
            one_map=f_map[::,i,::,::].view([bs,1,h,w])
            #scatter index value < dim
            out_label[::,i*nc:i*nc+nc,::,::].scatter_(1, one_map, 1.0)

        return out_label

    def discriminate(self, fake_image, real_image):

        real_group1 = self.group(real_image+1,[1.0,2.0,3.0,7.0],[4.0,5.0])
        real_group2 = self.group(real_image+1,[3.0,7.0],[1.0,2.0])

        fake_group1 = self.group_G(fake_image,[1.0,2.0,3.0,7.0],[4.0,5.0])
        fake_group2 = self.group_G(fake_image,[3.0,7.0],[1.0,2.0])

        fake_add = torch.cat([fake_group1, fake_group2], dim=1)
        real_add = torch.cat([real_group1, real_group2], dim=1)

        real_image = self.onehot(real_image)
        
        fake_concat = torch.cat([fake_add, fake_image[::,[0,1,2,3,4,6],::,::]], dim=1)
        #print(fake_concat.size())
        real_concat = torch.cat([real_add, real_image[::,[0,1,2,3,4,6],::,::]], dim=1)
        
        pred_fake = self.netD(fake_concat)
        pred_real = self.netD(real_concat)
        #print(pred_fake[0][0].size())
        return pred_fake, pred_real

    def group(self, inputTensor, group1, group2):
        inputTensor = inputTensor.float()
        bs, c, h, w = inputTensor.size()
        groupedTensor1 = self.FloatTensor(bs,1, h, w).zero_()
        groupedTensor2 = self.FloatTensor(bs,1, h, w).zero_()
        one = torch.ones_like(inputTensor)

        for i in group1:
            groupedTensor1 = torch.where(inputTensor==i,one,groupedTensor1)
        for i in group2:
            groupedTensor2 = torch.where(inputTensor==i,one,groupedTensor2)
        
        groupedTensor = torch.cat([groupedTensor1, groupedTensor2], dim=1)

        return groupedTensor

    def group_G(self,inputTensor, group1, group2):
        inputTensor = inputTensor.float()
        bs, c, h, w = inputTensor.size()
        out = torch.cuda.FloatTensor(bs, 2, h, w).zero_()
        for i in group1:
            out[::,0,::,::] += inputTensor[::,int(i-1),::,::]
        for i in group2:
            out[::,1,::,::] += inputTensor[::,int(i-1),::,::]

        return out