from models.gan_model import GANModel
import torch
import os

def save_optim(optim, label, epoch):
    save_filename = '%s_optim_%s.pth' % (epoch, label)
    save_path = os.path.join(save_filename)
    torch.save(optim.state_dict(),save_path)

def load_optim(optim, label, epoch):
    save_filename = '%s_optim_%s.pth' % (epoch, label)
    save_path = os.path.join(save_filename)
    weights = torch.load(save_path)
    optim.load_state_dict(weights)
    return optim

class GANTrainer():
    def __init__(self, args):
        self.args = args
        self.gan_model = GANModel(args)
        if len(args.gpu_ids) > 0:
            self.gan_model_on_one_gpu = self.gan_model.cuda()
        self.generated = None
        self.isTrain = args.isTrain
        
        if args.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.gan_model_on_one_gpu.create_optimizers(args)
            if args.is_continue:
                self.optimizer_G = load_optim(self.optimizer_G, 'G', args.which_epoch)
                self.optimizer_D = load_optim(self.optimizer_D, 'D', args.which_epoch)

    def test_generate(self, noise, data):
        generated = self.gan_model(noise, data, mode='inference')
        return generated

    def run_generator_one_step(self, noise, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.gan_model(noise, data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, noise, data):
        self.optimizer_D.zero_grad()
        d_losses = self.gan_model(noise, data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.gan_model_on_one_gpu.save(epoch)
        save_optim(self.optimizer_G, 'G', epoch)
        save_optim(self.optimizer_D, 'D', epoch)