import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
import os
from dataloader.geocube2d import Geocube
from models.gan_trainer import GANTrainer
import ReadGSLIB
cmap,norm = ReadGSLIB.color_map(mode='7')

def plot(input_tensor,epoch,num):
    plt.imsave("./generated/Epoch%d_%d.png"%(epoch,num),input_tensor[0][0],vmin=0,vmax=6,cmap=cmap)
    plt.imsave("./generated/Epoch%d_%d.png"%(epoch,num+1),input_tensor[1][0],vmin=0,vmax=6,cmap=cmap)

checkfolder = r'./generated'
isExist = os.path.exists(checkfolder)
if not isExist:
    os.makedirs(checkfolder)

path = r'./Dataset/Flumy'
batch_size=16
fold = 'train2d'
dataset = {
    x: Geocube(path, split=x,isaug=False) for x in [fold]
}

data = {
    x: torch.utils.data.DataLoader(dataset[x], 
                  batch_size=batch_size, 
                  shuffle=True, 
                  num_workers=0,
                  drop_last=True) for x in [fold]
}

# Create class to resemble argparse
class Args:
    def __init__(self,gen_input_size=256, gen_hidden_size=128, g_num_filter=32, 
                d_num_layer=4, d_num_filter=32, img_nc = 1,isTrain=True,
                lr=0.0002,is_continue=False,which_epoch=0,gpu_ids="0"):
        self.gen_input_size = gen_input_size
        self.gen_hidden_size = gen_hidden_size
        self.g_num_filter = g_num_filter
        self.d_num_layer = d_num_layer
        self.d_num_filter = d_num_filter
        self.img_nc = img_nc
        self.isTrain = isTrain
        self.is_continue = is_continue
        self.which_epoch = which_epoch
        self.lr = lr
        self.gpu_ids = gpu_ids

args = Args()

epochs = 200-args.which_epoch
con = args.which_epoch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise Exception('GPU not available')
torch.backends.cudnn.benchmark = True

def save_loss(filename, value):
    with open (filename,'a') as f:
        f.write(str(value))
        f.write('\n')

trainer = GANTrainer(args)

for epoch in tqdm(range(epochs)):
    for i, data_i in enumerate(data[fold]):
        noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)
        trainer.run_generator_one_step(noise,data_i)
        trainer.run_discriminator_one_step(noise, data_i)
        print("Iteration {}/{} started".format(i+1, len(data[fold])))
        if (i+1)%200 == 0:
            losses = trainer.get_latest_losses()
            save_loss('g_loss.txt',losses['GAN'].item())
            save_loss('dr_loss.txt',losses['D_real'].item())
            save_loss('df_loss.txt',losses['D_Fake'].item())
            save_loss('gp.txt',losses['D_GP'].item())

    if (epoch+1+con)%1 == 0:
        with torch.no_grad():
            fake_img = trainer.get_latest_generated()
            fake_img_np = fake_img.detach().cpu().numpy()
            plot(fake_img_np,epoch+1+con,1)

    if (epoch+1+con)%10 == 0:
        trainer.save(epoch+1+con)