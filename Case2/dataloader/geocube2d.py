import os
import numpy as np
import torch
from torch.utils.data import Dataset

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)]

def seperate_cube(input_path, num=64, isaug = False):
    output = []
    for i in range(len(input_path)):
        if isaug:
            for index in range(0,8):
                output.append([input_path[i],index])
        else:
            output.append([input_path[i],0])
    return output

class Geocube(Dataset):
    def __init__(self, 
                 root, 
                 split='train', 
                 args = None,
                 isaug = False,
                 cube_size=(1, 256, 256)):
        self.root = root
        self.split = split
        self.args = args
        self.isaug = isaug
        self.cube_size = cube_size
        self.files = {}
        self.data = {}
        self.cube_base = os.path.join(self.root, "img", self.split)

        self.files[split] = recursive_glob(rootdir=self.cube_base, suffix=".npy")
        self.data[split] = seperate_cube(self.files[split],num=int(self.cube_size[0]),isaug=self.isaug)
        print("Found %d %s images" % (len(self.files[split]), split))
        print("%d %s data"% (len(self.data[split]), split))
 
    def __len__(self):
        return len(self.data[self.split])
    
    def __getitem__(self, index):
        index_aug = self.data[self.split][index][1]
        cube_path = self.data[self.split][index][0].rstrip()
        #print(cube_path)
        NO = os.path.basename(cube_path)[:-11]
        no = NO[:4]
        num = int(NO[4:])
        image = np.load(cube_path).reshape([1,256,256])
        if index_aug == 1:#左转90
            image = np.rot90(image,k=1,axes=(1,2))
            image = np.ascontiguousarray(image)
        elif index_aug == 2:#左转180
            image = np.rot90(image,k=2,axes=(1,2))
            image = np.ascontiguousarray(image)
        elif index_aug == 3:#右转90
            image = np.rot90(image,k=-1,axes=(1,2))
            image = np.ascontiguousarray(image)
        elif index_aug == 4:#上下
            image = np.flip(image,axis=1)
            image = np.ascontiguousarray(image)
        elif index_aug == 5:#左右
            image = np.flip(image,axis=2)
            image = np.ascontiguousarray(image)
        elif index_aug == 6:#负对角线
            image = np.rot90(image,k=1,axes=(1,2))
            image = np.flip(image,axis=1)
            image = np.ascontiguousarray(image)
        elif index_aug == 7:#正对角线
            image = np.rot90(image,k=-1,axes=(1,2))       
            image = np.flip(image,axis=1)
            image = np.ascontiguousarray(image)

        grouped = np.zeros(np.shape(image))
        grouped[np.where(image==1)] = 0.0
        grouped[np.where(image==2)] = 1.0
        grouped[np.where(image==3)] = 2.0
        grouped[np.where(image==4)] = 3.0
        grouped[np.where(image==5)] = 3.0
        grouped[np.where(image==6)] = 3.0
        grouped[np.where(image==7)] = 4.0
        grouped[np.where(image==8)] = 5.0
        grouped[np.where(image==9)] = 6.0
        image_tensor = torch.from_numpy(grouped).view([1,256,256]).float()

        input_dict = {'image': image_tensor,
                      'No':no,
                      'index':num,
                      'index_aug':index_aug,
                      }

        return input_dict