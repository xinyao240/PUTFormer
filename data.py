import os

from torch.utils.data import Dataset
import h5py
import numpy as np
import random
from skimage import io
from skimage.transform import resize, rescale
import torch

def rescale_(im, range):
  """
  mini-max rescales the input image
  """
  im_std = (im - im.min()) / (im.max() - im.min()+1e-20)
  im_scaled = im_std * (range[1] - range[0]) + range[0]
  return im_scaled

def wrap(phi):
  """
  wraps the true phase signal within [-pi, pi]
  """
  return np.angle(np.exp(1j*phi))

class PUTrainData(Dataset):
    def __init__(self, path='', aug=True, mask_aug=False):
        super(PUTrainData, self).__init__()
        self.path=path
        file=h5py.File(path, 'r')
        self.wrapped=file['psi']
        self.gt = file['phi']

        self.aug=aug
        self.mask_aug=mask_aug

    def __len__(self):
        return self.wrapped.shape[0]

    def mask_out(self, size=256, min_box_wid=8, max_box_wid=64, min_box_num=1, max_box_num=4):
        box_num = random.randint(min_box_num, max_box_num)
        box_wids = [random.randint(min_box_wid, max_box_wid) for i in range(box_num)]
        corners = [(random.randint(0, size - 1 - box_wids[i]), random.randint(0, size - 1 - box_wids[i])) for i in
                   range(box_num)]
        mask=np.ones((size, size), dtype=np.float32)
        for corner, wid in zip(corners, box_wids):
            mask[corner[0]:corner[0] + wid, corner[1]:corner[1] + wid] = 0.

        return mask

    def __getitem__(self, i):
        wrapped=np.array(self.wrapped[i:i+1, ...], dtype=np.float32)
        gt=np.array(self.gt[i:i+1, ...], dtype=np.float32)
        maskornot=np.random.rand()
        if maskornot<=0.6 and self.mask_aug:
            mask=self.mask_out(min_box_num=4, max_box_num=24)
            wrapped=mask*wrapped
            gt=mask*gt

        if self.aug:
            flip = np.random.randint(0, 2)
            if flip:
                axis = np.random.randint(-2, 0)
                wrapped = np.flip(wrapped, axis=axis)
                gt = np.flip(gt, axis=axis)
            rotate = np.random.randint(0, 4)
            wrapped = np.rot90(wrapped, k=rotate, axes=(-2, -1))
            gt = np.rot90(gt, k=rotate, axes=(-2, -1))
        return {'wrapped':wrapped.copy(), 'gt':gt.copy()}


class ElevationMapData(Dataset):
    def __init__(self, num_samples=1000, path='', crop_size=256, max_range=10*2*np.pi, min_range=-10*2*np.pi, aug=True):
        super().__init__()
        self.num=num_samples
        self.path=path
        img_files_list=[]
        fns=os.listdir(path)
        self.img_list=[]
        for fn in fns:
            img_files_list.append(os.path.join(path, fn))
            self.img_list.append(io.imread(os.path.join(path, fn)))
        self.files_list=img_files_list

        self.crop_size=crop_size
        self.max_range=max_range
        self.min_range=min_range
        self.aug=aug

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        img = random.sample(self.img_list, 1)[0]
        scale=np.random.rand()*0.8+0.2

        # img=rescale(img, scale)

        h, w=img.shape

        x=np.random.randint(0, h-self.crop_size+1)
        y=np.random.randint(0, w-self.crop_size+1)

        phi=img[x:x+self.crop_size, y:y+self.crop_size]

        upperbound=np.random.rand()*(self.max_range-self.min_range)+self.min_range

        gt=rescale_(phi, (0, upperbound))
        wrapped=wrap(gt)

        if self.aug:
            flip = np.random.randint(0, 2)
            if flip:
                axis = np.random.randint(-2, 0)
                wrapped = np.flip(wrapped, axis=axis)
                gt = np.flip(gt, axis=axis)
            rotate = np.random.randint(0, 4)
            wrapped = np.rot90(wrapped, k=rotate, axes=(-2, -1))
            gt = np.rot90(gt, k=rotate, axes=(-2, -1))

        psi = torch.tensor(wrapped.copy(), dtype=torch.float32).cuda()
        phi = torch.tensor(gt.copy(), dtype=torch.float32).cuda()
        psi = psi.unsqueeze(0)
        phi = phi.unsqueeze(0)

        return {'wrapped': psi, 'gt': phi}
