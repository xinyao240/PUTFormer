from model_2080 import PUTFormer
import torch
from torch.utils.data import DataLoader
from data import PUTrainData
from tqdm import tqdm
import numpy as np
from utils import NRMSE
from matplotlib import pyplot as plt
from thop import profile
import time
import math

import matplotlib
matplotlib.use('Agg')

dataset=PUTrainData(f'data/test_data.hdf5', aug=False)

loader=DataLoader(dataset, shuffle=False, batch_size=1)
gtb_head=8
gtb_num=4
ltb_num=(1,1,1,1)
base_ch=8
net = PUTFormer(base_ch=base_ch, gtb_head=gtb_head, ltb_head=(8,4,2,1), gtb_num=gtb_num, ltb_num=ltb_num).cuda()
net.eval()

model_name=f'model_name'

net.load_state_dict(torch.load(f'checkpoints/{model_name}.pth'))

wrapped_y=[]
y=[]
y_pred=[]
y_pred_scaled=[]
nrmses=[]
naes=[]
start_time=time.time()
for i, batch in enumerate(tqdm(loader)):
    gt = batch['gt'].cuda()
    wrapped = batch['wrapped'].cuda()

    wrapped_y.append(wrapped.squeeze().detach().cpu().numpy())
    y.append(gt.squeeze().detach().cpu().numpy())

    out = net(wrapped)

    y_pred.append(out.squeeze().detach().cpu().numpy())

    nrmse, out_scaled, nae=NRMSE(gt, out)
    nrmses.append(nrmse.item())
    y_pred_scaled.append(out_scaled.squeeze().detach().cpu().numpy())
    naes.append(nae.squeeze().detach().cpu().numpy())
avg_time=(time.time()-start_time)/len(loader)
print(f'time={avg_time*1e3:.3f} ms')
avg_nrmse=sum(nrmses)/len(nrmses)
print(f'NRMSE={avg_nrmse:.2f}% ')

