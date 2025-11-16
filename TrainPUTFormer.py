from model_2080 import PUTFormer, PUNet_v8, PUTFormer_v1
import torch
from torch.utils.data import DataLoader
from data import PUTrainData
from tqdm import tqdm
import random
import numpy as np
from thop import profile
from utils import multiscale_tv_loss as loss_func
from utils import toRed, toBlue, toCyan, toGreen, toYellow

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed = 777
seed_everything(seed)

dataset_n='RME-Noisy_0_5_10_20_60_Phase_Data_5000_20pi_20pi'
# dataset_n='Noisy_0_5_10_20_60_Phase_Data_5000_20pi_20pi-glob_num_1~16'
# dataset_n='InSAR-train_True-Noisy_0_5_10_20_60_Phase_Data_5000_20pi_20pi'
dataset=PUTrainData(f'/home/lab535/data/yx/Data/SyntheticPhaseUnwrap/train/{dataset_n}.hdf5',  aug=True, mask_aug=False)
batch_size=8
acc_step=1
loader=DataLoader(dataset, shuffle=True, batch_size=batch_size//acc_step, num_workers=0)

gtb_head=8
gtb_num=4
ltb_num=(1,1,1,1)
base_ch=8
net = PUTFormer(base_ch=base_ch, gtb_head=gtb_head, ltb_head=(8,4,2,1), gtb_num=gtb_num, ltb_num=ltb_num).cuda()

net.train()
model_name=f'PUTFormer-{base_ch}-{dataset_n}-gtb_head_{gtb_head}-gtb_num_{gtb_num}-ltb_num_{ltb_num}'

lr=1e-3

start_epoch=0
epoch_n=500-start_epoch
sav_freq=50
torch.autograd.set_detect_anomaly(True)

f = open(f'{model_name}.txt', 'a')
# f.write(f'\n\nFLOPs: {flops/1e9:.2f}G\nParams: {params / 1e6:.2f}M\nlr:{lr}\n')
f.write(f'gtb head:{gtb_head}\n')
f.close()

optimizer=torch.optim.Adam(net.parameters(), lr=lr)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 80, 0.5)

epoch_train_loss=[]
epoch_train_psnr=[]

for epoch in range(epoch_n):
    sum_los = []

    with tqdm(total=len(loader),
              desc=f'epoch{start_epoch + epoch + 1}/{start_epoch + epoch_n} train', unit='it', ncols=150) as pbar:
        for i, batch in enumerate(loader):
            gt = batch['gt'].cuda()
            wrapped = batch['wrapped'].cuda()

            # import matplotlib.pyplot as plt
            # plt.figure(0)
            # plt.imshow(wrapped[0].squeeze().detach().cpu().numpy())
            # plt.show()
            #
            # plt.figure(1)
            # plt.imshow(gt[0].squeeze().detach().cpu().numpy())
            # plt.show()

            outs = net(wrapped)

            los, loses = loss_func(gt, outs)

            losss= los / acc_step

            losss.backward()

            pbar.set_postfix(
                {'bat_loss': toBlue(f'{loses[-1].item():.5f}'),
                 'learning rate': toYellow(f'{optimizer.param_groups[0]["lr"]}')})

            sum_los.append(loses[-1].item())

            if ((i + 1) % acc_step) == 0:
                optimizer.step()  
                optimizer.zero_grad()

            pbar.update(1)

    scheduler.step()

    epoch_avg_train_loss = sum(sum_los) / len(sum_los)
    print('epoch{0}/{1} avg train loss:{2}'.format(
        start_epoch + epoch + 1, start_epoch + epoch_n, epoch_avg_train_loss
    ))

    f = open(f'{model_name}.txt', 'a')
    f.write('epoch{0}/{1} avg train loss:{2}\n'.format(
        start_epoch + epoch + 1, start_epoch + epoch_n, epoch_avg_train_loss
    ))
    f.close()

    epoch_train_loss.append(epoch_avg_train_loss)

    if (start_epoch + epoch + 1) % sav_freq == 0:
        torch.save(net.state_dict(), f'checkpoints/{model_name}-epoch{start_epoch + epoch + 1}.pth')

    else:
        torch.save(net.state_dict(),
                   f'checkpoints/{model_name}.pth')
