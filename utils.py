import math

import matplotlib.pyplot as plt
import torch
import termcolor
import torch.nn.functional as F

def tv_loss_plus_var_loss(y_true, y_pred):
    """
    Define the composite loss function that includes total variation of errors
    loss and variance of errors loss
    """
    # total variation loss
    y_x = y_true[:, :, 1:256, :] - y_true[:, :, 0:255, :]
    y_y = y_true[:, :, :, 1:256] - y_true[:, :, :, 0:255]
    y_bar_x = y_pred[:, :, 1:256, :] - y_pred[:, :, 0:255, :]
    y_bar_y = y_pred[:, :, :, 1:256] - y_pred[:, :, :, 0:255]
    L_tv = torch.mean(torch.abs(y_x - y_bar_x)) + torch.mean(torch.abs(y_y - y_bar_y))

    # variance of the error loss
    # E = y_pred - y_true
    # L_var = torch.mean(torch.mean(torch.square(E), dim=(1, 2, 3)) - torch.square(torch.mean(E, dim=(1, 2, 3))))

    # loss = L_var + 0.1 * L_tv
    loss = L_tv
    return loss

def tv_loss(y_true, y_pred):
    # total variation loss
    y_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    y_y = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
    y_bar_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    y_bar_y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
    L_tv = torch.mean(torch.abs(y_x - y_bar_x)) + torch.mean(torch.abs(y_y - y_bar_y))

    return L_tv

def l2_tv_loss(y_true, y_pred):
    # total variation loss
    y_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    y_y = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
    y_bar_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    y_bar_y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
    L_tv = torch.mean(torch.square(y_x - y_bar_x)) + torch.mean(torch.square(y_y - y_bar_y))

    return L_tv

def multiscale_tv_loss(y_true, y_pred):
    ys=[F.interpolate(y_true, scale_factor=0.125, mode='bilinear'),
        F.interpolate(y_true, scale_factor=0.25, mode='bilinear'),
        F.interpolate(y_true, scale_factor=0.5, mode='bilinear'),
        y_true]

    weight=[0., 0., 0., 1.]  # supervise last scale only
    loss=0
    losses=[]
    for (y, yp, w) in zip(ys, y_pred, weight):
        loss_s=tv_loss(y, yp)
        loss+=loss_s*w
        losses.append(loss_s)

    return loss, losses

def multiscale_l2_tv_loss(y_true, y_pred):
    ys=[F.interpolate(y_true, scale_factor=0.125, mode='bilinear'),
        F.interpolate(y_true, scale_factor=0.25, mode='bilinear'),
        F.interpolate(y_true, scale_factor=0.5, mode='bilinear'),
        y_true]

    weight=[0., 0., 0., 1.]
    loss=0
    losses=[]
    for (y, yp, w) in zip(ys, y_pred, weight):
        loss_s=l2_tv_loss(y, yp)
        loss+=loss_s*w
        losses.append(loss_s)

    return loss, losses

def NRMSE(gt, out):
    gt_min, gt_max = gt.min(), gt.max()
    out_min, out_max = out.min(), out.max()
    gt=gt.squeeze()
    out=out.squeeze()
    out_scaled = (out - out_min) / (out_max - out_min) * (gt_max-gt_min) + gt_min

    error = gt - out_scaled
    # error = gt - out
    r = torch.max(gt) - torch.min(gt)
    nrmse = torch.mean(torch.sqrt(torch.mean(error ** 2, dim=(0, 1))) / r) * 100

    return nrmse, out_scaled, torch.abs(error)

def wrap(phi):
  """
  wraps the true phase signal within [-pi, pi]
  """
  return torch.angle(torch.exp(1j*phi))

def loss_func(x, target):
    grdY_x, grdX_x = torch.gradient(x, dim=(-2, -1))
    grd_x = torch.cat([grdX_x.unsqueeze(-1)
                       , grdY_x.unsqueeze(-1)], dim=-1)

    grdY_targ, grdX_targ = torch.gradient(target, dim=(-2, -1))
    grd_targ = torch.cat([grdX_targ.unsqueeze(-1)
                          , grdY_targ.unsqueeze(-1)], dim=-1)

    unweighted=torch.sqrt(torch.square_(grd_x - wrap(grd_targ)).sum(dim=-1) + 1e-18)  # 256*256
    # unweighted = torch.square_(grd_x - Wrap(grd_targ)).sum(dim=-1) # 256*256

    return unweighted.mean()

def gaussian_filter(x, sigma=2):
    ks=21
    xx, yy = torch.meshgrid([torch.arange(0, ks), torch.arange(0, ks)])
    k=torch.exp(-((xx-ks//2)/math.sqrt(2)/sigma)**2-((yy-ks//2)/math.sqrt(2)/sigma)**2)
    k=k/k.sum()
    k=k.cuda()
    # plt.figure(0)
    # plt.imshow(k)
    # plt.show()

    y=torch.conv2d(x, k.unsqueeze(0).unsqueeze(0), padding='same')

    return y


def toRed(content):
    return termcolor.colored(content,"red",attrs=["bold"])

def toGreen(content):
    return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])

def toGrey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])

def toWhite(content):
    return termcolor.colored(content,"white",attrs=["bold"])


if __name__=='__main__':
    x=torch.rand(1,1,256,256)
    gaussian_filter(x, 2)
