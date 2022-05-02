import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import nibabel as nib
import pickle
from tqdm import tqdm
import random
from torchvision.transforms import transforms
from scipy import ndimage
import matplotlib.pyplot as plt


def Dice(output, target, eps=1e-5):
        target = target.float()
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
        return 1.0 - num/den

def dice_score(output, target, eps=1e-5):
    # import pdb
    # pdb.set_trace()
    _, labels = torch.max(output, dim = 1)
    D0 = Dice((labels[...]==0).float(), (target == 0).float())
    D1 = Dice((labels[...]==1).float(), (target == 1).float())
    D2 = Dice((labels[...]==2).float(), (target == 2).float())
    D3 = Dice((labels[...]==3).float(), (target == 3).float())
    return 1-D0, 1-D1, 1-D2, 1-D3

class softmax_dice(nn.Module):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    def __init__(self):
        super(softmax_dice, self).__init__()
        
    def forward(self, output, target):
        target[target == 4] = 3 
        output = output.cuda()
        target = target.cuda()
        loss0 = Dice(output[:, 0, ...], (target == 0).float())
        loss1 = Dice(output[:, 1, ...], (target == 1).float())
        loss2 = Dice(output[:, 2, ...], (target == 2).float())
        loss3 = Dice(output[:, 3, ...], (target == 3).float())
        D0, D1, D2, D3 = dice_score(output, target)
        return loss1 + loss2 + loss3 + loss0, D0, D1, D2, D3