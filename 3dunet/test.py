import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import nibabel
import sys
import random
from torchvision.transforms import transforms
from scipy import ndimage
import matplotlib.pyplot as plt
from dataset import BraTS
from model import Unet3D
from criterion import softmax_dice
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

def validate_batch(model, data, criterion):
    with torch.no_grad():
        model.eval()
        images, targets = data
        preds = model(images)
        loss, dice0, dice1, dice2, dice3 = criterion(preds, targets)
        return loss.item(), dice0.item(), dice1.item(), dice2.item(), dice3.item()

print("Loading Data : ")

#train_dataset = BraTS("./train_pkl_all/",'train')
val_dataset = BraTS("./val_pkl_all/",'valid')

#train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = True)

print(f"Val Loader {len(val_loader)}")


model = Unet3D(4, 4, 64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = softmax_dice()


# load_checkpoint = True
# if load_checkpoint:
#     path = './model_f58.pth'
#     model.load_state_dict(torch.load(path))

nEpochs = 20
start_epoch = 40
log_interval = 10

def log(mode, epoch, i, loss, l0, l1, l2, l3):
    #loss, l0, l1, l2, l3 = [round(j/i,3) for j in [loss, l0, l1, l2, l3]]
    ET = l3
    TC = round((l1 + l2)/2,3)
    WT = round((l1 + l2 + l3)/3,3)
    print(f"{mode} | Epoch: {epoch+1} | Iter: {i+1} Overall Loss: {loss} | BackGround: {l0} | ET : {ET} | TC : {TC} | WT : {WT}")

for epoch in range(start_epoch, start_epoch+nEpochs):

    path = './model_f58.pth'
    # model = nn.DataParallel(model.cuda())
    train_loss = val_loss = 0
    dice_1_t = dice_2_t= dice_3_t = dice_0_v = dice_1_v = dice_2_v = dice_3_v = 0
    model.load_state_dict(torch.load(path))
    for i, data in enumerate(val_loader):
        import pdb
        #pdb.set_trace()
        loss, dice0, dice1, dice2, dice3 = validate_batch(model, data, criterion)
        if (i+1)%log_interval==0:
            log("Val", epoch, i, loss, dice0, dice1, dice2, dice3)
            # print(f"Val Epoch: {epoch}, Iter: {i} Overall Loss: {loss} | L1 Dice : {dice1} | L2 Dice : {dice2} | L3 Dice : {dice3}")
        val_loss += loss
        dice_0_v += dice0
        dice_1_v += dice1
        dice_2_v += dice2
        dice_3_v += dice3
    print('='*50)
    d = len(val_loader)
    log("Val End", epoch, i, val_loss/d, dice_0_v/d, dice_1_v/d, dice_2_v/d, dice_3_v/d)
    # avg_val = f"Val Epoch: {epoch}, Overall Loss: {val_loss/d} | L1 Dice : {dice_1_v/d} | L2 Dice : {dice_2_v/d} | L3 Dice : {dice_3_v/d}"
    # print(avg_val)
    # f.writelines(avg_val)