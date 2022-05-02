import cv2
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import torch
from model_ import UNet
from utils import visualize, log
from dataset import TumorDataset
from torch.utils.data import DataLoader
from criterion_ import softmax_dice, cal_Hausdoff
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 4

# train = TumorDataset(limit=100)
# trainLoader = DataLoader(train, batch_size = batch_size, shuffle=True)

val = TumorDataset(pickle_path_root = '/home/sarucrcv/projects/brats/val_pkl_all/', limit=100)
valLoader = DataLoader(val, batch_size= batch_size, shuffle = True)



# print(f"Train Loader {len(trainLoader)}")
print(f"Val Loader {len(valLoader)}")



model = UNet(4, 4).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = softmax_dice()
# nEpochs = 10
log_interval = 10

# Load Weights
PATH = './2dunet_best_50_Epochs_30k.pth'
model.load_state_dict(torch.load(PATH))

val_res = []



@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    images, targets = data
    images = images.to(device)
    targets = targets.to(device)
    preds = model(images.permute(0,3,1,2))
    loss, dice0, dice1, dice2, dice3 = criterion(preds, targets)
    return loss.item(), dice0.item(), dice1.item(), dice2.item(), dice3.item()

best_loss = 10000

epoch = 1
train_loss = val_loss = 0
dice_0_t = dice_1_t = dice_2_t= dice_3_t = dice_1_v = dice_2_v = dice_3_v = 0
for i, data_v in enumerate(valLoader):
    loss, dice0, dice1, dice2, dice3 = validate_batch(model, data_v , criterion)
    val_loss += loss
    dice_0_t += dice0
    dice_1_t += dice1
    dice_2_t += dice2
    dice_3_t += dice3
    if (i+1)%log_interval==0:
            log("Val", epoch, i, val_loss, dice_0_t, dice_1_t, dice_2_t, dice_3_t)      
log("Val", epoch, i, val_loss, dice_0_t, dice_1_t, dice_2_t, dice_3_t)      
