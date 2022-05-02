import cv2
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import torch
from model import AttU_Net
from utils import visualize, log
from dataset import TumorDataset
from torch.utils.data import DataLoader
from criterion import softmax_dice
from tqdm import tqdm
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 16

seed_ = 42
torch.manual_seed(seed_)
torch.cuda.manual_seed(seed_)
random.seed(seed_)
np.random.seed(seed_)


train = TumorDataset('../brats/train_pkl_all/', limit=80)
trainLoader = DataLoader(train, batch_size = batch_size, shuffle=False)

val = TumorDataset(pickle_path_root = '../brats/train_pkl_all/', limit=80)
valLoader = DataLoader(train, batch_size= batch_size, shuffle = False)



print(f"Train Loader {len(trainLoader)}")
print(f"Val Loader {len(valLoader)}")


ENAME = '100'
model = AttU_Net(4, 4).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.9)
criterion = softmax_dice()
nEpochs = 50
log_interval = 2
training_res = []
val_res = []
# print(model)

def train_batch(model, data, optimizer, criterion):
    model.train()
    # import pdb
    # pdb.set_trace()
    images, targets = data
    images = images.to(device)
    targets = targets.to(device)
    preds = model(images.permute(0,3,1,2))
    optimizer.zero_grad()
    loss, dice0, dice1, dice2, dice3 = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    return loss.item(), dice0.item(), dice1.item(), dice2.item(), dice3.item()

@torch.no_grad()
def validate_batch(model, data, criterion):
    with torch.no_grad():
        model.eval()
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images.permute(0,3,1,2))
        loss, dice0, dice1, dice2, dice3 = criterion(preds, targets)
        return loss.item(), dice0.item(), dice1.item(), dice2.item(), dice3.item()

best_loss = 10000

for epoch in tqdm(range(nEpochs)):
    train_loss = val_loss = 0
    dice_0_t = dice_1_t = dice_2_t= dice_3_t = dice_1_v = dice_2_v = dice_3_v = 0
    for i, data in enumerate(trainLoader):
        loss, dice0, dice1, dice2, dice3 = train_batch(model, data, opt, criterion)
        train_loss += loss
        dice_0_t += dice0
        dice_1_t += dice1
        dice_2_t += dice2
        dice_3_t += dice3 
    d = len(trainLoader)
    log("Train", epoch, i, train_loss, dice_0_t, dice_1_t, dice_2_t, dice_3_t)      
    training_res.append(round(train_loss/d,3))
    
    
    dice_1_t = dice_2_t= dice_3_t = dice_1_v = dice_2_v = dice_3_v = 0
    for i, data_v in enumerate(valLoader):
        loss, dice0, dice1, dice2, dice3 = validate_batch(model, data_v , criterion)
        val_loss += loss
        dice_0_t += dice0
        dice_1_t += dice1
        dice_2_t += dice2
        dice_3_t += dice3
              
    d = len(valLoader)
    log("Val", epoch, i, val_loss, dice_0_t, dice_1_t, dice_2_t, dice_3_t)      
    val_res.append(round(val_loss/d,3))
    
    
    if val_loss/d < best_loss:
        print("Saving Model")
        torch.save(model.state_dict(), 'saved_models/2dunet_best'+ENAME+'.pth')
        best_loss = val_loss/d
    else:
        torch.save(model.state_dict(), 'saved_models/2dunet_last_checkpoint'+ENAME+'.pth')
    
    
plt.plot(training_res, label='Training_loss')
plt.plot(val_res, label='Val Loss')
plt.legend()
plt.savefig(f'{ENAME}.png')