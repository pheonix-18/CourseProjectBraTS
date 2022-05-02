import numpy as np
import albumentations as A
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from glob import glob


class TumorDataset(Dataset):
    def __init__(self, pickle_path_root = '/home/sarucrcv/projects/brats/train_pkl_all/', limit = 10000, transforms=None ):
        """
        pickle_path_root = /home/sarucrcv/projects/brats/train_pkl_all
        """
        self.data = glob(pickle_path_root+"*/*.npy")[:limit]
        self.transforms = transforms
        self.ImageMaskTransforms = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)
            ]),
            A.RandomRotate90(p=0.5)

        ])
        self.ImageTransforms = A.Compose([
            A.RandomBrightnessContrast(p=0.8)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        d = np.load(self.data[ix], allow_pickle=True)
        img = d[0]/(np.max(d[0])+0.0001)
        label = d[1]
        return img, label
    
def collate_fn(train, batch):
        _imgs, _masks = list(zip(*batch))
        imgs, masks = [], []
        if train.transforms:
            for i in len(_imgs):
                augmented = train.ImageMaskTransforms(image = _imgs[i],mask =  _masks[i])
                imgs.append(torch.Tensor(augmented[0]).unsqueeze(0))
                masks.append(torch.Tensor(augmented[1]).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0).float().to(device)
            masks = torch.cat(masks, dim=0).float().to(device)
            return imgs, masks
        else:
            return _imgs, _masks
                # augmented2 = self.ImageTransforms(imgs[i])
                # imgs[i] = augmented2[0]
            

