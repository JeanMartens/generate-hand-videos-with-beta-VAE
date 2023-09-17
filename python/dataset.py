import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import torchvision.transforms.functional as F

from hyperparams import Hyperparams
from fastnumpyio import fastnumpyio

class PreprocessedDataset(Dataset):
    def __init__(self, metadata_df,all_images,normalise_tranform = None,train=True):
        self.metadata_df = metadata_df
        self.train = train
        # self.all_images = all_images
        self.all_images = F.adjust_sharpness(torch.tensor(all_images),1).squeeze(1).numpy()
        self.normalise_transform = normalise_tranform
        self.resize = A.Resize(Hyperparams.img_shape[0], Hyperparams.img_shape[1],interpolation=cv2.INTER_LANCZOS4, always_apply=True)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #Load input / input must be (Shape1, Shape2, channels)
        index = self.metadata_df.loc[idx, 'index']
        
        input = self.all_images[index,:,:]

        #Load label
        # label = np.array(self.metadata_df.loc[idx, 'labels'])

        #Load input weight
        weight = torch.tensor([1])

        #Augments 
        if self.train == True:
            transformed = Hyperparams.augment_transform(image=input)
            input = transformed["image"]

        #Resize
        # resized = self.resize(image=input)
        # input = resized["image"]

            
        #To tensor and (Shape1, Shape2, channels) -> (channels, Shape1, Shape2)
        input = torch.tensor(input).unsqueeze(0)

        #Apply final transform (Usually Normalisation)
        if self.normalise_transform:
            input = self.normalise_transform(input.float())
        label = input

        return input, label,weight